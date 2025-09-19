import abc
import logging
import queue
import threading
import time
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar, overload

from typing_extensions import override

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

_QUEUE_SENTINEL: Any = object()

_LOGGER = logging.getLogger(__name__)


# region Factory


def create_queue(kind: str) -> "ConsumerQueue[Any]":
    """Factory to create different types of queues based on the specified kind."""
    if kind == "sequential":
        return ImmediateQueue[Any]()
    elif kind == "threading":
        return ThreadingQueue[Any]()
    elif kind == "multiprocessing":
        raise NotImplementedError("MultiprocessingQueue not implemented yet")
    elif kind == "redis":
        raise NotImplementedError("RedisQueue not implemented yet")
    else:
        raise ValueError(f"Unknown queue kind: {kind}")


# region Base Classes


class BaseQueueProducer(abc.ABC, Generic[T_contra]):
    """Abstract base class for queues that support producing items."""

    @abc.abstractmethod
    def put(self, item: T_contra) -> None:
        """Put an item into the queue."""
        ...


class BaseQueueConsumer(abc.ABC, Generic[T_co]):
    """Abstract base class queues that support consuming items."""

    @overload
    def get(self) -> T_co: ...

    @overload
    def get(self, *, timeout: float) -> T_co: ...

    @abc.abstractmethod
    def get(self, *, timeout: float | None = None) -> T_co:
        """Get an item from the queue, wth optional timeout."""
        ...

    @abc.abstractmethod
    @contextmanager
    def consumer(self, callback: Callable[[T_co], None]) -> Iterator[None]:
        """Context manager to consume items from the queue."""
        ...


class ConsumerQueue(BaseQueueProducer[T], BaseQueueConsumer[T]):
    """Abstract base class for queues that support consumer contexts."""

    def close(self) -> None:
        """Close the queue and clean up resources. Optional - not all queues need it."""
        return  # No-op by default


# region Implementations


class ImmediateQueue(ConsumerQueue[T]):
    """Queue implementation that synchronously dispatches values to a consumer."""

    def __init__(self) -> None:
        self._consumer: Callable[[T], None] | None = None

    @override
    @contextmanager
    def consumer(self, callback: Callable[[T], None]) -> Iterator[None]:
        if self._consumer is not None:
            raise RuntimeError("ImmediateQueue already has a registered consumer.")
        self._consumer = callback
        try:
            yield
        finally:
            self._consumer = None


class BaseWorkerQueue(ConsumerQueue[T]):
    """Abstract base class for queues that run a consumer worker in a background thread."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the queue, used for thread naming and logging."""
        ...

    @abc.abstractmethod
    def put(self, item: T) -> None:
        """Put an item into the queue."""
        ...

    @abc.abstractmethod
    def get(self, *, timeout: float | None = None) -> T:
        """Get an item from the queue, with optional timeout."""
        ...


    @override
    @contextmanager
    def consumer(self, callback: Callable[[T], None]) -> Iterator[None]:
        """Context manager to consume items from the queue in a background thread."""
        stop_event = threading.Event()
        context = copy_context()

        def _worker() -> None:
            while not stop_event.is_set():
                try:
                    item = self.get(timeout=0.1)
                    if item is _QUEUE_SENTINEL:  # Deque exit signal
                        break
                    context.run(callback, item)
                except queue.Empty:
                    time.sleep(0.01)  # Small delay to avoid busy
                    continue
                except Exception as e:
                    # Log but don't crash consumer
                    _LOGGER.exception("%s consumer error: %s", self.name, e)

        thread = threading.Thread(target=_worker, name=self.name, daemon=True)
        thread.start()

        try:
            yield
        finally:
            stop_event.set()
            self.put(_QUEUE_SENTINEL)  # Enqueue exit signal
            thread.join()
            self.close()


class ThreadingQueue(BaseWorkerQueue[T]):
    """Wrapper around threading.Queue with consumer context."""

    def __init__(self) -> None:
        self._queue = queue.Queue[T]()

    @property
    @override
    def name(self) -> str:
        return "threading-queue"

    @override
    def put(self, item: T) -> None:
        self._queue.put(item)

    @override
    def get(self, *, timeout: float | None = None) -> T:
        return self._queue.get(timeout=timeout)


class MultiprocessingQueue(BaseWorkerQueue[T]):
    """Wrapper around multiprocessing.Queue with consumer context."""

    def __init__(self) -> None:
        import multiprocessing

        self._queue = multiprocessing.Queue[T]()

    @property
    @override
    def name(self) -> str:
        return "multiprocessing-queue"

    @override
    def put(self, item: T) -> None:
        self._queue.put(item)

    @override
    def get(self, *, timeout: float | None = None) -> T:
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
        """Clean up multiprocessing queue resources."""
        try:
            self._queue.close()
            self._queue.join_thread()
        except Exception:
            pass  # Best effort cleanup


class PickleFileQueue(BaseWorkerQueue[T]):
    """Queue using pickle files in shared directory."""

    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    @property
    @override
    def name(self) -> str:
        return "pickle-file-queue"

    @override
    def put(self, item: T) -> None:
        import pickle
        import time
        import uuid

        # Create unique filename
        filename = f"{int(time.time() * 1000000):020d}_{uuid.uuid4().hex[:8]}.pkl"
        temp_path = self.queue_dir / f"{filename}.tmp"
        final_path = self.queue_dir / filename

        # Atomic write (temp file → rename)
        with temp_path.open("wb") as f:
            pickle.dump(item, f)
        temp_path.rename(final_path)  # Atomic operation

    @override
    def get(self, *, timeout: float | None = None) -> T:
        import pickle

        # Small delay before checking again
        time.sleep(0.01)

        # Get oldest .pkl file
        pkl_files = sorted(self.queue_dir.glob("*.pkl"))
        if not pkl_files:
            raise queue.Empty()

        file_path = pkl_files[0]
        try:
            with file_path.open("rb") as f:
                item = pickle.load(f)
            file_path.unlink()  # Delete processed file
            return item
        except (FileNotFoundError, pickle.UnpicklingError):
            raise queue.Empty()

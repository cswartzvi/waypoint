import abc
import logging
import queue
import threading
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any, Callable, Iterator, Protocol, TypeVar

from typing_extensions import override

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

_QUEUE_SENTINEL = object()

_LOGGER = logging.getLogger(__name__)


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


class ConsumerQueue(Protocol[T]):
    """Protocol for queues that support consumer contexts."""

    @abc.abstractmethod
    def put(self, item: T) -> None:
        """Put an item into the queue."""
        ...

    @abc.abstractmethod
    def get(self) -> T:
        """Get an item from the queue."""
        ...

    @abc.abstractmethod
    @contextmanager
    def consumer(self, callback: Callable[[T], None]) -> Iterator[None]:
        """Context manager to consume items from the queue."""
        ...


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

    @override
    def put(self, item: T) -> None:
        if self._consumer is None:  # pragma: no cover - defensive guard
            raise RuntimeError("ImmediateQueue has no registered consumer.")
        self._consumer(item)

    @override
    def get(self) -> T:  # pragma: no cover - not expected to be called
        raise RuntimeError("ImmediateQueue does not support blocking get operations.")


class ThreadingQueue(ConsumerQueue[T]):
    """Wrapper around threading.Queue with consumer context."""

    def __init__(self) -> None:
        self._queue = queue.Queue[T]()

    @override
    def put(self, item: T) -> None:
        self._queue.put(item)

    @override
    def get(self) -> T:
        return self._queue.get()

    @contextmanager
    def consumer(self, callback: Callable[[T], None]) -> Iterator[None]:
        """Context manager to consume items from the queue in a background thread."""
        stop_event = threading.Event()
        context = copy_context()

        def _worker() -> None:
            while not stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                    if item is _QUEUE_SENTINEL:
                        break
                    context.run(callback, item)
                except queue.Empty:
                    continue
                except Exception as e:
                    # Log but don't crash consumer
                    logging.getLogger(__name__).exception("Consumer error: %s", e)

        thread = threading.Thread(target=_worker, name="queue-consumer", daemon=True)
        thread.start()

        try:
            yield
        finally:
            stop_event.set()
            self._queue.put(_QUEUE_SENTINEL)
            thread.join()

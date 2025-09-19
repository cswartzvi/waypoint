import abc
import logging
import logging.handlers
import queue
import threading
from contextlib import ExitStack
from contextlib import contextmanager
from contextvars import Context
from contextvars import copy_context
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Generic,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    final,
)

from typing_extensions import Self

from waypoint.futures import TaskFuture
from waypoint.logging import LogForwarder, get_logger
from waypoint.logging import iter_waypoint_loggers

R = TypeVar("R")
T_co = TypeVar("T_co")

if TYPE_CHECKING:
    from waypoint.tasks import TaskResultMessage

# NOTE: These are the default task runner types. Custom task runners can use any string
# identifier they want, but these are the ones built into Waypoint.
DefaultTaskRunner = Literal["sequential", "threading", "multiprocessing"]
DefaultTaskRunners: list[DefaultTaskRunner] = ["sequential", "threading", "multiprocessing"]


class EventLike(Protocol):
    """
    A protocol for an event-like object.

    This protocol is used to allow for the use of different event-like objects
    across different concurrency types. For example, the `threading.Event` class
    is used for sequential and concurrent task runners, while the
    `multiprocessing.Event` class is used for parallel task runners.
    """

    def is_set(self) -> bool:
        """Return true if and only if the internal flag is true."""
        ...

    def set(self) -> None:
        """Set the internal flag to true."""
        ...

    def clear(self) -> None:
        """Reset the internal flag to false."""
        ...


def create_runner_queues(
    kind: DefaultTaskRunner | str,
    *,
    log_queue: QueueType | None = None,
    result_queue: QueueType | None = None,
) -> tuple[QueueType, QueueType]:
    """Create log and result queues for the specified runner kind."""

    return (
        log_queue or _create_queue(kind),
        result_queue or _create_queue(kind),
    )



def _close_queue(q: QueueType) -> None:
    close = getattr(q, "close", None)
    if callable(close):  # pragma: no branch - attribute guarded
        close()
    join_thread = getattr(q, "join_thread", None)
    if callable(join_thread):  # pragma: no branch - attribute guarded
        join_thread()


@contextmanager
def _queue_consumer_context(
    q: QueueType,
    callback: Callable[[Any], None],
    *,
    name: str,
    close_on_exit: bool = True,
    context: Context | None = None,
) -> Iterator[None]:
    if isinstance(q, ImmediateQueue):
        with q.consumer(callback):
            yield
        return

    stop_event = threading.Event()
    run_context = context or copy_context()

    def _worker() -> None:
        while not stop_event.is_set():
            item = q.get()
            if item is _QUEUE_SENTINEL:
                break
            run_context.run(callback, item)

    thread = threading.Thread(target=_worker, name=name, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        q.put(_QUEUE_SENTINEL)
        thread.join()
        if close_on_exit:
            _close_queue(q)


class BaseTaskRunner(metaclass=abc.ABCMeta):
    """
    An abstract base class for task runners.

    Task runners are responsible for submitting tasks, as well as managing the state of
    submitted task futures. When applicable, they are also responsible for serializing
    and deserializing the results of tasks.
    """

    type: ClassVar[str]  # Used for looking up task runners by type
    """A string identifier for the type of task runner."""

    def __init__(self) -> None:
        from waypoint.hooks.manager import get_hook_manager

        self._started: bool = False
        self.logger = get_logger(f"task_runner.{self.name}")
        self._hook_manager = get_hook_manager()
        self._log_queue: QueueType | None = None
        self._result_queue: QueueType | None = None

    @final
    @property
    def is_running(self) -> bool:
        """Returns true if the task runner is running."""
        return self._started

    @final
    @property
    def name(self):
        """The name of the task runner."""
        return type(self).__name__.lower().replace("taskrunner", "")

    @abc.abstractmethod
    def duplicate(self) -> Self:
        """
        Create a duplicate of the task runner.

        This is used to create a new instance of the task runner with the same
        configuration as the original.
        """
        raise NotImplementedError

    @final
    @contextmanager
    def start(self) -> Iterator[Self]:
        """
        Start the task runner, preparing any resources necessary for task submission.

        Sub classes should implement `_start` to prepare and clean up resources.
        """
        if self._started:
            raise RuntimeError("The task runner has already been started.")

        with ExitStack() as stack:
            try:
                self._started = True
                self._setup_context(stack)
                yield self
            finally:
                self._started = False
                self._log_queue = None
                self._result_queue = None

    def _setup_context(self, stack: ExitStack) -> None:
        """Set up queue consumers and log forwarding for the task runner."""
        log_queue, result_queue = create_runner_queues(self.type)
        self._log_queue = log_queue
        self._result_queue = result_queue
        stack.enter_context(LogForwarder(log_queue))
        stack.enter_context(
            _queue_consumer_context(
                result_queue,
                self._process_task_result,
                name=f"{self.name}-result-consumer",
                close_on_exit=True,
            )
        )

    @property
    def log_queue(self) -> QueueType:
        if self._log_queue is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Task runner has not been started.")
        return self._log_queue

    @property
    def result_queue(self) -> QueueType:
        if self._result_queue is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Task runner has not been started.")
        return self._result_queue

    @final
    def submit(self, func: Callable[[], Any]) -> TaskFuture[Any]:
        """
        Submit a task to the task runner.

        Subclasses should implement `_submit` to handle the actual submission logic.

        Args:
            func (Callable[[None], Any]): Parameter-less function to be executed.

        Returns:
            A future representing the result of the task execution.
        """
        if not self._started:
            raise RuntimeError("The task runner has not been started.")

        return self._submit(func)

    @abc.abstractmethod
    def _submit(self, func: Callable[[], Any]) -> TaskFuture[Any]:
        """
        Submit a task to the task runner.

        Args:
            func (Callable[[None], Any]): Parameter-less function to be executed.

        Returns:
            A future representing the result of the task execution.
        """
        # NOTE: This method is intentionally not generic over R to avoid complicating
        # the type signature of task runners. Instead, we cast in `submit`.
        raise NotImplementedError

    def enqueue_completed_task(self, message: "TaskResultMessage") -> None:
        """Enqueue a completed task message for processing on the caller thread."""
        self.result_queue.put(message)

    def _process_task_result(self, message: "TaskResultMessage") -> None:
        from waypoint.tasks import process_task_result

        process_task_result(message)


@contextmanager
def log_execution(name: str, logger: logging.Logger):
    """Helper context manager to log task runner execution in `setup_context`."""
    logger.debug("Initializing '%s' task runner", name)
    yield
    logger.debug("Shutting down '%s' task runner", name)

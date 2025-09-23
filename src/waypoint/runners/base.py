import abc
from contextlib import ExitStack
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    final,
)

from typing_extensions import Self

from waypoint.futures import TaskFuture
from waypoint.logging import get_logger

if TYPE_CHECKING:
    from waypoint.utils.queues import ConsumerQueue
else:
    ConsumerQueue = Any

R = TypeVar("R")
T_co = TypeVar("T_co")


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
                self._result_queue = None

    def _setup_context(self, stack: ExitStack) -> None:
        """Set up queue consumers and log forwarding for the task runner."""

        # TODO: Add consumer queues for results

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

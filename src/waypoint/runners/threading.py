import asyncio
import concurrent.futures
import inspect
import queue
from contextlib import ExitStack
from contextlib import contextmanager
from contextvars import Context
from contextvars import copy_context
from threading import Event
from typing import Any, Callable, ClassVar, TypeVar
from uuid import UUID
from uuid import uuid4

from typing_extensions import Self, override

from waypoint.exceptions import TaskRunError
from waypoint.futures import TaskFuture
from waypoint.logging import setup_listener_context
from waypoint.logging import setup_worker_logging
from waypoint.runners.base import BaseTaskRunner
from waypoint.runners.base import DefaultTaskRunners
from waypoint.runners.base import EventLike

R = TypeVar("R")


class ThreadingTaskRunner(BaseTaskRunner):
    """
    An executor that runs tasks in a separate thread.

    This task runner uses a thread pool to execute tasks concurrently. It is suitable for
    I/O-bound tasks that can benefit from concurrency.

    Args:
        max_workers (int):
            The maximum number of threads to use in the thread pool. If None, the default
            value will be used (typically the number of processors on the machine, multiplied
            by 5).
        initializer (Callable[..., Any], optional):
            A callable that will be called once for each worker thread when it is started. If
            `initargs` is provided, it will be passed to the initializer as positional arguments,
            otherwise it will be called with no arguments.
        initargs (tuple[Any, ...], optional):
            A tuple of arguments that will be passed to the initializer as positional arguments.
    """

    type: ClassVar[str] = DefaultTaskRunners[1]

    def __init__(
        self,
        max_workers: int | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ):
        """Initialize the threading task runner."""
        super().__init__()
        self._raw_futures: dict[UUID, concurrent.futures.Future[Any]] = {}
        self._max_workers = max_workers
        self._initializer = initializer
        self._initargs = initargs or ()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._event: EventLike | None = None
        self._log_queue = queue.Queue[Any]()

    @override
    def duplicate(self) -> Self:
        """Create a duplicate of the task runner."""
        return type(self)(max_workers=self._max_workers)

    @override
    def _setup_context(self, stack: ExitStack) -> None:
        super()._setup_context(stack)
        stack.enter_context(self._log_context())
        stack.enter_context(setup_listener_context(self._log_queue))
        self._event = Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            initializer=self._initializer,
            initargs=self._initargs,
        )
        stack.enter_context(self._executor)
        stack.enter_context(self._raw_futures_context())

    @override
    def _submit(self, func: Callable[[], Any]) -> TaskFuture[Any]:
        if self._executor is None:  # pragma: no cover
            raise RuntimeError("The concurrent task runner cannot be used without an executor.")

        if self._event is None:  # pragma: no cover
            raise RuntimeError(
                "The concurrent task runner cannot be used without an event manager."
            )

        if self._event.is_set():  # pragma: no cover
            # Error has already occurred, don't submit any more work
            raise TaskRunError("The concurrent task runner is shutting down.")

        # NOTE: Unlike multiprocessing, Python threads share the same process. Therefore we can
        # simply run the task in the current context (e.q. no need to pass context to the engine).
        context = copy_context()

        run_in_thread_args: list[Any] = [_run_in_thread, context, self._log_queue]
        if inspect.iscoroutinefunction(func):
            run_in_thread_args.extend((asyncio.run, func()))
        else:
            run_in_thread_args.append(func)

        future = self._executor.submit(*run_in_thread_args)
        task_future = TaskFuture(future)
        key = uuid4()
        self._raw_futures[key] = future

        return task_future

    @contextmanager
    def _log_context(self):
        self.logger.debug("Initializing '%s' task runner", self.name)
        yield
        self.logger.debug("Shutting down '%s' task runner", self.name)

    @contextmanager
    def _raw_futures_context(self):
        self._raw_futures.clear()
        yield
        concurrent.futures.wait(self._raw_futures.values())


def _run_in_thread(context: Context, log_queue: queue.Queue, *args) -> Any:
    setup_worker_logging(log_queue)
    return context.run(*args)

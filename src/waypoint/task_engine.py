from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import AsyncExitStack
from contextlib import ExitStack
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterator,
    ParamSpec,
    TypeVar,
    cast,
)

from waypoint.context import TaskRunContext
from waypoint.hooks.manager import get_hook_manager
from waypoint.task_run import TaskRun
from waypoint.tasks import TaskData
from waypoint.utils.callables import call_with_arguments
from waypoint.utils.collections import aenumerate

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:
    from pluggy import PluginManager
else:
    PluginManager = object


# region API


def run_task_sync(task_function: Callable[P, R], task_data: TaskData, task_run: TaskRun) -> R:
    """
    Run a task synchronously.

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        The result of the task execution.
    """
    engine = SyncTaskRunEngine(task_function=task_function, task_data=task_data, task_run=task_run)
    with engine.initialize():
        return engine.call()


def run_generator_task_sync(
    task_function: Callable[P, Generator[R, None, None]],
    task_data: TaskData,
    task_run: TaskRun,
    _log_iterations: bool = True,
) -> Generator[R, None, None]:
    """
    Run a task that is a synchronous generator (yield-only).

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        A generator that yields results from the task execution.
    """
    engine = SyncGeneratorTaskRunEngine(
        task_function=task_function,
        task_data=task_data,
        task_run=task_run,
        _log_iterations=_log_iterations,
    )
    with engine.initialize():
        yield from engine.call()


def consume_generator_task_sync(
    task_function: Callable[P, Generator[R, None, None]], task_data: TaskData, task_run: TaskRun
) -> list[R]:
    """
    Consumes an asynchronous generator task and return all results as a list.

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        A list contains all results of the asynchronous generator.
    """
    return list(
        run_generator_task_sync(
            task_function=task_function,
            task_data=task_data,
            task_run=task_run,
            _log_iterations=False,
        )
    )


async def run_task_async(
    task_function: Callable[P, Coroutine[Any, Any, R]], task_data: TaskData, task_run: TaskRun
) -> R:
    """
    Run a task asynchronously.

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        A coroutine that resolves to the result of the task execution.
    """
    engine = AsyncTaskRunEngine(task_function=task_function, task_data=task_data, task_run=task_run)
    async with engine.initialize():
        return await engine.call_task_fn()


async def run_generator_task_async(
    task_function: Callable[P, AsyncGenerator[R, None]],
    task_data: TaskData,
    task_run: TaskRun,
    _log_iterations: bool = True,
) -> AsyncGenerator[R, None]:
    """
    Run a task that is an asynchronous generator (yield-only).

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        An async generator that yields results from the task execution.
    """
    engine = AsyncGeneratorTaskRunEngine(
        task_function=task_function,
        task_data=task_data,
        task_run=task_run,
        _log_iterations=_log_iterations,
    )
    async with engine.initialize():
        async for item in engine.call():
            yield item


async def consume_generator_task_async(
    task_function: Callable[P, AsyncGenerator[R, None]], task_data: TaskData, task_run: TaskRun
) -> list[R]:
    """
    Consumes an asynchronous generator task and return all results as a list.

    Args:
        task_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a task.
        task_data (waypoint.tasks.TaskData): Metadata related to the task being executed.
        task_run (waypoint.task_run.TaskRun): Execution data related to current task run.

    Returns:
        A list contains all results of the asynchronous generator.
    """
    results = []
    async for item in run_generator_task_async(
        task_function=task_function, task_data=task_data, task_run=task_run, _log_iterations=False
    ):
        results.append(item)
    return results


# region Engine: Base


@dataclass
class _BaseTaskRunEngine(Generic[P, R]):
    """Base class for task run engines."""

    task_function: Callable[P, R]
    task_data: TaskData
    task_run: TaskRun

    # The following attributes are set during initialization
    initialized: bool = field(init=False, default=False)

    @cached_property
    def _hook_manager(self) -> PluginManager:
        return get_hook_manager()

    def _run_hook(self, hook_name: str, **kwargs: Any) -> None:
        """
        Try to run a specified hook, with the provided arguments if it exists.

        Will automatically add `task_data` and `task_run` to the arguments if not provided.
        """
        from waypoint.hooks.manager import try_run_hook

        kwargs.setdefault("task_data", self.task_data)
        kwargs.setdefault("task_run", self.task_run)
        try_run_hook(self._hook_manager, hook_name, **kwargs)


# region Engine: Sync


@dataclass
class _BaseSyncTaskRunEngine(_BaseTaskRunEngine[P, R]):
    """Base class for synchronous task run engines."""

    @contextmanager
    def initialize(self) -> Iterator[None]:
        """Initialize the task run engine, sets attributes and validates state."""
        with ExitStack() as stack:
            task_run_context = TaskRunContext(task_data=self.task_data, task_run=self.task_run)
            stack.enter_context(task_run_context)

            self.initialized = True
            yield

        self.initialized = False

    def process(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the task.

        Args:
            result (Any): Result of the task iteration.
            iteration (int, optional): Current iteration index.
        """
        pass


@dataclass
class SyncTaskRunEngine(_BaseSyncTaskRunEngine[P, R]):
    """Synchronous task run engine for blocking task execution."""

    def call(self) -> R:
        """Calls the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task engine must be initialized before starting context.")

        self._run_hook("before_task_run")
        result, error = None, None
        try:
            result = call_with_arguments(self.task_function, self.task_run.parameters)
        except Exception as exception:
            error = exception
        finally:
            self._run_hook("after_task_run", result=result, error=error)
            if error:
                raise error from None

        self.process(result)
        return cast(R, result)


@dataclass
class SyncGeneratorTaskRunEngine(_BaseSyncTaskRunEngine[P, Generator[R, None, None]]):
    """Synchronous generator (yield-only) task run engine for blocking task execution."""

    _log_iterations: bool = True

    def call(self) -> Generator[R, None, None]:
        """Calls the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task engine must be initialized before starting context.")

        self._run_hook("before_task_run")
        error = None
        try:
            items = call_with_arguments(self.task_function, self.task_run.parameters)
            for idx, item in enumerate(items):
                if self._log_iterations:
                    self._run_hook("after_task_iteration", result=item, index=idx)
                self.process(item, iteration=idx)
                yield item
        except Exception as exception:
            error = exception
        finally:
            self._run_hook("after_task_run", result=None, error=error)
            if error:
                raise error from None


# region Engine: Async


@dataclass
class _BaseAsyncTaskRunEngine(_BaseTaskRunEngine[P, R]):
    """Base class for asynchronous task run engines."""

    @asynccontextmanager
    async def initialize(self) -> AsyncGenerator[None, None]:
        """Initialize the task run engine, sets attributes and validates state."""
        async with AsyncExitStack() as stack:
            task_run_context = TaskRunContext(task_data=self.task_data, task_run=self.task_run)
            stack.enter_context(task_run_context)

            self.initialized = True
            yield

        self.initialized = False

    async def process(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the task.

        Args:
            result (Any): Result of the task iteration.
            iteration (int): Current iteration index.
        """
        pass


@dataclass
class AsyncTaskRunEngine(_BaseAsyncTaskRunEngine[P, Coroutine[Any, Any, R]]):
    """Asynchronous task run engine for non-blocking task execution."""

    async def call_task_fn(self) -> R:
        """Calls the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task run engine must be initialized before starting context.")

        self._run_hook("before_task_run")
        result, error = None, None
        try:
            result = await call_with_arguments(self.task_function, self.task_run.parameters)
        except Exception as exception:
            error = exception
        finally:
            self._run_hook("after_task_run", result=result, error=error)
            if error:
                raise error from None
        await self.process(result)
        return cast(R, result)


@dataclass
class AsyncGeneratorTaskRunEngine(_BaseAsyncTaskRunEngine[P, AsyncGenerator[R, None]]):
    """Asynchronous generator task run engine for non-blocking task execution."""

    _log_iterations: bool = True

    async def call(self) -> AsyncGenerator[R, None]:
        """Calls the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task run engine must be initialized before starting context.")

        self._run_hook("before_task_run")
        error = None
        try:
            items = call_with_arguments(self.task_function, self.task_run.parameters)
            async for idx, item in aenumerate(items):
                if self._log_iterations:
                    self._run_hook("after_task_iteration", result=item, index=idx)
                await self.process(item, iteration=idx)
                yield item
        except Exception as exc:
            error = exc
        finally:
            self._run_hook("after_task_run", result=None, error=error)
            if error:
                raise error from None

from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import AsyncExitStack
from contextlib import ExitStack
from contextlib import _BaseExitStack
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from logging import Logger
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

from waypoint.context import FlowRunContext
from waypoint.context import TaskRunContext
from waypoint.exceptions import TaskRunError
from waypoint.hooks.manager import get_hook_manager
from waypoint.logging import get_run_logger
from waypoint.task_run import TaskRun
from waypoint.tasks import TaskData
from waypoint.utils.callables import call_with_arguments
from waypoint.utils.collections import aenumerate
from waypoint.utils.timing import format_duration

P = ParamSpec("P")
R = TypeVar("R")

_MISSING = object()

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
    with engine.setup_run_context():
        return engine.run()


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
    with engine.setup_run_context():
        yield from engine.run()


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
    async with engine.setup_run_context():
        return await engine.run()


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
    async with engine.setup_run_context():
        async for item in engine.run():
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

    initialized: bool = field(init=False, default=False)
    _hook_manager: PluginManager = field(init=False, default_factory=get_hook_manager)

    # NOTE: Logger for the current run context prior to initialization
    _logger: Logger = field(init=False, default_factory=lambda: get_run_logger("engine"))

    def _run_hook(self, hook_name: str, **kwargs: Any) -> None:
        """
        Try to run a specified hook, with the provided arguments if it exists.

        Will automatically add `task_data` and `task_run` to the arguments if not provided.
        """
        from waypoint.hooks.manager import try_run_hook

        kwargs.setdefault("task_data", self.task_data)
        kwargs.setdefault("task_run", self.task_run)
        try_run_hook(manager=self._hook_manager, hook_name=hook_name, **kwargs)

    @contextmanager
    def _initialize_run(self, stack: _BaseExitStack) -> Iterator[None]:
        """Adds items to the given context stack to initialize the task run."""
        task_run_context = TaskRunContext(task_data=self.task_data, task_run=self.task_run)
        stack.enter_context(task_run_context)

        try:
            self._run_hook("before_task_run")
            self.task_run.start_time = datetime.now()
            self._logger.debug("Beginning task run %s", self.task_run.task_id)
            self.initialized = True
            yield

        except Exception as error:
            self._run_hook("after_task_run", result=None, error=error)
            self._logger.error("Task run %s failed with error: %s", self.task_run.task_id, error)
            raise TaskRunError(self.task_run.task_id, error) from None
        else:
            # NOTE: When successful, the after_task_run hook should be called in the run
            # method to ensure it is with the final result of the task.
            self.task_run.end_time = datetime.now()
            duration = format_duration(self.task_run.start_time, self.task_run.end_time)
            self._logger.info("Completed task run %s in %s", self.task_run.task_id, duration)

        self.initialized = False


# region Engine: Sync


@dataclass
class _BaseSyncTaskRunEngine(_BaseTaskRunEngine[P, R]):
    """Base class for synchronous task run engines."""

    @contextmanager
    def setup_run_context(self) -> Iterator[None]:
        """Initialize the task run engine, sets attributes and validates state."""
        with ExitStack() as stack:
            stack.enter_context(self._initialize_run(stack))
            yield

    def process_result(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the task.

        Args:
            result (Any): Result of the task iteration.
            iteration (int, optional): Current iteration index.
        """
        if iteration is not None:
            return

        mapper = self.task_data.mapper
        if mapper is None:
            return

        flow_context = FlowRunContext.get()
        if flow_context is None or flow_context.asset_store is None:  # pragma: no cover
            return

        key = mapper.save(result, store=flow_context.asset_store)
        logger = get_run_logger()
        logger.info("Saved result to asset store '%s'", key)


@dataclass
class SyncTaskRunEngine(_BaseSyncTaskRunEngine[P, R]):
    """Synchronous task run engine for blocking task execution."""

    def run(self) -> R:
        """Runs the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task engine must be initialized before starting context.")

        result = call_with_arguments(self.task_function, self.task_run.parameters)
        self._run_hook("after_task_run", result=result, error=None)
        self.process_result(result)
        return cast(R, result)


@dataclass
class SyncGeneratorTaskRunEngine(_BaseSyncTaskRunEngine[P, Generator[R, None, None]]):
    """Synchronous generator (yield-only) task run engine for blocking task execution."""

    _log_iterations: bool = True

    def run(self) -> Generator[R, None, None]:
        """Runs the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task engine must be initialized before starting context.")

        # NOTE: Logger and intermediate time must be refresh to ensure correct task context
        logger = get_run_logger()
        intermediate = datetime.now()

        items = call_with_arguments(self.task_function, self.task_run.parameters)
        for idx, item in enumerate(items):
            self.process_result(item, iteration=idx)
            end_time = datetime.now()
            duration = format_duration(intermediate, end_time)
            yield item
            self._run_hook("after_task_iteration", result=item, index=idx)
            logger.info("Completed iteration %d in %s", idx, duration)
            intermediate = end_time

        self._run_hook("after_task_run", result=None, error=None)


# region Engine: Async


@dataclass
class _BaseAsyncTaskRunEngine(_BaseTaskRunEngine[P, R]):
    """Base class for asynchronous task run engines."""

    @asynccontextmanager
    async def setup_run_context(self) -> AsyncGenerator[None, None]:
        """Initialize the task run engine, sets attributes and validates state."""
        async with AsyncExitStack() as stack:
            stack.enter_context(self._initialize_run(stack))
            yield

    async def process_result(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the task.

        Args:
            result (Any): Result of the task iteration.
            iteration (int): Current iteration index.
        """
        if iteration is not None:
            return

        mapper = self.task_data.mapper
        if mapper is None:
            return

        flow_context = FlowRunContext.get()
        if flow_context is None or flow_context.asset_store is None:  # pragma: no cover
            return

        key = mapper.save(result, store=flow_context.asset_store)
        logger = get_run_logger()
        logger.info("Saved result to asset store '%s'", key)


@dataclass
class AsyncTaskRunEngine(_BaseAsyncTaskRunEngine[P, Coroutine[Any, Any, R]]):
    """Asynchronous task run engine for non-blocking task execution."""

    async def run(self) -> R:
        """Runs the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task run engine must be initialized before starting context.")

        result = await call_with_arguments(self.task_function, self.task_run.parameters)
        self._run_hook("after_task_run", result=result, error=None)
        await self.process_result(result)
        return cast(R, result)


@dataclass
class AsyncGeneratorTaskRunEngine(_BaseAsyncTaskRunEngine[P, AsyncGenerator[R, None]]):
    """Asynchronous generator task run engine for non-blocking task execution."""

    _log_iterations: bool = True

    async def run(self) -> AsyncGenerator[R, None]:
        """Runs the task function with the provided parameters."""
        if not self.initialized:  # pragma: no cover
            raise RuntimeError("Task run engine must be initialized before starting context.")

        # NOTE: Logger and intermediate time must be refresh to ensure correct task context
        logger = get_run_logger()
        intermediate = datetime.now()

        items = call_with_arguments(self.task_function, self.task_run.parameters)
        async for idx, item in aenumerate(items):
            await self.process_result(item, iteration=idx)
            end_time = datetime.now()
            duration = format_duration(intermediate, end_time)
            yield item
            self._run_hook("after_task_iteration", result=item, index=idx)
            logger.info("Completed iteration %d in %s", idx, duration)
            intermediate = end_time

        self._run_hook("after_task_run", result=None, error=None)

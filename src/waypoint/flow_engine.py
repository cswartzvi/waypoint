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
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Generic,
    Iterator,
    ParamSpec,
    TypeVar,
    cast,
)

from waypoint.context import FlowRunContext
from waypoint.context import TaskRunContext
from waypoint.exceptions import FlowRunError
from waypoint.flow_run import FlowRun
from waypoint.flows import FlowData
from waypoint.hooks.manager import get_hook_manager
from waypoint.logging import get_run_logger
from waypoint.runners import get_task_runner
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


def run_flow_sync(flow_function: Callable[P, R], flow_data: FlowData, flow_run: FlowRun) -> R:
    """
    Run a flow synchronously.

    Args:
        flow_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a workflow.
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.

    Returns:
        The result of the flow execution.
    """
    engine = _SyncFunctionFlowRunEngine(
        flow_function=flow_function, flow_data=flow_data, flow_run=flow_run
    )
    with engine.setup_run_context():
        return engine.run()


def run_generator_flow_sync(
    flow_function: Callable[P, Generator[R, None, None]], flow_data: FlowData, flow_run: FlowRun
) -> Generator[R, None, None]:
    """
    Run a flow that is a synchronous generator (yield-only).

    Args:
        flow_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a workflow.
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.

    Returns:
        A generator that yields results from the flow execution.
    """
    engine = SyncGeneratorFlowRunEngine(
        flow_function=flow_function, flow_data=flow_data, flow_run=flow_run
    )
    with engine.setup_run_context():
        yield from engine.run()


async def run_flow_async(
    flow_function: Callable[P, Coroutine[Any, Any, R]], flow_data: FlowData, flow_run: FlowRun
) -> Coroutine[Any, Any, R]:
    """
    Run a flow asynchronously.

    Args:
        flow_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a workflow.
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.

    Returns:
        A coroutine that resolves to the result of the flow execution.
    """
    engine = _AsyncFunctionFlowRunEngine(
        flow_data=flow_data, flow_function=flow_function, flow_run=flow_run
    )
    async with engine.setup_run_context():
        return await engine.run()


async def run_generator_flow_async(
    flow_function: Callable[P, AsyncGenerator[R, None]], flow_data: FlowData, flow_run: FlowRun
) -> AsyncGenerator[R, None]:
    """
    Run a flow that is an asynchronous generator (yield-only).

    Args:
        flow_function (Callable[P, AsyncGenerator[R, None]]): Callable to execute as a workflow.
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.

    Returns:
        An async generator that yields results from the flow execution.
    """
    engine = _AsyncGeneratorFlowRunEngine(
        flow_data=flow_data, flow_function=flow_function, flow_run=flow_run
    )
    async with engine.setup_run_context():
        async for item in engine.run():
            yield item


@contextmanager
def create_flow_session(flow_data: FlowData, flow_run: FlowRun) -> Iterator[None]:
    """
    Context manager to create a flow run context without executing a flow.

    Args:
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.
    """
    task_runner = get_task_runner(flow_data.task_runner).duplicate()

    with task_runner.start():
        asset_store = flow_data.asset_store.duplicate()
        with FlowRunContext(
            flow_data=flow_data,
            flow_run=flow_run,
            task_runner=task_runner,
            asset_store=asset_store,
        ):
            yield


# region Engine: Base


@dataclass
class BaseFlowRunEngine(Generic[P, R]):
    """Base class for flow run engines."""

    flow_function: Callable[P, R]
    flow_data: FlowData
    flow_run: FlowRun

    _initialized: bool = field(init=False, default=False)
    _hook_manager: PluginManager = field(init=False, default_factory=get_hook_manager)

    # NOTE: Logger for the current run context prior to initialization
    _logger: Logger = field(init=False, default_factory=lambda: get_run_logger("engine"))

    def _run_hook(self, hook_name: str, **kwargs: Any) -> None:
        """
        Try to run a specified hook, with the provided arguments if it exists.

        Will automatically add `flow_data` and `flow_run` to the arguments if not provided.
        """
        from waypoint.hooks.manager import try_run_hook

        kwargs.setdefault("flow_data", self.flow_data)
        kwargs.setdefault("flow_run", self.flow_run)
        try_run_hook(manager=self._hook_manager, hook_name=hook_name, **kwargs)

    @contextmanager
    def _initialize_run(self, stack: _BaseExitStack) -> Iterator[None]:
        """Adds items to the given context stack to initialize the flow run."""
        # NOTE: Task runner is duplicated to ensure a the original definition is preserved
        task_runner = self.flow_data.task_runner.duplicate()
        stack.enter_context(task_runner.start())

        asset_store = self.flow_data.asset_store.duplicate()

        flow_run_context = FlowRunContext(
            flow_data=self.flow_data,
            flow_run=self.flow_run,
            task_runner=task_runner,
            asset_store=asset_store,
        )
        stack.enter_context(flow_run_context)

        self._logger.info("Beginning flow run %s", self.flow_run.flow_id)

        try:
            self._run_hook("before_flow_run")
            self.flow_run.start_time = datetime.now()
            self._initialized = True
            yield

        except Exception as err:
            self.flow_run.end_time = datetime.now()
            self._run_hook("after_flow_run", result=None, error=err)
            self._logger.error("Flow run %s failed with error: %s", self.flow_run.flow_id, err)
            raise FlowRunError(str(self.flow_run.flow_id), err) from None
        else:
            # NOTE: When successful, the after_flow_run hook should be called in the call()
            # method to ensure it is with the final result of the flow.
            self.flow_run.end_time = datetime.now()
            duration = format_duration(self.flow_run.start_time, self.flow_run.end_time)
            self._logger.info("Completed flow run %s in %s", self.flow_run.flow_id, duration)

        self._initialized = False


# region Engine: Sync


@dataclass
class _BaseSyncFlowRunEngine(BaseFlowRunEngine[P, R]):
    """Base class for synchronous flow run engines."""

    @contextmanager
    def setup_run_context(self, is_session: bool = False) -> Iterator[None]:
        """Setup the flow run engine, sets attributes and validates state."""
        if TaskRunContext.get():
            raise RuntimeError("Cannot start a flow run context within a task.")

        with ExitStack() as stack:
            stack.enter_context(self._initialize_run(stack))
            yield

    def process_result(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the flow.

        Args:
            result (Any): Result of the flow iteration.
            iteration (int, optional): Current iteration index (if applicable).
        """
        if iteration is not None:
            return

        mapper = self.flow_data.mapper
        if mapper is None:
            return

        flow_context = FlowRunContext.get()
        if flow_context is None or flow_context.asset_store is None:  # pragma: no cover
            return

        mapper.save(result, store=flow_context.asset_store)


@dataclass
class _SyncFunctionFlowRunEngine(_BaseSyncFlowRunEngine[P, R]):
    """Synchronous flow run engine for blocking flow execution."""

    def run(self) -> R:
        """Runs the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow engine must be initialized before starting context.")

        result = call_with_arguments(self.flow_function, self.flow_run.parameters)
        self._run_hook("after_flow_run", result=result, error=None)
        self.process_result(result)
        return cast(R, result)


@dataclass
class SyncGeneratorFlowRunEngine(_BaseSyncFlowRunEngine[P, Generator[R, None, None]]):
    """Synchronous generator (yield-only) flow run engine for blocking flow execution."""

    def run(self) -> Generator[R, None, None]:
        """Runs the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow engine must be initialized before starting context.")

        # NOTE: Logger and intermediate time must be refresh to ensure correct flow context
        logger = get_run_logger()
        intermediate = datetime.now()

        items = call_with_arguments(self.flow_function, self.flow_run.parameters)
        for idx, item in enumerate(items):
            self.process_result(item, iteration=idx)
            end_time = datetime.now()
            duration = format_duration(intermediate, end_time)
            yield item
            self._run_hook("after_flow_iteration", result=item, index=idx)
            logger.info("Completed iteration %d in %s", idx, duration)
            intermediate = end_time

        self._run_hook("after_flow_run", result=None, error=None)


# region Engine: Async


@dataclass
class _BaseAsyncFlowRunEngine(BaseFlowRunEngine[P, R]):
    """Base class for asynchronous flow run engines."""

    @asynccontextmanager
    async def setup_run_context(self, is_session: bool = False) -> AsyncGenerator[None, None]:
        """Initialize the flow run engine, sets attributes and validates state."""
        if TaskRunContext.get():
            raise RuntimeError("Cannot start a flow run context within a task.")

        async with AsyncExitStack() as stack:
            stack.enter_context(self._initialize_run(stack))
            yield

    async def process_result(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the flow.

        Args:
            result (Any): Result of the flow iteration.
            iteration (int, optional): Current iteration index (if applicable).
        """
        if iteration is not None:
            return

        mapper = self.flow_data.mapper
        if mapper is None:
            return

        flow_context = FlowRunContext.get()
        if flow_context is None or flow_context.asset_store is None:  # pragma: no cover
            return

        mapper.save(result, store=flow_context.asset_store)


@dataclass
class _AsyncFunctionFlowRunEngine(_BaseAsyncFlowRunEngine[P, Coroutine[Any, Any, R]]):
    """Asynchronous flow run engine for non-blocking flow execution."""

    async def run(self) -> Coroutine[Any, Any, R]:
        """Runs the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow run engine must be initialized before starting context.")

        result = await call_with_arguments(self.flow_function, self.flow_run.parameters)
        self._run_hook("after_flow_run", result=result, error=None)
        await self.process_result(result)
        return cast(Coroutine[Any, Any, R], result)


@dataclass
class _AsyncGeneratorFlowRunEngine(_BaseAsyncFlowRunEngine[P, AsyncGenerator[R, None]]):
    """Asynchronous generator (yield-only) flow run engine for non-blocking flow execution."""

    async def run(self) -> AsyncGenerator[R, None]:
        """Runs the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow run engine must be initialized before starting context.")

        # NOTE: Logger and intermediate time must be refresh to ensure correct flow context
        logger = get_run_logger()
        intermediate = datetime.now()

        items = call_with_arguments(self.flow_function, self.flow_run.parameters)
        async for idx, item in aenumerate(items):
            await self.process_result(item, iteration=idx)
            end_time = datetime.now()
            duration = format_duration(intermediate, end_time)
            yield item
            self._run_hook("after_flow_iteration", result=item, index=idx)
            logger.info("Completed iteration %d in %s", idx, duration)
            intermediate = end_time

        self._run_hook("after_flow_run", result=None, error=None)

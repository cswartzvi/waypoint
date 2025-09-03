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
from waypoint.utils.callables import call_with_arguments
from waypoint.utils.collections import aenumerate

P = ParamSpec("P")
R = TypeVar("R")

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
    with engine.initialize():
        return engine.call()


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
    with engine.initialize():
        yield from engine.call()


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
    async with engine.initialize():
        return await engine.call()


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
    async with engine.initialize():
        async for item in engine.call():
            yield item


@contextmanager
def create_flow_session(flow_data: FlowData, flow_run: FlowRun) -> Iterator[None]:
    """
    Context manager to create a flow run context without executing a flow.

    Args:
        flow_data (waypoint.flows.FlowData): Metadata related to the workflow being executed.
        flow_run (waypoint.flow_run.FlowRun): Execution data related to current flow run.
    """
    engine = _BaseSyncFlowRunEngine(lambda: None, flow_data=flow_data, flow_run=flow_run)
    with engine.initialize():
        yield


# region Engine: Base


@dataclass
class BaseFlowRunEngine(Generic[P, R]):
    """Base class for flow run engines."""

    flow_function: Callable[P, R]
    flow_data: FlowData
    flow_run: FlowRun

    _initialized: bool = field(init=False, default=False)
    """Whether the engine has been initialized."""

    @cached_property
    def _hook_manager(self) -> PluginManager:
        return get_hook_manager()

    def _run_hook(self, hook_name: str, **kwargs: Any) -> None:
        """
        Try to run a specified hook, with the provided arguments if it exists.

        Will automatically add `flow_data` and `flow_run` to the arguments if not provided.
        """
        from waypoint.hooks.manager import try_run_hook

        kwargs.setdefault("flow_data", self.flow_data)
        kwargs.setdefault("flow_run", self.flow_run)
        try_run_hook(manager=self._hook_manager, hook_name=hook_name, **kwargs)


# region Engine: Sync


@dataclass
class _BaseSyncFlowRunEngine(BaseFlowRunEngine[P, R]):
    """Base class for synchronous flow run engines."""

    @contextmanager
    def initialize(self) -> Iterator[None]:
        """Initialize the flow run engine, sets attributes and validates state."""
        if TaskRunContext.get():
            raise RuntimeError("Cannot start a flow run context within a task.")

        with ExitStack() as stack:
            # NOTE: Task runner is duplicated to ensure a the original definition is preserved
            task_runner = self.flow_data.task_runner.duplicate()
            stack.enter_context(task_runner.start())

            flow_run_context = FlowRunContext(
                flow_data=self.flow_data,
                flow_run=self.flow_run,
                task_runner=task_runner,
            )
            stack.enter_context(flow_run_context)

            self._initialized = True
            yield

        self._initialized = False

    def process(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the flow.

        Args:
            result (Any): Result of the flow iteration.
            iteration (int, optional): Current iteration index (if applicable).
        """
        pass


@dataclass
class _SyncFunctionFlowRunEngine(_BaseSyncFlowRunEngine[P, R]):
    """Synchronous flow run engine for blocking flow execution."""

    def call(self) -> R:
        """Calls the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow engine must be initialized before starting context.")

        self._run_hook("before_flow_run")

        result, error = None, None
        try:
            result = call_with_arguments(self.flow_function, self.flow_run.parameters)
        except Exception as exception:
            error = exception
        finally:
            self._run_hook("after_flow_run", result=result, error=error)
            if error:
                raise error from None

        self.process(result)
        return cast(R, result)


@dataclass
class SyncGeneratorFlowRunEngine(_BaseSyncFlowRunEngine[P, Generator[R, None, None]]):
    """Synchronous generator (yield-only) flow run engine for blocking flow execution."""

    def call(self) -> Generator[R, None, None]:
        """Calls the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow engine must be initialized before starting context.")

        self._run_hook("before_flow_run")
        error = None
        try:
            items = call_with_arguments(self.flow_function, self.flow_run.parameters)
            for idx, item in enumerate(items):
                self._run_hook("after_flow_iteration", result=item, index=idx)
                self.process(item, iteration=idx)
                yield item
        except Exception as exception:
            error = FlowRunError(f"Error during flow run: {exception}", exc=exception)
        finally:
            self._run_hook("after_flow_run", result=None, error=error)
            if error:
                raise error from None


# region Engine: Async


@dataclass
class _BaseAsyncFlowRunEngine(BaseFlowRunEngine[P, R]):
    """Base class for asynchronous flow run engines."""

    @asynccontextmanager
    async def initialize(self) -> AsyncGenerator[None, None]:
        """Initialize the flow run engine, sets attributes and validates state."""
        if TaskRunContext.get():
            raise RuntimeError("Cannot start a flow run context within a task.")

        async with AsyncExitStack() as stack:
            # NOTE: Task runner is duplicated to ensure a the original definition is preserved
            task_runner = self.flow_data.task_runner.duplicate()
            stack.enter_context(task_runner.start())

            flow_run_context = FlowRunContext(
                flow_data=self.flow_data,
                flow_run=self.flow_run,
                task_runner=task_runner,
            )
            stack.enter_context(flow_run_context)

            self._initialized = True
            yield

        self._initialized = False

    async def process(self, result: Any, iteration: int | None = None) -> None:
        """
        Handle the successful completion of the flow.

        Args:
            result (Any): Result of the flow iteration.
            iteration (int, optional): Current iteration index (if applicable).
        """
        pass


@dataclass
class _AsyncFunctionFlowRunEngine(_BaseAsyncFlowRunEngine[P, Coroutine[Any, Any, R]]):
    """Asynchronous flow run engine for non-blocking flow execution."""

    async def call(self) -> Coroutine[Any, Any, R]:
        """Calls the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow run engine must be initialized before starting context.")

        self._run_hook("before_flow_run")

        result, error = None, None
        try:
            result = await call_with_arguments(self.flow_function, self.flow_run.parameters)
        except Exception as exception:
            error = exception
        finally:
            self._run_hook("after_flow_run", result=result, error=error)
            if error:
                raise error from None

        await self.process(result)
        return cast(Coroutine[Any, Any, R], result)


@dataclass
class _AsyncGeneratorFlowRunEngine(_BaseAsyncFlowRunEngine[P, AsyncGenerator[R, None]]):
    """Asynchronous generator (yield-only) flow run engine for non-blocking flow execution."""

    async def call(self) -> AsyncGenerator[R, None]:
        """Calls the flow function with the provided parameters."""
        if not self._initialized:  # pragma: no cover
            raise RuntimeError("Flow run engine must be initialized before starting context.")

        self._run_hook("before_flow_run")
        error = None
        try:
            items = call_with_arguments(self.flow_function, self.flow_run.parameters)
            async for idx, item in aenumerate(items):
                self._run_hook("after_flow_iteration", result=item, index=idx)
                await self.process(item, iteration=idx)
                yield item
        except Exception as exception:
            error = FlowRunError(f"Error during flow run: {exception}", exc=exception)
        finally:
            self._run_hook("after_flow_run", result=None, error=error)
            if error:
                raise error from None

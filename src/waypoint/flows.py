from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    ParamSpec,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from waypoint.runners import BaseTaskRunner
    from waypoint.runners import DefaultTaskRunner
else:
    BaseTaskRunner = object
    DefaultTaskRunner = object

P = ParamSpec("P")
R = TypeVar("R")

# region API


@overload
def flow(__func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def flow(
    __func: None = None,
    *,
    name: str | None = None,
    task_runner: BaseTaskRunner | DefaultTaskRunner | str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def flow(
    __func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    task_runner: BaseTaskRunner | DefaultTaskRunner | str | None = None,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Wraps a callable as a workflow - the orchestration layer of the Waypoint framework.

    Workflows can be composed of multiple waypoint tasks (or other operations) that are executed
    in a specific order, allowing for complex operations to be performed in a structured manner.
    Additionally, workflows can be nested as sub-workflows, enabling modular design and reuse of
    task sequences.

    Task are executed via the specified task runner, which can be customized to use different
    backends or execution strategies.

    Args:
        fn (Callable[P, R]):
            Callable (sync or async) to be executed as a workflow.
        name (str, optional):
            Name for the workflow. If not provided, the function's name is used.
        task_runner (BaseTaskRunner | DefaultTaskRunners | str, optional):
            Task runner to use for executing tasks in the workflow. Defaults to "sequential".
    """
    from waypoint.flow_engine import run_flow_async
    from waypoint.flow_engine import run_flow_sync
    from waypoint.flow_engine import run_generator_flow_async
    from waypoint.flow_engine import run_generator_flow_sync
    from waypoint.flow_run import create_flow_run
    from waypoint.runners import get_task_runner
    from waypoint.utils.callables import get_call_arguments
    from waypoint.utils.callables import is_asynchronous
    from waypoint.utils.callables import is_generator

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        flow_data = FlowData(
            name=name if name is not None else func.__name__,
            task_runner=get_task_runner(task_runner if task_runner else "sequential"),
            is_async=is_asynchronous(func),
            is_generator=is_generator(func),
        )

        # Common engine API parameters
        common: dict[str, Any] = dict(flow_function=func, flow_data=flow_data)

        if flow_data.is_async:
            if flow_data.is_generator:
                # Flow engine delegation wrapper for async generators. Note that async generators
                # need to be iterated with `async for` and can't be awaited directly.
                @wraps(func)
                async def async_generator_wrapper(*args, **kwargs) -> AsyncGenerator[Any, None]:
                    parameters = get_call_arguments(func, args, kwargs)
                    flow_run = create_flow_run(flow_data, parameters)
                    async for item in run_generator_flow_async(flow_run=flow_run, **common):
                        yield item

                _add_flow_data(async_generator_wrapper, flow_data)  # Attach flow data
                return async_generator_wrapper
            else:
                # Flow engine delegation wrapper for async coroutines
                @wraps(func)
                async def async_wrapper(*args, **kwargs) -> Any:
                    parameters = get_call_arguments(func, args, kwargs)
                    flow_run = create_flow_run(flow_data, parameters)
                    return await run_flow_async(flow_run=flow_run, **common)

                _add_flow_data(async_wrapper, flow_data)  # Attach flow data
                return async_wrapper

        else:
            # Task engine delegation wrapper for synchronous workflows. Note that we can use the
            # same wrapper for both functions and generators (unlike the async case above).
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                parameters = get_call_arguments(func, args, kwargs)
                flow_run = create_flow_run(flow_data, parameters)
                if flow_data.is_generator:
                    return run_generator_flow_sync(flow_run=flow_run, **common)
                return run_flow_sync(flow_run=flow_run, **common)

            _add_flow_data(sync_wrapper, flow_data)  # Attach flow data
            return sync_wrapper

    if __func is None:
        return decorator
    return decorator(__func)


@contextmanager
def flow_session(
    name: str = "waypoint-flow-session",
    task_runner: BaseTaskRunner | DefaultTaskRunner | str = "sequential",
):
    """
    Creates an interactive flow session context.

    Primarily useful for running tasks outside of a flow such as in a REPL or notebook. This can
    also be useful for testing task code in isolation.

    Args:
        name (str, optional):
            Name for the temporary flow session. Defaults to "waypoint-flow-session".
        task_runner (BaseTaskRunner | DefaultTaskRunners | str, optional):
            Task runner to use for executing tasks in the flow session. Defaults to "sequential".

    Yields:
        None
    """
    from waypoint.flow_engine import create_flow_session
    from waypoint.flow_run import create_flow_run
    from waypoint.runners import get_task_runner

    flow_data = FlowData(
        name=name,
        task_runner=get_task_runner(task_runner if task_runner else "sequential"),
        is_async=False,
        is_generator=False,
        log_prints=False,
    )

    flow_run = create_flow_run(flow_data, {})

    with create_flow_session(flow_data=flow_data, flow_run=flow_run) as _:
        yield None


# region Metadata


@dataclass(frozen=True)
class FlowData:
    """Data about a Waypoint workflow without the execution logic."""

    name: str
    """Name of the workflow."""

    task_runner: BaseTaskRunner
    """The task runner used to execute tasks in this workflow."""

    is_async: bool
    """True if the flow is async (coroutine or async generator), False otherwise."""

    is_generator: bool
    """True if the flow is a generator (sync or async), False otherwise."""

    log_prints: bool = False
    """Whether to log print statements during the flow run."""


def _add_flow_data(func: Callable[P, R], flow_data: FlowData) -> Callable[P, R]:
    """Attach flow metadata to a function."""
    setattr(func, "__waypoint_flow_data__", flow_data)
    return func


def get_flow_data(func: Callable[P, R]) -> FlowData:
    """
    Get flow metadata from a decorated function, raise an exception if not a valid flow.

    Valid flow functions are those decorated with the `@flow` decorator.

    Args:
        func (Callable[P, R]): Function to get flow data from.

    Raises:
        InvalidFlowError: If the function is not a valid flow - e.g. not decorated with a `@flow`.

    Returns:
        Flow data associated with the function.
    """
    from waypoint.exceptions import InvalidFlowError

    flow_data = getattr(func, "__waypoint_flow_data__", None)
    if flow_data is None:
        name = func.__name__ if hasattr(func, "__name__") else str(func)
        raise InvalidFlowError(f"Callable '{name}' was not decorated with `@flow`")
    return flow_data


def is_flow(func: Callable[P, R]) -> bool:
    """
    Determines if a function is a Waypoint workflow.

    Waypoint workflows are functions decorated with the `@flow` decorator.

    Args:
        func (Callable[P, R]): Function to check if flow.

    Returns:
        bool: True if the function is a flow, False otherwise.
    """
    return hasattr(func, "__waypoint_flow_data__")

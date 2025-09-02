from collections.abc import AsyncGenerator
from collections.abc import Generator
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, ParamSpec, TypeVar, overload

from waypoint.futures import TaskFuture

if TYPE_CHECKING:
    from waypoint.runners import BaseTaskRunner
    from waypoint.runners import DefaultTaskRunner
else:
    BaseTaskRunner = Any
    DefaultTaskRunner = Any


R = TypeVar("R")
P = ParamSpec("P")


# region API


@overload
def task(__func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def task(
    __func: None = None,
    *,
    name: str | None = None,
    log_prints: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def task(
    __func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    log_prints: bool = False,
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
        log_prints (bool, optional):
            Whether to log print statements during the task run. Defaults to False.
    """
    from waypoint.task_engine import run_generator_task_async
    from waypoint.task_engine import run_generator_task_sync
    from waypoint.task_engine import run_task_async
    from waypoint.task_engine import run_task_sync
    from waypoint.task_run import create_task_run
    from waypoint.utils.callables import get_call_arguments
    from waypoint.utils.callables import get_function_name
    from waypoint.utils.callables import is_asynchronous
    from waypoint.utils.callables import is_generator

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        task_name = get_function_name(func) if name is None else name

        task_data = TaskData(
            name=task_name,
            is_async=is_asynchronous(func),
            is_generator=is_generator(func),
            log_prints=log_prints,
            _original_function=func,
        )

        # Common engine API parameters
        common: dict[str, Any] = dict(task_function=func, task_data=task_data)

        if task_data.is_async:
            if task_data.is_generator:
                # Task engine delegation wrapper for async generators. Note that async generators
                # need to be iterated with `async for` and can't be awaited directly.
                @wraps(func)
                async def async_generator_wrapper(*args, **kwargs) -> AsyncGenerator[Any, None]:
                    parameters = get_call_arguments(func, args, kwargs)
                    task_run = create_task_run(task_data, parameters)
                    async for item in run_generator_task_async(task_run=task_run, **common):
                        yield item

                _add_task_data(async_generator_wrapper, task_data)
                return async_generator_wrapper
            else:
                # Task engine delegation wrapper for async coroutines
                @wraps(func)
                async def async_wrapper(*args, **kwargs) -> Any:
                    parameters = get_call_arguments(func, args, kwargs)
                    task_run = create_task_run(task_data, parameters)
                    return await run_task_async(task_run=task_run, **common)

                _add_task_data(async_wrapper, task_data)
                return async_wrapper

        else:
            # Task engine delegation wrapper for synchronous workflows. Note that we can use the
            # same wrapper for both functions and generators (unlike the async case above).
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                parameters = get_call_arguments(func, args, kwargs)
                task_run = create_task_run(task_data, parameters)
                if task_data.is_generator:
                    return run_generator_task_sync(task_run=task_run, **common)
                return run_task_sync(task_run=task_run, **common)

            _add_task_data(sync_wrapper, task_data=task_data)
            return sync_wrapper

    if __func is None:
        return decorator
    return decorator(__func)


@overload
def submit_task(
    task: Callable[P, Generator[R, None, None]], *args: P.args, **kwargs: P.kwargs
) -> TaskFuture[list[R]]: ...


@overload
def submit_task(
    task: Callable[P, AsyncGenerator[R, None]], *args: P.args, **kwargs: P.kwargs
) -> TaskFuture[list[R]]: ...


@overload
def submit_task(
    task: Callable[P, Coroutine[Any, Any, R]], *args: P.args, **kwargs: P.kwargs
) -> TaskFuture[R]: ...


@overload
def submit_task(task: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> TaskFuture[R]: ...


def submit_task(task: Callable[..., Any], *args, **kwargs) -> TaskFuture[Any]:
    """
    Submit a task for execution to the current in-scope task runner.

    The current in-scope task runner is determined by the active flow run context. If no flow
    run context is active, an exception will be raised.

    Args:
        task (Callable[P, R]):
            The synchronous or asynchronous task function to be executed. If not a valid Waypoint
            task, a best-effort attempt will be made to create a task from the callable. Note that
            generator tasks (either sync or async) are not supported and will raise an exception.
        *args (P.args):
            Positional arguments to pass to the task function.
        **kwargs (P.kwargs):
            Keyword arguments to pass to the task function.

    Raises:
        InvalidTaskError: If the provided function is not a valid Waypoint task.

    Returns:
        R: The result of the task execution.
    """
    from waypoint.context import FlowRunContext
    from waypoint.context import TaskRunContext
    from waypoint.exceptions import MissingContextError
    from waypoint.hooks.manager import get_hook_manager
    from waypoint.runners.sequential import SequentialTaskRunner
    from waypoint.task_engine import consume_generator_task_async
    from waypoint.task_engine import consume_generator_task_sync
    from waypoint.task_engine import run_task_async
    from waypoint.task_engine import run_task_sync
    from waypoint.task_run import create_task_run
    from waypoint.utils.callables import get_call_arguments
    from waypoint.utils.callables import is_asynchronous
    from waypoint.utils.callables import is_generator

    flow_run_context = FlowRunContext.get()
    if flow_run_context is None:
        raise MissingContextError(
            "No active flow is running to submit the task to. "
            "Tasks must be called within a flow run or flow session."
        )
    task_runner = flow_run_context.task_runner

    # Make a best-effort attempt to create task data if task is not valid
    if not is_task(task):
        task_data = TaskData(
            name=task.__name__ if hasattr(task, "__name__") else str(task),
            is_async=is_asynchronous(task),
            is_generator=is_generator(task),
            _original_function=task,
        )
        _add_task_data(task, task_data)
    else:
        task_data = get_task_data(task)

    parameters = get_call_arguments(task_data._original_function, args, kwargs)
    task_run = create_task_run(task_data, parameters)
    engine_params: dict[str, Any] = {
        "task_function": task_data._original_function,
        "task_data": task_data,
        "task_run": task_run,
    }

    # Create wrapper sync or async function to run the task with the task engine. Note that
    # generator should be consumed fully within the wrapper, as task runners expect a final
    # result (not a generator).
    wrapper: Callable[[], Any]
    if task_data.is_async:

        @wraps(task)
        async def async_task_wrapper() -> Any:
            if task_data.is_generator:
                return await consume_generator_task_async(**engine_params)
            else:
                return await run_task_async(**engine_params)

        wrapper = async_task_wrapper

    else:

        @wraps(task)
        def sync_task_wrapper() -> Any:
            if task_data.is_generator:
                return consume_generator_task_sync(**engine_params)
            else:
                return run_task_sync(**engine_params)

        wrapper = sync_task_wrapper

    # NOTE: Currently, nested tasks only support sequential execution. Changing this would
    # require a more complex setup that allows tasks to forward a nested tasks to the task runner
    # defined on the parent flow (possible distributed and many levels up and on).
    if TaskRunContext.get() is not None:
        task_runner = SequentialTaskRunner()
        with task_runner.start():
            return task_runner.submit(wrapper)

    # Should fire event immediately before submitting to the task runner, but after creating
    # the task run. Note that we skip this for sequential task runners as they run in-line
    # and the hook would be redundant.
    if not isinstance(task_runner, SequentialTaskRunner):
        hook_manager = get_hook_manager()
        if hook := getattr(hook_manager.hook, "before_task_submit", None):  # pragma: no branch
            hook(task_data=task_data, task_run=task_run, runner=task_runner)

    return task_runner.submit(wrapper)


def map_task(task: Callable[P, R], *args) -> Iterator[R]:
    """
    Map a task over multiple sets of arguments, yielding results as they complete.

    This function is similar to the built-in `map`, but is designed to work with Waypoint tasks.

    Args:
        task (Callable[P, R]):
            The task function to be executed. Must be a valid Waypoint task.
        *args (P.args):
            Arugments to map over. Each argument should be an iterable of values.

    Raises:
        InvalidTaskError: If the provided function is not a valid Waypoint task.

    Returns:
        Iterator[R]: An iterator over the results of the task executions.
    """
    yield from []  # Placeholder for future implementation


# region Metadata


@dataclass(frozen=True)
class TaskData:
    """Metadata about a defined Waypoint task."""

    name: str
    """Name of the task."""

    is_async: bool
    """True if the task is asynchronous (coroutine or async generator), False otherwise."""

    is_generator: bool
    """True if the task is a generator (sync or async), False otherwise."""

    _original_function: Callable[..., Any] = field(repr=False, hash=False)

    log_prints: bool = False
    """Whether to log print statements during the task run."""


def _add_task_data(func: Callable[P, R], task_data: TaskData) -> Callable[P, R]:
    """Attach task metadata to a function."""
    setattr(func, "__waypoint_task_data__", task_data)
    return func


def get_task_data(func: Callable[P, R]) -> TaskData:
    """
    Get task metadata from a decorated function, raise an exception if not a valid task.

    Valid task functions are those decorated with the `@task` decorator.

    Args:
        func (Callable[P, R]): Function to get task data from.

    Raises:
        InvalidFlowError: If the function is not a valid task - e.g. not decorated with a `@task`.

    Returns:
        Task data associated with the function.
    """
    from waypoint.exceptions import InvalidTaskError

    task_data = getattr(func, "__waypoint_task_data__", None)
    if task_data is None:
        name = func.__name__ if hasattr(func, "__name__") else str(func)
        raise InvalidTaskError(f"Callable '{name}' was not decorated with `@task`")
    return task_data


def is_task(func: Callable[P, R]) -> bool:
    """
    Determines if a function is a Waypoint task.

    Waypoint tasks are functions decorated with the `@task` decorator.

    Args:
        func (Callable[P, R]): Function to check.

    Returns:
        bool: True if the function is a task, False otherwise.
    """
    return hasattr(func, "__waypoint_task_data__")

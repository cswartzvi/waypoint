from collections.abc import AsyncGenerator
from collections.abc import Generator
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, ParamSpec, TypeVar, overload

from waypoint.futures import TaskFuture

if TYPE_CHECKING:
    from concurrent.futures import Future

    from waypoint.runners import BaseTaskRunner
    from waypoint.runners import DefaultTaskRunner
    from waypoint.task_run import TaskRun
else:
    Future = Any
    BaseTaskRunner = Any
    DefaultTaskRunner = Any
    TaskRun = Any


R = TypeVar("R")
P = ParamSpec("P")

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


# region Decorator


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


# region Submit


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
    from waypoint.exceptions import MissingContextError
    from waypoint.utils.callables import get_call_arguments

    flow_run_context = FlowRunContext.get()
    if flow_run_context is None:
        raise MissingContextError(
            "No active flow is running to submit the task to. "
            "Tasks must be called within a flow run or flow session."
        )

    task_runner = flow_run_context.task_runner
    parameters = get_call_arguments(task, args, kwargs)
    return _submit_task(task_function=task, arguments=parameters, task_runner=task_runner)


# region Map


@overload
def map_task(
    task: Callable[..., Coroutine[Any, Any, R]], *args: Any, **kwargs: Any
) -> Iterator[R]: ...


@overload
def map_task(
    task: Callable[..., AsyncGenerator[R, None]], *args: Any, **kwargs: Any
) -> Iterator[list[R]]: ...


@overload
def map_task(
    task: Callable[..., Generator[R, None, None]], *args: Any, **kwargs: Any
) -> Iterator[list[R]]: ...


@overload
def map_task(task: Callable[..., R], *args: Any, **kwargs: Any) -> Iterator[R]: ...


def map_task(task: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterator[Any]:
    """
    Map a task over multiple sets of arguments, yielding results as they complete.

    This function is similar to the built-in `map`, but is designed to work with Waypoint tasks.

    Args:
        task (Callable[P, R]):
            The task function to be executed. Must be a valid Waypoint task.
        *args (P.args):
            Positional arguments to pass to the task function. At least one argument must be
            iterable (e.g., list, tuple). Use `unmapped` to pass static arguments.
        **kwargs (P.kwargs):
            Keyword arguments to pass to the task function. At least one argument must be
            iterable (e.g., list, tuple). Use `unmapped` to pass static arguments.

    Raises:
        InvalidTaskError: If the provided function is not a valid Waypoint task.

    Returns:
        Iterator[R]: An iterator over the results of the task executions.
    """
    from waypoint.context import FlowRunContext
    from waypoint.exceptions import MappingLengthMismatch
    from waypoint.exceptions import MappingMissingIterable
    from waypoint.exceptions import MissingContextError
    from waypoint.utils.annotations import unmapped
    from waypoint.utils.callables import collapse_variadic_arguments
    from waypoint.utils.callables import explode_variadic_arguments
    from waypoint.utils.callables import get_call_arguments
    from waypoint.utils.callables import get_parameter_defaults
    from waypoint.utils.collections import is_iterable

    flow_run_context = FlowRunContext.get()
    if flow_run_context is None:
        raise MissingContextError(
            "No active flow is running to map the task to. "
            "Tasks must be called within a flow run or flow session."
        )

    task_runner: BaseTaskRunner = flow_run_context.task_runner
    arguments = get_call_arguments(task, args, kwargs, apply_defaults=False)
    arguments = explode_variadic_arguments(task, arguments)

    # Separate static and iterable arguments
    iterable_arguments: dict[str, Any] = {}
    static_arguments: dict[str, Any] = {}
    for key, val in arguments.items():
        if isinstance(val, unmapped):
            static_arguments[key] = val.value
        elif is_iterable(val):
            iterable_arguments[key] = list(val)
        else:
            static_arguments[key] = val

    if not len(iterable_arguments):
        raise MappingMissingIterable(
            "No iterable arguments were received. Arguments for map must "
            f"include at least one iterable. Arguments: {arguments}"
        )

    # All iterable arguments are the same length
    iterable_arguments_lengths = {key: len(val) for key, val in iterable_arguments.items()}
    lengths = set(iterable_arguments_lengths.values())
    if len(lengths) > 1:
        raise MappingLengthMismatch(
            "Received iterable arguments with different lengths. Arguments for map"
            f" must all be the same length. Got lengths: {iterable_arguments_lengths}"
        )
    map_length = list(lengths)[0]

    # Submit tasks for each set of arguments
    futures: list[TaskFuture[Any]] = []
    for i in range(map_length):
        call_args: dict[str, Any] = {key: value[i] for key, value in iterable_arguments.items()}
        call_args.update({key: value for key, value in static_arguments.items()})

        # Add default values for parameters; these were skipped previously
        for key, value in get_parameter_defaults(task).items():
            call_args.setdefault(key, value)

        # Collapse any previously exploded kwargs
        call_args = collapse_variadic_arguments(task, call_args)

        future = _submit_task(task_function=task, arguments=call_args, task_runner=task_runner)
        futures.append(future)

    for future in futures:
        yield future.result()


# region Internal


def _submit_task(
    task_function: Callable[..., R], arguments: dict[str, Any], task_runner: BaseTaskRunner
) -> TaskFuture[R]:
    """Internal implementation - submit a task function and arguments to a specified task runner."""
    from waypoint.context import TaskRunContext
    from waypoint.hooks.manager import try_run_hook
    from waypoint.logging import get_run_logger
    from waypoint.runners.sequential import SequentialTaskRunner
    from waypoint.task_engine import consume_generator_task_async
    from waypoint.task_engine import consume_generator_task_sync
    from waypoint.task_engine import run_task_async
    from waypoint.task_engine import run_task_sync
    from waypoint.task_run import create_task_run

    logger = get_run_logger()

    # Make a best-effort attempt to create task data if task is not valid
    if not is_task(task_function):
        task_data = _create_best_effort_task_data(task_function)
    else:
        task_data = get_task_data(task_function)

    original_function = task_data._original_function
    task_run = create_task_run(task_data, arguments)

    params: dict = {
        "task_function": original_function,
        "task_data": task_data,
        "task_run": task_run,
    }
    if task_data.is_async:
        # Wrapper async function to run the task with the task engine. Async generator should be
        # consumed fully within the wrapper, as task runners expect a final result.
        @wraps(original_function)
        async def wrapper() -> Any:  # pyright: ignore
            if task_data.is_generator:
                return await consume_generator_task_async(**params)
            else:
                return await run_task_async(**params)
    else:
        # Wrapper sync function to run the task with the task engine. Generator should be
        # consumed fully within the wrapper, as task runners expect a final result.
        @wraps(original_function)
        def wrapper() -> Any:
            if task_data.is_generator:
                return consume_generator_task_sync(**params)
            else:
                return run_task_sync(**params)

    logger.debug(f"Submitting task '{task_run.task_id}' to '{task_runner.name}' runner")

    try_run_hook(
        hook_name="before_task_submit",
        task_data=task_data,
        task_run=task_run,
        task_runner=task_runner.name,
    )

    # NOTE: Currently, nested tasks only support sequential execution. Changing this would
    # require a more complex setup that allows tasks to forward a nested tasks to the task runner
    # defined on the parent flow (possible distributed and many levels up and on).
    if TaskRunContext.get() is not None:
        task_runner = SequentialTaskRunner()
        with task_runner.start():
            return task_runner.submit(wrapper)

    future = task_runner.submit(wrapper)
    return future


def _create_best_effort_task_data(func: Callable[..., Any]) -> "TaskData":
    """Create task data for a callable that is not a valid Waypoint task."""
    from waypoint.utils.callables import is_asynchronous
    from waypoint.utils.callables import is_generator

    task_data = TaskData(
        name=func.__name__ if hasattr(func, "__name__") else str(func),
        is_async=is_asynchronous(func),
        is_generator=is_generator(func),
        _original_function=func,
    )
    _add_task_data(func, task_data)

    return task_data

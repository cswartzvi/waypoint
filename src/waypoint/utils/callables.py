import inspect
from collections.abc import Coroutine
from typing import Any, Callable, ParamSpec, TypeVar

from waypoint.exceptions import ParameterBindError
from waypoint.exceptions import SignatureMismatchError

P = ParamSpec("P")
R = TypeVar("R")


def is_asynchronous(fn: Callable[..., Any]) -> bool:
    """
    Check if a function is asynchronous.

    Args:
        fn: The function to check.

    Returns:
        bool: True if the function is asynchronous, False otherwise.
    """
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def is_generator(fn: Callable[..., Any]) -> bool:
    """
    Check if a function is a generator.

    Args:
        fn: The function to check.

    Returns:
        bool: True if the function is a generator, False otherwise.
    """
    return inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn)


def get_call_arguments(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    apply_defaults: bool = True,
) -> dict[str, Any]:
    """
    Bind a call to a function to get parameter/argument mapping.

    Default values on the signature will be included if not overridden. Raises a
    ParameterBindError if the arguments/kwargs are not valid for the function
    """
    try:
        bound_signature = inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as exc:
        raise ParameterBindError.from_bind_failure(fn, exc, args, kwargs)

    if apply_defaults:
        bound_signature.apply_defaults()

    return dict(bound_signature.arguments)


def arguments_to_args_kwargs(
    fn: Callable, arguments: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Convert a `arguments` dictionary to positional and keyword arguments.

    The function _must_ have an identical signature to the original function or this
    will return an empty tuple and dict.
    """
    parameters = dict(inspect.signature(fn).parameters).keys()
    # Check for arguments that are not present in the function signature
    unknown_arguments = arguments.keys() - parameters
    if unknown_arguments:
        raise SignatureMismatchError.from_bad_params(list(parameters), list(arguments.keys()))
    bound_signature = inspect.signature(fn).bind_partial()
    bound_signature.arguments = arguments  # type: ignore

    return bound_signature.args, bound_signature.kwargs


def call_with_arguments(fn: Callable[..., R], parameters: dict[str, Any]) -> R:
    """
    Call a function with parameters extracted with `get_call_arguments`.

    The function _must_ have an identical signature to the original function or this
    will fail. If you need to send to a function with a different signature, extract
    the args/kwargs using `parameters_to_positional_and_keyword` directly.
    """
    args, kwargs = arguments_to_args_kwargs(fn, parameters)
    return fn(*args, **kwargs)


def explode_variadic_arguments(fn: Callable, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Explode keyword arguments (**kwargs) in variadic arguments (unpacked).

    Example:
        ```python
        def foo(a, b, **kwargs):
            pass


        arguments = {"a": 1, "b": 2, "kwargs": {"c": 3, "d": 4}}
        explode_variadic_parameter(foo, arguments)
        # {"a": 1, "b": 2, "c": 3, "d": 4}
        ```

    See Also:
        collapse_variadic_arguments
    """
    variadic_key = None
    for key, parameter in inspect.signature(fn).parameters.items():
        if parameter.kind == parameter.VAR_KEYWORD:
            variadic_key = key
            break

    if not variadic_key:
        return arguments

    new_arguments = arguments.copy()
    for key, value in new_arguments.pop(variadic_key, {}).items():
        new_arguments[key] = value

    return new_arguments


def collapse_variadic_arguments(fn: Callable, arguments: dict[str, Any]) -> dict:
    """
    Collapse variadic arguments (unpacked) in keyword arguments (**kwargs).

    Example:
        ```python
        def foo(a, b, **kwargs):
            pass


        arguments = {"a": 1, "b": 2, "c": 3, "d": 4}
        collapse_variadic_arguments(foo, arguments)
        # {"a": 1, "b": 2, "kwargs": {"c": 3, "d": 4}}
        ```

    See Also:
        explode_variadic_arguments
    """
    parameters = inspect.signature(fn).parameters
    variadic_key = None
    for key, parameter in parameters.items():
        if parameter.kind == parameter.VAR_KEYWORD:
            variadic_key = key
            break

    missing = set(arguments.keys()) - set(parameters.keys())

    if not variadic_key and missing:
        raise ValueError(
            f"Signature for {fn} does not include any variadic keyword argument "
            "but parameters were given that are not present in the signature."
        )

    if variadic_key and not missing:
        # variadic key is present but no missing parameters, return parameters unchanged
        return arguments

    new_arguments: dict = arguments.copy()
    if variadic_key:
        new_arguments[variadic_key] = {}

    for key in missing:
        new_arguments[variadic_key][key] = new_arguments.pop(key)

    return new_arguments


def get_parameter_defaults(fn: Callable, include_empty: bool = False) -> dict[str, Any]:
    """Get default parameter values for a callable."""
    signature = inspect.signature(fn)

    parameter_defaults = {}

    for name, param in signature.parameters.items():
        if param.default is not signature.empty or include_empty:
            parameter_defaults[name] = param.default

    return parameter_defaults


def get_function_name(fn: Callable[..., Any]) -> str:
    """
    Get a human-readable name for a function.

    This will return the `__qualname__` of the function if it exists, otherwise
    it will return the `__name__`. If neither exists, it will return the string
    representation of the function.

    Args:
        fn: The function to get the name of.

    Returns:
        str: The name of the function.
    """
    if hasattr(fn, "__qualname__"):
        return fn.__qualname__
    elif hasattr(fn, "__name__"):  # pragma: no cover
        return fn.__name__
    else:
        return str(fn)


def get_docstring_summary(fn: Callable[..., Any]) -> str | None:
    """
    Extract the summary from a pipeline docstring, (allows multiple line).

    Args:
        fn: The function to extract the docstring summary from.

    Returns:
        The docstring summary or None if no summary was found.
    """
    docstring = inspect.getdoc(fn)
    if docstring is None:
        return None
    parts = docstring.split("\n")
    extracted = parts[0].strip()
    for part in parts[1:]:
        part = part.strip()
        if not part:
            break
        extracted += " " + part
    return extracted

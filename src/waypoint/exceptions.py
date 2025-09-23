import inspect
from typing import Any, Callable

from typing_extensions import Self


class WaypointException(Exception):
    """Base exception type for Waypoint errors."""


class InvalidTaskError(WaypointException, ValueError):
    """Raised when an invalid task is encountered."""


class InvalidFlowError(WaypointException, ValueError):
    """Raised when an invalid flow is encountered."""


class FlowRunError(WaypointException, RuntimeError):
    """Raised when a flow run fails; wraps the original exception."""

    def __init__(self, flow_id: str, exc: Exception):
        self.flow_id = flow_id
        self.exc = exc


class TaskRunError(WaypointException, RuntimeError):
    """Raised when a task run fails; wraps the original exception."""

    def __init__(self, task_id: str, exc: Exception | None = None):
        self.task_id = task_id
        self.exc = exc


class MissingContextError(WaypointException, RuntimeError):
    """Raised when no task or flow run context to be can be found (but is required)."""


class MappingMissingIterable(WaypointException):
    """Raised when attempting to call Task.map with all static arguments."""


class MappingLengthMismatch(WaypointException):
    """Raised when attempting to call Task.map with arguments of different lengths."""


class ParameterBindError(TypeError, WaypointException):
    """Raised when args and kwargs cannot be converted to parameters."""

    def __init__(self, msg: str):
        super().__init__(msg)

    @classmethod
    def from_bind_failure(
        cls,
        fn: Callable[..., Any],
        exc: TypeError,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Self:
        """
        Creates a ParameterBindError from a failed bind attempt.

        Args:
            fn (Callable[..., Any]): The function that failed to bind.
            exc (TypeError): The exception raised during binding.
            call_args (tuple[Any, ...]): The positional arguments passed to the function.
            call_kwargs (dict[str, Any]): The keyword arguments passed to the function.
        """
        fn_signature = str(inspect.signature(fn)).strip("()")

        base = f"Error binding parameters for function '{fn.__name__}': {exc}"
        signature = f"Function '{fn.__name__}' has signature '{fn_signature}'"
        received = f"received args: {call_args} and kwargs: {list(call_kwargs.keys())}"
        msg = f"{base}.\n{signature} but {received}."
        return cls(msg)


class SignatureMismatchError(WaypointException, TypeError):
    """Raised when parameters passed to a function do not match its signature."""

    def __init__(self, msg: str):
        super().__init__(msg)

    @classmethod
    def from_bad_params(cls, expected_params: list[str], provided_params: list[str]) -> Self:
        """
        Creates a SignatureMismatchError from mismatched parameters.

        Args:
            expected_params (list[str]): The parameters expected by the function.
            provided_params (list[str]): The parameters provided to the function.
        """
        msg = (
            f"Function expects parameters {expected_params} but was provided with"
            f" parameters {provided_params}"
        )
        return cls(msg)

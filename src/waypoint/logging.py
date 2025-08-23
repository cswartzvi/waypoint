"""waypoint logging configuration and logger management."""

import io
import logging
import sys
from builtins import print  # required for patching
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, Literal, cast

from waypoint.exceptions import MissingContextError

LOGGERS = Literal["waypoint", "waypoint.flow", "waypoint.task"]
"""Base logger names used by Waypoint."""

# NOTE: Python `Literal` can not contain variables, therefore we must also define variables
# for the base loggers. The value of these variables MUST match the values in `LOGGERS`.
_BASE_LOGGER = "waypoint"
_FLOW_LOGGER = f"{_BASE_LOGGER}.flow"
_TASK_LOGGER = f"{_BASE_LOGGER}.task"


@lru_cache()
def get_logger(name: str | None = None) -> logging.Logger:
    """
    Returns a named logger, or the root logger if no name is specified.

    See `get_run_logger` for retrieving loggers for use within task or flow runs.

    Args:
        name: The name of the logger to retrieve. If not supplied, the root logger
            will be returned. If the name is relative, it will be prefixed with the
            "waypoint" logger name. If the name is absolute, it will be used as is.
    """
    parent_logger = logging.getLogger(_BASE_LOGGER)

    if name:
        if not name.startswith(parent_logger.name + "."):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger

    return logger


def get_run_logger(default: str | None = None) -> logging.Logger:
    """
    Returns a logger for the current run or the default (if supplied).

    Args:
        default: Default logger to use if there is no active run context. If not
            supplied, an error will be raised if there is no active run context.
    """
    from waypoint.context import FlowRunContext
    from waypoint.context import TaskRunContext

    # Check for existing contexts
    task_run_context = TaskRunContext.get()
    flow_run_context = FlowRunContext.get()

    task_run = task_run_context.task_run if task_run_context else None
    flow_run = flow_run_context.flow_run if flow_run_context else None

    logger: logging.Logger | logging.LoggerAdapter[Any]
    if task_run:
        logger = logging.getLogger(_TASK_LOGGER)
        logger = logging.LoggerAdapter(logger, {"task_run_name": task_run.task_id})
    elif flow_run:
        logger = logging.getLogger(_FLOW_LOGGER)
        logger = logging.LoggerAdapter(logger, {"flow_run_name": flow_run.name})
    elif default is not None:
        logger = get_logger(default)
    else:
        raise MissingContextError("There is no active flow or task run context.")

    return cast(logging.Logger, logger)


@contextmanager
def patch_print(enable: bool = True) -> Iterator[None]:
    """
    Patches the Python builtin `print` method to use `print_as_log`.

    Args:
        enable: Indicates whether to enable or disable the patching of `print`.
    """
    import builtins

    if not enable:
        yield
        return

    original = builtins.print

    try:
        builtins.print = _print_as_log
        yield
    finally:
        builtins.print = original


def _print_as_log(*args, **kwargs):
    """
    Patches `print` to send printed messages to a run logger.

    If no run is active, `print` will behave as if it were not patched. If `print`
    sends data to a file other than `sys.stdout` or `sys.stderr`, it will not be
    forwarded to the run logger either.
    """
    from waypoint.context import FlowRunContext
    from waypoint.context import TaskRunContext

    def is_log_print_enabled(context):
        if isinstance(context, FlowRunContext):
            return context.flow_data.log_prints
        elif isinstance(context, TaskRunContext):
            return context.task_data.log_prints
        return False

    context = TaskRunContext.get() or FlowRunContext.get()
    if (
        not context
        or not is_log_print_enabled(context)
        or kwargs.get("file") not in {None, sys.stdout, sys.stderr}
    ):
        return print(*args, **kwargs)

    logger = get_run_logger()

    # Print to an in-memory buffer; so we do not need to implement `print`
    buffer = io.StringIO()
    kwargs["file"] = buffer
    print(*args, **kwargs)

    # Remove trailing whitespace to prevent duplicates
    logger.info(buffer.getvalue().rstrip())

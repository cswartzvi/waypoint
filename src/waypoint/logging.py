"""waypoint logging configuration and logger management."""

import io
import logging
import logging.handlers
import sys
from builtins import print  # required for patching
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Iterator, MutableMapping, cast

from waypoint.exceptions import MissingContextError

_LOGGER = logging.getLogger(__name__)


# Define base logger names
_BASE_LOGGER = "waypoint"
_FLOW_LOGGER = f"{_BASE_LOGGER}.flow"
_TASK_LOGGER = f"{_BASE_LOGGER}.task"
_APP_LOGGERS: Final[tuple[str, ...]] = (_BASE_LOGGER, _FLOW_LOGGER, _TASK_LOGGER)

# Define the log formats for different loggers
_CONSOLE_FORMATS: dict[str, str] = {
    _BASE_LOGGER: "%(message)s",
    _FLOW_LOGGER: "Flow run '%(flow_run_name)s' - %(message)s",
    _TASK_LOGGER: "Task run '%(task_run_name)s' - %(message)s",
}

# Base file format for logging
_BASE_FILE_FORMAT = "%(asctime)s | %(levelname)-7s |"

# Define the file formats for different loggers
_FILE_FORMATS: dict[str, str] = {
    "waypoint": _BASE_FILE_FORMAT + " %(name)s - %(message)s",
    "waypoint.flow": _BASE_FILE_FORMAT + " %(name)s '%(flow_run_name)s' - %(message)s",
    "waypoint.task": _BASE_FILE_FORMAT + " %(name)s '%(task_run_name)s' - %(message)s",
}


class EnhancedLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter that ensures extra kwargs are passed through correctly.

    Without this, the `extra` fields set on the adapter would overshadow any provided
    on a log-by-log basis.

    See https://bugs.python.org/issue32732 — the Python team has declared that this is
    not a bug in the LoggingAdapter and subclassing is the intended workaround.
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Process the logging call to merge extra context correctly."""
        # Merge adapter's extra with call-specific extra, preferring call-specific
        kwargs["extra"] = {**(self.extra or {}), **(kwargs.get("extra") or {})}
        return (msg, kwargs)

    def getChild(
        self, suffix: str, extra: dict[str, Any] | None = None
    ) -> "EnhancedLoggerAdapter":  # pragma: no cover
        """Create a child adapter with merged extra context."""
        _extra: dict[str, Any] = extra or {}

        return EnhancedLoggerAdapter(
            self.logger.getChild(suffix),
            extra={
                **(self.extra or {}),
                **_extra,
            },
        )


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

    logger: logging.Logger | EnhancedLoggerAdapter
    if task_run:
        logger = logging.getLogger(_TASK_LOGGER)
        logger = EnhancedLoggerAdapter(logger, {"task_run_name": task_run.task_id})
    elif flow_run:
        logger = logging.getLogger(_FLOW_LOGGER)
        logger = EnhancedLoggerAdapter(logger, {"flow_run_name": flow_run.name})
    elif default is not None:
        logger = get_logger(default)
    else:
        raise MissingContextError("There is no active flow or task run context.")

    return cast(logging.Logger, logger)


def setup_console_logging(
    level: int = logging.INFO,
    traceback: bool = False,
    use_rich: bool = False,
    loggers: Iterable[str] | None = None,
) -> None:
    """
    Sets up console logging with specified traceback.

    Args:
        level (int): Logging level to use for the console handler.
        traceback (bool): Indicates whether to include tracebacks in the console output.
        use_rich (bool): Indicates whether to use the Rich library for console logging. Library
            must be installed for this to work (use `waypoint[rich]`). Defaults to False.
        loggers (Iterable[str]): Additional logger names to configure beyond the default loggers.
    """
    _setup_loggers(level=level, loggers=loggers)

    logger_handlers = defaultdict(set)
    for name, fmt in _CONSOLE_FORMATS.items():
        logger = logging.getLogger(name)
        handler = _console_handler(fmt, traceback, use_rich)
        handler.set_name(f"{name} - console")
        handler.setLevel(level)

        # We need to take care not to add the same handler multiple times to
        # the same logger. This can happen when the context manager is used
        # multiple times in the same process.
        if not any(handler.name == h.name for h in logger.handlers):
            logger_handlers[logger].add(handler)
            logger.addHandler(handler)

    return


def setup_file_logging(
    base: Path, level: int = logging.INFO, loggers: Iterable[str] | None = None
) -> None:
    """
    Sets up file logging for the specified file path.

    Args:
        base (pathlib.Path): Path to the directory where the log file should be written.
        level (int): Logging level to use for the file handler.
        loggers (Iterable[str]): Additional logger names to configure beyond the default loggers.
    """
    _setup_loggers(level=level, loggers=loggers)

    # Because we are altering the logging configuration at runtime, we need to
    # ensure that the log file exists before we start logging to it.
    file_path = Path(base).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)

    logger_handlers = defaultdict(set)
    for name, fmt in _FILE_FORMATS.items():
        logger = logging.getLogger(name)
        handler = _file_handler(str(file_path), fmt)
        handler.set_name(f"{name} - {file_path}")
        handler.setLevel(level)

        # We need to take care not to add the same handler multiple times to
        # the same logger. This can happen when the context manager is used
        # multiple times in the same process.
        if not any(handler.name == h.name for h in logger.handlers):
            logger_handlers[logger].add(handler)
            logger.addHandler(handler)

    return


# TODO: Remove pragma when feature is fully implemented
@contextmanager  # pragma: no cover
def setup_listener_context(queue: Any) -> Iterator[None]:
    """Sets up a logging listener context."""
    yield


# TODO: Remove pragma when feature is fully implemented
def setup_worker_logging(queue: Any) -> None:  # pragma: no cover
    """Sets up worker logging to send logs to the specified queue."""
    pass


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


def _setup_loggers(level: Any = logging.DEBUG, loggers: Iterable[str] | None = None) -> None:
    """Setups update base loggers for the applications."""
    loggers = loggers or []
    for name in _APP_LOGGERS + tuple(loggers):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(level)


def _console_handler(format: str, traceback: bool, use_rich: bool) -> logging.Handler:
    """Creates a console handler for the specified console and format."""
    handler: logging.Handler

    if not use_rich:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        return handler

    try:  # pragma: no cover
        import rich.logging
        from rich import get_console
    except ImportError:  # pragma: no cover
        raise ImportError("Rich library is required for logging in Waypoint.") from None

    console = get_console()
    handler = rich.logging.RichHandler(
        rich_tracebacks=traceback, omit_repeated_times=False, console=console
    )
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    return handler


def _file_handler(filename: str, format: str) -> logging.Handler:
    """Creates a file handler for the specified filename and format."""
    handler = logging.handlers.RotatingFileHandler(
        filename,
        mode="a",
        maxBytes=10485760,
        encoding="utf-8",
        delay=True,
    )
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    return handler


# TODO: Remove pragma when feature is fully implemented
def _print_as_log(*args, **kwargs):  # pragma: no cover
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

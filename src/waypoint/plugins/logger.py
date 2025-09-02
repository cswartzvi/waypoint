import logging.config
import logging.handlers
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

try:  # pragma: no cover
    import rich.logging
    from rich import get_console
except ImportError:  # pragma: no cover
    raise ImportError("Rich library is required for logging in Waypoint.") from None

from waypoint.hooks import hook_impl
from waypoint.logging import LOGGERS
from waypoint.logging import get_run_logger

if TYPE_CHECKING:
    from waypoint.flow_run import FlowRun
    from waypoint.flows import FlowData
    from waypoint.runners import BaseTaskRunner
    from waypoint.task_run import TaskRun
    from waypoint.tasks import TaskData
else:
    FlowData = object
    FlowRun = object
    TaskData = object
    TaskRun = object
    BaseTaskRunner = object


# Define the log formats for different loggers
_CONSOLE_FORMATS: dict[LOGGERS, str] = {
    "waypoint": "%(message)s",
    "waypoint.flow": "Flow run '%(flow_run_name)s' - %(message)s",
    "waypoint.task": "Task run '%(task_run_name)s' - %(message)s",
}

# Base file format for logging
_BASE_FILE_FORMAT = "%(asctime)s | %(levelname)-7s |"

# Define the file formats for different loggers
_FILE_FORMATS: dict[LOGGERS, str] = {
    "waypoint": _BASE_FILE_FORMAT + " %(name)s - %(message)s",
    "waypoint.flow": _BASE_FILE_FORMAT + " %(name)s '%(flow_run_name)s' - %(message)s",
    "waypoint.task": _BASE_FILE_FORMAT + " %(name)s '%(task_run_name)s' - %(message)s",
}


class WaypointLogger:
    """
    A logger that integrates with Waypoint's task and flow execution context.

    Args:
        console_enabled (bool, optional):
            If True, enables console logging. Defaults to True.
        file_path (Path | str | None, optional):
            If provided, sets up file logging to the specified path. Defaults to None, which means
            no file logging will be set up.
    """

    def __init__(
        self,
        console_level: int = logging.INFO,
        disable_console: bool = False,
        file_path: Path | str | None = None,
        file_level: int = logging.INFO,
        traceback: bool = False,
    ):
        file_path = Path(file_path) if file_path else None

        if not disable_console:
            _setup_console_logging(level=console_level, traceback=traceback)

        if file_path:
            _setup_file_logging(file_path, level=file_level)

    @hook_impl
    def before_flow_run(
        self,
        flow_data: FlowData,
        flow_run: FlowRun,
    ) -> None:
        """Hook that is called before a flow run starts."""
        logger = get_run_logger()
        logger.info(f"Beginning flow run {flow_run.flow_id}")

    @hook_impl
    def after_flow_iteration(
        self,
        flow_data: FlowData,
        flow_run: FlowRun,
        result: Any,
        index: int,
    ) -> None:
        """Hook that is called after each iteration of a flow."""
        logger = get_run_logger()
        logger.info(f"Finished iteration {index} [OK]")

    @hook_impl
    def after_flow_run(
        self,
        flow_data: FlowData,
        flow_run: FlowRun,
        error: Exception | None,
    ) -> None:
        """Hook that is called after a flow run completes."""
        logger = get_run_logger()
        if error:
            logger.error("Task run failed with error")
        logger.info(f"Finished flow run {flow_run.flow_id}")

    @hook_impl
    def before_task_submit(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        task_runner: str,
    ) -> None:
        """Hook that is called before a task is submitted for execution."""
        logger = get_run_logger()
        message = f"Submitting task '{task_run.task_id}' to '{task_runner}' runner"
        if task_runner in {"sequential", "threading"}:
            logger.debug(message)
        else:
            logger.info(message)

    @hook_impl
    def before_task_run(
        self,
        task_data: TaskData,
        task_run: TaskRun,
    ) -> None:
        """Hook that is called before a task run starts."""
        logger = get_run_logger()
        params = set(task_run.parameters.keys())
        logger.debug(f"Beginning task run with parameters: {params}")

    @hook_impl
    def after_task_iteration(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        result: Any,
        index: int,
    ) -> None:
        """Hook that is called after each iteration of a task."""
        logger = get_run_logger()
        logger.info(f"Finished iteration {index} [OK]")

    @hook_impl
    def after_task_run(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        error: Exception | None,
    ) -> None:
        """Hook that is called after a task run completes."""
        logger = get_run_logger()
        if error:
            logger.error("Task run failed with error")
        else:
            logger.info("Finished task run [OK]")

    @hook_impl
    def after_task_future_result(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        error: Exception | None,
        cancelled: bool,
        result: object | None,
        task_runner: str,
    ) -> None:
        """Hook that is called after a task future completes."""
        logger = get_run_logger()
        if cancelled:
            logger.warning(f"Task '{task_run.task_id} was cancelled")
        elif error:
            logger.error(f"Task '{task_run.task_id}' failed with error: {error}")
        else:
            message = f"Task '{task_run.task_id}' returned from '{task_runner}' runner [OK]"
            if task_runner in {"sequential", "threading"}:
                logger.debug(message)
            else:
                logger.info(message)


def _setup_console_logging(level: int = logging.INFO, traceback: bool = False) -> None:
    """
    Sets up console logging with specified traceback.

    Args:
        level: Logging level to use for the console handler.
        traceback: Indicates whether to include tracebacks in the console output.
    """
    _setup_app_loggers()

    logger_handlers = defaultdict(set)
    for name, fmt in _CONSOLE_FORMATS.items():
        logger = logging.getLogger(name)
        handler = _console_handler(fmt, traceback)
        handler.set_name(f"{name} - console")
        handler.setLevel(level)

        # We need to take care not to add the same handler multiple times to
        # the same logger. This can happen when the context manager is used
        # multiple times in the same process.
        if not any(handler.name == h.name for h in logger.handlers):
            logger_handlers[logger].add(handler)
            logger.addHandler(handler)

    return


def _setup_file_logging(base: Path, level: int = logging.INFO) -> None:
    """
    Sets up file logging for the specified file path.

    Args:
        base: Path to the directory where the log file should be written.
        level: Logging level to use for the file handler.
    """
    _setup_app_loggers()

    # Because we are altering the logging configuration at runtime, we need to
    # ensure that the log file exists before we start logging to it.
    file_path = Path(base).joinpath(".log").resolve()
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


def _setup_app_loggers(level: Any = logging.DEBUG) -> None:
    """Setups update base loggers for the applications."""
    for name in get_args(LOGGERS):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(level)


def _console_handler(format: str, traceback: bool) -> logging.Handler:
    """Creates a console handler for the specified console and format."""
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

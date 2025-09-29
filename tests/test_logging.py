import logging
import logging.handlers
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from waypoint.exceptions import MissingContextError
from waypoint.flows import flow_session
from waypoint.logging import get_logger
from waypoint.logging import get_run_logger
from waypoint.logging import patch_print
from waypoint.logging import setup_console_logging
from waypoint.logging import setup_file_logging
from waypoint.tasks import task


class TestGetLogger:
    def test_get_logger_default(self) -> None:
        """Test getting the default logger."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "waypoint"

    def test_get_logger_with_explicit_name(self) -> None:
        """Test getting a logger with an explicit name."""
        logger = get_logger("waypoint.test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "waypoint.test_logger"

    def test_get_logger_with_implicit_name(self) -> None:
        """Test getting a logger with an implicit name."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "waypoint.test_logger"


class TestRungLogger:
    def test_get_run_logger_with_default(self) -> None:
        """Test getting the run logger with no active context but with a default."""
        default = "waypoint.test_default_logger"
        logger = get_run_logger(default=default)
        assert isinstance(logger, logging.Logger)
        assert logger.name == default

    def test_get_run_logger_raises_error_with_no_context_and_no_default(self) -> None:
        """Test getting the run logger with no active context and no default raises an error."""
        with pytest.raises(MissingContextError):
            _ = get_run_logger()

    def test_get_run_logger_inside_flow_context(self) -> None:
        """Test getting the run logger inside a flow context."""
        with flow_session(name="test_flow"):
            logger = get_run_logger()
            assert logger.extra == {"flow_run_name": "test_flow"}  # type: ignore

    def test_get_run_logger_inside_task_context(self) -> None:
        """Test getting the run logger inside a task context."""

        @task
        def test_task():
            logger = get_run_logger()
            assert logger.extra["task_run_name"].endswith("test_task-1")  # pyright: ignore

        with flow_session(name="test_flow"):
            with flow_session(name="session_flow"):
                logger = get_run_logger()
                assert logger.extra == {"flow_run_name": "session_flow"}  # type: ignore
                test_task()


class TestPatchPrint:
    """Test the patch_print context manager functionality."""

    def test_patch_print_disabled_by_default(self, capfd) -> None:
        """Test that patch_print disabled doesn't affect normal print."""
        with patch_print(enable=False):
            print("test message")

        captured = capfd.readouterr()
        assert captured.out == "test message\n"
        assert captured.err == ""

    def test_patch_print_with_no_active_context(self, capfd) -> None:
        """Test patch_print behavior when no flow/task context is active."""
        with patch_print(enable=True):
            print("no context message")

        captured = capfd.readouterr()
        assert captured.out == "no context message\n"

    def test_patch_print_context_manager_behavior(self, capfd) -> None:
        """Test that patch_print context manager properly manages print function."""
        # Store original print
        original_print = print

        with patch_print(enable=False):
            # print should be unchanged when disabled
            assert print is original_print

        # print should be restored after context
        assert print is original_print

    def test_patch_print_restores_print_after_exception(self, capfd) -> None:
        """Test that patch_print restores original print even if exception occurs."""
        original_print = print

        try:
            with patch_print(enable=True):
                print("before exception")
                raise ValueError("test exception")
        except ValueError:
            pass

        # print should be restored even after exception
        assert print is original_print

        # Verify we can still print normally
        print("after exception")
        captured = capfd.readouterr()
        assert "after exception\n" in captured.out


class TestSetupConsoleLogging:
    """Test setup_console_logging function."""

    def test_setup_console_logging_basic(self):
        """Test basic console logging setup."""
        # Clear existing handlers first
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()

        # Act
        setup_console_logging(level=logging.WARNING)

        # Assert - check that loggers have handlers
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            assert len(logger.handlers) > 0
            # Check that at least one handler has the expected level
            handler_levels = [h.level for h in logger.handlers]
            assert logging.WARNING in handler_levels

    def test_setup_console_logging_prevents_duplicates(self):
        """Test that setup_console_logging prevents duplicate handlers."""
        # Clear existing handlers first
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()

        # Act - call setup twice
        setup_console_logging(level=logging.INFO)
        initial_handler_counts = {}
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            initial_handler_counts[logger_name] = len(logger.handlers)

        setup_console_logging(level=logging.INFO)

        # Assert - handler counts should not increase
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            assert len(logger.handlers) == initial_handler_counts[logger_name]

    @patch("rich.logging.RichHandler")
    @patch("rich.get_console")
    def test_setup_console_logging_with_rich(self, mock_get_console, mock_rich_handler_class):
        """Test console logging setup with Rich enabled."""
        # Clear existing handlers first
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()

        mock_console = Mock()
        mock_get_console.return_value = mock_console
        mock_rich_handler = Mock()
        mock_rich_handler_class.return_value = mock_rich_handler

        # Act
        setup_console_logging(level=logging.DEBUG, traceback=True, use_rich=True)

        # Assert - Rich handlers should be created with correct parameters
        expected_calls = len(["waypoint", "waypoint.flow", "waypoint.task"])
        assert mock_rich_handler_class.call_count == expected_calls
        mock_rich_handler_class.assert_called_with(
            rich_tracebacks=True, omit_repeated_times=False, console=mock_console
        )


class TestSetupFileLogging:
    """Test setup_file_logging function."""

    def test_setup_file_logging_basic(self):
        """Test basic file logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clear existing handlers first - remove before closing to avoid logging errors
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                # Remove only file handlers to avoid interfering with other tests
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                for handler in file_handlers:
                    logger.removeHandler(handler)  # Remove first
                    handler.close()  # Then close

            log_file = Path(temp_dir) / "run.log"
            # Act
            setup_file_logging(log_file, level=logging.ERROR)

            # Assert - check that log file was created and loggers have file handlers
            assert log_file.exists()

            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                assert len(file_handlers) > 0
                # Check that at least one file handler has the expected level
                handler_levels = [h.level for h in file_handlers]
                assert logging.ERROR in handler_levels

            # Clean up handlers to prevent issues with temp directory cleanup
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                for handler in file_handlers:
                    logger.removeHandler(handler)
                    handler.close()

    def test_setup_file_logging_prevents_duplicates(self):
        """Test that setup_file_logging prevents duplicate handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clear existing handlers first - do this more carefully to avoid logging errors
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                for handler in file_handlers:
                    logger.removeHandler(handler)  # Remove first
                    handler.close()  # Then close

            # Act - call setup twice
            setup_file_logging(Path(temp_dir), level=logging.INFO)
            initial_handler_counts = {}
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                initial_handler_counts[logger_name] = len(file_handlers)

            setup_file_logging(Path(temp_dir), level=logging.INFO)

            # Assert - file handler counts should not increase
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                assert len(file_handlers) == initial_handler_counts[logger_name]

            # Clean up handlers to prevent issues with temp directory cleanup
            for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
                logger = logging.getLogger(logger_name)
                file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
                for handler in file_handlers:
                    logger.removeHandler(handler)
                    handler.close()

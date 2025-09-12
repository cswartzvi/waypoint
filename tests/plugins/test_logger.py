import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from waypoint.flows import flow
from waypoint.flows import flow_session
from waypoint.hooks.manager import register_hooks
from waypoint.plugins.logger import WaypointLogger
from waypoint.plugins.logger import _console_handler
from waypoint.plugins.logger import _file_handler
from waypoint.plugins.logger import _setup_app_loggers
from waypoint.plugins.logger import _setup_console_logging
from waypoint.plugins.logger import _setup_file_logging
from waypoint.tasks import submit_task
from waypoint.tasks import task


@pytest.fixture(autouse=True, scope="function")
def plugin():
    """Create and register a WaypointLogger plugin for each test."""
    plugin_ = WaypointLogger()
    register_hooks(plugin_)
    yield plugin_


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file logging tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestLoggerPluginInit:
    """Test suite for WaypointLogger plugin initialization."""

    @patch("waypoint.plugins.logger._setup_console_logging")
    def test_logger_init_with_console_enabled(self, mock_setup_console):
        """Test that WaypointLogger initializes console logging when not disabled."""
        # Act
        WaypointLogger(console_level=logging.DEBUG, traceback=True)

        # Assert
        mock_setup_console.assert_called_once_with(level=logging.DEBUG, traceback=True)

    @patch("waypoint.plugins.logger._setup_console_logging")
    def test_logger_init_with_console_disabled(self, mock_setup_console):
        """Test that WaypointLogger skips console logging when disabled."""
        # Act
        WaypointLogger(disable_console=True)

        # Assert
        mock_setup_console.assert_not_called()

    @patch("waypoint.plugins.logger._setup_file_logging")
    def test_logger_init_with_file_logging(self, mock_setup_file, temp_dir):
        """Test that WaypointLogger initializes file logging when file_path provided."""
        # Act
        WaypointLogger(file_path=temp_dir, file_level=logging.ERROR)

        # Assert
        mock_setup_file.assert_called_once_with(temp_dir, level=logging.ERROR)

    @patch("waypoint.plugins.logger._setup_file_logging")
    def test_logger_init_with_string_file_path(self, mock_setup_file):
        """Test that WaypointLogger converts string path to Path object."""
        # Act
        WaypointLogger(file_path="/tmp/logs")

        # Assert
        mock_setup_file.assert_called_once_with(Path("/tmp/logs"), level=logging.INFO)

    @patch("waypoint.plugins.logger._setup_file_logging")
    def test_logger_init_without_file_logging(self, mock_setup_file):
        """Test that WaypointLogger skips file logging when no file_path provided."""
        # Act
        WaypointLogger()

        # Assert
        mock_setup_file.assert_not_called()


class TestLoggerIntegrationFlows:
    """Integration tests for logger plugin with flows."""

    def test_logger_plugin_with_simple_flow(self, plugin):
        """Test that logger plugin works with a simple flow execution."""

        @flow
        def simple_flow():
            return "simple_result"

        with flow_session():
            result = simple_flow()
            assert result == "simple_result"

    def test_logger_plugin_with_parameterized_flow(self, plugin):
        """Test that logger plugin works with parameterized flows."""

        @flow
        def param_flow(x: int, y: str):
            return f"{y}_{x}"

        with flow_session():
            result = param_flow(x=42, y="test")
            assert result == "test_42"

    def test_logger_plugin_with_failing_flow(self, plugin):
        """Test that logger plugin handles flow errors correctly."""

        @flow
        def failing_flow():
            raise ValueError("Test error")

        with flow_session():
            with pytest.raises(ValueError, match="Test error"):
                failing_flow()


class TestLoggerIntegrationTasks:
    """Integration tests for logger plugin with tasks."""

    def test_logger_plugin_with_simple_task(self, plugin):
        """Test that logger plugin works with simple task execution."""

        @task
        def simple_task(x: int) -> int:
            return x * 2

        with flow_session():
            future = submit_task(simple_task, x=5)
            result = future.result()
            assert result == 10

    def test_logger_plugin_with_multiple_tasks(self, plugin):
        """Test that logger plugin works with multiple task executions."""

        @task
        def add_task(x: int, y: int) -> int:
            return x + y

        @task
        def multiply_task(x: int) -> int:
            return x * 3

        with flow_session():
            future1 = submit_task(add_task, x=2, y=3)
            future2 = submit_task(multiply_task, x=future1.result())

            assert future1.result() == 5
            assert future2.result() == 15

    def test_logger_plugin_with_failing_task_sequential(self, plugin):
        """Test that logger plugin handles task errors correctly with sequential runner."""

        @task
        def failing_task():
            raise RuntimeError("Task failed")

        # Sequential runner now defers execution like other runners for consistency
        with flow_session(task_runner="sequential"):
            future = submit_task(failing_task)
            with pytest.raises(Exception):
                future.result()

    def test_logger_plugin_with_failing_task_threading(self, plugin):
        """Test that logger plugin handles task errors correctly with threading runner."""

        @task
        def failing_task():
            raise RuntimeError("Task failed")

        # Threading runner raises exception only when calling .result()
        with flow_session(task_runner="threading"):
            future = submit_task(failing_task)
            with pytest.raises(Exception):  # TaskRunError wraps the original exception
                future.result()

    def test_logger_plugin_with_flow_error(self, plugin):
        """Test logger plugin with flow that has errors."""

        @flow
        def failing_flow():
            raise ValueError("Flow error")

        with pytest.raises(ValueError):
            failing_flow()

    def test_logger_plugin_with_custom_task_runner(self, plugin):
        """Test logger plugin with non-standard task runner name."""
        from unittest.mock import Mock

        from waypoint.context import FlowRunContext
        from waypoint.flows import flow_session

        @task
        def test_task(x: int) -> int:
            return x + 1

        with flow_session():
            # Mock a custom task runner name
            context = FlowRunContext.get()
            if context:  # Ensure context exists
                original_runner = context.task_runner
                mock_runner = Mock()
                mock_runner.name = "custom_runner"
                mock_runner.submit = original_runner.submit
                context.task_runner = mock_runner

                future = submit_task(test_task, 5)
                result = future.result()
                assert result == 6

    def test_logger_plugin_with_cancelled_future(self, plugin):
        """Test logger plugin handles cancelled futures."""
        from unittest.mock import Mock

        from waypoint.plugins.logger import WaypointLogger
        from waypoint.task_run import TaskRun
        from waypoint.tasks import TaskData

        # Create mocks
        task_data = Mock(spec=TaskData)
        task_run = Mock(spec=TaskRun)
        task_run.task_id = "test-task-id"

        # Test cancelled future within a flow context
        with flow_session():
            logger_plugin = WaypointLogger()
            logger_plugin.after_task_future_result(
                task_data=task_data,
                task_run=task_run,
                error=None,
                cancelled=True,
                result=None,
                task_runner="threading",
            )

    def test_logger_plugin_with_failed_future(self, plugin):
        """Test logger plugin handles failed futures."""
        from unittest.mock import Mock

        from waypoint.plugins.logger import WaypointLogger
        from waypoint.task_run import TaskRun
        from waypoint.tasks import TaskData

        # Create mocks
        task_data = Mock(spec=TaskData)
        task_run = Mock(spec=TaskRun)
        task_run.task_id = "test-task-id"
        error = RuntimeError("Task failed")

        # Test failed future within a flow context
        with flow_session():
            logger_plugin = WaypointLogger()
            logger_plugin.after_task_future_result(
                task_data=task_data,
                task_run=task_run,
                error=error,
                cancelled=False,
                result=None,
                task_runner="custom_runner",  # Non-standard runner
            )


class TestSetupFunctions:
    """Test suite for the setup functions used by WaypointLogger."""

    @patch("waypoint.plugins.logger._setup_app_loggers")
    @patch("waypoint.plugins.logger._console_handler")
    @patch("logging.getLogger")
    def test_setup_console_logging(self, mock_get_logger, mock_console_handler, mock_setup_app):
        """Test that _setup_console_logging configures console handlers correctly."""
        # Arrange
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger
        mock_handler = Mock()
        mock_handler.name = "test_handler"
        mock_console_handler.return_value = mock_handler

        # Act
        _setup_console_logging(level=logging.WARNING, traceback=True)

        # Assert
        mock_setup_app.assert_called_once()
        mock_console_handler.assert_called()
        mock_handler.setLevel.assert_called_with(logging.WARNING)
        mock_logger.addHandler.assert_called()

    @patch("waypoint.plugins.logger._setup_app_loggers")
    @patch("waypoint.plugins.logger._file_handler")
    @patch("logging.getLogger")
    def test_setup_file_logging(self, mock_get_logger, mock_file_handler, mock_setup_app, temp_dir):
        """Test that _setup_file_logging configures file handlers correctly."""
        # Arrange
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger
        mock_handler = Mock()
        mock_handler.name = "test_handler"
        mock_file_handler.return_value = mock_handler

        # Act
        _setup_file_logging(temp_dir, level=logging.ERROR)

        # Assert
        mock_setup_app.assert_called_once()
        mock_file_handler.assert_called()
        mock_handler.setLevel.assert_called_with(logging.ERROR)
        mock_logger.addHandler.assert_called()

    @patch("logging.getLogger")
    def test_setup_app_loggers(self, mock_get_logger):
        """Test that _setup_app_loggers configures all required loggers."""
        # Arrange
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Act
        _setup_app_loggers(level=logging.WARNING)

        # Assert
        assert mock_get_logger.call_count == 3  # waypoint, waypoint.flow, waypoint.task
        mock_logger.setLevel.assert_called_with(logging.WARNING)
        assert mock_logger.propagate is False

    def test_console_handler(self):
        """Test that _console_handler creates a properly configured handler."""
        # Act
        handler = _console_handler("%(message)s", traceback=True)

        # Assert
        assert handler is not None
        assert hasattr(handler, "setLevel")
        assert hasattr(handler, "setFormatter")

    def test_file_handler(self, temp_dir):
        """Test that _file_handler creates a properly configured handler."""
        # Arrange
        log_file = temp_dir / "test.log"

        # Act
        handler = _file_handler(str(log_file), "%(asctime)s - %(message)s")

        # Assert
        assert handler is not None
        assert hasattr(handler, "setLevel")
        assert hasattr(handler, "setFormatter")

    def test_setup_file_logging_prevents_duplicate_handlers(self, temp_dir):
        """Test that _setup_file_logging prevents adding duplicate handlers."""
        # Act - call _setup_file_logging twice with the same path
        _setup_file_logging(temp_dir, level=logging.INFO)

        # Get the current handlers for one of the loggers
        logger = logging.getLogger("waypoint")
        initial_handler_count = len(logger.handlers)

        # Call again with the same path
        _setup_file_logging(temp_dir, level=logging.INFO)

        # Assert - handler count should not increase
        final_handler_count = len(logger.handlers)
        assert final_handler_count == initial_handler_count

        # Clean up file handlers to prevent race conditions with temp directory cleanup
        for logger_name in ["waypoint", "waypoint.flow", "waypoint.task"]:
            test_logger = logging.getLogger(logger_name)
            # Remove file handlers that point to the temp directory
            handlers_to_remove = [
                h
                for h in test_logger.handlers
                if (
                    hasattr(h, "baseFilename")
                    and str(temp_dir) in str(getattr(h, "baseFilename", ""))
                )
            ]
            for handler in handlers_to_remove:
                handler.close()
                test_logger.removeHandler(handler)

import logging

import pytest

from waypoint.exceptions import MissingContextError
from waypoint.flows import flow_session
from waypoint.logging import get_logger
from waypoint.logging import get_run_logger
from waypoint.logging import patch_print
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

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

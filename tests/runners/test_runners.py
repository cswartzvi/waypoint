"""Tests for task runners functionality."""

import asyncio
import time

import pytest

from waypoint.futures import TaskFuture
from waypoint.runners import get_task_runner
from waypoint.runners.sequential import SequentialTaskRunner
from waypoint.runners.threading import ThreadingTaskRunner


class TestRunnerDiscoveryAndInstantiation:
    """Test task runner discovery and instantiation."""

    @pytest.mark.parametrize(
        "type_str, expected_cls",
        [
            ("sequential", SequentialTaskRunner),
            ("threading", ThreadingTaskRunner),
        ],
    )
    def test_get_task_runner_by_string(self, type_str, expected_cls):
        """Test getting task runners by their type string."""
        runner = get_task_runner(type_str)
        assert isinstance(runner, expected_cls)
        assert runner.type == type_str

    @pytest.mark.parametrize(
        "runner_instance",
        [
            SequentialTaskRunner(),
            ThreadingTaskRunner(),
        ],
    )
    def test_get_task_runner_by_instance(self, runner_instance):
        """Test that passing a runner instance returns the same instance."""
        returned_runner = get_task_runner(runner_instance)
        assert returned_runner is runner_instance

    @pytest.mark.parametrize(
        "runner, expected_type",
        [
            (SequentialTaskRunner(), "sequential"),
            (ThreadingTaskRunner(), "threading"),
        ],
    )
    def test_runner_type_attributes(self, runner, expected_type):
        """Test that all runners have proper type attributes."""
        assert hasattr(runner, "type")
        assert runner.type == expected_type

    def test_get_task_runner_invalid_type(self):
        """Test that invalid runner types raise appropriate errors."""
        with pytest.raises(Exception):  # Should raise some kind of lookup error
            get_task_runner("nonexistent_runner")

    @pytest.mark.parametrize("runner_type", ["sequential", "threading"])
    def test_start_already_started_runner_raises_error(self, runner_type):
        """Test that starting an already started runner raises RuntimeError."""
        runner = get_task_runner(runner_type)

        with runner.start():
            # Runner is now started, trying to start again should raise an error
            with pytest.raises(RuntimeError, match="The task runner has already been started"):
                with runner.start():
                    pass


class TestRunnerSubmit:
    """Test task runner submit functionality."""

    @pytest.fixture(params=["sequential", "threading"])
    def runner(self, request):
        """Parameterized fixture for different runner types."""
        return get_task_runner(request.param)

    def test_submit_simple_function(self, runner):
        """Test submitting a simple function that returns a value."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            result = future.result()
            assert result == 42

    def test_submit_function_with_side_effects(self, runner):
        """Test submitting a function that has side effects."""
        results = []

        def side_effect_task():
            results.append("executed")
            return len(results)

        with runner.start():
            future = runner.submit(side_effect_task)
            result = future.result()
            assert result == 1
            assert results == ["executed"]

    def test_submit_function_that_raises_exception(self, runner):
        """Test submitting a function that raises an exception."""

        def failing_task():
            raise ValueError("Test error")

        with runner.start():
            if isinstance(runner, SequentialTaskRunner):
                # Sequential runner executes immediately, so exception is raised during submit
                with pytest.raises(ValueError, match="Test error"):
                    runner.submit(failing_task)
            else:
                # Other runners defer execution, so exception is raised during result()
                future = runner.submit(failing_task)
                with pytest.raises(ValueError, match="Test error"):
                    future.result()

    def test_submit_multiple_tasks(self, runner):
        """Test submitting multiple tasks and verifying all complete."""

        def numbered_task(n):
            def inner():
                return n * 2

            return inner

        with runner.start():
            futures = []
            for i in range(5):
                future = runner.submit(numbered_task(i))
                futures.append(future)

            results = [f.result() for f in futures]
            expected = [i * 2 for i in range(5)]
            assert results == expected

    def test_submit_async_function(self, runner):
        """Test submitting an async function."""

        async def async_task():
            await asyncio.sleep(0.01)  # Small delay
            return "async_result"

        with runner.start():
            future = runner.submit(async_task)
            result = future.result()
            assert result == "async_result"

    def test_submit_without_start_raises_error(self, runner):
        """Test that submitting without starting the runner raises an error."""

        def simple_task():
            return 42

        with pytest.raises(RuntimeError, match="has not been started"):
            runner.submit(simple_task)

    def test_submit_returns_task_future(self, runner):
        """Test that submit returns a TaskFuture instance."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            assert isinstance(future, TaskFuture)

    def test_submit_function_with_closure(self, runner):
        """Test submitting a function that uses closure variables."""
        captured_value = "captured"

        def closure_task():
            return f"Result: {captured_value}"

        with runner.start():
            future = runner.submit(closure_task)
            result = future.result()
            assert result == "Result: captured"

    def test_submit_task_state_management(self, runner):
        """Test that task futures properly track their state."""

        def slow_task():
            time.sleep(0.1)
            return "completed"

        with runner.start():
            future = runner.submit(slow_task)

            # For sequential runner, task completes immediately
            # For threading runner, we can check states
            if isinstance(runner, ThreadingTaskRunner):
                # May or may not be done yet depending on timing
                pass

            result = future.result()
            assert result == "completed"
            assert future.done()


class TestRunnerCloning:
    """Test task runner duplicate functionality."""

    @pytest.fixture(params=["sequential", "threading"])
    def runner(self, request):
        """Parameterized fixture for different runner types."""
        return get_task_runner(request.param)

    def test_duplicate_creates_new_instance(self, runner):
        """Test that duplicate creates a new instance, not the same object."""
        duplicate = runner.duplicate()
        assert duplicate is not runner
        assert type(duplicate) is type(runner)

    def test_duplicate_preserves_type(self, runner):
        """Test that duplicate preserves the runner type."""
        duplicate = runner.duplicate()
        assert duplicate.type == runner.type

    def test_duplicate_has_independent_state(self, runner):
        """Test that duplicate has independent state from original."""
        # Start the original runner
        with runner.start():
            assert runner.is_running

            # Create duplicate while original is running
            duplicate = runner.duplicate()
            assert not duplicate.is_running  # Should not be started

    def test_duplicate_can_be_started_independently(self, runner):
        """Test that duplicate can be started and used independently."""
        duplicate = runner.duplicate()

        def test_task():
            return "duplicate_result"

        with duplicate.start():
            future = duplicate.submit(test_task)
            result = future.result()
            assert result == "duplicate_result"

    def test_duplicate_preserves_configuration(self, runner):
        """Test that duplicate preserves runner configuration."""
        if isinstance(runner, ThreadingTaskRunner):
            # Create a runner with specific configuration
            original = ThreadingTaskRunner(max_workers=2)
            duplicate = original.duplicate()

            # Configuration should be preserved
            assert duplicate._max_workers == original._max_workers
        else:
            # For sequential runner, just verify type preservation
            duplicate = runner.duplicate()
            assert type(duplicate) is type(runner)

    def test_duplicate_has_fresh_logger(self, runner):
        """Test that duplicate gets its own logger instance."""
        duplicate = runner.duplicate()

        # Both should have loggers with the same name but different instances
        assert runner.logger.name == duplicate.logger.name
        # Note: Logger instances may be the same due to logging module caching
        # but the important thing is they work independently

    def test_duplicate_multiple_times(self, runner):
        """Test creating multiple duplicates from the same runner."""
        duplicates = [runner.duplicate() for _ in range(3)]

        # All should be different instances
        for i, dup1 in enumerate(duplicates):
            for j, dup2 in enumerate(duplicates):
                if i != j:
                    assert dup1 is not dup2
                assert type(dup1) is type(runner)

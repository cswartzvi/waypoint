"""Tests for task runners functionality."""

import asyncio
import time
from typing import AsyncGenerator, Generator

import pytest

from waypoint import flow
from waypoint import submit_task
from waypoint import task
from waypoint.exceptions import TaskRunError
from waypoint.flows import flow_session
from waypoint.runners import get_task_runner
from waypoint.runners.sequential import SequentialTaskRunner
from waypoint.runners.threading import ThreadingTaskRunner


class TestTaskRunnerDiscovery:
    """Test task runner discovery and instantiation."""

    def test_get_task_runner_by_string(self):
        """Test getting task runners by their type string."""
        sequential = get_task_runner("sequential")
        assert isinstance(sequential, SequentialTaskRunner)
        assert sequential.type == "sequential"

        threading = get_task_runner("threading")
        assert isinstance(threading, ThreadingTaskRunner)
        assert threading.type == "threading"

    def test_get_task_runner_by_instance(self):
        """Test that passing a runner instance returns the same instance."""
        original_runner = SequentialTaskRunner()
        returned_runner = get_task_runner(original_runner)
        assert returned_runner is original_runner

    def test_get_task_runner_invalid_type(self):
        """Test that invalid runner types raise appropriate errors."""
        with pytest.raises(Exception):  # Should raise some kind of lookup error
            get_task_runner("nonexistent_runner")

    def test_runner_type_attributes(self):
        """Test that all runners have proper type attributes."""
        sequential = SequentialTaskRunner()
        assert hasattr(sequential, "type")
        assert sequential.type == "sequential"

        threading = ThreadingTaskRunner()
        assert hasattr(threading, "type")
        assert threading.type == "threading"


class TestSequentialTaskRunner:
    """Test the sequential task runner specifically."""

    def test_sequential_execution_order(self):
        """Test that tasks execute in submission order with sequential runner."""
        results = []

        @task
        def ordered_task(value: int, delay: float = 0.01) -> int:
            time.sleep(delay)
            results.append(value)
            return value

        with flow_session(name="test_sequential", task_runner="sequential"):
            # Submit tasks that should execute in order
            futures = []
            for i in range(5):
                future = submit_task(ordered_task, i, delay=0.01)
                futures.append(future)

            # Wait for all to complete
            final_results = [f.result() for f in futures]

        # Results should be in submission order
        assert results == [0, 1, 2, 3, 4]
        assert final_results == [0, 1, 2, 3, 4]

    def test_sequential_runner_duplicate(self):
        """Test that duplicating a sequential runner works correctly."""
        original = SequentialTaskRunner()
        duplicate = original.duplicate()

        assert isinstance(duplicate, SequentialTaskRunner)
        assert duplicate is not original
        assert duplicate.type == original.type


class TestThreadingTaskRunner:
    """Test the threading task runner specifically."""

    def test_threading_runner_default_workers(self):
        """Test threading runner with default number of workers."""
        runner = ThreadingTaskRunner()
        # Default should be None (which uses ThreadPoolExecutor default)
        assert runner._max_workers is None

    def test_threading_runner_custom_workers(self):
        """Test threading runner with custom number of workers."""
        runner = ThreadingTaskRunner(max_workers=4)
        assert runner._max_workers == 4

    def test_threading_runner_duplicate(self):
        """Test that duplicating a threading runner preserves settings."""
        original = ThreadingTaskRunner(max_workers=3)
        duplicate = original.duplicate()

        assert isinstance(duplicate, ThreadingTaskRunner)
        assert duplicate is not original
        assert duplicate._max_workers == original._max_workers

    def test_threading_concurrent_execution(self):
        """Test that threading runner can execute tasks concurrently."""
        start_times = {}
        end_times = {}

        @task
        def concurrent_task(task_id: int) -> int:
            start_times[task_id] = time.time()
            time.sleep(0.1)  # Simulate work
            end_times[task_id] = time.time()
            return task_id

        with flow_session(name="test_threading", task_runner="threading"):
            # Submit multiple tasks
            futures = []
            _ = time.time()

            for i in range(3):
                future = submit_task(concurrent_task, i)
                futures.append(future)

            # Wait for all to complete
            results = [f.result() for f in futures]

        # All tasks should have started roughly at the same time
        # (within a reasonable window, accounting for thread startup)
        start_time_range = max(start_times.values()) - min(start_times.values())
        assert start_time_range < 0.05  # Should start within 50ms of each other

        # Results should be present (order might vary due to concurrency)
        assert set(results) == {0, 1, 2}


class TestTaskRunnerWithDifferentTaskTypes:
    """Test task runners with various task types."""

    def test_sync_tasks_with_different_runners(self):
        """Test synchronous tasks with different runners."""

        @task
        def sync_add(a: int, b: int) -> int:
            return a + b

        # Test with sequential runner
        with flow_session(name="sync_sequential", task_runner="sequential"):
            result1 = sync_add(3, 4)
            assert result1 == 7

        # Test with threading runner
        with flow_session(name="sync_threading", task_runner="threading"):
            result2 = sync_add(5, 6)
            assert result2 == 11

    def test_async_tasks_with_different_runners(self):
        """Test asynchronous tasks with different runners."""

        @task
        async def async_multiply(a: int, b: int) -> int:
            await asyncio.sleep(0.001)
            return a * b

        # Test with sequential runner
        with flow_session(name="async_sequential", task_runner="sequential"):
            result1 = asyncio.run(async_multiply(3, 4))
            assert result1 == 12

        # Test with threading runner
        with flow_session(name="async_threading", task_runner="threading"):
            result2 = asyncio.run(async_multiply(5, 6))
            assert result2 == 30

    def test_generator_tasks_with_different_runners(self):
        """Test generator tasks with different runners."""

        @task
        def sync_generator(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i

        @task
        async def async_generator(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i

        # Test sync generator with different runners
        with flow_session(name="gen_sequential", task_runner="sequential"):
            sync_result = list(sync_generator(3))
            assert sync_result == [0, 1, 2]

        with flow_session(name="gen_threading", task_runner="threading"):
            sync_result2 = list(sync_generator(3))
            assert sync_result2 == [0, 1, 2]

        # Test async generator with different runners
        async def collect_async_gen(generator):
            results = []
            async for item in generator:
                results.append(item)
            return results

        with flow_session(name="async_gen_sequential", task_runner="sequential"):
            async_result = asyncio.run(collect_async_gen(async_generator(3)))
            assert async_result == [0, 1, 2]

        with flow_session(name="async_gen_threading", task_runner="threading"):
            async_result2 = asyncio.run(collect_async_gen(async_generator(3)))
            assert async_result2 == [0, 1, 2]


class TestTaskRunnerExceptionHandling:
    """Test exception handling across different task runners."""

    def test_sync_task_exceptions_with_runners(self):
        """Test synchronous task exception handling with different runners."""

        @task
        def failing_sync_task() -> None:
            raise ValueError("Sync task failed")

        # Test with sequential runner
        with flow_session(name="sync_fail_sequential", task_runner="sequential"):
            with pytest.raises(TaskRunError, match="Sync task failed"):
                failing_sync_task()

        # Test with threading runner
        with flow_session(name="sync_fail_threading", task_runner="threading"):
            with pytest.raises(TaskRunError, match="Sync task failed"):
                failing_sync_task()

    def test_async_task_exceptions_with_runners(self):
        """Test asynchronous task exception handling with different runners."""

        @task
        async def failing_async_task() -> None:
            await asyncio.sleep(0.001)
            raise RuntimeError("Async task failed")

        # Test with sequential runner
        with flow_session(name="async_fail_sequential", task_runner="sequential"):
            with pytest.raises(RuntimeError, match="Async task failed"):
                asyncio.run(failing_async_task())

        # Test with threading runner
        with flow_session(name="async_fail_threading", task_runner="threading"):
            with pytest.raises(RuntimeError, match="Async task failed"):
                asyncio.run(failing_async_task())

    def test_submitted_task_exceptions_with_runners(self):
        """Test exception handling in submitted tasks with different runners."""

        @task
        def failing_submitted_task() -> None:
            raise ConnectionError("Submitted task failed")

        # Test with sequential runner (should fail on submission)
        with flow_session(name="submit_fail_sequential", task_runner="sequential"):
            with pytest.raises(TaskRunError, match="Submitted task failed"):
                future = submit_task(failing_submitted_task)

        # Test with threading runner
        with flow_session(name="submit_fail_threading", task_runner="threading"):
            future = submit_task(failing_submitted_task)
            with pytest.raises(TaskRunError, match="Submitted task failed"):
                future.result()


class TestTaskRunnerIntegration:
    """Test task runners in integration scenarios."""

    def test_mixed_task_types_in_single_flow(self):
        """Test a flow that uses multiple task types with a single runner."""

        @task
        def sync_task(x: int) -> int:
            return x * 2

        @task
        async def async_task(x: int) -> int:
            await asyncio.sleep(0.001)
            return x + 10

        @task
        def generator_task(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i

        @flow
        def mixed_flow(value: int) -> dict:
            # Use different task types in one flow
            doubled = sync_task(value)
            added = asyncio.run(async_task(doubled))
            generated = list(generator_task(3))

            return {"original": value, "doubled": doubled, "added": added, "generated": generated}

        with flow_session(name="mixed_integration", task_runner="sequential"):
            result = mixed_flow(5)
            expected = {"original": 5, "doubled": 10, "added": 20, "generated": [0, 1, 2]}
            assert result == expected

    def test_nested_flow_sessions_with_different_runners(self):
        """Test nested flow sessions using different task runners."""

        @task
        def runner_aware_task(multiplier: int) -> str:
            # This task doesn't actually need to know about the runner,
            # but we can test that it works in nested contexts
            return f"result_{multiplier}"

        with flow_session(name="outer", task_runner="sequential"):
            outer_result = runner_aware_task(1)

            with flow_session(name="inner", task_runner="threading"):
                inner_result = runner_aware_task(2)

        assert outer_result == "result_1"
        assert inner_result == "result_2"

    def test_flow_with_concurrent_task_submissions(self):
        """Test a flow that submits multiple tasks concurrently."""

        @task
        def parallel_work(task_id: int, work_amount: int) -> dict:
            start = time.time()
            # Simulate some work
            total = sum(range(work_amount))
            end = time.time()
            return {"task_id": task_id, "result": total, "duration": end - start}

        @flow
        def parallel_flow() -> list[dict]:
            # Submit multiple tasks concurrently
            futures = []
            for i in range(3):
                future = submit_task(parallel_work, i, 100)
                futures.append(future)

            # Collect all results
            results = [f.result() for f in futures]
            return results

        with flow_session(name="parallel_integration", task_runner="threading"):
            results = parallel_flow()

            # Should have 3 results
            assert len(results) == 3

            # Each result should have the expected structure
            for i, result in enumerate(results):
                assert result["task_id"] == i
                assert "result" in result
                assert "duration" in result

    def test_task_runner_context_management(self):
        """Test that task runners properly manage their execution contexts."""

        @task
        def context_test_task() -> str:
            return "context_test"

        # Test that tasks work within flow sessions
        with flow_session(name="context_test", task_runner="sequential"):
            result = context_test_task()
            assert result == "context_test"

        # Test that task submission works within contexts
        with flow_session(name="context_submit_test", task_runner="threading"):
            future = submit_task(context_test_task)
            result = future.result()
            assert result == "context_test"

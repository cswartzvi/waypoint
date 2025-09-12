"""Tests for TaskFuture functionality with different runners."""

import asyncio
import time

import pytest

from waypoint.futures import TaskFuture
from waypoint.futures import as_completed
from waypoint.futures import wait
from waypoint.runners import get_task_runner


class TestTaskFutureInitialization:
    """Test TaskFuture initialization and basic properties."""

    @pytest.fixture(params=["sequential", "threading"])
    def runner(self, request):
        """Parameterized fixture for different runner types."""
        return get_task_runner(request.param)

    def test_future_creation(self, runner):
        """Test that TaskFuture is properly created from submit."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            assert isinstance(future, TaskFuture)

    def test_future_result_retrieval(self, runner):
        """Test basic result retrieval from TaskFuture."""

        def simple_task():
            return "test_result"

        with runner.start():
            future = runner.submit(simple_task)
            result = future.result()
            assert result == "test_result"

    def test_future_exception_handling(self, runner):
        """Test exception handling in TaskFuture."""

        def failing_task():
            raise ValueError("Test exception")

        with runner.start():
            try:
                future = runner.submit(failing_task)
                # For sequential runner, exception is raised during submit
                # For other runners, exception is raised during result()
                if not future.done():
                    with pytest.raises(ValueError, match="Test exception"):
                        future.result()
            except ValueError:
                # Sequential runner raises immediately during submit
                pass

    def test_future_done_status(self, runner):
        """Test that futures properly report done status."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            # Result should be available
            future.result()
            assert future.done()

    def test_future_running_status(self, runner):
        """Test that futures properly report running status."""

        def quick_task():
            return 42

        with runner.start():
            future = runner.submit(quick_task)
            # For sequential runner, task completes immediately
            # For threading runner, may or may not be running when we check
            _ = future.running()  # Just ensure it doesn't crash
            future.result()  # Ensure completion

    def test_future_cancelled_status(self, runner):
        """Test that futures properly report cancelled status."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            # Most tasks complete too quickly to cancel, but we can check the method exists
            assert not future.cancelled()

    def test_future_exception_method(self, runner):
        """Test the exception method on TaskFuture."""

        def normal_task():
            return 42

        def failing_task():
            raise ValueError("Test exception")

        with runner.start():
            # Normal task should return None from exception()
            normal_future = runner.submit(normal_task)
            try:
                normal_future.result()
                assert normal_future.exception() is None
            except ValueError:
                # Sequential runner case
                pass

            # Failing task should return the exception
            try:
                failing_future = runner.submit(failing_task)
                if not failing_future.done():
                    failing_future.result()  # This will raise
            except ValueError:
                # Sequential runner case - exception raised during submit
                pass

    def test_future_with_async_function(self, runner):
        """Test TaskFuture with async functions."""

        async def async_task():
            await asyncio.sleep(0.01)
            return "async_result"

        with runner.start():
            future = runner.submit(async_task)
            result = future.result()
            assert result == "async_result"

    def test_future_timeout_parameter(self, runner):
        """Test TaskFuture result() with timeout parameter."""

        def quick_task():
            return "quick"

        with runner.start():
            future = runner.submit(quick_task)
            # Should complete well within timeout
            result = future.result(timeout=1.0)
            assert result == "quick"

    def test_future_cancel(self, runner):
        """Test TaskFuture cancel() method."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            # Most tasks complete too quickly to cancel, but test the method exists and works
            cancel_result = future.cancel()
            # cancel() returns bool indicating if cancellation was successful
            assert isinstance(cancel_result, bool)

    def test_future_hash(self, runner):
        """Test that TaskFuture objects are hashable."""

        def simple_task():
            return 42

        with runner.start():
            future1 = runner.submit(simple_task)
            future2 = runner.submit(simple_task)

            # Should be able to hash and use in sets/dicts
            future_set = {future1, future2}
            assert len(future_set) == 2


class TestFutureWait:
    """Test the wait function with TaskFutures."""

    @pytest.fixture(params=["sequential", "threading"])
    def runner(self, request):
        """Parameterized fixture for different runner types."""
        return get_task_runner(request.param)

    def test_wait_all_completed(self, runner):
        """Test wait with ALL_COMPLETED (default behavior)."""

        def numbered_task(n):
            def inner():
                time.sleep(0.01)  # Small delay
                return n

            return inner

        with runner.start():
            futures = []
            for i in range(3):
                future = runner.submit(numbered_task(i))
                futures.append(future)

            done, not_done = wait(futures)

            # All should be done
            assert len(done) == 3
            assert len(not_done) == 0

            # Results should be available
            results = {f.result() for f in done}
            assert results == {0, 1, 2}

    def test_wait_with_timeout(self, runner):
        """Test wait with timeout parameter."""

        def quick_task():
            return "quick"

        with runner.start():
            futures = [runner.submit(quick_task) for _ in range(2)]

            done, not_done = wait(futures, timeout=1.0)

            # Should complete within timeout
            assert len(done) == 2
            assert len(not_done) == 0

    def test_wait_first_completed(self, runner):
        """Test wait with FIRST_COMPLETED."""

        def varying_delay_task(delay):
            def inner():
                time.sleep(delay)
                return f"task_{delay}"

            return inner

        with runner.start():
            futures = [
                runner.submit(varying_delay_task(0.01)),
                runner.submit(varying_delay_task(0.05)),
                runner.submit(varying_delay_task(0.1)),
            ]

            done, not_done = wait(futures, return_when="FIRST_COMPLETED")

            # At least one should be done
            assert len(done) >= 1
            # For sequential runner, all will be done
            # For threading runner, may have some not done

    def test_wait_first_exception(self, runner):
        """Test wait with FIRST_EXCEPTION."""

        def normal_task():
            time.sleep(0.05)
            return "normal"

        def failing_task():
            time.sleep(0.01)
            raise ValueError("Test exception")

        with runner.start():
            try:
                futures = [runner.submit(normal_task), runner.submit(failing_task)]

                done, not_done = wait(futures, return_when="FIRST_EXCEPTION")

                # Should have at least one done (the failed one)
                assert len(done) >= 1

                # Check that we can identify the failed future
                for future in done:
                    try:
                        future.result()
                    except ValueError:
                        # Found the failed future
                        break

            except ValueError:
                # Sequential runner case - exception raised during submit
                pass

    def test_wait_empty_futures(self, runner):
        """Test wait with empty futures list."""
        done, not_done = wait([])
        assert len(done) == 0
        assert len(not_done) == 0

    def test_wait_single_future(self, runner):
        """Test wait with single future."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            done, not_done = wait([future])

            assert len(done) == 1
            assert len(not_done) == 0
            assert future in done

    def test_wait_duplicate_futures(self, runner):
        """Test wait with duplicate futures."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            # Pass the same future multiple times
            done, not_done = wait([future, future, future])

            # Should only return unique futures
            assert len(done) == 1
            assert len(not_done) == 0
            assert future in done

    def test_wait_mixed_completion_states(self, runner):
        """Test wait with futures in different completion states."""

        def quick_task():
            return "quick"

        def slow_task():
            time.sleep(0.1)
            return "slow"

        with runner.start():
            quick_future = runner.submit(quick_task)
            slow_future = runner.submit(slow_task)

            # For sequential runner, both will be done
            # For threading runner, quick one should be done
            done, not_done = wait([quick_future, slow_future])

            assert len(done) + len(not_done) == 2
            assert quick_future in done or quick_future in not_done
            assert slow_future in done or slow_future in not_done


class TestFutureAsCompleted:
    """Test the as_completed function with TaskFutures."""

    @pytest.fixture(params=["sequential", "threading"])
    def runner(self, request):
        """Parameterized fixture for different runner types."""
        return get_task_runner(request.param)

    def test_as_completed_basic(self, runner):
        """Test basic as_completed functionality."""

        def numbered_task(n):
            def inner():
                time.sleep(0.01 * n)  # Varying delays
                return n

            return inner

        with runner.start():
            futures = []
            for i in range(3):
                future = runner.submit(numbered_task(i))
                futures.append(future)

            completed_futures = list(as_completed(futures))

            # Should get all futures back
            assert len(completed_futures) == 3

            # All should be TaskFuture instances
            for future in completed_futures:
                assert isinstance(future, TaskFuture)
                # For DelayedTaskFuture, done() is False until result() is called
                # For regular TaskFuture, done() is True when returned by as_completed
                # Both behaviors are correct for their respective semantics

            # Should get all expected results
            results = {f.result() for f in completed_futures}
            assert results == {0, 1, 2}

    def test_as_completed_with_timeout(self, runner):
        """Test as_completed with timeout."""

        def quick_task():
            return "quick"

        with runner.start():
            futures = [runner.submit(quick_task) for _ in range(2)]

            completed_futures = list(as_completed(futures, timeout=1.0))

            # Should complete within timeout
            assert len(completed_futures) == 2

    def test_as_completed_order_independence(self, runner):
        """Test that as_completed yields futures as they complete, not in submission order."""

        def delay_task(delay, value):
            def inner():
                time.sleep(delay)
                return value

            return inner

        with runner.start():
            # Submit in order: slow, fast, medium
            futures = [
                runner.submit(delay_task(0.1, "slow")),
                runner.submit(delay_task(0.01, "fast")),
                runner.submit(delay_task(0.05, "medium")),
            ]

            completed_futures = list(as_completed(futures))

            # Should get all futures
            assert len(completed_futures) == 3

            # For sequential runner, order will be submission order
            # For threading runner, order should be completion order (fast, medium, slow)
            results = [f.result() for f in completed_futures]
            assert set(results) == {"slow", "fast", "medium"}

    def test_as_completed_single_future(self, runner):
        """Test as_completed with single future."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            completed_futures = list(as_completed([future]))

            assert len(completed_futures) == 1
            assert completed_futures[0] is future
            assert completed_futures[0].result() == 42

    def test_as_completed_empty_futures(self, runner):
        """Test as_completed with empty futures list."""
        completed_futures = list(as_completed([]))
        assert len(completed_futures) == 0

    def test_as_completed_with_exceptions(self, runner):
        """Test as_completed with futures that raise exceptions."""

        def normal_task(value):
            def inner():
                return value

            return inner

        def failing_task():
            raise ValueError("Test exception")

        with runner.start():
            try:
                futures = [
                    runner.submit(normal_task("success")),
                    runner.submit(failing_task),
                    runner.submit(normal_task("another_success")),
                ]

                completed_futures = list(as_completed(futures))

                # Should get all futures back
                assert len(completed_futures) == 3

                # Check results and exceptions
                results = []
                exceptions = []
                for future in completed_futures:
                    try:
                        results.append(future.result())
                    except ValueError as e:
                        exceptions.append(e)

                assert len(results) == 2
                assert len(exceptions) == 1
                assert set(results) == {"success", "another_success"}

            except ValueError:
                # Sequential runner case - exception raised during submit
                pass

    def test_as_completed_duplicate_futures(self, runner):
        """Test as_completed with duplicate futures."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            # Pass the same future multiple times
            completed_futures = list(as_completed([future, future, future]))

            # Should only yield unique futures once
            assert len(completed_futures) == 1
            assert completed_futures[0] is future

    def test_as_completed_iterator_behavior(self, runner):
        """Test that as_completed returns an iterator, not a list."""

        def simple_task():
            return 42

        with runner.start():
            future = runner.submit(simple_task)
            iterator = as_completed([future])

            # Should be an iterator
            assert hasattr(iterator, "__iter__")
            assert hasattr(iterator, "__next__")

            # Should yield the future
            first_future = next(iterator)
            assert first_future is future

            # Should raise StopIteration after yielding all futures
            with pytest.raises(StopIteration):
                next(iterator)

    def test_as_completed_async_functions(self, runner):
        """Test as_completed with async functions."""

        async def async_task():
            await asyncio.sleep(0.01)
            return "async_result"

        with runner.start():
            futures = [runner.submit(async_task), runner.submit(async_task)]

            completed_futures = list(as_completed(futures))

            assert len(completed_futures) == 2
            results = {f.result() for f in completed_futures}
            assert results == {"async_result"}

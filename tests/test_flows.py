"""Tests for the flow decorator and flow execution functionality."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest

from waypoint import flow
from waypoint import task
from waypoint.exceptions import FlowRunError
from waypoint.flows import FlowData
from waypoint.flows import flow_session
from waypoint.flows import get_flow_data


@pytest.fixture(autouse=True)
def flow_context():
    """Automatically provide a flow context for all flow tests."""
    with flow_session(name="test_flow", task_runner="sequential"):
        yield


class TestFlowDecorator:
    """Test the @flow decorator functionality."""

    def test_decorator_with_custom_name(self):
        """Test @flow decorator with custom name parameter."""

        @flow(name="custom_flow_name")
        def named_flow() -> str:
            return "test"

        flow_data = get_flow_data(named_flow)
        assert flow_data.name == "custom_flow_name"

    def test_decorator_without_custom_name_uses_function_name(self):
        """Test @flow decorator uses function name when no custom name provided."""

        @flow
        def my_flow_function() -> str:
            return "test"

        flow_data = get_flow_data(my_flow_function)
        assert "my_flow_function" in flow_data.name

    def test_decorator_preserves_function_metadata(self):
        """Test that @flow decorator preserves original function metadata."""

        @flow
        def documented_flow(x: int) -> int:
            """This is a documented flow."""
            return x

        assert "documented_flow" in documented_flow.__name__
        assert documented_flow.__doc__ and "documented flow" in documented_flow.__doc__

    def test_flow_data_attachment(self):
        """Test that FlowData is properly attached to decorated functions."""

        @flow
        def test_flow() -> None:
            pass

        flow_data = get_flow_data(test_flow)
        assert isinstance(flow_data, FlowData)
        assert "test_flow" in flow_data.name
        assert flow_data.is_async is False
        assert flow_data.is_generator is False

    def test_async_flow_data_detection(self):
        """Test that async functions are correctly detected in FlowData."""

        @flow
        async def async_test_flow() -> None:
            pass

        flow_data = get_flow_data(async_test_flow)
        assert flow_data.is_async is True
        assert flow_data.is_generator is False

    def test_generator_flow_data_detection(self):
        """Test that generator functions are correctly detected in FlowData."""

        @flow
        def generator_test_flow() -> Generator[int, None, None]:
            yield 1

        flow_data = get_flow_data(generator_test_flow)
        assert flow_data.is_async is False
        assert flow_data.is_generator is True

    def test_async_generator_flow_data_detection(self):
        """Test that async generator functions are correctly detected in FlowData."""

        @flow
        async def async_generator_test_flow() -> AsyncGenerator[int, None]:
            yield 1

        flow_data = get_flow_data(async_generator_test_flow)
        assert flow_data.is_async is True
        assert flow_data.is_generator is True

    def test_positional_arguments(self):
        """Test flow execution with positional arguments."""

        @flow
        def concat_flow(a: str, b: str, c: str) -> str:
            return f"{a}-{b}-{c}"

        result = concat_flow("hello", "world", "flow")
        assert result == "hello-world-flow"

    def test_keyword_arguments(self):
        """Test flow execution with keyword arguments."""

        @flow
        def format_flow(message: str, prefix: str = "FLOW", suffix: str = "!") -> str:
            return f"[{prefix}] {message}{suffix}"

        result = format_flow("Test message", prefix="DEBUG", suffix=".")
        assert result == "[DEBUG] Test message."

    def test_mixed_arguments(self):
        """Test flow execution with mixed positional and keyword arguments."""

        @flow
        def mixed_flow(a: int, b: int, multiplier: int = 1, offset: int = 0) -> int:
            return (a + b) * multiplier + offset

        result = mixed_flow(10, 20, multiplier=2, offset=5)
        assert result == 65  # (10 + 20) * 2 + 5

    def test_default_parameter_values(self):
        """Test that default parameter values are properly handled."""

        @flow
        def with_defaults_flow(value: int, factor: int = 3, add: int = 0) -> int:
            return value * factor + add

        # Test with all defaults
        result1 = with_defaults_flow(5)
        assert result1 == 15  # 5 * 3 + 0

        # Test with partial defaults
        result2 = with_defaults_flow(5, factor=4)
        assert result2 == 20  # 5 * 4 + 0

        # Test with no defaults
        result3 = with_defaults_flow(5, factor=4, add=2)
        assert result3 == 22  # 5 * 4 + 2

    def test_get_flow_data_invalid_function(self):
        """Test get_flow_data with non-flow function raises error."""

        def regular_function():
            pass

        from waypoint.exceptions import InvalidFlowError

        with pytest.raises(InvalidFlowError):
            get_flow_data(regular_function)

    def test_is_flow_with_flow_function(self):
        """Test is_flow returns True for decorated functions."""
        from waypoint.flows import is_flow

        @flow
        def flow_function():
            pass

        assert is_flow(flow_function) is True

    def test_is_flow_with_regular_function(self):
        """Test is_flow returns False for regular functions."""
        from waypoint.flows import is_flow

        def regular_function():
            pass

        assert is_flow(regular_function) is False


class TestFlowExecution:
    """Test flow execution behavior."""

    def test_sync_flow_execution(self):
        """Test synchronous flow execution."""

        @flow
        def simple_flow(x: int, y: int) -> int:
            return x + y

        result = simple_flow(3, 4)
        assert result == 7

    def test_sync_generator_flow_execution(self):
        """Test synchronous generator flow execution."""

        @flow
        def generator_flow(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i * 3

        result = list(generator_flow(3))
        assert result == [0, 3, 6]

    def test_nested_flows(self):
        """Test flows calling other flows."""

        @flow
        def inner_flow(x: int) -> int:
            return x * 2

        @flow
        def outer_flow(x: int) -> int:
            inner_result = inner_flow(x)
            return inner_result + 1

        result = outer_flow(5)
        assert result == 11  # (5 * 2) + 1

    def test_async_flow_execution(self):
        """Test asynchronous flow execution."""

        @flow
        async def async_flow(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x + y

        result = asyncio.run(async_flow(3, 4))
        assert result == 7

    def test_async_generator_flow_execution(self):
        """Test asynchronous generator flow execution."""

        @flow
        async def async_generator_flow(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i * 3

        async def collect_results():
            results = []
            async for value in async_generator_flow(3):
                results.append(value)
            return results

        result = asyncio.run(collect_results())
        assert result == [0, 3, 6]

    def test_nested_async_flows(self):
        """Test async flows calling other async flows."""

        @flow
        async def inner_async_flow(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        @flow
        async def outer_async_flow(x: int) -> int:
            inner_result = await inner_async_flow(x)
            return inner_result + 1

        result = asyncio.run(outer_async_flow(5))
        assert result == 11  # (5 * 2) + 1

    def test_sync_flow_exception_propagation(self):
        """Test that exceptions in flows are properly propagated."""

        @flow
        def failing_flow() -> None:
            raise ValueError("Flow failed")

        with pytest.raises(ValueError, match="Flow failed"):
            failing_flow()

    def test_sync_generator_flow_exception_propagation(self):
        """Test that exceptions in generator flows are properly propagated."""

        @flow
        def failing_generator_flow() -> Generator[int, None, None]:
            yield 1
            raise ValueError("Generator flow failed")

        gen = failing_generator_flow()
        assert next(gen) == 1
        with pytest.raises(FlowRunError, match="Generator flow failed"):
            next(gen)

    def test_async_flow_exception_propagation(self):
        """Test that exceptions in async flows are properly propagated."""

        @flow
        async def async_failing_flow() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("Async flow failed")

        with pytest.raises(ValueError, match="Async flow failed"):
            asyncio.run(async_failing_flow())

    def test_async_generator_flow_exception_propagation(self):
        """Test that exceptions in async generator flows are properly propagated."""

        @flow
        async def async_failing_generator_flow() -> AsyncGenerator[int, None]:
            yield 1
            await asyncio.sleep(0.01)
            raise ValueError("Async generator flow failed")

        async def run_gen():
            gen = async_failing_generator_flow()
            value = await gen.__anext__()
            assert value == 1
            await gen.__anext__()

        with pytest.raises(FlowRunError, match="Async generator flow failed"):
            asyncio.run(run_gen())

    def test_flow_with_no_return(self):
        """Test flow that does not return a value."""

        @flow
        def no_return_flow(x: int) -> None:
            return

        result = no_return_flow(5)
        assert result is None

    def test_async_flow_with_no_return(self):
        """Test async flow that does not return a value."""

        @flow
        async def async_no_return_flow(x: int) -> None:
            await asyncio.sleep(0.01)
            return

        result = asyncio.run(async_no_return_flow(5))
        assert result is None


class TestFlowWithTasks:
    """Test flows that use tasks internally."""

    def test_flow_calling_tasks(self):
        """Test a flow that calls multiple tasks."""

        @task
        def add_task(x: int, y: int) -> int:
            return x + y

        @task
        def multiply_task(x: int, y: int) -> int:
            return x * y

        @flow
        def compute_flow(a: int, b: int, c: int) -> int:
            # Flow orchestrates multiple tasks
            sum_result = add_task(a, b)
            final_result = multiply_task(sum_result, c)
            return final_result

        result = compute_flow(2, 3, 4)
        assert result == 20  # (2 + 3) * 4

    def test_async_flow_calling_async_tasks(self):
        """Test an async flow that calls async tasks."""

        @task
        async def async_add_task(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        @task
        async def async_multiply_task(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x * y

        @flow
        async def async_compute_flow(a: int, b: int, c: int) -> int:
            sum_result = await async_add_task(a, b)
            final_result = await async_multiply_task(sum_result, c)
            return final_result

        result = asyncio.run(async_compute_flow(2, 3, 4))
        assert result == 20  # (2 + 3) * 4

    def test_flow_with_generator_tasks(self):
        """Test flow that uses generator tasks."""

        @task
        def number_generator(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i

        @flow
        def sum_generated_flow(n: int) -> int:
            total = 0
            for num in number_generator(n):
                total += num
            return total

        result = sum_generated_flow(5)
        assert result == 10  # 0 + 1 + 2 + 3 + 4

    def test_flow_with_async_generator_tasks(self):
        """Test flow that uses async generator tasks."""

        @task
        async def async_number_generator(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i

        @flow
        async def sum_async_generated_flow(n: int) -> int:
            total = 0
            async for num in async_number_generator(n):
                total += num
            return total

        result = asyncio.run(sum_async_generated_flow(5))
        assert result == 10

    def test_flow_with_task_submissions(self):
        """Test a flow that submits tasks for concurrent execution."""
        from waypoint import submit_task

        @task
        def square_task(x: int) -> int:
            return x * x

        @flow
        def parallel_squares_flow(numbers: list[int]) -> list[int]:
            # Submit all tasks concurrently
            futures = [submit_task(square_task, num) for num in numbers]
            # Collect results
            results = [future.result() for future in futures]
            return results

        result = parallel_squares_flow([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_flow_with_mixed_tasks(self):
        """Test a flow that mixes sync and async tasks."""

        @task
        def sync_task(x: int) -> int:
            return x + 1

        @task
        async def async_task(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        @flow
        async def mixed_flow(x: int) -> int:
            step1 = sync_task(x)
            step2 = await async_task(step1)
            return step2

        result = asyncio.run(mixed_flow(3))
        assert result == 8

    def test_complex_flow_orchestration(self):
        """Test a complex flow that orchestrates multiple operations."""

        @task
        def data_processor(data: list[int], operation: str) -> list[int]:
            if operation == "double":
                return [x * 2 for x in data]
            elif operation == "filter_even":
                return [x for x in data if x % 2 == 0]
            else:
                return data

        @flow
        def data_pipeline_flow(input_data: list[int]) -> dict[str, list[int]]:
            # Multi-stage data processing pipeline
            doubled = data_processor(input_data, "double")
            filtered = data_processor(doubled, "filter_even")

            return {"original": input_data, "doubled": doubled, "filtered": filtered}

        result = data_pipeline_flow([1, 2, 3, 4, 5])
        expected = {
            "original": [1, 2, 3, 4, 5],
            "doubled": [2, 4, 6, 8, 10],
            "filtered": [2, 4, 6, 8, 10],
        }
        assert result == expected

    def test_sync_flow_from_inside_sync_task_error(self):
        """Test that calling a sync flow from inside a sync task raises an error."""

        @flow
        def inner_flow(x: int) -> int:
            return x * 2

        @task
        def calling_task(x: int) -> int:
            # This should raise an error
            return inner_flow(x)

        with pytest.raises(RuntimeError, match="Cannot start a flow run context within a task"):
            calling_task(5)

    def test_async_flow_from_inside_sync_task_error(self):
        """Test that calling an async flow from inside a sync task raises an error."""

        @flow
        async def inner_async_flow(x: int) -> int:
            return x * 2

        @task
        def calling_task(x: int) -> int:
            # This should raise an error
            return asyncio.run(inner_async_flow(x))

        with pytest.raises(RuntimeError, match="Cannot start a flow run context within a task"):
            calling_task(5)


class TestFlowParameterBinding:
    """Test parameter binding and argument handling in flows."""


class TestFlowSession:
    """Test flow session functionality."""

    def test_flow_session_context_manager(self):
        """Test that flow_session provides proper context."""

        # Test without explicit flow context (should work due to autouse fixture)
        @flow
        def simple_flow() -> str:
            return "success"

        result = simple_flow()
        assert result == "success"

    def test_flow_session_with_different_runners(self):
        """Test flow session with different task runners."""

        @task
        def test_task(x: int) -> int:
            return x * 2

        # Test with threading runner
        with flow_session(name="threading_test", task_runner="threading"):

            @flow
            def threading_flow(x: int) -> int:
                return test_task(x)

            result = threading_flow(5)
            assert result == 10

    def test_nested_flow_sessions(self):
        """Test that nested flow sessions work correctly."""

        @task
        def test_task(x: int) -> int:
            return x + 1

        @flow
        def outer_session_flow(x: int) -> int:
            return test_task(x)

        # Nested sessions
        with flow_session(name="outer", task_runner="sequential"):
            outer_result = outer_session_flow(5)

            with flow_session(name="inner", task_runner="threading"):

                @flow
                def inner_session_flow(x: int) -> int:
                    return test_task(x)

                inner_result = inner_session_flow(10)

        assert outer_result == 6  # 5 + 1
        assert inner_result == 11  # 10 + 1

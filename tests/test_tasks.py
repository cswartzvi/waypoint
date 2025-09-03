"""Tests for the task decorator and task execution functionality."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest

from waypoint import submit_task
from waypoint import task
from waypoint.exceptions import TaskRunError
from waypoint.flows import flow_session
from waypoint.tasks import TaskData
from waypoint.tasks import get_task_data
from waypoint.tasks import map_task


@pytest.fixture(autouse=True)
def flow_context(request):
    """Automatically provide a flow context for all task tests."""
    if "noautouse" in request.keywords:
        yield
    else:
        # Get task_runner from request.param if available, otherwise use default
        task_runner = getattr(request, "param", "sequential")
        with flow_session(name="test_flow", task_runner=task_runner):
            yield


class TestTaskDecorator:
    """Test the @task decorator functionality."""

    def test_decorator_with_custom_name(self):
        """Test @task decorator with custom name parameter."""

        @task(name="custom_task_name")
        def named_task() -> str:
            return "test"

        task_data = get_task_data(named_task)
        assert task_data.name == "custom_task_name"

    def test_decorator_without_custom_name_uses_function_name(self):
        """Test @task decorator uses function name when no custom name provided."""

        @task
        def my_function_name() -> str:
            return "test"

        task_data = get_task_data(my_function_name)
        # The full qualified name includes the test context
        assert "my_function_name" in task_data.name

    def test_decorator_preserves_function_metadata(self):
        """Test that @task decorator preserves original function metadata."""

        @task
        def documented_task(x: int) -> int:
            """This is a documented task."""
            return x

        # The wrapper function should preserve the original name pattern
        assert "documented_task" in documented_task.__name__
        assert documented_task.__doc__ == "This is a documented task."

    def test_task_data_attachment(self):
        """Test that TaskData is properly attached to decorated functions."""

        @task
        def test_task() -> None:
            pass

        task_data = get_task_data(test_task)
        assert isinstance(task_data, TaskData)
        assert "test_task" in task_data.name
        assert task_data.is_async is False
        assert task_data.is_generator is False

    def test_async_task_data_detection(self):
        """Test that async functions are correctly detected in TaskData."""

        @task
        async def async_test_task() -> None:
            pass

        task_data = get_task_data(async_test_task)
        assert task_data.is_async is True
        assert task_data.is_generator is False

    def test_generator_task_data_detection(self):
        """Test that generator functions are correctly detected in TaskData."""

        @task
        def generator_test_task() -> Generator[int, None, None]:
            yield 1

        task_data = get_task_data(generator_test_task)
        assert task_data.is_async is False
        assert task_data.is_generator is True

    def test_async_generator_task_data_detection(self):
        """Test that async generator functions are correctly detected in TaskData."""

        @task
        async def async_generator_test_task() -> AsyncGenerator[int, None]:
            yield 1

        task_data = get_task_data(async_generator_test_task)
        assert task_data.is_async is True
        assert task_data.is_generator is True

    def test_positional_arguments(self):
        """Test task execution with positional arguments."""

        @task
        def concat_strings(a: str, b: str, c: str) -> str:
            return f"{a}-{b}-{c}"

        result = concat_strings("hello", "world", "test")
        assert result == "hello-world-test"

    def test_keyword_arguments(self):
        """Test task execution with keyword arguments."""

        @task
        def format_message(message: str, prefix: str = "INFO", suffix: str = "!") -> str:
            return f"[{prefix}] {message}{suffix}"

        result = format_message("Test message", prefix="DEBUG", suffix=".")
        assert result == "[DEBUG] Test message."

    def test_mixed_arguments(self):
        """Test task execution with mixed positional and keyword arguments."""

        @task
        def mixed_args(a: int, b: int, multiplier: int = 1, offset: int = 0) -> int:
            return (a + b) * multiplier + offset

        result = mixed_args(10, 20, multiplier=2, offset=5)
        assert result == 65  # (10 + 20) * 2 + 5

    def test_default_parameter_values(self):
        """Test that default parameter values are properly handled."""

        @task
        def with_defaults(value: int, factor: int = 2, add: int = 0) -> int:
            return value * factor + add

        # Test with all defaults
        result1 = with_defaults(5)
        assert result1 == 10  # 5 * 2 + 0

        # Test with partial defaults
        result2 = with_defaults(5, factor=3)
        assert result2 == 15  # 5 * 3 + 0

        # Test with no defaults
        result3 = with_defaults(5, factor=3, add=2)
        assert result3 == 17  # 5 * 3 + 2

    def test_is_task_with_task_function(self):
        """Test is_task returns True for decorated functions."""
        from waypoint.tasks import is_task

        @task
        def task_function():
            pass

        assert is_task(task_function) is True

    def test_is_task_with_regular_function(self):
        """Test is_task returns False for regular functions."""
        from waypoint.tasks import is_task

        def regular_function():
            pass

        assert is_task(regular_function) is False

    def test_get_task_data_invalid_function(self):
        """Test get_task_data with non-task function raises error."""

        def regular_function():
            pass

        from waypoint.exceptions import InvalidTaskError

        with pytest.raises(InvalidTaskError):
            get_task_data(regular_function)


class TestTaskExecution:
    """Test task execution behavior."""

    def test_sync_task_execution(self):
        """Test synchronous task execution."""

        @task
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(3, 4)
        assert result == 7

    def test_async_task_execution(self):
        """Test asynchronous task execution."""

        @task
        async def async_add_numbers(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = asyncio.run(async_add_numbers(3, 4))
        assert result == 7

    def test_sync_generator_task_execution(self):
        """Test synchronous generator task execution."""

        @task
        def generator_task(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i * 2

        result = list(generator_task(3))
        assert result == [0, 2, 4]

    def test_async_generator_task_execution(self):
        """Test asynchronous generator task execution."""

        @task
        async def async_generator_task(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i * 2

        async def collect_results():
            results = []
            # The task returns an async generator directly
            async for value in async_generator_task(3):
                results.append(value)
            return results

        result = asyncio.run(collect_results())
        assert result == [0, 2, 4]

    def test_task_with_complex_parameters(self):
        """Test task execution with complex parameter types."""

        @task
        def process_data(data: dict, multiplier: int = 2) -> dict:
            return {k: v * multiplier for k, v in data.items()}

        input_data = {"a": 1, "b": 2, "c": 3}
        result = process_data(input_data, multiplier=3)
        expected = {"a": 3, "b": 6, "c": 9}
        assert result == expected

    def test_task_chaining(self):
        """Test chaining task results."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_by_two(x: int) -> int:
            return x * 2

        # Chain tasks manually
        intermediate = add_one(5)  # 6
        result = multiply_by_two(intermediate)  # 12
        assert result == 12

    def test_nested_task_calls(self):
        """Test tasks calling other tasks."""

        @task
        def base_task(x: int) -> int:
            return x * 3

        @task
        def wrapper_task(x: int) -> int:
            # Task calling another task
            result = base_task(x)
            return result + 1

        result = wrapper_task(5)
        assert result == 16  # (5 * 3) + 1

    def test_task_exception_propagation(self):
        """Test that exceptions in tasks are properly propagated."""

        @task
        def failing_task() -> None:
            raise ValueError("Task failed")

        with pytest.raises(TaskRunError, match="Task failed"):
            failing_task()

    def test_task_generator_exception_propagation(self):
        """Test that exceptions in generator tasks are properly propagated."""

        @task
        def failing_generator_task() -> Generator[int, None, None]:
            yield 1
            raise RuntimeError("Generator task failed")

        gen = failing_generator_task()
        assert next(gen) == 1
        with pytest.raises(RuntimeError, match="Generator task failed"):
            next(gen)

    def test_nested_task_error_propagation(self):
        """Test that exceptions in nested tasks are properly propagated."""

        @task
        def inner_task() -> None:
            raise RuntimeError("Inner task error")

        @task
        def outer_task() -> None:
            inner_task()

        with pytest.raises(RuntimeError, match="Inner task error"):
            outer_task()

    def test_async_task_exception_propagation(self):
        """Test that exceptions in async tasks are properly propagated."""

        @task
        async def async_failing_task() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("Async task failed")

        with pytest.raises(TaskRunError, match="Async task failed"):
            asyncio.run(async_failing_task())

    def test_async_generator_task_exception_propagation(self):
        """Test that exceptions in async generator tasks are properly propagated."""

        @task
        async def async_failing_generator_task() -> AsyncGenerator[int, None]:
            yield 1
            await asyncio.sleep(0.01)
            raise RuntimeError("Async generator task failed")

        async def collect_results():
            results = []
            agen = async_failing_generator_task()
            results.append(await agen.__anext__())
            with pytest.raises(RuntimeError, match="Async generator task failed"):
                await agen.__anext__()
            return results

        result = asyncio.run(collect_results())
        assert result == [1]

    def test_nested_async_task_error_propagation(self):
        """Test that exceptions in nested async tasks are properly propagated."""

        @task
        async def inner_async_task() -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("Inner async task error")

        @task
        async def outer_async_task() -> None:
            await inner_async_task()

        with pytest.raises(RuntimeError, match="Inner async task error"):
            asyncio.run(outer_async_task())

    def test_task_returning_none(self):
        """Test that tasks returning None behave as expected."""

        @task
        def none_task() -> None:
            return None

        result = none_task()
        assert result is None

    @pytest.mark.noautouse
    def test_task_outside_flow_context_raises(self):
        """Test that executing a task outside a flow context raises an error."""

        @task
        def simple_task() -> int:
            return 42

        from waypoint.exceptions import MissingContextError

        with pytest.raises(MissingContextError):
            simple_task()


class TestTaskSubmission:
    """Test task submission and future handling."""

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_sync_task(self):
        """Test submitting a synchronous task."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        future = submit_task(multiply, 4, 5)
        result = future.result()
        assert result == 20

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_sync_generator_task(self):
        """Test submitting a synchronous generator task evaluates to a list."""

        @task
        def generate_numbers(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i + 1

        future = submit_task(generate_numbers, 3)
        result = future.result()
        assert result == [1, 2, 3]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_async_task(self):
        """Test submitting an asynchronous task."""

        @task
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y

        future = submit_task(async_multiply, 4, 5)
        result = future.result()
        assert result == 20

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_async_generator_task(self):
        """Test submitting an asynchronous generator task evaluates to a list."""

        @task
        async def async_generate_numbers(n: int) -> AsyncGenerator[int, None]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i + 1

        future = submit_task(async_generate_numbers, 3)
        result = future.result()
        assert result == [1, 2, 3]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_task_with_kwargs(self):
        """Test submitting a task with keyword arguments."""

        @task
        def calculate(base: int, power: int = 2, offset: int = 0) -> int:
            return (base**power) + offset

        future = submit_task(calculate, 3, power=3, offset=1)
        result = future.result()
        assert result == 28  # 3^3 + 1

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_multiple_tasks(self):
        """Test submitting multiple tasks and collecting results."""

        @task
        def square(x: int) -> int:
            return x * x

        futures = [submit_task(square, i) for i in range(1, 5)]
        results = [future.result() for future in futures]
        assert results == [1, 4, 9, 16]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_nested_task_submission(self):
        """Test submitting a task that submits another task."""

        @task
        def inner_task(x: int) -> int:
            return x + 10

        @task
        def outer_task(x: int) -> int:
            future = submit_task(inner_task, x)
            return future.result() * 2

        future = submit_task(outer_task, 5)
        result = future.result()
        assert result == 30

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_submit_non_task_function(self):
        """Test submitting a non-task function creates task data automatically."""

        def regular_function(x: int) -> int:
            return x * 2

        # This should work by creating task data automatically
        future = submit_task(regular_function, 5)
        result = future.result()
        assert result == 10

    @pytest.mark.noautouse
    def test_submit_nested_task_use_sequential_runner(self):
        """Test that nested task submissions use the sequential task runner."""
        from unittest.mock import patch

        from waypoint.flows import flow_session
        from waypoint.runners.sequential import SequentialTaskRunner

        sequential_submissions = 0
        original_submit = SequentialTaskRunner._submit

        def mock_submit(*args, **kwargs):
            nonlocal sequential_submissions
            sequential_submissions += 1
            return original_submit(*args, **kwargs)

        @task
        def inner_task(x: int) -> int:
            return x + 10

        @task
        def outer_task(x: int) -> int:
            future = submit_task(inner_task, x)  # This should use SequentialTaskRunner
            return future.result() * 2

        # Mock the SequentialTaskRunner submit method to track calls
        with patch.object(SequentialTaskRunner, "_submit", side_effect=mock_submit, autospec=True):
            with flow_session(name="test_flow", task_runner="threading"):
                _ = submit_task(outer_task, 5)
                _ = submit_task(outer_task, 5)
                future = submit_task(outer_task, 5)
                result = future.result()

        assert result == 30
        assert sequential_submissions == 3

    def test_submit_task_exception_handling(self):
        """Test exception handling in submitted tasks."""

        @task
        def error_task() -> None:
            raise RuntimeError("Submitted task error")

        # With proper deferred execution, submit returns immediately
        with pytest.raises(RuntimeError, match="Submitted task error"):
            _ = submit_task(error_task)

    @pytest.mark.noautouse
    def test_submit_task_outside_flow_context_raises(self):
        """Test that submitting a task outside a flow context raises an error."""

        @task
        def simple_task() -> int:
            return 42

        from waypoint.exceptions import MissingContextError

        with pytest.raises(MissingContextError):
            submit_task(simple_task)


class TestTaskMapping:
    """Test task mapping functionality."""

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_single_iterable_positional_arg(self):
        """Test mapping over a single iterable positional argument."""

        @task
        def square(x: int) -> int:
            return x * x

        numbers = [1, 2, 3, 4, 5]
        results = list(map_task(square, numbers))
        assert results == [1, 4, 9, 16, 25]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_multiple_iterable_positional_args(self):
        """Test mapping over multiple iterable positional arguments."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        x_values = [1, 2, 3]
        y_values = [10, 20, 30]
        results = list(map_task(add, x_values, y_values))
        assert results == [11, 22, 33]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_single_iterable_keyword_arg(self):
        """Test mapping over a single iterable keyword argument."""

        @task
        def multiply_by_factor(value: int, factor: int = 2) -> int:
            return value * factor

        factors = [1, 2, 3, 4]
        results = list(map_task(multiply_by_factor, value=10, factor=factors))
        assert results == [10, 20, 30, 40]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_multiple_iterable_keyword_args(self):
        """Test mapping over multiple iterable keyword arguments."""

        @task
        def calculate(base: int, multiplier: int, offset: int = 0) -> int:
            return base * multiplier + offset

        bases = [1, 2, 3]
        multipliers = [5, 10, 15]
        offsets = [1, 2, 3]
        results = list(map_task(calculate, base=bases, multiplier=multipliers, offset=offsets))
        assert results == [6, 22, 48]  # (1*5+1), (2*10+2), (3*15+3)

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_mixed_iterable_and_static_args(self):
        """Test mapping with both iterable and static arguments."""
        from waypoint.utils.annotations import unmapped

        @task
        def format_message(prefix: str, value: int, suffix: str) -> str:
            return f"{prefix}{value}{suffix}"

        prefixes = ["A:", "B:", "C:"]
        static_suffix = "!"
        results = list(map_task(format_message, prefixes, value=42, suffix=unmapped(static_suffix)))
        assert results == ["A:42!", "B:42!", "C:42!"]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_unmapped_positional_args(self):
        """Test mapping with unmapped positional arguments."""
        from waypoint.utils.annotations import unmapped

        @task
        def combine(prefix: str, value: int, multiplier: int) -> str:
            return f"{prefix}{value * multiplier}"

        values = [1, 2, 3]
        static_prefix = "Item:"
        static_multiplier = 10
        results = list(
            map_task(combine, unmapped(static_prefix), values, unmapped(static_multiplier))
        )
        assert results == ["Item:10", "Item:20", "Item:30"]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_unmapped_keyword_args(self):
        """Test mapping with unmapped keyword arguments."""
        from waypoint.utils.annotations import unmapped

        @task
        def process_data(data: int, scale: int, prefix: str = "Result") -> str:
            return f"{prefix}: {data * scale}"

        data_values = [1, 2, 3, 4]
        static_scale = 5
        static_prefix = "Output"
        results = list(
            map_task(
                process_data,
                data=data_values,
                scale=unmapped(static_scale),
                prefix=unmapped(static_prefix),
            )
        )
        assert results == ["Output: 5", "Output: 10", "Output: 15", "Output: 20"]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_default_parameters(self):
        """Test mapping with tasks that have default parameters."""

        @task
        def calculate_with_defaults(value: int, multiplier: int = 2, offset: int = 0) -> int:
            return value * multiplier + offset

        values = [1, 2, 3]
        # Only mapping over value, using defaults for multiplier and offset
        results = list(map_task(calculate_with_defaults, values))
        assert results == [2, 4, 6]  # value * 2 + 0

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_async_task(self):
        """Test mapping with asynchronous tasks."""

        @task
        async def async_multiply(x: int, factor: int) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return x * factor

        values = [1, 2, 3]
        factors = [10, 20, 30]
        results = list(map_task(async_multiply, values, factors))
        assert results == [10, 40, 90]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_generator_task(self):
        """Test mapping with generator tasks."""

        @task
        def generate_sequence(start: int, count: int) -> Generator[int, None, None]:
            for i in range(count):
                yield start + i

        starts = [0, 10, 20]
        counts = [3, 2, 4]
        results = list(map_task(generate_sequence, starts, counts))
        expected = [
            [0, 1, 2],  # start=0, count=3
            [10, 11],  # start=10, count=2
            [20, 21, 22, 23],  # start=20, count=4
        ]
        assert results == expected

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_async_generator_task(self):
        """Test mapping with async generator tasks."""

        @task
        async def async_generate_sequence(start: int, count: int) -> AsyncGenerator[int, None]:
            for i in range(count):
                await asyncio.sleep(0.001)
                yield start + i

        starts = [0, 5]
        counts = [2, 3]
        results = list(map_task(async_generate_sequence, starts, counts))
        expected = [
            [0, 1],  # start=0, count=2
            [5, 6, 7],  # start=5, count=3
        ]
        assert results == expected

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_empty_iterable(self):
        """Test mapping with empty iterables."""

        @task
        def double(x: int) -> int:
            return x * 2

        empty_list = []
        results = list(map_task(double, empty_list))
        assert results == []

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_complex_data_types(self):
        """Test mapping with complex data types like dictionaries."""

        @task
        def process_dict(data: dict, key: str) -> str:
            return f"{key}: {data.get(key, 'missing')}"

        data_list = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie"}]
        keys = ["name", "age", "name"]
        results = list(map_task(process_dict, data_list, keys))
        assert results == ["name: Alice", "age: 25", "name: Charlie"]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_non_task_function(self):
        """Test mapping with a non-task function (should create task data automatically)."""

        def regular_function(x: int) -> int:
            return x**2

        values = [2, 3, 4]
        results = list(map_task(regular_function, values))
        assert results == [4, 9, 16]

    def test_map_task_missing_iterable_arguments_raises(self):
        """Test that mapping without iterable arguments raises MappingMissingIterable."""
        from waypoint.exceptions import MappingMissingIterable
        from waypoint.utils.annotations import unmapped

        @task
        def static_task(x: int, y: str) -> str:
            return f"{y}: {x}"

        with pytest.raises(MappingMissingIterable, match="No iterable arguments were received"):
            list(map_task(static_task, unmapped(42), unmapped("prefix")))

    def test_map_task_length_mismatch_raises(self):
        """Test that mapping with mismatched iterable lengths raises MappingLengthMismatch."""
        from waypoint.exceptions import MappingLengthMismatch

        @task
        def add(x: int, y: int) -> int:
            return x + y

        x_values = [1, 2, 3]
        y_values = [10, 20]  # Different length

        with pytest.raises(MappingLengthMismatch, match="different lengths"):
            list(map_task(add, x_values, y_values))

    def test_map_task_length_mismatch_with_kwargs_raises(self):
        """Test mapping with mismatched keyword iterable lengths raises MappingLengthMismatch."""
        from waypoint.exceptions import MappingLengthMismatch

        @task
        def calculate(base: int, multiplier: int, offset: int) -> int:
            return base * multiplier + offset

        with pytest.raises(MappingLengthMismatch, match="different lengths"):
            list(
                map_task(
                    calculate,
                    base=[1, 2, 3],
                    multiplier=[5, 10],  # Length 2
                    offset=[1, 2, 3, 4],  # Length 4
                )
            )

    @pytest.mark.noautouse
    def test_map_task_outside_flow_context_raises(self):
        """Test that mapping outside a flow context raises MissingContextError."""
        from waypoint.exceptions import MissingContextError

        @task
        def simple_task(x: int) -> int:
            return x * 2

        with pytest.raises(MissingContextError):
            list(map_task(simple_task, [1, 2, 3]))

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_with_tuple_iterable(self):
        """Test mapping with tuple iterables."""

        @task
        def join_strings(a: str, b: str) -> str:
            return f"{a}-{b}"

        first_parts = ("hello", "good", "nice")
        second_parts = ("world", "bye", "day")
        results = list(map_task(join_strings, first_parts, second_parts))
        assert results == ["hello-world", "good-bye", "nice-day"]

    @pytest.mark.parametrize("flow_context", ["sequential", "threading"], indirect=True)
    def test_map_task_single_item_iterables(self):
        """Test mapping with single-item iterables."""

        @task
        def format_value(value: int, template: str) -> str:
            return template.format(value)

        values = [42]
        templates = ["Value: {}"]
        results = list(map_task(format_value, values, templates))
        assert results == ["Value: 42"]

    def test_map_task_error_propagation_sequential(self):
        """Test that exceptions in mapped tasks are properly propagated with sequential runner."""

        @task
        def failing_task(x: int) -> int:
            if x == 2:
                raise ValueError(f"Error with value {x}")
            return x * 2

        with flow_session(name="test_flow", task_runner="sequential"):
            values = [1, 2, 3]
            # With sequential runner, the error occurs during task submission/execution
            with pytest.raises(TaskRunError, match="Error with value 2"):
                list(map_task(failing_task, values))

    @pytest.mark.parametrize("flow_context", ["threading"], indirect=True)
    def test_map_task_error_propagation_concurrent(self):
        """Test that exceptions in mapped tasks are properly propagated with concurrent runners."""

        @task
        def failing_task(x: int) -> int:
            if x == 2:
                raise ValueError(f"Error with value {x}")
            return x * 2

        values = [1, 2, 3]
        iterator = map_task(failing_task, values)

        # First result should work
        assert next(iterator) == 2  # 1 * 2

        # Second result should raise the exception (wrapped in TaskRunError)
        with pytest.raises(TaskRunError, match="Error with value 2"):
            next(iterator)

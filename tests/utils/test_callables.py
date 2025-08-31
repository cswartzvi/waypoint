"""Comprehensive tests for waypoint.utils.callables module."""

import inspect

import pytest

from waypoint.exceptions import ParameterBindError
from waypoint.exceptions import SignatureMismatchError
from waypoint.utils.callables import arguments_to_args_kwargs
from waypoint.utils.callables import call_with_arguments
from waypoint.utils.callables import collapse_variadic_arguments
from waypoint.utils.callables import explode_variadic_arguments
from waypoint.utils.callables import get_call_arguments
from waypoint.utils.callables import get_docstring_summary
from waypoint.utils.callables import get_function_name
from waypoint.utils.callables import get_parameter_defaults
from waypoint.utils.callables import is_asynchronous
from waypoint.utils.callables import is_generator


class TestFunctionIntrospection:
    """Test function introspection utilities."""

    def test_is_asynchronous_with_sync_function(self):
        """Test is_asynchronous returns False for sync functions."""

        def sync_func():
            pass

        assert is_asynchronous(sync_func) is False

    def test_is_asynchronous_with_async_function(self):
        """Test is_asynchronous returns True for async functions."""

        async def async_func():
            pass

        assert is_asynchronous(async_func) is True

    def test_is_asynchronous_with_async_generator(self):
        """Test is_asynchronous returns True for async generators."""

        async def async_gen():
            yield 1

        assert is_asynchronous(async_gen) is True

    def test_is_generator_with_regular_function(self):
        """Test is_generator returns False for regular functions."""

        def regular_func():
            return 1

        assert is_generator(regular_func) is False

    def test_is_generator_with_sync_generator(self):
        """Test is_generator returns True for sync generators."""

        def sync_gen():
            yield 1

        assert is_generator(sync_gen) is True

    def test_is_generator_with_async_generator(self):
        """Test is_generator returns True for async generators."""

        async def async_gen():
            yield 1

        assert is_generator(async_gen) is True

    def test_get_function_name_with_regular_function(self):
        """Test get_function_name returns function name."""

        def test_function():
            pass

        name = get_function_name(test_function)
        assert "test_function" in name

    def test_get_function_name_with_class_method(self):
        """Test get_function_name with class methods."""

        class TestClass:
            def method(self):
                pass

        name = get_function_name(TestClass.method)
        assert "TestClass.method" in name

    def test_get_function_name_with_builtin(self):
        """Test get_function_name with builtin functions."""
        name = get_function_name(len)
        assert name == "len"

    def test_get_docstring_summary_with_single_line(self):
        """Test get_docstring_summary with single line docstring."""

        def documented_func():
            """This is a single line docstring."""
            pass

        summary = get_docstring_summary(documented_func)
        assert summary == "This is a single line docstring."

    def test_get_docstring_summary_with_multiline(self):
        """Test get_docstring_summary with multiline docstring."""

        def documented_func():
            """This is a multiline docstring.

            This should not be included in the summary.
            """
            pass

        summary = get_docstring_summary(documented_func)
        assert summary == "This is a multiline docstring."

    def test_get_docstring_summary_with_no_docstring(self):
        """Test get_docstring_summary returns None for undocumented functions."""

        def undocumented_func():
            pass

        summary = get_docstring_summary(undocumented_func)
        assert summary is None

    def test_get_docstring_summary_with_multiline_summary(self):
        """Test get_docstring_summary handles multiline summaries."""

        def documented_func():
            """This is a multiline summary
            that continues on the next line.

            This is the detailed description.
            """
            pass

        summary = get_docstring_summary(documented_func)
        assert summary == "This is a multiline summary that continues on the next line."

    def test_get_docstring_summary_with_empty_docstring(self):
        """Test get_docstring_summary with empty docstring."""

        def empty_doc_func():
            """"""
            pass

        summary = get_docstring_summary(empty_doc_func)
        assert summary == ""

    def test_get_function_name_with_callable_object_no_name_attrs(self):
        """Test get_function_name with callable object lacking __name__ and __qualname__."""

        class CallableWithoutNameAttrs:
            def __call__(self, x):
                return x * 2

        callable_obj = CallableWithoutNameAttrs()
        name = get_function_name(callable_obj)
        # Should fall back to str() representation
        assert "CallableWithoutNameAttrs" in name
        assert "object at" in name

    def test_get_function_name_with_lambda(self):
        """Test get_function_name with lambda function."""
        lambda_func = lambda x: x + 1
        name = get_function_name(lambda_func)
        assert "lambda" in name or "<lambda>" in name

    def test_get_function_name_with_partial_function(self):
        """Test get_function_name with functools.partial."""
        from functools import partial

        def multiply(x, y):
            return x * y

        partial_func = partial(multiply, 2)
        name = get_function_name(partial_func)
        # partial objects typically have __name__ but test the fallback case
        assert isinstance(name, str)
        assert len(name) > 0


class TestParameterHandling:
    """Test parameter binding and argument handling."""

    def test_get_call_arguments_basic(self):
        """Test get_call_arguments with basic function."""

        def func(a, b, c=10):
            pass

        args = get_call_arguments(func, (1, 2), {})
        assert args == {"a": 1, "b": 2, "c": 10}

    def test_get_call_arguments_with_kwargs(self):
        """Test get_call_arguments with keyword arguments."""

        def func(a, b, c=10):
            pass

        args = get_call_arguments(func, (1,), {"b": 2, "c": 5})
        assert args == {"a": 1, "b": 2, "c": 5}

    def test_get_call_arguments_without_defaults(self):
        """Test get_call_arguments with apply_defaults=False."""

        def func(a, b, c=10):
            pass

        args = get_call_arguments(func, (1, 2), {}, apply_defaults=False)
        assert args == {"a": 1, "b": 2}

    def test_get_call_arguments_with_no_parameters(self):
        """Test get_call_arguments with function that has no parameters."""

        def func():
            pass

        args = get_call_arguments(func, (), {})
        assert args == {}

    def test_get_call_arguments_bind_error_missing_required(self):
        """Test get_call_arguments raises ParameterBindError for missing required args."""

        def func(a, b):
            pass

        with pytest.raises(ParameterBindError):
            get_call_arguments(func, (1,), {})  # Missing 'b'

    def test_get_call_arguments_bind_error_excess_args(self):
        """Test get_call_arguments raises ParameterBindError for excess args."""

        def func(a, b):
            pass

        with pytest.raises(ParameterBindError):
            get_call_arguments(func, (1, 2, 3), {})  # Too many args

    def test_get_parameter_defaults_basic(self):
        """Test get_parameter_defaults extracts default values."""

        def func(a, b=10, c="hello"):
            pass

        defaults = get_parameter_defaults(func)
        assert defaults == {"b": 10, "c": "hello"}

    def test_get_parameter_defaults_include_empty(self):
        """Test get_parameter_defaults with include_empty=True."""

        def func(a, b=10):
            pass

        defaults = get_parameter_defaults(func, include_empty=True)
        assert defaults == {"a": inspect.Parameter.empty, "b": 10}

    def test_get_parameter_defaults_no_defaults(self):
        """Test get_parameter_defaults with function that has no defaults."""

        def func(a, b):
            pass

        defaults = get_parameter_defaults(func)
        assert defaults == {}

    def test_get_parameter_defaults_all_defaults(self):
        """Test get_parameter_defaults with function where all parameters have defaults."""

        def func(a=1, b=2, c=3):
            pass

        defaults = get_parameter_defaults(func)
        assert defaults == {"a": 1, "b": 2, "c": 3}


class TestArgumentConversion:
    """Test argument conversion utilities."""

    def test_arguments_to_args_kwargs_basic(self):
        """Test arguments_to_args_kwargs with basic function."""

        def func(a, b, c=10):
            pass

        args, kwargs = arguments_to_args_kwargs(func, {"a": 1, "b": 2, "c": 5})
        # Should be callable
        func(*args, **kwargs)  # Should not raise

    def test_arguments_to_args_kwargs_signature_mismatch(self):
        """Test arguments_to_args_kwargs raises error for unknown parameters."""

        def func(a, b):
            pass

        with pytest.raises(SignatureMismatchError):
            arguments_to_args_kwargs(func, {"a": 1, "b": 2, "unknown": 3})

    def test_arguments_to_args_kwargs_empty_parameters(self):
        """Test arguments_to_args_kwargs with no parameters."""

        def func():
            pass

        args, kwargs = arguments_to_args_kwargs(func, {})
        assert args == ()
        assert kwargs == {}

    def test_call_with_arguments_basic(self):
        """Test call_with_arguments executes function correctly."""

        def func(a, b, c=10):
            return a + b + c

        result = call_with_arguments(func, {"a": 1, "b": 2, "c": 5})
        assert result == 8

    def test_call_with_arguments_with_defaults(self):
        """Test call_with_arguments works with default parameters."""

        def func(a, b, c=10):
            return a + b + c

        result = call_with_arguments(func, {"a": 1, "b": 2})
        assert result == 13

    def test_call_with_arguments_complex_types(self):
        """Test call_with_arguments with complex parameter types."""

        def func(data: dict, multiplier: int = 2):
            return {k: v * multiplier for k, v in data.items()}

        result = call_with_arguments(func, {"data": {"a": 1, "b": 2}, "multiplier": 3})
        assert result == {"a": 3, "b": 6}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_function_with_varargs(self):
        """Test utilities work with *args functions."""

        def func(a, *args, **kwargs):
            return a, args, kwargs

        arguments = get_call_arguments(func, (1, 2, 3), {"key": "value"})
        assert "a" in arguments

    def test_function_with_keyword_only_args(self):
        """Test utilities work with keyword-only arguments."""

        def func(a, *, b, c=10):
            return a + b + c

        arguments = get_call_arguments(func, (1,), {"b": 2})
        assert arguments == {"a": 1, "b": 2, "c": 10}

        result = call_with_arguments(func, arguments)
        assert result == 13

    def test_callable_object(self):
        """Test utilities work with callable objects."""

        class CallableClass:
            def __call__(self, x, y=5):
                return x + y

        callable_obj = CallableClass()

        # Test introspection works
        assert is_asynchronous(callable_obj) is False
        assert is_generator(callable_obj) is False

        # Test parameter handling works
        arguments = get_call_arguments(callable_obj, (10,), {})
        assert arguments == {"x": 10, "y": 5}

        result = call_with_arguments(callable_obj, arguments)
        assert result == 15

    def test_builtin_function(self):
        """Test utilities work with builtin functions."""
        # Test with len builtin
        name = get_function_name(len)
        assert name == "len"

        # get_call_arguments should work with builtins too
        arguments = get_call_arguments(len, ([1, 2, 3],), {})
        result = call_with_arguments(len, arguments)
        assert result == 3


class TestExplodeVariadicParameter:
    def test_no_error_if_no_variadic_parameter(self):
        def foo(a, b):
            pass

        parameters = {"a": 1, "b": 2}
        new_params = explode_variadic_arguments(foo, parameters)

        assert parameters == new_params

    def test_no_error_if_variadic_parameter_and_kwargs_provided(self):
        def foo(a, b, **kwargs):
            pass

        parameters = {"a": 1, "b": 2, "kwargs": {"c": 3, "d": 4}}
        new_params = explode_variadic_arguments(foo, parameters)

        assert new_params == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_no_error_if_variadic_parameter_and_no_kwargs_provided(self):
        def foo(a, b, **kwargs):
            pass

        parameters = {"a": 1, "b": 2}
        new_params = explode_variadic_arguments(foo, parameters)

        assert new_params == parameters


class TestCollapseVariadicParameter:
    def test_no_error_if_no_variadic_parameter(self):
        def foo(a, b):
            pass

        parameters = {"a": 1, "b": 2}
        new_params = collapse_variadic_arguments(foo, parameters)

        assert new_params == parameters

    def test_no_error_if_variadic_parameter_and_kwargs_provided(self):
        def foo(a, b, **kwargs):
            pass

        parameters = {"a": 1, "b": 2, "c": 3, "d": 4}
        new_params = collapse_variadic_arguments(foo, parameters)

        assert new_params == {"a": 1, "b": 2, "kwargs": {"c": 3, "d": 4}}

    def test_params_unchanged_if_variadic_parameter_and_no_kwargs_provided(self):
        def foo(a, b, **kwargs):
            pass

        parameters = {"a": 1, "b": 2}
        new_params = collapse_variadic_arguments(foo, parameters)

        assert new_params == parameters

    def test_value_error_raised_if_extra_args_but_no_variadic_parameter(self):
        def foo(a, b):
            pass

        parameters = {"a": 1, "b": 2, "kwargs": {"c": 3, "d": 4}}

        with pytest.raises(ValueError):
            collapse_variadic_arguments(foo, parameters)

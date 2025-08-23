"""Tests for utils.collections utility functions."""

import asyncio

import pytest

from waypoint.utils.collections import aenumerate, ensure_iterable, is_iterable


class TestAenumerate:
    """Test the aenumerate function."""

    @pytest.mark.asyncio
    async def test_empty_async_iterable(self):
        """Test aenumerate with empty async iterable."""

        async def empty_generator():
            return
            yield  # This will never execute

        result = []
        async for idx, item in aenumerate(empty_generator()):
            result.append((idx, item))

        assert result == []

    @pytest.mark.asyncio
    async def test_single_item_async_iterable(self):
        """Test aenumerate with single item."""

        async def single_generator():
            yield "item"

        result = []
        async for idx, item in aenumerate(single_generator()):
            result.append((idx, item))

        assert result == [(0, "item")]

    @pytest.mark.asyncio
    async def test_multiple_items_async_iterable(self):
        """Test aenumerate with multiple items."""

        async def multi_generator():
            yield "first"
            yield "second"
            yield "third"

        result = []
        async for idx, item in aenumerate(multi_generator()):
            result.append((idx, item))

        assert result == [(0, "first"), (1, "second"), (2, "third")]

    @pytest.mark.asyncio
    async def test_custom_start_value(self):
        """Test aenumerate with custom start value."""

        async def generator():
            yield "a"
            yield "b"
            yield "c"

        result = []
        async for idx, item in aenumerate(generator(), start=10):
            result.append((idx, item))

        assert result == [(10, "a"), (11, "b"), (12, "c")]

    @pytest.mark.asyncio
    async def test_negative_start_value(self):
        """Test aenumerate with negative start value."""

        async def generator():
            yield "x"
            yield "y"

        result = []
        async for idx, item in aenumerate(generator(), start=-5):
            result.append((idx, item))

        assert result == [(-5, "x"), (-4, "y")]

    @pytest.mark.asyncio
    async def test_with_async_list(self):
        """Test aenumerate with async iterable from list."""

        async def list_generator():
            for item in ["apple", "banana", "cherry"]:
                yield item

        result = []
        async for idx, item in aenumerate(list_generator()):
            result.append((idx, item))

        assert result == [(0, "apple"), (1, "banana"), (2, "cherry")]

    @pytest.mark.asyncio
    async def test_with_different_types(self):
        """Test aenumerate with different item types."""

        async def mixed_generator():
            yield 1
            yield "string"
            yield [1, 2, 3]
            yield {"key": "value"}

        result = []
        async for idx, item in aenumerate(mixed_generator()):
            result.append((idx, item))

        expected = [(0, 1), (1, "string"), (2, [1, 2, 3]), (3, {"key": "value"})]
        assert result == expected

    @pytest.mark.asyncio
    async def test_with_delays(self):
        """Test aenumerate with async generator that has delays."""

        async def delayed_generator():
            yield "first"
            await asyncio.sleep(0.001)
            yield "second"
            await asyncio.sleep(0.001)
            yield "third"

        result = []
        async for idx, item in aenumerate(delayed_generator()):
            result.append((idx, item))

        assert result == [(0, "first"), (1, "second"), (2, "third")]

    def test_aenumerate_sync_usage(self):
        """Test running aenumerate in sync context."""

        async def generator():
            yield "a"
            yield "b"

        async def collect():
            result = []
            async for idx, item in aenumerate(generator()):
                result.append((idx, item))
            return result

        result = asyncio.run(collect())
        assert result == [(0, "a"), (1, "b")]

    @pytest.mark.asyncio
    async def test_large_sequence(self):
        """Test aenumerate with larger sequence."""

        async def large_generator():
            for i in range(100):
                yield i * 2

        result = []
        async for idx, item in aenumerate(large_generator()):
            result.append((idx, item))

        # Check first few and last few items
        assert result[0] == (0, 0)
        assert result[1] == (1, 2)
        assert result[2] == (2, 4)
        assert result[-1] == (99, 198)
        assert len(result) == 100


class TestIsIterable:
    @pytest.mark.parametrize("obj", [[1, 2, 3], (1, 2, 3)])
    def test_is_iterable(self, obj):
        assert is_iterable(obj)

    @pytest.mark.parametrize("obj", [5, Exception(), True, "hello", bytes()])
    def test_not_iterable(self, obj):
        assert not is_iterable(obj)


class TestEnsureIterable:
    @pytest.mark.parametrize("obj", [[1, 2, 3], (1, 2, 3), set([1, 2, 3])])
    def test_is_iterable(self, obj):
        assert ensure_iterable(obj) == obj

    @pytest.mark.parametrize("obj", [5, "a", None])
    def test_not_iterable(self, obj):
        assert ensure_iterable(obj) == [obj]

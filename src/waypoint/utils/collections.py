import io
from collections.abc import Collection
from collections.abc import Iterable
from typing import Any, AsyncGenerator, AsyncIterable, Tuple, TypeVar, cast

T = TypeVar("T")


def is_iterable(obj: Any) -> bool:
    """
    Return a boolean indicating if an object is iterable.

    Excludes types that are iterable but typically used as singletons:
    - str
    - bytes
    - IO objects
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return not isinstance(obj, (str, bytes, io.IOBase))


def ensure_iterable(obj: T | Iterable[T]) -> Collection[T]:
    """Ensure that an object is returned as an iterable collection."""
    if is_iterable(obj):
        return cast(Collection[T], obj)
    obj = cast(T, obj)  # No longer in the iterable case
    return [obj]


async def aenumerate(
    async_iterable: AsyncIterable[T], start: int = 0
) -> AsyncGenerator[Tuple[int, T], None]:
    """Async version of enumerate."""
    idx = start
    async for item in async_iterable:
        yield idx, item
        idx += 1

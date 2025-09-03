from operator import itemgetter
from typing import Any, TypeVar, cast

from typing_extensions import Self

T = TypeVar("T")


class BaseAnnotation(tuple[T]):
    """
    Base class for Waypoint annotation types.

    Inherits from `tuple` for unpacking support in other tools.
    """

    __slots__ = ()

    def __new__(cls, value: T) -> Self:  # noqa: D102
        return super().__new__(cls, (value,))

    # use itemgetter to minimize overhead, just like namedtuple generated code would
    value: T = cast(T, property(itemgetter(0)))

    def unwrap(self) -> T:
        """Returns the wrapped value."""
        return self[0]

    def rewrap(self, value: T) -> Self:
        """Returns a new instance of the annotation wrapping the provided value."""
        return type(self)(value)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self[0]!r})"


class unmapped(BaseAnnotation[T]):
    """
    Wrapper for iterables.

    Indicates that this input should be sent as-is to all runs created during a mapping
    operation instead of being split.
    """

    def __getitem__(self, _: object) -> T:  # type: ignore[override]
        # Internally, this acts as an infinite array where all items are the same value
        return cast(T, super().__getitem__(0))

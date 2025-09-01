from typing import Callable, Iterator, TypeVar

T = TypeVar("T", contravariant=True)


def lenient_issubclass(cls: type, class_or_tuple: type | tuple[type, ...]) -> bool:
    """
    Check if a class is a subclass of a class or any of a tuple of classes.

    This similar to the built-in issubclass, but does not raise a TypeError if the
    first argument is not a class, instead returning False.
    """
    try:
        return issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def iter_subclasses(cls: type[T]) -> Iterator[type[T]]:
    """Iterate over all subclasses of a class."""
    for sub in cls.__subclasses__():
        yield sub
        yield from iter_subclasses(sub)


def get_subclass(cls: type, key: Callable[[type], bool]) -> type:
    """Get the first subclass of a class that matches a key function."""
    sub: type
    for sub in iter_subclasses(cls):
        if key(sub):
            return sub
    raise TypeError(f"No subclass found for {cls.__name__!r}.")

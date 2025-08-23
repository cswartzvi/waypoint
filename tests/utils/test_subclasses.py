import pytest

from waypoint.utils.subclasses import get_subclass
from waypoint.utils.subclasses import iter_subclasses
from waypoint.utils.subclasses import lenient_issubclass


class TestLenientIsSubclass:
    def test_lenient_issubclass_with_valid_subclass(self):
        class Foo:
            pass

        class Bar(Foo):
            pass

        assert lenient_issubclass(Bar, Foo)

    def test_lenient_issubclass_with_invalid_subclass(self):
        class Foo:
            pass

        foo = Foo()

        assert not lenient_issubclass(Foo, foo)  # pyright: ignore
        assert not lenient_issubclass(foo, Foo)  # pyright: ignore
        assert not lenient_issubclass(foo, foo)  # pyright: ignore


class TestIterSubclasses:
    def test_iter_subclasses_with_single_subclass(self):
        class Foo:
            pass

        class Bar(Foo):
            pass

        assert list(iter_subclasses(Foo)) == [Bar]

    def test_iter_subclasses_with_multiple_subclasses(self):
        class Foo:
            pass

        class Bar(Foo):
            pass

        class Baz(Foo):
            pass

        assert list(iter_subclasses(Foo)) == [Bar, Baz]

    def test_iter_subclasses_with_multiple_levels_of_subclasses(self):
        class Foo:
            pass

        class Bar(Foo):
            pass

        class Baz(Bar):
            pass

        assert list(iter_subclasses(Foo)) == [Bar, Baz]

    def test_iter_subclasses_with_pydantic_base_model(self):
        from pydantic import BaseModel

        class Foo(BaseModel):
            pass

        class Bar(Foo):
            pass

        class Baz(Foo):
            pass

        assert list(iter_subclasses(Foo)) == [Bar, Baz]

    def test_iter_subclasses_with_generic_pydantic_base_model(self):
        from typing import Generic, TypeVar

        from pydantic import BaseModel

        T = TypeVar("T")

        class Foo(BaseModel, Generic[T]):
            pass

        class Bar(Foo[int]):
            pass

        class Baz(Foo[str]):
            pass

        subclasses = list(iter_subclasses(Foo))
        expected = [Bar, Baz]

        # Pydantic creates additional subclasses for generic models, so we
        # expect the number of subclasses to be greater than the expected.
        assert subclasses != expected
        assert len(subclasses) > len(expected)
        assert all(lenient_issubclass(sub, Foo) for sub in subclasses)


class TestGetSubclass:
    def test_get_subclass_with_valid_subclass(self):
        class Foo:
            pass

        class Bar(Foo):
            pass

        assert get_subclass(Foo, lambda x: x.__name__ == "Bar") == Bar

    def test_get_subclass_with_invalid_subclass_raises_error(self):
        class Foo:
            pass

        with pytest.raises(TypeError, match="No subclass found for 'Foo'."):
            get_subclass(Foo, lambda x: x.__name__ == "Bar")

from waypoint.utils.annotations import BaseAnnotation
from waypoint.utils.annotations import unmapped


class TestBaseAnnotation:
    """Tests for BaseAnnotation and its behavior."""

    def test_creation_and_value_access(self):
        """Test creating a BaseAnnotation and accessing its value."""
        annotation = BaseAnnotation("test_value")
        assert annotation.value == "test_value"
        assert annotation[0] == "test_value"

    def test_unwrap(self):
        """Test unwrapping the annotation value."""
        annotation = BaseAnnotation(42)
        assert annotation.unwrap() == 42

    def test_rewrap(self):
        """Test creating a new annotation with a different value."""
        annotation = BaseAnnotation("original")
        new_annotation = annotation.rewrap("new_value")
        assert new_annotation.value == "new_value"
        assert annotation.value == "original"  # Original unchanged

    def test_equality(self):
        """Test equality comparison between annotations."""
        ann1 = BaseAnnotation("test")
        ann2 = BaseAnnotation("test")
        ann3 = BaseAnnotation("different")
        ann4 = unmapped("test")  # Different annotation type

        assert ann1 == ann2
        assert ann1 != ann3
        assert ann1 != "test"  # Different type
        # Test the __eq__ method directly to cover line 34
        assert ann1.__eq__(ann4) is False

    def test_repr(self):
        """Test string representation."""
        annotation = BaseAnnotation("test")
        assert repr(annotation) == "BaseAnnotation('test')"


class TestUnmapped:
    """Tests for the `unmapped` annotation."""

    def test_creation_and_value_access(self):
        """Test creating an unmapped annotation."""
        annotation = unmapped([1, 2, 3])
        assert annotation.value == [1, 2, 3]

    def test_getitem_behavior(self):
        """Test that unmapped acts like an infinite array of the same value."""
        annotation = unmapped("static_value")
        assert annotation[0] == "static_value"
        assert annotation[100] == "static_value"  # pyright: ignore
        assert annotation[-1] == "static_value"

    def test_with_different_types(self):
        """Test unmapped with different value types."""
        str_annotation = unmapped("text")
        int_annotation = unmapped(42)
        list_annotation = unmapped([1, 2, 3])

        assert str_annotation.value == "text"
        assert int_annotation.value == 42
        assert list_annotation.value == [1, 2, 3]

    def test_repr(self):
        """Test string representation."""
        annotation = unmapped("test")
        assert repr(annotation) == "unmapped('test')"

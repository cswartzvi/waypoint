import tempfile
from pathlib import Path

import pytest

from waypoint.stores.filesystem import FileSystemAssetStore
from waypoint.stores.memory import InMemoryAssetStore


@pytest.fixture(
    params=[
        ("memory", InMemoryAssetStore),
        ("filesystem", FileSystemAssetStore),
    ]
)
def store_factory(request):
    """Parametrized fixture providing different store types."""
    store_type, store_class = request.param

    if store_type == "memory":
        return lambda: store_class()
    else:  # filesystem

        def _create_fs_store():
            # Create temp directory with more explicit error handling for CI
            temp_dir = tempfile.mkdtemp(prefix="waypoint_test_")

            # Ensure the directory is properly accessible
            temp_path = Path(temp_dir)
            if not temp_path.exists() or not temp_path.is_dir():
                raise RuntimeError(f"Failed to create test directory: {temp_dir}")

            return store_class(temp_dir)

        return _create_fs_store


@pytest.fixture
def store(store_factory):
    """Create a fresh store for each test."""
    return store_factory()

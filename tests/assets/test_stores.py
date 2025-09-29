import tempfile
from pathlib import Path

import cloudpickle
import pytest

from waypoint.exceptions import AssetNotFoundError
from waypoint.exceptions import AssetStoreError
from waypoint.stores.filesystem import FileSystemAssetStore


@pytest.fixture
def sample_data():
    """Realistic test data representing ML experiment results."""
    return {
        "experiment_id": "exp_001",
        "model": "transformer",
        "accuracy": 0.95,
        "parameters": {"lr": 0.001, "batch_size": 32},
        "metadata": {"timestamp": "2024-01-01"},
    }


class TestAssetStores:
    """Test asset store functionality across all implementations."""

    def test_write_read_roundtrip(self, store, sample_data):
        """Test basic write/read operations work correctly."""
        key = "experiments/model_v1.pkl"
        serialized = cloudpickle.dumps(sample_data)

        store.write(key, serialized)
        retrieved = store.read(key)

        assert retrieved == serialized
        assert cloudpickle.loads(retrieved) == sample_data

    @pytest.mark.asyncio
    async def test_async_write_read_roundtrip(self, store, sample_data):
        """Test async write/read operations work correctly."""
        key = "experiments/model_v2.pkl"
        serialized = cloudpickle.dumps(sample_data)

        await store.awrite(key, serialized)
        retrieved = await store.aread(key)

        assert retrieved == serialized
        assert cloudpickle.loads(retrieved) == sample_data

    def test_exists_behavior(self, store):
        """Test exists() correctly identifies present and missing keys."""
        key = "test/check_exists.bin"

        assert not store.exists(key)
        store.write(key, b"test data")
        assert store.exists(key)
        store.delete(key)
        assert not store.exists(key)

    @pytest.mark.asyncio
    async def test_async_exists_behavior(self, store):
        """Test async exists() correctly identifies present and missing keys."""
        key = "test/async_check_exists.bin"

        assert not await store.aexists(key)
        await store.awrite(key, b"async test data")
        assert await store.aexists(key)
        store.delete(key)
        assert not await store.aexists(key)

    def test_missing_key_raises_error(self, store):
        """Test that reading missing keys raises appropriate errors."""
        with pytest.raises(AssetNotFoundError):
            store.read("missing/key.bin")

    def test_delete_operations(self, store):
        """Test deletion of existing and non-existing keys."""
        key = "test/to_delete.bin"
        store.write(key, b"temporary data")
        assert store.exists(key)

        store.delete(key)
        assert not store.exists(key)

        # Delete non-existing key should be safe
        store.delete("never/existed.bin")

    @pytest.mark.parametrize(
        "key,data",
        [
            ("simple.bin", b"simple data"),
            ("with-dashes.bin", b"dashed data"),
            ("with_underscores.bin", b"underscore data"),
            ("deep/nested/path.bin", b"nested data"),
        ],
    )
    def test_various_key_formats(self, store, key, data):
        """Test stores handle various valid key formats."""
        store.write(key, data)
        assert store.read(key) == data


class TestFileSystemAssetStore:
    """Test FileSystemAssetStore specific functionality."""

    def test_directory_creation(self):
        """Test that stores create necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "new_store"
            assert not base_path.exists()

            FileSystemAssetStore(base_path, create=True)
            assert base_path.exists()
            assert base_path.is_dir()

    def test_create_false_with_missing_directory(self):
        """Test that create=False fails when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing"

            with pytest.raises(AssetStoreError, match="not a directory"):
                FileSystemAssetStore(missing_path, create=False)

    def test_path_traversal_prevention(self):
        """Test that stores prevent path traversal attacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileSystemAssetStore(temp_dir)

            with pytest.raises(AssetStoreError, match="outside of store root"):
                store.write("../../../etc/passwd", b"malicious")

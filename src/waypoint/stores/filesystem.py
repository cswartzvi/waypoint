from __future__ import annotations

from pathlib import Path

from waypoint.exceptions import AssetNotFoundError
from waypoint.exceptions import AssetStoreError

from ._base import BaseAssetStore


class FileSystemAssetStore(BaseAssetStore):
    """File-system backed asset store."""

    name = "filesystem"

    def __init__(self, base_path: str | Path, *, create: bool = True) -> None:
        path = Path(base_path)
        path = path.resolve() if not path.is_absolute() else path
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.exists() or not path.is_dir():  # pragma: no cover - defensive
            raise AssetStoreError(f"Asset store path '{path}' is not a directory")
        self._base_path = path

    def _resolve_key(self, key: str) -> Path:
        """Resolve ``key`` to a concrete path within the base directory."""
        candidate = (self._base_path / key).resolve()
        try:
            candidate.relative_to(self._base_path)
        except ValueError as exc:  # pragma: no cover - defensive
            raise AssetStoreError(
                f"Asset key '{key}' resolves outside of store root '{self._base_path}'"
            ) from exc
        return candidate

    def write(self, key: str, data: bytes) -> None:
        """Write raw bytes to the file backing ``key``."""
        path = self._resolve_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, key: str) -> bytes:
        """Read and return bytes from the file backing ``key``."""
        path = self._resolve_key(key)
        if not path.exists():
            raise AssetNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        """Return ``True`` if a file exists for ``key``."""
        path = self._resolve_key(key)
        return path.exists()

    def delete(self, key: str) -> None:
        """Delete the file backing ``key`` if present."""
        path = self._resolve_key(key)
        if path.exists():
            path.unlink()

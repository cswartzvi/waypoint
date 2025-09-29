from __future__ import annotations

from threading import RLock

from waypoint.exceptions import AssetNotFoundError

from ._base import BaseAssetStore


class InMemoryAssetStore(BaseAssetStore):
    """Simple asset store that keeps assets in-memory in the current process."""

    name = "in-memory"

    def __init__(self) -> None:
        self._lock = RLock()
        self._data: dict[str, bytes] = {}

    def write(self, key: str, data: bytes) -> None:
        """Store raw bytes for the specified key in memory."""
        with self._lock:
            self._data[key] = data

    def read(self, key: str) -> bytes:
        """Return the bytes stored for ``key`` or raise if missing."""
        with self._lock:
            try:
                return self._data[key]
            except KeyError:  # pragma: no cover - simple guard
                raise AssetNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        """Check whether ``key`` has a stored value."""
        with self._lock:
            return key in self._data

    def delete(self, key: str) -> None:
        """Remove any data stored for ``key``."""
        with self._lock:
            self._data.pop(key, None)

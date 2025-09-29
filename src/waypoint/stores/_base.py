from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseAssetStore(ABC):
    """Abstract base class for persistent asset storage backends."""

    name: str = "asset-store"

    def duplicate(self) -> "BaseAssetStore":
        """Return a store instance appropriate for a new flow run scope."""
        return self

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Persist the raw bytes for a given asset key."""

    async def awrite(self, key: str, data: bytes) -> None:
        """Async version of write. Default implementation calls sync version."""
        return self.write(key, data)

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Retrieve the raw bytes associated with an asset key."""

    async def aread(self, key: str) -> bytes:
        """Async version of read. Default implementation calls sync version."""
        return self.read(key)

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if the asset key is present in the store."""

    async def aexists(self, key: str) -> bool:
        """Async version of exists. Default implementation calls sync version."""
        return self.exists(key)

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the asset at the provided key if it exists."""

    @abstractmethod
    def __str__(self) -> str:
        pass

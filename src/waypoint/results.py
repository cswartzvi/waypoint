"""Utilities for storing and retrieving task results."""

from __future__ import annotations

import importlib
from typing import Any, Protocol


class ResultStore(Protocol):
    """Protocol for storing and retrieving task results."""

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to the given path."""
        ...

    def read_bytes(self, path: str) -> bytes:
        """Read bytes from the given path."""
        ...

    def exists(self, path: str) -> bool:
        """Return ``True`` if the given path exists."""
        ...


class FSSpecResultStore(ResultStore):
    """
    A :class:`ResultStore` backed by an ``fsspec`` filesystem.

    Args:
        fs: Optional ``fsspec`` filesystem. If not provided, a local
            filesystem is created.
    """

    def __init__(self, fs: Any | None = None) -> None:
        if fs is None:
            try:
                fsspec = importlib.import_module("fsspec")
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
                raise RuntimeError("fsspec is required to use FSSpecResultStore") from exc
            fs = fsspec.filesystem("file")
        self._fs: Any = fs

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path``."""
        with self._fs.open(path, "wb") as file:
            file.write(data)

    def read_bytes(self, path: str) -> bytes:
        """Read and return bytes from ``path``."""
        with self._fs.open(path, "rb") as file:
            return file.read()

    def exists(self, path: str) -> bool:
        """Return ``True`` if ``path`` exists."""
        return self._fs.exists(path)

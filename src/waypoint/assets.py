from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from string import Formatter
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeAlias, TypeVar, overload

import cloudpickle

from waypoint.exceptions import AssetKeyFormatError
from waypoint.exceptions import AssetMapperError
from waypoint.exceptions import AssetNotFoundError
from waypoint.exceptions import AssetStoreError
from waypoint.stores.filesystem import BaseAssetStore

if TYPE_CHECKING:
    from waypoint.context import FlowRunContext
    from waypoint.context import TaskRunContext
else:  # pragma: no cover
    FlowRunContext = Any
    TaskRunContext = Any

T = TypeVar("T")

AssetKeyBuilder: TypeAlias = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]
"""Builds an asset storage key from flow run context, task run context and parameters."""

_FORMATTER = Formatter()


@overload
def asset_mapper(key: str | AssetKeyBuilder) -> BoundAssetMapper[Any]: ...


@overload
def asset_mapper(
    key: str | AssetKeyBuilder,
    *,
    serializer: Callable[[T], bytes] | None = None,
    deserializer: Callable[[bytes], T] | None = None,
) -> BoundAssetMapper[T]: ...


@overload
def asset_mapper(
    key: None = None,
    *,
    serializer: Callable[[T], bytes] | None = None,
    deserializer: Callable[[bytes], T] | None = None,
) -> AssetMapper[T]: ...


def asset_mapper(
    key: str | AssetKeyBuilder | None = None,
    *,
    serializer: Callable[[T], bytes] | None = None,
    deserializer: Callable[[bytes], T] | None = None,
) -> BoundAssetMapper[T] | BoundAssetMapper[Any] | AssetMapper[T]:
    """
    Create an asset mapper that uses `key` to compute storage keys.

    Args:
        key (str | AssetKeyBuilder):
            A format string or callable that computes a storage key from in-context parameters.
            If a format string is provided, it may include replacement fields that correspond
            to parameters in the current flow or task run, as well as `flow` and `task`
            fields that provide access to the current run metadata. For example, a key of
            `"results/{name}.bin"` would use the `name` parameter from the current run to
            compute a key like `"results/world.bin"`.
        serializer (Callable[[T], bytes], optional):
            A callable that serializes a value of type `T` into bytes for storage.
            If not provided, defaults to using `cloudpickle.dumps`.
        deserializer (Callable[[bytes], T], optional):
            A callable that deserializes bytes from storage back into a value of type `T`.
            If not provided, defaults to using `cloudpickle.loads`.

    Returns:
        An `AssetMapper` instance configured with the provided parameters or factory that will
        materialize an `AssetMapper` from an asset key.

    Example:
        >>> from waypoint.assets import asset_mapper
        >>>
        >>> def custom_serializer(value: str) -> bytes:
        ...     return value.encode("utf-8")
        >>>
        >>> def custom_deserializer(data: bytes) -> str:
        ...     return data.decode("utf-8")
        >>>
        >>> mapper = asset_mapper(
        ...     key="data/{id}.txt",
        ...     serializer=custom_serializer,
        ...     deserializer=custom_deserializer,
        ...     name="text-file-mapper",
        ... )
    """
    if key is not None:
        return BoundAssetMapper(
            key_or_func=key,
            serializer=serializer or _default_serializer,
            deserializer=deserializer or _default_deserializer,
        )

    return AssetMapper(
        serializer=serializer or _default_serializer,
        deserializer=deserializer or _default_deserializer,
    )


def _default_serializer(value: Any) -> bytes:
    try:
        return cloudpickle.dumps(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise AssetMapperError("cloudpickle", f"failed to serialize value: {exc}") from exc


def _default_deserializer(data: bytes) -> Any:
    try:
        return cloudpickle.loads(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise AssetMapperError("cloudpickle", f"failed to deserialize value: {exc}") from exc


def _validate_explicit_key(key: str) -> None:
    """Validate an explicitly provided asset key."""
    if not key or not isinstance(key, str):
        raise AssetMapperError("explicit-key", "Asset key must be a non-empty string")

    # Basic security: no path traversal
    if ".." in key:
        raise AssetMapperError("explicit-key", "Asset key cannot contain '..' (path traversal)")

    # No absolute paths
    if key.startswith("/") or (len(key) > 1 and key[1] == ":"):  # Unix absolute or Windows C:
        raise AssetMapperError("explicit-key", "Asset key cannot be an absolute path")


@dataclass
class AssetMapper(Generic[T]):
    """Handles serialization and storage of assets using a backend store."""

    serializer: Callable[[T], bytes] = field(default=_default_serializer)
    deserializer: Callable[[bytes], T] = field(default=_default_deserializer)

    def __call__(self, key_or_func: str | AssetKeyBuilder) -> BoundAssetMapper[T]:
        """Bind a key or key-building function to this mapper, returning a BoundAssetMapper."""
        return BoundAssetMapper(
            key_or_func=key_or_func,
            serializer=self.serializer,
            deserializer=self.deserializer,
            _mapper=self,
        )

    def exists(self, key: str, *, store: BaseAssetStore | None = None) -> bool:
        """Check if an asset exists, with optional explicit key and store."""
        _validate_explicit_key(key)
        resolved_store = self._resolve_store(store)
        return resolved_store.exists(key)

    def save(self, key: str, value: T, *, store: BaseAssetStore | None = None) -> str:
        """Save a value to storage, with optional explicit key and store."""
        _validate_explicit_key(key)
        resolved_store = self._resolve_store(store)
        try:
            resolved_store.write(key, self.serializer(value))
        except Exception as exc:  # pragma: no cover
            raise AssetMapperError(f"failed to write asset '{key}': {exc}") from exc
        return key

    def load(self, key: str, *, store: BaseAssetStore | None = None, default: T | None = None) -> T:
        """Load a value from storage, with optional explicit key, store, and default."""
        _validate_explicit_key(key)
        resolved_store = self._resolve_store(store)
        if not resolved_store.exists(key):
            if default is not None:
                return default
            raise AssetNotFoundError(key)
        try:
            raw = resolved_store.read(key)
        except Exception as exc:  # pragma: no cover
            raise AssetMapperError(f"failed to read asset '{key}': {exc}") from exc

        try:
            return self.deserializer(raw)
        except Exception as exc:  # pragma: no cover
            raise AssetMapperError(f"failed to decode asset '{key}': {exc}") from exc

    def _resolve_store(self, store: BaseAssetStore | None = None) -> BaseAssetStore:
        """Resolve the backend store from parameter or context."""
        if store is not None:
            return store

        from waypoint.context import FlowRunContext

        flow_context = FlowRunContext.get()
        if flow_context is None or flow_context.asset_store is None:  # pragma: no cover
            raise AssetStoreError(
                "Asset operations require a configured asset store either directly "
                "specified or from the current flow run context"
            )
        return flow_context.asset_store


@dataclass
class BoundAssetMapper(Generic[T]):
    """Associates a logical key template with serialization logic for a value."""

    key_or_func: str | AssetKeyBuilder
    serializer: Callable[[T], bytes] = field(default=_default_serializer, repr=False)
    deserializer: Callable[[bytes], T] = field(default=_default_deserializer, repr=False)
    _mapper: AssetMapper[T] | None = None

    @cached_property
    def mapper(self) -> AssetMapper[T]:
        """Underlying AssetMapper used for serialization and storage operations."""
        if self._mapper is None:
            self._mapper = AssetMapper(serializer=self.serializer, deserializer=self.deserializer)
        return self._mapper

    def compute_key(
        self,
        *,
        parameters: dict[str, Any] | None = None,
        flow_context: FlowRunContext | None = None,
        task_context: TaskRunContext | None = None,
    ) -> str:
        """Compute the storage key with optional context and parameter overrides."""
        from waypoint.context import FlowRunContext
        from waypoint.context import TaskRunContext

        # Use provided contexts or fall back to current contexts
        flow_context = flow_context if flow_context else FlowRunContext.get()
        task_context = task_context if task_context else TaskRunContext.get()

        parameters = parameters or {}

        # Build parameters with explicit precedence: task > flow > manual
        base_params: dict[str, Any] = {}
        if task_context is not None:
            base_params.update(task_context.task_run.parameters)
        elif flow_context is not None:
            base_params.update(flow_context.flow_run.parameters)
        if parameters:
            base_params.update(parameters)  # Manual parameters override context

        # Compute key using contexts and parameters
        if callable(self.key_or_func):
            serialized_flow = flow_context.serialize() if flow_context else {}
            serialized_task = task_context.serialize() if task_context else {}
            key = self.key_or_func(serialized_flow, serialized_task, base_params)
        else:
            try:
                format_kwargs = self._format_kwargs(flow_context, task_context, base_params)
                key = _FORMATTER.vformat(self.key_or_func, (), format_kwargs)
            except KeyError as exc:
                missing = exc.args[0] if exc.args else str(exc)
                available = set(base_params.keys())
                raise AssetKeyFormatError(
                    self.key_or_func,
                    f"Missing parameter '{missing}' for key template. Available: {available}",
                )

        if not isinstance(key, str) or not key:
            raise AssetMapperError(str(self), "computed key must be a non-empty string")

        return key

    def exists(
        self,
        *,
        store: BaseAssetStore | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """Check if an asset exists, with optional explicit key and store."""
        computed_key = self.compute_key(parameters=parameters)
        return self.mapper.exists(computed_key, store=store)

    def save(
        self,
        value: T,
        *,
        store: BaseAssetStore | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Save a value to storage, with optional explicit key and store."""
        computed_key = self.compute_key(parameters=parameters)
        return self.mapper.save(computed_key, value, store=store)

    def load(
        self,
        *,
        store: BaseAssetStore | None = None,
        default: T | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> T:
        """Load a value from storage, with optional explicit key, store, and default."""
        computed_key = self.compute_key(parameters=parameters)
        return self.mapper.load(computed_key, store=store, default=default)

    def _format_kwargs(
        self,
        flow_run_context: FlowRunContext | None,
        task_run_context: TaskRunContext | None,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the format kwargs for the key template."""
        data = dict(parameters)
        if flow_run_context is not None:
            if flow_run := flow_run_context.flow_run:  # pragma: no cover
                data.setdefault("flow_run", flow_run)
        if task_run_context is not None:
            if task_run := task_run_context.task_run:  # pragma: no cover
                data.setdefault("task_run", task_run)
        return data

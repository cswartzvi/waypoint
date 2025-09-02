from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import ExitStack
from contextlib import contextmanager
from contextvars import ContextVar
from contextvars import Token
from typing import TYPE_CHECKING, Any, ClassVar

from pluggy import PluginManager
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr
from typing_extensions import Self

from waypoint.runners.base import BaseTaskRunner
from waypoint.tasks import TaskData
from waypoint.utils.subclasses import iter_subclasses

if TYPE_CHECKING:
    from waypoint.flow_run import FlowRun
    from waypoint.flows import FlowData
    from waypoint.task_run import TaskRun
    from waypoint.tasks import TaskData
else:
    FlowData = Any
    FlowRun = Any
    Task = Any
    TaskRun = Any

# region Base


class ContextModel(BaseModel):
    """A base model for managing the context state."""

    if TYPE_CHECKING:
        # subclasses can pass through keyword arguments to the pydantic base model
        def __init__(self, **kwargs: Any) -> None: ...

    # The context variable for storing data must be defined by the child class
    __var__: ClassVar[ContextVar[Any]]
    _token: Token[Self] | None = PrivateAttr(None)
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("Context already entered. Context enter calls cannot be nested.")
        self._token = self.__var__.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._token:
            raise RuntimeError("Asymmetric use of context. Context exit called without an enter.")
        self.__var__.reset(self._token)
        self._token = None

    @classmethod
    def get(cls: type[Self]) -> Self | None:
        """Gets the current context instance."""
        return cls.__var__.get(None)

    def model_copy(
        self: Self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """
        Duplicates the context model, optionally choose which fields to update.

        Args:
            update (Mapping[str, Any], optional):
                Values to change/add in the new model. Note: the data is not validated before
                creating the new model - you should trust this data.
            deep (bool):
                Set to `True` to make a deep copy of the model. Defaults to `False`.

        Returns:
            A new model instance.
        """
        new = super().model_copy(update=update, deep=deep)
        # Remove the token on copy to avoid re-entrance errors
        new._token = None
        return new

    def serialize(self, include_secrets: bool = True) -> dict[str, Any]:
        """Serialize the context model to a dictionary that can be pickled with cloudpickle."""
        return self.model_dump(exclude_unset=True, context={"include_secrets": include_secrets})


# region API


class TaskRunContext(ContextModel):
    """A context model for task run state."""

    __var__ = ContextVar("task_run", default=None)

    task_data: TaskData
    """Data related to the underlying task."""

    task_run: TaskRun
    """Data related to the current task run."""


class FlowRunContext(ContextModel):
    """A context model for workflow run state."""

    __var__ = ContextVar("flow_run", default=None)

    flow_data: FlowData
    """Data related to the underlying workflow."""

    flow_run: FlowRun
    """Data related to the current workflow run."""

    task_runner: BaseTaskRunner
    """
    Task runner used for executing tasks in the current workflow run.

    Note that the task runner may be duplicated for each flow run to ensure
    that the task runner's state is isolated to the flow run.
    """


class HooksContext(ContextModel):
    """A context model for managing hooks."""

    __var__ = ContextVar("hooks", default=None)

    manager: PluginManager | None = None


FlowRunContext.model_rebuild()


# region Helpers


def serialize_context() -> dict[str, Any]:
    """Serialize the current context for use in a remote execution environment."""
    data = {}

    # Get all ContextModel subclasses and serialize their current instances
    for subclass in iter_subclasses(ContextModel):
        context = subclass.get()
        if context is None:
            continue
        key = getattr(subclass, "__var__", None)
        if key is None:  # pragma: no cover
            continue  # NOTE: Normally shouldn't happen, `ContextModel` has subclasses
        data[key.name] = context.serialize()

    return data


@contextmanager
def hydrated_context(serialized_context: dict[str, Any] | None = None) -> Iterator[None]:
    """Context manager to hydrate the context models from a serialized state."""
    # We need to rebuild the models because we might be hydrating in a remote
    # environment where the models are not available.
    FlowRunContext.model_rebuild()
    TaskRunContext.model_rebuild()
    HooksContext.model_rebuild()

    with ExitStack() as stack:
        if serialized_context:
            for subclass in iter_subclasses(ContextModel):
                key = getattr(subclass, "__var__", None)
                if key is None:  # pragma: no cover
                    continue  # NOTE: Normally shouldn't happen, `ContextModel` has subclasses

                data = serialized_context.get(key.name)
                if data is None:
                    continue

                instance = subclass(**data)
                stack.enter_context(instance)
        yield

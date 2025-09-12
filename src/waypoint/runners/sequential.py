from typing import Any, Callable, ClassVar

from typing_extensions import Self, override

from waypoint.futures import DelayedTaskFuture
from waypoint.runners.base import BaseTaskRunner
from waypoint.runners.base import DefaultTaskRunners


class SequentialTaskRunner(BaseTaskRunner):
    """
    An executor that runs tasks sequentially.

    This task runner executes tasks one at a time in the order they are submitted.
    """

    type: ClassVar[str] = DefaultTaskRunners[0]

    @override
    def duplicate(self) -> Self:
        """Create a duplicate of the task runner."""
        return type(self)()

    @override
    def _submit(self, func: Callable[[], Any]) -> DelayedTaskFuture[Any]:
        delayed_future = DelayedTaskFuture(func)
        return delayed_future

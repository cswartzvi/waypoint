import asyncio
import inspect
from concurrent.futures import Future
from typing import Any, Callable, ClassVar

from typing_extensions import Self, override

from waypoint.futures import TaskFuture
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
    def _submit(self, func: Callable[[], Any]) -> TaskFuture[Any]:
        raw_future: Future[Any] = Future()

        # No error handling here; if the function raises it should propagate to the caller
        if inspect.iscoroutinefunction(func):
            result = asyncio.run(func())
        else:
            result = func()
        raw_future.set_result(result)
        return TaskFuture(raw_future)

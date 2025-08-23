from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from waypoint.runners.base import BaseTaskRunner
else:
    BaseTaskRunner = object


DefaultTaskRunners = Literal["sequential", "concurrent", "parallel"]


def get_task_runner(type_: DefaultTaskRunners | str | BaseTaskRunner) -> "BaseTaskRunner":
    """Look up a task runner based on their type attribute."""
    # NOTE: We must import the runners here in order for them to be discovered.
    from waypoint.runners.base import BaseTaskRunner
    from waypoint.runners.sequential import SequentialTaskRunner  # noqa: F401
    from waypoint.runners.threading import ThreadingTaskRunner  # noqa: F401
    from waypoint.utils.subclasses import get_subclass

    if isinstance(type_, BaseTaskRunner):
        return type_

    runner = get_subclass(BaseTaskRunner, lambda x: getattr(x, "type") == type_)
    return runner()

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from waypoint.tasks import TaskData
else:
    TaskData = Any


@dataclass(frozen=True)
class TaskRun:
    """Data about a specific execution of a Waypoint task."""

    name: str
    """Qualified name of the underlying task being executed."""

    task_id: str
    """Unique identifier for the task run within a workflow run."""

    flow_run_id: UUID | None = None
    """Identifier for the parent workflow run, if applicable."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Parameters passed to the flow run."""


def create_task_run(task_data: TaskData, parameters: dict[str, Any] | None = None) -> TaskRun:
    """
    Create a new task run instance for the given task.

    Args:
        task_data (TaskData): Metadata related to the underlying task.
        parameters (dict[str, Any], optional): Parameters to pass to the task run. Defaults to None.

    Raises:
        MissingContextError: If there is no active flow run context.

    Returns:
        TaskRun: A new instance of TaskRun with the provided metadata.
    """
    from waypoint.context import FlowRunContext
    from waypoint.exceptions import MissingContextError

    parent_flow_context = FlowRunContext.get()

    if parent_flow_context is None:
        raise MissingContextError(
            "Task runs can only be created from inside flow run or session context."
        )

    # Collect parent flow information
    parent_flow = parent_flow_context.flow_data
    parent_flow_run = parent_flow_context.flow_run
    parent_flow_id = parent_flow_run.flow_id
    task_run_index = parent_flow_run.next_task_run_index(task_data.name)
    task_run_name = f"{parent_flow.name}.{task_data.name}"
    task_run_id = f"{task_run_name}-{task_run_index}"

    return TaskRun(
        name=task_run_name,
        task_id=task_run_id,
        flow_run_id=parent_flow_id,
        parameters=parameters or {},
    )

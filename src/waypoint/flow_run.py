from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID

if TYPE_CHECKING:
    from waypoint.flows import FlowData
else:
    FlowData = Any


@dataclass(frozen=True)
class FlowRun:
    """Data about a specific execution of a Waypoint workflow."""

    name: str
    """Qualified name of the underlying workflow."""

    flow_id: UUID
    """Unique identifier for the flow run."""

    parent_flow_id: UUID | None = None
    """Identifier for the parent flow run, if this is a sub-flow."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Parameters passed to the flow run."""

    task_run_counter: Mapping[str, int] = field(default_factory=dict)
    """Counter for task runs within this flow run, keyed by task name."""

    def next_task_run_index(self, task_name: str) -> int:
        """
        Get the next task run ID for a given task name within this flow run.

        This method increments the internal counter for the specified task name and
        returns the new value. If the task name has not been seen before, it initializes
        its counter to 1.

        Args:
            task_name (str): Qualified name of the task.

        Returns:
            int: The next task run ID for the specified task name.
        """
        current_count = self.task_run_counter.get(task_name, 0)
        next_count = current_count + 1

        # Use replace to create a new instance with updated counter
        new_counters = dict(self.task_run_counter)
        new_counters[task_name] = next_count
        object.__setattr__(self, "task_run_counter", new_counters)

        return next_count


def create_flow_run(flow_data: FlowData, parameters: dict[str, Any] | None = None) -> "FlowRun":
    """
    Create a new flow run instance for the given workflow.

    Args:
        flow_data (FlowData): Metadata related to the underlying workflow.
        parameters (dict[str, Any], optional): Parameters to pass to the flow run. Defaults to None.

    Raises:
        InvalidFlowError: If the provided function is not a valid flow - not decorated with`@flow`.

    Returns:
        FlowRun: New flow run instance.

    """
    from uuid import uuid4

    from waypoint.context import FlowRunContext

    parent_flow_context = FlowRunContext.get()
    parent_flow_run = parent_flow_context.flow_run if parent_flow_context else None
    parent_flow_id = parent_flow_run.flow_id if parent_flow_run else None

    return FlowRun(
        name=flow_data.name,
        flow_id=uuid4(),
        parent_flow_id=parent_flow_id,
        parameters=parameters or {},
    )

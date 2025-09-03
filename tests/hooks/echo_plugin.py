from typing import TYPE_CHECKING, Any

from waypoint.hooks.markers import hook_impl

if TYPE_CHECKING:
    from waypoint.flow_run import FlowRun
    from waypoint.flows import FlowData
    from waypoint.runners import BaseTaskRunner
    from waypoint.task_run import TaskRun
    from waypoint.tasks import TaskData
else:
    FlowData = object
    FlowRun = object
    TaskData = object
    TaskRun = object
    BaseTaskRunner = object


class EchoPlugin:
    def __init__(self) -> None:
        self.lines: list[str] = []

    @hook_impl
    def before_flow_run(self, flow_data: FlowData, flow_run: FlowRun) -> None:
        self.lines.append(f"before_flow_run called - {flow_run.name}")

    @hook_impl
    def after_flow_iteration(
        self, flow_data: FlowData, flow_run: FlowRun, result: Any, index: int
    ) -> None:
        self.lines.append(f"after_flow_iteration called - {flow_run.name} - {index}")

    @hook_impl
    def after_flow_run(
        self,
        flow_data: FlowData,
        flow_run: FlowRun,
        error: Exception | None,
        result: object | None,
    ) -> None:
        self.lines.append(f"after_flow_run called - {flow_run.name}")

    @hook_impl
    def before_task_submit(
        self, task_data: TaskData, task_run: TaskRun, task_runner: BaseTaskRunner
    ) -> None:
        self.lines.append(f"before_task_submit called - {task_run.task_id}")

    @hook_impl
    def after_task_iteration(
        self, task_data: TaskData, task_run: TaskRun, result: Any, index: int
    ) -> None:
        self.lines.append(f"after_task_iteration called - {task_run.task_id} - {index}")

    @hook_impl
    def before_task_run(self, task_data: TaskData, task_run: TaskRun) -> None:
        self.lines.append(f"before_task_run called - {task_run.task_id}")

    @hook_impl
    def after_task_run(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        error: Exception | None,
        result: object | None,
    ) -> None:
        self.lines.append(f"after_task_run called - {task_run.task_id}")

    @hook_impl
    def after_task_future_result(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        error: Exception | None,
        result: object | None,
    ) -> None:
        self.lines.append(f"after_task_future_result called - {task_run.task_id}")

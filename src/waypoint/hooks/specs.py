"""
Contains the specifications for all callable hooks in the Waypoint framework.

For more information, please seettps://pluggy.readthedocs.io/en/stable/#specs
"""

from typing import TYPE_CHECKING, Any

from pluggy import PluginManager

from waypoint.runners.base import BaseTaskRunner

from .markers import hook_spec

if TYPE_CHECKING:
    from waypoint.flow_run import FlowRun
    from waypoint.flows import FlowData
    from waypoint.task_run import TaskRun
    from waypoint.tasks import TaskData
else:
    FlowData = object
    FlowRun = object
    TaskData = object
    TaskRun = object


class FlowSpec:
    """Specifies the hooks related to flow execution."""

    @hook_spec
    def before_flow_run(self, flow_data: FlowData, flow_run: FlowRun) -> None:
        """
        Hook called before a flow run starts.

        Args:
            flow_data (waypoint.flows.FlowData): Metadata related to the underlying flow.
            flow_run (waypoint.flows.FlowRune): Context related data about the current flow run.
        """
        pass

    @hook_spec
    def after_flow_iteration(
        self, flow_data: FlowData, flow_run: FlowRun, result: Any, index: int
    ) -> None:
        """
        Hook called after each iteration of a generator based flow.

        Only applies to flows that are generators.

        Args:
            flow_data (waypoint.flows.FlowData): Metadata related to the underlying flow.
            flow_run (waypoint.flows.FlowRune): Context related data about the current flow run.
            result (Any): Result of the current iteration.
            index (int): Current iteration index.
        """
        pass

    @hook_spec
    def after_flow_run(
        self,
        flow_data: FlowData,
        flow_run: FlowRun,
        error: Exception | None,
        result: object | None,
    ) -> None:
        """
        Hook called after a flow run completes.

        Args:
            flow_data (waypoint.flows.FlowData): Metadata related to the underlying flow.
            flow_run (waypoint.flows.FlowRune): Context related data about the current flow run.
            error (Exception, optional): Exception that occurred during the flow execution, if any.
            result (Any): The result of the flow execution.
        """
        pass

    @hook_spec
    def after_result_handling(self, flow_data: FlowData, flow_run: FlowRun, result: Any) -> None:
        """
        Hook called after the result of a flow run is handled.

        Args:
            flow_data (waypoint.flows.FlowData): Metadata related to the underlying flow.
            flow_run (waypoint.flows.FlowRune): Context related data about the current flow run.
            result (Any): The result of the flow execution.
        """
        pass


class TaskSpec:
    """Specifies the hooks related to task execution."""

    @hook_spec
    def before_task_submit(
        self, task_data: TaskData, task_run: TaskRun, runner: BaseTaskRunner
    ) -> None:
        """
        Hook called before a task is submitted for execution.

        Args:
            task_data (waypoint.tasks.TaskData): Metadata related to the underlying task.
            task_run (waypoint.tasks.TaskRun): Details about current task execution.
            runner (waypoint.runners.base.BaseTaskRunner): Task runner that will execute the task.
        """
        pass

    @hook_spec
    def after_task_iteration(
        self, task_data: TaskData, task_run: TaskRun, result: Any, index: int
    ) -> None:
        """
        Hook called after each iteration of a generator based task.

        Args:
            task_data (waypoint.tasks.TaskData): Metadata related to the underlying task.
            task_run (waypoint.tasks.TaskRun): Details about current task execution.
            result (Any): Result of the current iteration.
            index (int): Current iteration index.
        """
        pass

    @hook_spec
    def before_task_run(self, task_data: TaskData, task_run: TaskRun) -> None:
        """
        Hook called before a task run starts.

        Args:
            task_data (waypoint.tasks.TaskData): Metadata related to the underlying task.
            task_run (waypoint.tasks.TaskRun): Details about current task execution.
        """
        pass

    @hook_spec
    def after_task_run(
        self,
        task_data: TaskData,
        task_run: TaskRun,
        error: Exception | None,
        result: object | None,
    ) -> None:
        """
        Hook called after a task run completes.

        Args:
            task_data (waypoint.tasks.TaskData): Metadata related to the underlying task.
            task_run (waypoint.tasks.TaskRun): Details about current task execution.
            error (Exception, optional): Exception that occurred during the task execution, if any.
            result (Any): The result of the task execution.
        """
        pass

    @hook_spec
    def after_result_handling(self, task_data: TaskData, task_run: TaskRun, result: object) -> None:
        """
        Hook called after the result of a task run is handled.

        Args:
            task_data (waypoint.tasks.TaskData): Metadata related to the underlying task.
            task_run (waypoint.tasks.TaskRun): Details about current task execution.
            result (Any): The result of the task execution.
        """
        pass


def try_run_hook(manager: PluginManager, hook_name: str, **kwargs: Any) -> None:
    """
    Tries to run a specified hook, with the provided arguments if it exists.

    Note that this function is intended for internal use only.

    Args:
        manager (Any): The hook manager containing the hooks.
        hook_name (str): The name of the hook to run.
        **kwargs (Any): Keyword arguments to pass to the hook.
    """
    if hook := getattr(manager.hook, hook_name, None):
        hook(**kwargs)

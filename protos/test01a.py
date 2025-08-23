from typing import Any

from waypoint import Flow
from waypoint import FlowRun
from waypoint import flow
from waypoint.hooks import hook_impl
from waypoint.hooks import register_hooks


class MyHooks:
    @hook_impl
    def before_flow_run(self, flow: Flow[Any, Any], flow_run: FlowRun) -> None:
        print(f"Before flow {flow.name} - {flow_run.flow_id}")

    @hook_impl
    def after_flow_run(self, flow: Flow[Any, Any], flow_run: FlowRun, result: object) -> None:
        print(f"After flow {flow.name} - {flow_run.flow_id} - '{result}'")


register_hooks(MyHooks())


@flow
def test_flow(name: str) -> str:
    """A simple test flow.

    Args:
        name (str): The name to greet.
    """
    print("Inside `test_flow`")
    return f"Hello, {name}!"


if __name__ == "__main__":
    result = test_flow("World")

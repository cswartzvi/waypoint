import logging
import time

from waypoint.flows import flow
from waypoint.hooks import register_hooks
from waypoint.plugins.logger import WaypointLogger
from waypoint.tasks import map_task
from waypoint.tasks import task

register_hooks(WaypointLogger(console_level=logging.INFO))


@task
def greet(name: str) -> str:
    time.sleep(1)
    return f"Hello, {name}!"


@flow(task_runner="sequential")
def main(names: list[str]) -> list[str]:
    greetings: list[str] = []
    for name in map_task(greet, names):
        greetings.append(name)
    return greetings


if __name__ == "__main__":
    result = main(["Chuck", "Amy", "Charlie", "Harrison"])
    print(result)

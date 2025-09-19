import logging
import time
from typing import Any

from rich import print

from waypoint.context import serialize_context
from waypoint.flows import flow
from waypoint.hooks import register_hooks
from waypoint.logging import setup_console_logging
from waypoint.plugins.logger import WaypointLogger
from waypoint.tasks import map_task
from waypoint.tasks import task

setup_console_logging(level=logging.DEBUG, traceback=True, use_rich=True)


@task
def testing() -> dict[str, Any]:
    return serialize_context()


# @flow(task_runner="threading")
@flow(task_runner="sequential")
def main() -> dict[str, Any]:
    return testing()


if __name__ == "__main__":
    result = main()
    print(result)

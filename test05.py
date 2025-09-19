import logging
import time
from collections.abc import Iterator
from typing import Any

from rich import print

from waypoint.context import serialize_context
from waypoint.flows import flow
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.logging import setup_console_logging
from waypoint.plugins.logger import WaypointLogger
from waypoint.tasks import map_task
from waypoint.tasks import task

setup_console_logging(level=logging.INFO, traceback=True, use_rich=True)


@task
def greet(name: str) -> None:
    logger = get_run_logger()
    logger.info(f"Hello, {name}!")


# @flow(task_runner="threading")
@flow(task_runner="sequential")
def main() -> Iterator[str]:
    greet("World")
    for char in ["X", "Y", "Z"]:
        time.sleep(1)
        yield char
    greet("World")


if __name__ == "__main__":
    result = list(main())
    print(result)

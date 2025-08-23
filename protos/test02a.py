import logging
import time

from waypoint import flow
from waypoint import task
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.plugins.logger import WaypointLogger

register_hooks(WaypointLogger(console_level=logging.DEBUG))


@task
def greet(name: str) -> str:
    """A simple task that greets a user."""
    time.sleep(1)
    return f"Hello, {name}!"


@flow
def test_flow(name: str) -> str:
    """A simple flow that greets a user."""
    logger = get_run_logger()
    result = greet(name)
    logger.info("First print %s", result)
    result = greet(name)
    logger.info("Second print %s", result)
    return result


if __name__ == "__main__":
    result = test_flow("World")

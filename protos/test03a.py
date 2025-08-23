import logging
import time

from waypoint import flow
from waypoint import task
from waypoint.futures import wait
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.plugins.logger import WaypointLogger

register_hooks(WaypointLogger(console_level=logging.DEBUG))


@task
def greet(name: str) -> None:
    """A simple task that greets a user."""
    logger = get_run_logger()
    logger.debug("Sleeping before greeting %s", name)
    time.sleep(1)
    logger.info("Greeting %s", name)


@flow
def test_flow(name: str) -> None:
    """A simple flow that greets a user."""
    futures = [greet.submit(name) for name in ["chuck", "amy", "charlie", "harrison"]]
    wait(futures)


if __name__ == "__main__":
    test_flow("World")

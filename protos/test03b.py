import asyncio
import logging

from waypoint import flow
from waypoint import task
from waypoint.futures import wait
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.plugins.logger import WaypointLogger
from waypoint.tasks import async_task

register_hooks(WaypointLogger(console_level=logging.DEBUG))


@async_task
async def greet(name: str) -> None:
    logger = get_run_logger()
    logger.debug("Sleeping before greeting %s", name)
    await asyncio.sleep(3)
    logger.info("Greeting %s", name)


@flow
def test_flow(name: str) -> None:
    futures = [greet.submit(name) for name in ["chuck", "amy", "charlie", "harrison"]]
    wait(futures)


if __name__ == "__main__":
    test_flow("World")

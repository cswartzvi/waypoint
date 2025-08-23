import asyncio
import logging

from waypoint import flow
from waypoint import task
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.plugins.logger import WaypointLogger

register_hooks(WaypointLogger(console_level=logging.DEBUG))


@task
async def greet(name: str) -> None:
    logger = get_run_logger()
    await asyncio.sleep(3)  # Simulate a delay
    logger.info("Greeting '%s'", name)


@flow
async def test_flow(name: str) -> None:
    await asyncio.gather(greet(name), greet(name))


if __name__ == "__main__":
    asyncio.run(test_flow("World"))

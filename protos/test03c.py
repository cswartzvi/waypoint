import asyncio
import logging

from waypoint import flow
from waypoint import task
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


@async_task
async def test_task(names: list[str]) -> None:
    tasks = [greet(name) for name in names]
    await asyncio.gather(*tasks)


@flow
def test_flow(names: list[str]) -> None:
    future = test_task.submit(names)
    future.result()

reveal_type(greet)
reveal_type(greet.submit)
reveal_type(test_task)
reveal_type(test_task.submit)
reveal_type(test_flow)

if __name__ == "__main__":
    test_flow(["chuck", "amy", "charlie", "harrison"])

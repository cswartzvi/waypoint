import asyncio
import logging

from waypoint import flow
from waypoint import task
from waypoint.futures import wait
from waypoint.hooks import register_hooks
from waypoint.logging import get_run_logger
from waypoint.plugins.logger import WaypointLogger
from waypoint.runners.threading import ThreadingTaskRunner

register_hooks(WaypointLogger(console_level=logging.INFO))


@task
async def greet(name: str) -> None:
    logger = get_run_logger()
    logger.info("Sleeping before greeting %s", name)
    await asyncio.sleep(3)
    logger.info("Greeting %s", name)


@task
async def test_task(names: list[str]) -> None:
    tasks = [greet(name) for name in names]
    await asyncio.gather(*tasks)


@flow(task_runner=ThreadingTaskRunner(max_workers=1))
def test_flow(names: list[str]) -> None:
    future1 = test_task.submit(names)
    future2 = test_task.submit(names)
    wait([future1, future2])


if __name__ == "__main__":
    test_flow(["chuck", "amy", "charlie", "harrison"])

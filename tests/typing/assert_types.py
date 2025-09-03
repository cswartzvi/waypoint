"""Tests for various type assertions fro waypoint tasks and flows."""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
from typing import ParamSpec, TypeVar

from typing_extensions import assert_type

from waypoint.flows import flow
from waypoint.futures import TaskFuture
from waypoint.futures import as_completed
from waypoint.futures import wait
from waypoint.tasks import map_task
from waypoint.tasks import submit_task
from waypoint.tasks import task

P = ParamSpec("P")
R = TypeVar("R")

# TODO: We really should divide these tests into multiple files, but for now
# this is fine.


@task
def sync_task(x: int, y: int) -> int:
    return x + y


@task
def sync_generator_task(x: int, y: int) -> Generator[int, None, None]:
    yield x + y


@task
async def async_task(x: int, y: int) -> int:
    return x * y


@task
async def async_generator_task(x: int, y: int) -> AsyncGenerator[int, None]:
    yield x * y


@flow
def my_sync_flow(x: int, y: int) -> int:
    # Call sync task
    result1 = sync_task(x, y)
    assert_type(result1, int)

    # Call sync generator task
    result2 = 0
    for value in sync_generator_task(x, y):
        assert_type(value, int)
        result2 += value

    # Submit sync tasks
    futures1 = [submit_task(sync_task, i, i + 1) for i in range(5)]
    assert_type(futures1, list[TaskFuture[int]])

    # Map sync tasks
    result3 = list(map_task(sync_task, range(5), range(1, 6)))
    assert_type(result3, list[int])

    # Iterate sync tasks as completed
    futures2 = [submit_task(sync_task, i, i + 1) for i in range(5)]
    results4 = [f.result() for f in as_completed(futures2)]
    assert_type(results4, list[int])

    # Submit sync generator task
    future3 = submit_task(sync_generator_task, x, y)
    assert_type(future3, TaskFuture[list[int]])

    # Map sync generator task
    results5 = [val for val in map_task(sync_generator_task, range(5), range(1, 6))]
    assert_type(results5, list[list[int]])

    # Wait for sync tasks
    futures6 = [submit_task(sync_task, i, i + 1) for i in range(5)]
    done, not_done = wait(futures6)
    assert_type(done, set[TaskFuture[int]])
    assert_type(not_done, set[TaskFuture[int]])

    return x + y


@flow
async def my_async_flow(x: int, y: int) -> int:
    # Call sync task
    result1 = sync_task(x, y)
    assert_type(result1, int)

    # Call sync generator task
    result2 = 0
    for value in sync_generator_task(x, y):
        assert_type(value, int)
        result2 += value

    # Call async task
    result3 = await async_task(x, y)
    assert_type(result3, int)

    # Call async generator task
    result4 = 0
    async for value in async_generator_task(x, y):
        assert_type(value, int)
        result4 += value

    # Submit async tasks
    futures1 = [submit_task(async_task, i, i + 1) for i in range(5)]
    assert_type(futures1, list[TaskFuture[int]])

    # Map async tasks
    result5 = list(map_task(async_task, range(5), range(1, 6)))
    assert_type(result5, list[int])

    # Iterate async tasks as completed
    futures2 = [submit_task(async_task, i, i + 1) for i in range(5)]
    results6 = [f.result() for f in as_completed(futures2)]
    assert_type(results6, list[int])

    # Submit async generator task
    future3 = submit_task(async_generator_task, x, y)
    assert_type(future3, TaskFuture[list[int]])

    # Map sync async generator task
    results7 = list(map_task(async_generator_task, range(5), range(1, 6)))
    assert_type(results7, list[list[int]])

    # Wait for async tasks
    futures4 = [submit_task(async_task, i, i + 1) for i in range(5)]
    done, not_done = wait(futures4)
    assert_type(done, set[TaskFuture[int]])
    assert_type(not_done, set[TaskFuture[int]])

    return x + y


flow_result1 = my_sync_flow(3, 4)
assert_type(flow_result1, int)

flow_result2 = asyncio.run(my_async_flow(3, 4))
assert_type(flow_result2, int)

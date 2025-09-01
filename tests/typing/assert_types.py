"""Tests for various type assertions fro waypoint tasks and flows."""

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import ParamSpec, TypeVar

from typing_extensions import assert_type

from waypoint.flows import flow
from waypoint.futures import TaskFuture
from waypoint.tasks import submit_task
from waypoint.tasks import task

P = ParamSpec("P")
R = TypeVar("R")


@task
def sync_task(x: int, y: int) -> int:
    return x + y


@task
def sync_generator_task(x: int, y: int) -> Iterator[int]:
    yield x + y


@task
async def async_task(x: int, y: int) -> int:
    return x * y


@task
async def async_generator_task(x: int, y: int) -> AsyncIterator[int]:
    yield x * y


@flow
def my_sync_flow(x: int, y: int) -> int:
    result = sync_task(x, y)
    assert_type(result, int)

    for value in sync_generator_task(x, y):
        assert_type(value, int)
        result += value

    futures = [submit_task(sync_task, i, i + 1) for i in range(5)]
    assert_type(futures, list[TaskFuture[int]])

    future = submit_task(sync_generator_task, x, y)
    assert_type(future, TaskFuture[Iterator[int]])

    return x + y


@flow
async def my_async_flow(x: int, y: int) -> int:
    result = sync_task(x, y)
    assert_type(result, int)

    for value in sync_generator_task(x, y):
        assert_type(value, int)
        result += value

    result = await async_task(x, y)
    assert_type(result, int)

    async for value in async_generator_task(x, y):
        assert_type(value, int)
        result += value

    futures = [submit_task(sync_task, i, i + 1) for i in range(5)]
    assert_type(futures, list[TaskFuture[int]])

    future = submit_task(sync_generator_task, x, y)
    assert_type(future, TaskFuture[Iterator[int]])

    return x + y


flow_result1 = my_sync_flow(3, 4)
assert_type(flow_result1, int)

flow_result2 = asyncio.run(my_async_flow(3, 4))
assert_type(flow_result2, int)

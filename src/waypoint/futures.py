"""
Task-based futures for use with Waypoint task runners.

This module intends to provide an interface similar to `concurrent.futures.Future`.
Note that the classes defined here are not subclasses of `concurrent.futures.Future`
and cannot be used with other libraries that expect a `Future` object.
"""

import asyncio
import concurrent.futures
import inspect
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, TypeVar, cast, overload

from typing_extensions import override

R = TypeVar("R")
_WAIT_STATES = Literal["FIRST_COMPLETED", "ALL_COMPLETED", "FIRST_EXCEPTION"]


class TaskFuture(Generic[R]):
    """
    Represents the result of a Waypoint task.

    This class acts as a proxy to a `concurrent.futures.Future` object, allowing the
    user to check the status of the task and retrieve the result. Note that this class
    attempts to mimic the interface of `concurrent.futures.Future` as closely as
    possible, but it is not a subclass of that class and cannot be used with other
    libraries that expect a `Future` object.

    Args:
        raw_future (concurrent.futures.Future): Raw future object to be wrapped.
    """

    _raw_future: concurrent.futures.Future[Any]

    def __init__(self, raw_future: concurrent.futures.Future[R]) -> None:
        self._raw_future = raw_future

    def result(self, timeout: float | None = None) -> R:
        """
        Return the result of the task.

        Args:
            timeout (float, optional):
                Maximum number of seconds to wait. If None, there is no limit on the wait time.

        Returns:
            The result of the task.
        """
        return cast(R, self._raw_future.result(timeout))

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """
        Return the exception raised by the task.

        Args:
            timeout (float, optional):
                Maximum number of seconds to wait. If None, there is no limit on the wait time.

        Returns:
            The exception raised by the task, or None if it completed successfully.
        """
        return self._raw_future.exception(timeout)

    def cancel(self) -> bool:
        """
        Attempt to cancel the task.

        Returns:
            True if the task was successfully cancelled, False otherwise.
        """
        return self._raw_future.cancel()

    def cancelled(self) -> bool:
        """Return True if the task was successfully cancelled."""
        return self._raw_future.cancelled()

    def running(self) -> bool:
        """Return True if the task is currently running."""
        return self._raw_future.running()

    def done(self) -> bool:
        """Return True if the task completed successfully or was cancelled."""
        return self._raw_future.done()

    def add_done_callback(self, fn: Callable[[concurrent.futures.Future], None]) -> None:
        """
        Attaches a callable that will be called when the task is completed.

        The callable will be called with this `TaskFuture` object as its only argument.

        Args:
            fn (Callable[[concurrent.futures.Future], None]):
                Callable to be called when the task is completed.
        """
        self._raw_future.add_done_callback(fn)

    def __hash__(self) -> int:
        return hash(self._raw_future)


class DelayedTaskFuture(TaskFuture[R]):
    """
    Represents the result of a Waypoint task - internally delayed.

    This future does not execute the task until `result()` is called. This is useful
    for deferring execution of tasks until their results are actually needed.

    Args:
        func (Callable[[], R]):
            Parameter-less function to be executed when the result is requested.
    """

    def __init__(self, func: Callable[[], R]) -> None:
        self._func = func
        self._executed = False
        self._cancelled = False
        self._raw_future = concurrent.futures.Future[R]()

    def _ensure_executed(self) -> None:
        """Execute the delayed function if not already executed."""
        if not self._executed and not self._cancelled:
            try:
                if inspect.iscoroutinefunction(self._func):
                    result = asyncio.run(self._func())
                else:
                    result = self._func()
                self._raw_future.set_result(result)
            except Exception as exception:
                self._raw_future.set_exception(exception)
            finally:
                self._executed = True

    @override
    def result(self, timeout: float | None = None) -> R:
        self._ensure_executed()
        return cast(R, self._raw_future.result(timeout))

    @override
    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._ensure_executed()
        return self._raw_future.exception(timeout)

    @override
    def cancel(self) -> bool:
        """Cancel the delayed task if not yet executed."""
        if not self._executed and not self._cancelled:
            self._cancelled = True
            self._raw_future.cancel()
            return True
        return False

    @override
    def cancelled(self) -> bool:
        """Return True if the task was successfully cancelled."""
        return self._cancelled or self._raw_future.cancelled()

    @override
    def done(self) -> bool:
        return self._executed

    @override
    def running(self) -> bool:
        return False


class SerializedTaskFuture(TaskFuture[R]):
    """
    Represents the result of a Waypoint task - internally serialized.

    Args:
        raw_future (concurrent.futures.Future): Raw future object to be wrapped.
    """

    def __init__(self, raw_future: concurrent.futures.Future[bytes]) -> None:
        self._raw_future = raw_future

    @override
    def result(self, timeout: float | None = None) -> R:
        import cloudpickle

        payload = self._raw_future.result(timeout)
        result = cloudpickle.loads(payload)
        return cast(R, result)


@overload
def as_completed(
    futures: Iterable[TaskFuture[R]], timeout: float | None = None
) -> Iterator[TaskFuture[R]]: ...


@overload
def as_completed(
    futures: Iterable[TaskFuture[Any]], timeout: float | None = None
) -> Iterator[TaskFuture[Any]]: ...


def as_completed(
    futures: Iterable[TaskFuture[Any]], timeout: float | None = None
) -> Iterator[TaskFuture[Any]]:
    """
    An iterator over the given `TaskFuture` that yields each as it completes.

    Args:
        futures (Iterable[TaskFuture[Any]]):
            An iterable of futures. Note that futures need not come from the same executor.
            Note that if any futures are duplicated, they will be treated as a single future.
        timeout (float, optional):
            Maximum number of seconds to wait. If None, there is no limit on the wait time.
            Does not apply to DelayedTaskFuture instances, which are executed immediately.

    Returns:
        An iterator that yields the given Futures as they complete (finished or
        cancelled). If any given Futures are duplicated, they will be returned once.
    """
    futures_list = list(futures)

    regular_futures, delayed_futures = set(), set()
    for future in futures_list:
        if future in delayed_futures:
            continue
        if isinstance(future, DelayedTaskFuture):
            _ = future.result()
            delayed_futures.add(future)
            yield future
        else:
            regular_futures.add(future)

    # Handle regular futures with concurrent.futures.as_completed
    if regular_futures:
        future_lookup = {future._raw_future: future for future in regular_futures}
        for raw_future in concurrent.futures.as_completed(future_lookup.keys(), timeout=timeout):
            yield future_lookup[raw_future]


@overload
def wait(
    futures: Iterable[TaskFuture[R]],
    *,
    timeout: float | None = None,
    return_when: _WAIT_STATES = "ALL_COMPLETED",
) -> tuple[set[TaskFuture[R]], set[TaskFuture[R]]]: ...


@overload
def wait(
    futures: Iterable[TaskFuture[Any]],
    *,
    timeout: float | None = None,
    return_when: _WAIT_STATES = "ALL_COMPLETED",
) -> tuple[set[TaskFuture[Any]], set[TaskFuture[Any]]]: ...


def wait(
    futures: Iterable[TaskFuture[Any]],
    *,
    timeout: float | None = None,
    return_when: _WAIT_STATES = "ALL_COMPLETED",
) -> tuple[set[TaskFuture[Any]], set[TaskFuture[Any]]]:
    """
    Wait for the `TaskFuture` given by fs to complete.

    Args:
        futures (Iterable[TaskFuture[Any]]):
            An iterable of futures. Note that futures need not come from the same executor.
            Note that if any futures are duplicated, they will be treated as a single future.
        timeout (float, optional):
            Maximum number of seconds to wait. If None, there is no limit on the wait time.
        return_when (Literal["FIRST_COMPLETED", "ALL_COMPLETED", "FIRST_EXCEPTION"]):
            Indicates when this function should return. The options are
            - FIRST_COMPLETED: Return when any future finishes or is cancelled.
            - FIRST_EXCEPTION: Return when any future finishes by raising an exception.
            - ALL_COMPLETED: Return when all futures finish or are cancelled.

    Returns:
        A named 2-tuple of sets. The first set, named 'done', contains the futures that
        completed (is finished or cancelled) before the wait completed. The second set,
        named 'not_done', contains uncompleted futures. Duplicate futures are removed
        and will be returned only once.
    """
    futures_list = list(futures)

    regular_futures, delayed_futures = set(), set()
    for future in futures_list:
        if future in delayed_futures:
            continue
        if isinstance(future, DelayedTaskFuture):
            _ = future.result()
            delayed_futures.add(future)
        else:
            regular_futures.add(future)

    # All delayed futures are now "done"
    done_futures: set[TaskFuture[Any]] = set(delayed_futures)
    not_done_futures: set[TaskFuture[Any]] = set()

    # Handle regular futures with concurrent.futures.wait if any exist
    if regular_futures:
        raw_futures: dict[concurrent.futures.Future, TaskFuture[Any]] = {
            future._raw_future: future for future in regular_futures
        }
        done_raw, not_done_raw = concurrent.futures.wait(
            raw_futures.keys(), timeout=timeout, return_when=return_when
        )

        done_futures.update(raw_futures[raw_future] for raw_future in done_raw)
        not_done_futures.update(raw_futures[raw_future] for raw_future in not_done_raw)

    return (done_futures, not_done_futures)

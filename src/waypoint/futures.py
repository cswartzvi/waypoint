"""
Task-based futures for use with Waypoint task runners.

This module intends to provide an interface similar to `concurrent.futures.Future`.
Note that the classes defined here are not subclasses of `concurrent.futures.Future`
and cannot be used with other libraries that expect a `Future` object.
"""

import concurrent.futures
from typing import Any, Generic, Iterable, Iterator, Literal, TypeVar, cast, overload

from typing_extensions import override

R = TypeVar("R")
_WAIT_STATES = Literal["FIRST_COMPLETED", "ALL_COMPLETED", "FIRST_EXCEPTION"]


class TaskFuture(Generic[R]):
    """
    Represents the future result of a Waypoint task.

    This class acts as a proxy to a `concurrent.futures.Future` object, allowing the
    user to check the status of the task and retrieve the result. Note that this class
    attempts to mimic the interface of `concurrent.futures.Future` as closely as
    possible, but it is not a subclass of that class and cannot be used with other
    libraries that expect a `Future` object.

    Args:
        raw_future: The raw future object.
    """

    _raw_future: concurrent.futures.Future[Any]

    def __init__(self, raw_future: concurrent.futures.Future[R]) -> None:
        self._raw_future = raw_future

    def result(self, timeout: float | None = None) -> R:
        """
        Return the result of the task.

        Args:
            timeout: The maximum number of seconds to wait. If None, then there is no
                limit on the wait time.

        Raises:
            concurrent.futures.TimeoutError: If the future didn't finish executing
                before the given timeout.
            concurrent.futures.CancelledError: If the future was cancelled.

        Returns:
            The result of the task.
        """
        return cast(R, self._raw_future.result(timeout))

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """
        Return the exception raised by the task.

        Args:
            timeout: The maximum number of seconds to wait. If None, then there is no
                limit on the wait time.

        Raises:
            concurrent.futures.TimeoutError: If the future didn't finish executing
                before the given timeout.
            concurrent.futures.CancelledError: If the future was cancelled.

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

    def __hash__(self) -> int:
        return hash(self._raw_future)


class SerializedTaskFuture(TaskFuture[R]):
    """
    Represents the future result of a Waypoint task that returns a serialized data.

    Args:
        raw_future: The raw future object.
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
        futures:
            An iterable of futures. Note that futures need not come from the same executor.
        timeout: The maximum number of seconds to wait. If None, then there is no limit on the wait
            time.

    Returns:
        An iterator that yields the given Futures as they complete (finished or
        cancelled). If any given Futures are duplicated, they will be returned once.
    """
    future_lookup = {future._raw_future: future for future in futures}

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
        futures: An iterable of futures. Note that futures need not come from the
            same executor.
        timeout: The maximum number of seconds to wait. If None, then there is no
            limit on the wait time.
        return_when: Indicates when this function should return. The options are:
            FIRST_COMPLETED: Return when any future finishes or is cancelled.
            FIRST_EXCEPTION: Return when any future finishes by raising an exception.
            ALL_COMPLETED: Return when all futures finish or are cancelled.

    Returns:
        A named 2-tuple of sets. The first set, named 'done', contains the futures that
        completed (is finished or cancelled) before the wait completed. The second set,
        named 'not_done', contains uncompleted futures. Duplicate futures are removed
        and will be returned only once.
    """
    raw_futures = {future._raw_future: future for future in futures}
    done, not_done = concurrent.futures.wait(
        raw_futures.keys(), timeout=timeout, return_when=return_when
    )
    return (
        {raw_futures[future] for future in done},
        {raw_futures[future] for future in not_done},
    )

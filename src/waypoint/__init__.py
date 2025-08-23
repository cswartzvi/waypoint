"""Waypoint: A lightweight, local-first workflow orchestration framework."""

from waypoint import futures
from waypoint.flows import flow
from waypoint.hooks import initialize_hooks as _initialize_hooks
from waypoint.tasks import submit_task
from waypoint.tasks import task

__version__ = "0.0.1"

# NOTE: Must call first in order to establish an initial hook context.
_initialize_hooks()

__all__ = ["flow", "task", "submit_task", "futures"]

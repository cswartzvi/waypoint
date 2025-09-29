"""Waypoint: A lightweight, local-first workflow orchestration framework."""

from waypoint import futures
from waypoint.assets import asset_mapper
from waypoint.flows import flow
from waypoint.hooks import initialize_hooks as _initialize_hooks
from waypoint.tasks import map_task
from waypoint.tasks import submit_task
from waypoint.tasks import task

__version__ = "0.2.0"

# NOTE: Must call first in order to establish an initial hook context.
_initialize_hooks()

__all__ = [
    "asset_mapper",
    "flow",
    "futures",
    "map_task",
    "submit_task",
    "task",
]

"""Utility functions to manage the project-wide hook configuration."""

import logging
from inspect import isclass
from typing import Any

from pluggy import PluginManager

from waypoint.context import HooksContext

from .markers import HOOK_NAMESPACE
from .specs import FlowSpec

logger = logging.getLogger(__name__)

_PLUGIN_HOOKS = "waypoint.hooks"  # entry-point to load hooks from for installed plugins


def initialize_hooks() -> None:
    """Initializes hooks for the waypoint library."""
    if hook_context := HooksContext.get():
        hook_context.__exit__()  # Resets "global" context

    manager = _create_hook_manager()

    hook_context = HooksContext(manager=manager)
    hook_context.__enter__()  # Sets "global" context


def get_hook_manager() -> PluginManager:
    """Returns initialized hook plugin manager or raises an exception."""
    hook_context = HooksContext.get()
    if not hook_context:
        raise RuntimeError("Attempted access of Hook plugin manager without initialization.")
    hook_manager = hook_context.manager
    assert hook_manager is not None
    return hook_manager


def register_hooks(*hooks: Any) -> None:
    """Register specified Waypoint pluggy hooks."""
    hook_manager = get_hook_manager()
    for hooks_collection in hooks:
        if not hook_manager.is_registered(hooks_collection):
            if isclass(hooks_collection):
                raise TypeError(
                    "Waypoint expects hooks to be registered as instances. "
                    "Have you forgotten the `()` when registering a hook class?"
                )
            hook_manager.register(hooks_collection)


def register_hooks_entry_points() -> None:
    """Register Waypoint pluggy hooks from Python package entrypoints."""
    hook_manager = get_hook_manager()
    hook_manager.load_setuptools_entrypoints(_PLUGIN_HOOKS)  # Despite name setuptools not required


def _create_hook_manager() -> PluginManager:
    """Create a new PluginManager instance and register Waypoint's hook specs."""
    manager = PluginManager(HOOK_NAMESPACE)
    manager.trace.root.setwriter(
        logger.debug if logger.getEffectiveLevel() == logging.DEBUG else None
    )
    manager.enable_tracing()
    manager.add_hookspecs(FlowSpec)
    return manager

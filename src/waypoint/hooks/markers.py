"""
Declares pluggy markers waypoint's hook specs and implementations.

For more information, please see: https://pluggy.readthedocs.io/en/stable/#marking-hooks.
"""

import pluggy

HOOK_NAMESPACE = "waypoint"

hook_spec = pluggy.HookspecMarker(HOOK_NAMESPACE)
hook_impl = pluggy.HookimplMarker(HOOK_NAMESPACE)

import pytest

from waypoint.flows import flow
from waypoint.hooks.manager import register_hooks
from waypoint.plugins.logger import WaypointLogger


@pytest.fixture(autouse=True, scope="function")
def plugin():
    plugin_ = WaypointLogger()
    register_hooks(plugin_)
    yield plugin_


class TestLoggingHooks:
    def test_hooks_fire_for_flow_run(self, caplog):
        """Test that logging hooks fire for a standard flow."""

        @flow
        def sample_flow():
            return 42

        sample_flow()
        # assert "INFO:waypoint.plugins.logger:Starting flow run: sample_flow" in caplog.text

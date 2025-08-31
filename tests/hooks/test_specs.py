import pytest

from waypoint.flows import flow
from waypoint.flows import flow_session
from waypoint.hooks.manager import register_hooks
from waypoint.tasks import submit_task
from waypoint.tasks import task

from .echo_plugin import EchoPlugin


@pytest.fixture(scope="function")
def plugin():
    plugin_ = EchoPlugin()
    register_hooks(plugin_)
    yield plugin_


class TestFlowHooks:
    def test_hooks_fire_for_flow_run(self, plugin):
        """Test that flow hooks fire for a standard flow."""

        @flow
        def sample_flow():
            return 42

        with flow_session():
            result = sample_flow()
            assert result == 42

        assert len(plugin.lines) == 2
        assert "before_flow_run called - sample_flow" == plugin.lines[0]
        assert "after_flow_run called - sample_flow" == plugin.lines[1]

    def test_hooks_fire_for_generator_flow(self, plugin):
        """Test that flow hooks fire for a generator flow."""

        @flow
        def sample_flow():
            for i in range(3):
                yield i

        with flow_session():
            results = list(sample_flow())
            assert results == [0, 1, 2]

        assert len(plugin.lines) == 5
        assert "before_flow_run called - sample_flow" == plugin.lines[0]
        assert "after_flow_iteration called - sample_flow - 0" == plugin.lines[1]
        assert "after_flow_iteration called - sample_flow - 1" == plugin.lines[2]
        assert "after_flow_iteration called - sample_flow - 2" == plugin.lines[3]
        assert "after_flow_run called - sample_flow" == plugin.lines[4]


class TestTaskHooks:
    def test_hooks_fire_for_task_run(self, plugin):
        """Test that task hooks fire for a standard task."""

        @task(name="sample_task")
        def sample_task():
            return 99

        with flow_session(name="test_session"):
            result = sample_task()
            assert result == 99

        assert len(plugin.lines) == 2
        assert "before_task_run called - test_session.sample_task-1" == plugin.lines[0]
        assert "after_task_run called - test_session.sample_task-1" == plugin.lines[1]

    def test_hooks_fire_for_generator_task(self, plugin):
        """Test that task hooks fire for a generator task."""

        @task(name="sample_task")
        def sample_task():
            for i in range(2):
                yield i * 10

        with flow_session(name="test_session"):
            results = list(sample_task())
            assert results == [0, 10]

        assert len(plugin.lines) == 4
        assert "before_task_run called - test_session.sample_task-1" == plugin.lines[0]
        assert "after_task_iteration called - test_session.sample_task-1 - 0" == plugin.lines[1]
        assert "after_task_iteration called - test_session.sample_task-1 - 1" == plugin.lines[2]
        assert "after_task_run called - test_session.sample_task-1" == plugin.lines[3]

    def test_hooks_fire_for_task_submission_with_sequential(self, plugin):
        """
        Test that task submission hooks fire when using the sequential runner.

        Note: The sequential runner should not, by design, fire the task submission hook.
        """

        @task(name="sample_task")
        def sample_task():
            return "submitted"

        with flow_session(name="test_session", task_runner="sequential"):
            future = submit_task(sample_task)
            assert future.result() == "submitted"

        assert len(plugin.lines) == 2  # No submission hooks should fire
        assert "before_task_run called - test_session.sample_task-1" == plugin.lines[0]
        assert "after_task_run called - test_session.sample_task-1" == plugin.lines[1]

    def test_hooks_fire_for_task_submission_with_threaded(self, plugin):
        """Test that task submission hooks fire when using the threaded runner."""

        @task(name="sample_task")
        def sample_task():
            return "submitted"

        with flow_session(name="test_session", task_runner="threading"):
            future = submit_task(sample_task)
            assert future.result() == "submitted"

        assert len(plugin.lines) == 3
        assert "before_task_submit called - test_session.sample_task-1" == plugin.lines[0]
        assert "before_task_run called - test_session.sample_task-1" == plugin.lines[1]
        assert "after_task_run called - test_session.sample_task-1" == plugin.lines[2]


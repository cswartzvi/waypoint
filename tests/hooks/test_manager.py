from unittest.mock import Mock
from unittest.mock import patch

import pytest

from waypoint.hooks.manager import _PLUGIN_HOOKS
from waypoint.hooks.manager import _clear_hooks
from waypoint.hooks.manager import _create_hook_manager
from waypoint.hooks.manager import get_hook_manager
from waypoint.hooks.manager import initialize_hooks
from waypoint.hooks.manager import register_hooks
from waypoint.hooks.manager import register_hooks_entry_points
from waypoint.hooks.manager import try_run_hook


class TestInitializeHooks:
    """Test suite for initialize_hooks function."""

    @patch("waypoint.hooks.manager.HooksContext")
    @patch("waypoint.hooks.manager._create_hook_manager")
    def test_initialize_hooks_creates_new_context(self, mock_create_manager, mock_hooks_context):
        """Test that initialize_hooks creates and enters a new HooksContext."""
        # Arrange
        mock_manager = Mock()
        mock_create_manager.return_value = mock_manager
        mock_existing_context = Mock()
        mock_existing_context.__exit__ = Mock()  # Configure the __exit__ method
        mock_hooks_context.get.return_value = mock_existing_context
        mock_new_context = Mock()
        mock_new_context.__enter__ = Mock()  # Configure the __enter__ method
        mock_hooks_context.return_value = mock_new_context

        # Act
        initialize_hooks()

        # Assert
        mock_existing_context.__exit__.assert_called_once()
        mock_hooks_context.assert_called_once_with(manager=mock_manager)
        mock_new_context.__enter__.assert_called_once()

    @patch("waypoint.hooks.manager.HooksContext")
    @patch("waypoint.hooks.manager._create_hook_manager")
    def test_initialize_hooks_no_existing_context(self, mock_create_manager, mock_hooks_context):
        """Test that initialize_hooks works when no existing context exists."""
        # Arrange
        mock_manager = Mock()
        mock_create_manager.return_value = mock_manager
        mock_hooks_context.get.return_value = None  # No existing context
        mock_new_context = Mock()
        mock_new_context.__enter__ = Mock()  # Configure the __enter__ method
        mock_hooks_context.return_value = mock_new_context

        # Act
        initialize_hooks()

        # Assert
        mock_hooks_context.assert_called_once_with(manager=mock_manager)
        mock_new_context.__enter__.assert_called_once()


class TestRegisterHooks:
    """Test suite for register_hooks function."""

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_register_hooks_registers_instance(self, mock_get_hook_manager):
        """Test that register_hooks successfully registers hook instances."""
        # Arrange
        mock_hook_manager = Mock()
        mock_hook_manager.is_registered.return_value = False
        mock_get_hook_manager.return_value = mock_hook_manager
        mock_hook_instance = Mock()

        # Act
        register_hooks(mock_hook_instance)

        # Assert
        mock_hook_manager.register.assert_called_once_with(mock_hook_instance)

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_register_hooks_raises_error_for_class(self, mock_get_hook_manager):
        """Test that register_hooks raises TypeError when passed a class instead of instance."""
        # Arrange
        mock_hook_manager = Mock()
        mock_hook_manager.is_registered.return_value = False
        mock_get_hook_manager.return_value = mock_hook_manager

        class MockHookClass:
            pass

        # Act & Assert
        with pytest.raises(TypeError, match="Waypoint expects hooks to be registered as instances"):
            register_hooks(MockHookClass)


class TestClearHooks:
    """Test suite for clear_hooks function."""

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_clear_hooks_unregisters_all_plugins(self, mock_get_hook_manager):
        """Test that _clear_hooks unregisters all plugins from hook manager."""
        # Arrange
        mock_hook_manager = Mock()
        mock_get_hook_manager.return_value = mock_hook_manager

        # Act
        _clear_hooks()

        # Assert
        mock_hook_manager.unregister.assert_called_once_with(name=None)


class TestTryRunHook:
    """Test suite for try_run_hook function."""

    def test_try_run_hook_calls_existing_hook(self):
        """Test that try_run_hook calls the hook when it exists."""
        # Arrange
        mock_hook = Mock()
        mock_manager = Mock()
        mock_manager.hook.test_hook = mock_hook

        # Act
        try_run_hook(manager=mock_manager, hook_name="test_hook", arg1="value1", arg2="value2")

        # Assert
        mock_hook.assert_called_once_with(arg1="value1", arg2="value2")

    def test_try_run_hook_ignores_missing_hook(self):
        """Test that try_run_hook does nothing when hook doesn't exist."""
        # Arrange
        mock_manager = Mock()
        mock_manager.hook.missing_hook = None

        # Act (should not raise)
        try_run_hook(manager=mock_manager, hook_name="missing_hook", arg1="value1")

        # Assert - no exception raised is the test


class TestRegisterHooksEntryPoints:
    """Test suite for register_hooks_entry_points function."""

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_register_hooks_entry_points_calls_load_setuptools_entrypoints(
        self, mock_get_hook_manager
    ):
        """Test register_hooks_entry_points calls load_setuptools_entrypoints."""
        # Arrange
        mock_hook_manager = Mock()
        mock_get_hook_manager.return_value = mock_hook_manager

        # Act
        register_hooks_entry_points()

        # Assert
        mock_get_hook_manager.assert_called_once()
        mock_hook_manager.load_setuptools_entrypoints.assert_called_once_with(_PLUGIN_HOOKS)

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_register_hooks_entry_points_handles_hook_manager_exception(
        self, mock_get_hook_manager
    ):
        """Test that exceptions from hook manager are propagated."""
        # Arrange
        mock_hook_manager = Mock()
        mock_hook_manager.load_setuptools_entrypoints.side_effect = Exception("Entry point error")
        mock_get_hook_manager.return_value = mock_hook_manager

        # Act & Assert
        with pytest.raises(Exception, match="Entry point error"):
            register_hooks_entry_points()


class TestGetHookManager:
    """Test suite for get_hook_manager function."""

    @patch("waypoint.hooks.manager.HooksContext")
    def test_get_hook_manager_raises_runtime_error_when_no_context(self, mock_hooks_context):
        """Test that get_hook_manager raises RuntimeError when no context exists."""
        # Arrange
        mock_hooks_context.get.return_value = None

        # Act & Assert
        with pytest.raises(RuntimeError, match="Attempted access of Hook plugin manager"):
            get_hook_manager()

    @patch("waypoint.hooks.manager.HooksContext")
    def test_get_hook_manager_returns_manager_when_context_exists(self, mock_hooks_context):
        """Test that get_hook_manager returns the hook manager when context exists."""
        # Arrange
        mock_manager = Mock()
        mock_context = Mock()
        mock_context.manager = mock_manager
        mock_hooks_context.get.return_value = mock_context

        # Act
        result = get_hook_manager()

        # Assert
        assert result == mock_manager


class TestRegisterHooksAlreadyRegistered:
    """Test suite for register_hooks when hooks are already registered."""

    @patch("waypoint.hooks.manager.get_hook_manager")
    def test_register_hooks_skips_already_registered(self, mock_get_hook_manager):
        """Test that register_hooks skips hooks that are already registered."""
        # Arrange
        mock_hook_manager = Mock()
        mock_hook_manager.is_registered.return_value = True  # Already registered
        mock_get_hook_manager.return_value = mock_hook_manager
        mock_hook_instance = Mock()

        # Act
        register_hooks(mock_hook_instance)

        # Assert
        mock_hook_manager.register.assert_not_called()


class TestCreateHookManager:
    """Test suite for _create_hook_manager function."""

    @patch("waypoint.hooks.manager.PluginManager")
    @patch("waypoint.hooks.manager.logger")
    def test_create_hook_manager_creates_manager_with_specs(self, mock_logger, mock_plugin_manager):
        """Test that _create_hook_manager creates a PluginManager with proper configuration."""
        # Arrange
        mock_manager_instance = Mock()
        mock_plugin_manager.return_value = mock_manager_instance
        mock_logger.getEffectiveLevel.return_value = 10  # DEBUG level

        # Act
        result = _create_hook_manager()

        # Assert
        mock_plugin_manager.assert_called_once_with("waypoint")
        mock_manager_instance.enable_tracing.assert_called_once()
        mock_manager_instance.add_hookspecs.assert_called_once()
        assert result == mock_manager_instance

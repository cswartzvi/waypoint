import json

import pytest

# from waypoint import flow, task  # For future integration tests
from waypoint.assets import asset_mapper
from waypoint.exceptions import AssetKeyFormatError
from waypoint.exceptions import AssetMapperError
from waypoint.exceptions import AssetNotFoundError
from waypoint.flows import flow
from waypoint.stores.memory import InMemoryAssetStore
from waypoint.tasks import task


@pytest.fixture
def sample_data():
    return {"experiment_id": "exp_001", "model": "transformer", "accuracy": 0.95}


def test_basic_functionality():
    store = InMemoryAssetStore()
    bound_mapper = asset_mapper("test/{name}.pkl")
    data = {"test": "value"}
    key = bound_mapper.save(data, parameters={"name": "basic"}, store=store)
    loaded = bound_mapper.load(parameters={"name": "basic"}, store=store)
    assert loaded == data
    assert key == "test/basic.pkl"


class TestAssetMapper:
    """Test AssetMapper operations with explicit keys and stores."""

    def test_save_and_load_with_explicit_key(self, store, sample_data):
        """Test save/load operations using explicit keys."""
        mapper = asset_mapper()
        explicit_key = "manual/explicit_save.pkl"
        returned_key = mapper.save(explicit_key, sample_data, store=store)
        assert returned_key == explicit_key
        loaded_data = mapper.load(explicit_key, store=store)
        assert loaded_data == sample_data

    def test_key_validation_prevents_attacks(self, store):
        """Test that key validation prevents directory traversal attacks."""
        mapper = asset_mapper()

        # Test path traversal prevention
        with pytest.raises(AssetMapperError, match="path traversal"):
            mapper.save("../../../etc/passwd", {"data": "bad"}, store=store)

        # Test absolute path prevention (Unix style)
        with pytest.raises(AssetMapperError, match="absolute path"):
            mapper.save("/etc/passwd", {"data": "bad"}, store=store)

        # Test absolute path prevention (Windows style - covers line 128)
        with pytest.raises(AssetMapperError, match="absolute path"):
            mapper.save("C:\\windows\\system32", {"data": "bad"}, store=store)

    def test_custom_serialization(self, store):
        """Test custom serialization and deserialization functions."""

        def json_serializer(data: dict) -> bytes:
            return json.dumps(data).encode()

        def json_deserializer(data: bytes) -> dict:
            return json.loads(data.decode())

        mapper = asset_mapper(serializer=json_serializer, deserializer=json_deserializer)

        test_data = {"name": "test", "value": 42}
        key = "custom/test.json"

        mapper.save(key, test_data, store=store)
        loaded = mapper.load(key, store=store)
        assert loaded == test_data

    def test_error_handling_behavior(self, store):
        """Test error conditions in asset mapper operations."""
        mapper = asset_mapper()

        # Test loading non-existent asset without default
        with pytest.raises(AssetNotFoundError):
            mapper.load("nonexistent_key.pkl", store=store)

        # Test path traversal prevention
        with pytest.raises(AssetMapperError, match="path traversal"):
            mapper.save("../forbidden.pkl", {"data": "bad"}, store=store)

        # Test absolute path prevention
        with pytest.raises(AssetMapperError, match="absolute path"):
            mapper.save("/forbidden.pkl", {"data": "bad"}, store=store)

        # Test Windows-style absolute path prevention (covers line 128)
        with pytest.raises(AssetMapperError, match="absolute path"):
            mapper.save("C:/forbidden.pkl", {"data": "bad"}, store=store)

        # Test empty key validation
        with pytest.raises(AssetMapperError, match="non-empty string"):
            mapper.save("", {"data": "bad"}, store=store)


class TestBoundAssetMapper:
    """Test BoundAssetMapper operations with key templates and context."""

    def test_compute_key_with_parameters(self):
        """Test key computation with manual parameters."""
        mapper = asset_mapper("data/{experiment_id}/{model}.pkl")
        params = {"experiment_id": "exp_001", "model": "transformer"}
        key = mapper.compute_key(parameters=params)
        assert key == "data/exp_001/transformer.pkl"

    def test_missing_parameter_error(self):
        """Test that missing parameters raise appropriate errors."""
        mapper = asset_mapper("data/{experiment_id}/{missing_param}.pkl")

        # Should raise error when required parameter is missing
        with pytest.raises(AssetKeyFormatError, match="Missing parameter 'missing_param'"):
            mapper.compute_key(parameters={"experiment_id": "exp_001"})

    def test_template_based_operations(self, store, sample_data):
        """Test save/load/exists operations with template-based keys."""
        mapper = asset_mapper("test/{name}.pkl")
        params = {"name": "exists_test"}

        # Initially should not exist
        assert not mapper.exists(parameters=params, store=store)

        # Save data and check it exists
        key = mapper.save(sample_data, parameters=params, store=store)
        assert mapper.exists(parameters=params, store=store)
        assert key == "test/exists_test.pkl"

        # Load data back
        loaded = mapper.load(parameters=params, store=store)
        assert loaded == sample_data

    def test_default_value_on_missing_asset(self, store):
        """Test loading with a default value when asset is missing."""
        mapper = asset_mapper("defaults/{name}.pkl")
        params = {"name": "nonexistent"}

        default_data = {"default": True}
        loaded = mapper.load(parameters=params, store=store, default=default_data)
        assert loaded == default_data

        # Verify the nonexistent key doesn't exist in store (since we got default)
        key = mapper.compute_key(parameters={"name": "nonexistent"})
        assert not store.exists(key)

    def test_callable_key_function(self, store):
        """Test asset mapper with callable key function (covers lines 249-251)."""

        def dynamic_key(flow_ctx, task_ctx, params):
            return f"computed/{params.get('name', 'unknown')}"

        mapper = asset_mapper(dynamic_key)

        # Test compute_key with callable
        key = mapper.compute_key(parameters={"name": "test"})
        assert key == "computed/test"

        # Test save and load with callable key
        data = {"test": "data"}
        saved_key = mapper.save(data, parameters={"name": "callable_test"}, store=store)
        assert saved_key == "computed/callable_test"

        loaded = mapper.load(parameters={"name": "callable_test"}, store=store)
        assert loaded == data

    def test_invalid_computed_key_validation(self, store):
        """Test validation of computed keys (covers line 265)."""

        def bad_key_func(flow_ctx, task_ctx, params):
            return ""  # Empty string should cause validation error

        mapper = asset_mapper(bad_key_func)

        # Should raise error for empty computed key
        with pytest.raises(AssetMapperError, match="non-empty string"):
            mapper.compute_key(parameters={"name": "test"})

    def test_format_kwargs_with_context(self, store):
        """Test _format_kwargs method with flow and task context (covers lines 310-315)."""
        mapper = asset_mapper("context/{experiment_id}.pkl")

        # Test manual parameters (baseline case)
        key = mapper.compute_key(parameters={"experiment_id": "manual"})
        assert key == "context/manual.pkl"

        # Test with actual flow context to exercise the flow_run branches
        @flow(store=store)
        def test_flow(experiment_id: str):
            # Test mapper inside flow context to trigger the flow_run context path
            inner_mapper = asset_mapper("flow_context/{experiment_id}.pkl")
            return inner_mapper.compute_key(parameters={"experiment_id": experiment_id})

        # This will exercise the flow_run_context branches in _format_kwargs
        flow_key = test_flow("flow_test")
        assert "flow_test" in flow_key

        # Test with task context to exercise task_run branches
        @task
        def test_task(experiment_id: str):
            # Test mapper inside task context to trigger the task_run context path
            inner_mapper = asset_mapper("task_context/{experiment_id}.pkl")
            return inner_mapper.compute_key(parameters={"experiment_id": experiment_id})

        @flow(store=store)
        def test_flow_with_task(experiment_id: str):
            return test_task(experiment_id)

        # This will exercise the task_run_context branches in _format_kwargs
        task_key = test_flow_with_task("task_test")
        assert "task_test" in task_key


class TestIntegrationWithFlowsAndTasks:
    """Integration tests using AssetMappers with flows and tasks."""

    def test_sync_flow_level_mapper(self, store):
        """Test using an AssetMapper at the flow level."""
        mapper = asset_mapper("flow_level/{name}.pkl")

        @flow(store=store, mapper=mapper)
        def flow_with_task(name: str):
            return simple_task(name)

        @task
        def simple_task(name: str):
            return {"name": name, "result": "task_result"}

        # Execute flow
        result = flow_with_task("test_flow")
        assert result == {"name": "test_flow", "result": "task_result"}

        # Verify asset saved in store
        key = mapper.compute_key(parameters={"name": "test_flow"})
        assert store.exists(key)

        # Load asset directly from store
        loaded = mapper.load(store=store, parameters={"name": "test_flow"})
        assert loaded == result

    @pytest.mark.asyncio
    async def test_async_flow_level_mapper(self, store):
        """Test using an AssetMapper at the flow level in an async flow."""
        mapper = asset_mapper("async_flow_level/{name}.pkl")

        @flow(store=store, mapper=mapper)
        async def async_flow_with_task(name: str):
            return await simple_task(name)

        @task
        async def simple_task(name: str):
            return {"name": name, "result": "async_task_result"}

        # Execute async flow
        result = await async_flow_with_task("test_async_flow")
        assert result == {"name": "test_async_flow", "result": "async_task_result"}

        # Verify asset saved in store
        key = mapper.compute_key(parameters={"name": "test_async_flow"})
        assert store.exists(key)

        # Load asset directly from store
        loaded = mapper.load(store=store, parameters={"name": "test_async_flow"})
        assert loaded == result

    def test_sync_flow_level_mapper_factory(self, store):
        """Test using an AssetMapper factory at the flow level."""

        mapper = asset_mapper()

        @flow(store=store, mapper=mapper("flow_factory/{name}.pkl"))
        def flow_with_task(name: str):
            return simple_task(name)

        @task
        def simple_task(name: str):
            return {"name": name, "result": "task_result"}

        # Execute flow
        result = flow_with_task("test_flow_factory")
        assert result == {"name": "test_flow_factory", "result": "task_result"}

        # Verify asset exists in store
        assert mapper.exists("flow_factory/test_flow_factory.pkl", store=store)

        # Load asset directly from store
        value = mapper.load("flow_factory/test_flow_factory.pkl", store=store)
        assert value == result

    @pytest.mark.asyncio
    async def test_async_flow_level_mapper_factory(self, store):
        """Test using an AssetMapper factory at the flow level in an async flow."""

        mapper = asset_mapper()

        @flow(store=store, mapper=mapper("async_flow_factory/{name}.pkl"))
        async def async_flow_with_task(name: str):
            return await simple_task(name)

        @task
        async def simple_task(name: str):
            return {"name": name, "result": "async_task_result"}

        # Execute async flow
        result = await async_flow_with_task("test_async_flow_factory")
        assert result == {"name": "test_async_flow_factory", "result": "async_task_result"}

        # Verify asset exists in store
        assert mapper.exists("async_flow_factory/test_async_flow_factory.pkl", store=store)

        # Load asset directly from store
        value = mapper.load("async_flow_factory/test_async_flow_factory.pkl", store=store)
        assert value == result

    def test_sync_task_level_mapper(self, store):
        """Test using an AssetMapper at the task level."""
        mapper = asset_mapper("task_level/{name}.pkl")

        @flow(store=store)
        def flow_with_task(name: str):
            return simple_task(name)

        @task(mapper=mapper)
        def simple_task(name: str):
            return {"name": name, "result": "task_result"}

        # Execute flow
        result = flow_with_task("test_task")
        assert result == {"name": "test_task", "result": "task_result"}

        # Verify asset saved in store
        key = mapper.compute_key(parameters={"name": "test_task"})
        assert store.exists(key)

        # Load asset directly from store
        loaded = mapper.load(store=store, parameters={"name": "test_task"})
        assert loaded == result

    @pytest.mark.asyncio
    async def test_async_task_level_mapper(self, store):
        """Test using an AssetMapper at the task level in an async flow."""
        mapper = asset_mapper("async_task_level/{name}.pkl")

        @flow(store=store)
        async def async_flow_with_task(name: str):
            return await simple_task(name)

        @task(mapper=mapper)
        async def simple_task(name: str):
            return {"name": name, "result": "async_task_result"}

        # Execute async flow
        result = await async_flow_with_task("test_async_task")
        assert result == {"name": "test_async_task", "result": "async_task_result"}

        # Verify asset saved in store
        key = mapper.compute_key(parameters={"name": "test_async_task"})
        assert store.exists(key)

        # Load asset directly from store
        loaded = mapper.load(store=store, parameters={"name": "test_async_task"})
        assert loaded == result

    def test_sync_task_level_mapper_factory(self, store):
        """Test using an AssetMapper factory at the task level."""

        mapper = asset_mapper()

        @flow(store=store)
        def flow_with_task(name: str):
            return simple_task(name)

        @task(mapper=mapper("task_factory/{name}.pkl"))
        def simple_task(name: str):
            return {"name": name, "result": "task_result"}

        # Execute flow
        result = flow_with_task("test_task_factory")
        assert result == {"name": "test_task_factory", "result": "task_result"}

        # Verify asset exists in store
        assert mapper.exists("task_factory/test_task_factory.pkl", store=store)

        # Load asset directly from store
        value = mapper.load("task_factory/test_task_factory.pkl", store=store)
        assert value == result

    @pytest.mark.asyncio
    async def test_async_task_level_mapper_factory(self, store):
        """Test using an AssetMapper factory at the task level in an async flow."""

        mapper = asset_mapper()

        @flow(store=store)
        async def async_flow_with_task(name: str):
            return await simple_task(name)

        @task(mapper=mapper("async_task_factory/{name}.pkl"))
        async def simple_task(name: str):
            return {"name": name, "result": "async_task_result"}

        # Execute async flow
        result = await async_flow_with_task("test_async_task_factory")
        assert result == {"name": "test_async_task_factory", "result": "async_task_result"}

        # Verify asset exists in store
        assert mapper.exists("async_task_factory/test_async_task_factory.pkl", store=store)

        # Load asset directly from store
        value = mapper.load("async_task_factory/test_async_task_factory.pkl", store=store)
        assert value == result

    def test_store_is_inferred_from_flow(self, store):
        mapper = asset_mapper()

        @task
        def task_saving_asset(name: str):
            mapper.save(f"{name}.pkl", {"name": name})

        @flow(store=store)
        def flow_with_task(name: str):
            task_saving_asset(name)

        flow_with_task("test_store")

        # Verify asset saved in store
        key = "test_store.pkl"
        assert store.exists(key)

        # Load asset directly from store
        loaded = mapper.load(key, store=store)
        assert loaded == {"name": "test_store"}

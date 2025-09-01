from contextvars import ContextVar

from waypoint.context import ContextModel
from waypoint.context import hydrated_context
from waypoint.context import serialize_context


class DummyContext(ContextModel):
    __var__ = ContextVar("dummy", default=None)

    value: int


class TestDummyContext:
    def test_context_lifecycle(self):
        assert DummyContext.get() is None

        with DummyContext(value=5):
            ctx = DummyContext.get()
            assert ctx is not None
            assert ctx.value == 5

        assert DummyContext.get() is None

    def test_context_model_copy(self):
        with DummyContext(value=5):
            ctx = DummyContext.get()
            assert ctx is not None
            new_ctx = ctx.model_copy(update={"value": 10})
            assert new_ctx.value == 10
            assert ctx.value == 5

    def test_context_serialize(self):
        with DummyContext(value=5):
            ctx = DummyContext.get()
            assert ctx is not None
            ser = ctx.serialize()
            assert ser == {"value": 5}


class TestSerializeContext:
    def test_context_serialize_default_context(self):
        ser = serialize_context()
        assert ser.keys() == {"hooks"}

    def test_context_serialize_and_hydrate(self):
        with DummyContext(value=5):
            ctx = DummyContext.get()
            data = serialize_context()

        with hydrated_context(data):
            ctx = DummyContext.get()
            assert ctx is not None
            assert ctx.value == 5

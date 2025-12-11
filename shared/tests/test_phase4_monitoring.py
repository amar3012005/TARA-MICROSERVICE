import asyncio
import json
import logging
import sys
from pathlib import Path

import types
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /.../TARA-MICROSERVICE
REPO_ROOT = PROJECT_ROOT.parents[0]  # /home/prometheus/leibniz_agent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _alias_shared_modules():
    """Alias shared modules similarly to other phase tests."""
    shared_dir = PROJECT_ROOT / "shared"
    events_path = shared_dir / "events.py"
    broker_path = shared_dir / "event_broker.py"
    redis_path = shared_dir / "redis_client.py"

    sys.modules.setdefault("leibniz_agent", types.ModuleType("leibniz_agent"))
    sys.modules.setdefault("leibniz_agent.services", types.ModuleType("leibniz_agent.services"))
    sys.modules.setdefault("leibniz_agent.services.shared", types.ModuleType("leibniz_agent.services.shared"))

    def load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    load_mod("leibniz_agent.services.shared.events", events_path)
    load_mod("leibniz_agent.services.shared.event_broker", broker_path)
    load_mod("leibniz_agent.services.shared.redis_client", redis_path)

    # Stub livekit for orchestrator imports
    livekit_mod = types.ModuleType("livekit")
    livekit_api_mod = types.ModuleType("livekit.api")

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def with_grants(self, g):
            return self

        def with_identity(self, i):
            return self

        def with_name(self, n):
            return self

        def to_jwt(self):
            return "jwt"

    livekit_api_mod.AccessToken = _Dummy
    livekit_api_mod.VideoGrants = _Dummy
    sys.modules["livekit"] = livekit_mod
    sys.modules["livekit.api"] = livekit_api_mod


def _load_orchestrator_modules():
    """
    Load only the orchestrator modules needed for Phase 4 monitoring tests.

    We avoid loading the full app / FSM stack here to keep imports simple and
    focused on StructuredLogger + StateManager.
    """
    sys.modules.setdefault(
        "leibniz_agent.services.orchestrator",
        types.ModuleType("leibniz_agent.services.orchestrator"),
    )

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    module_map = {
        "structured_logger": PROJECT_ROOT / "orchestrator" / "structured_logger.py",
        "state_manager": PROJECT_ROOT / "orchestrator" / "state_manager.py",
    }

    loaded = {}
    for name, path in module_map.items():
        full_name = f"leibniz_agent.services.orchestrator.{name}"
        loaded[name] = load(full_name, path)
        # Local alias
        sys.modules.setdefault(f"orchestrator.{name}", loaded[name])

    return loaded["state_manager"], loaded["structured_logger"]


_alias_shared_modules()
state_mgr_mod, structured_logger_mod = _load_orchestrator_modules()
StateManager = state_mgr_mod.StateManager
State = state_mgr_mod.State
StructuredLogger = structured_logger_mod.StructuredLogger


class _ListHandler(logging.Handler):
    """Helper logging handler to capture log messages."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_structured_logger_emits_json():
    """StructuredLogger should emit JSON payloads with expected fields."""
    logger = logging.getLogger("test_structured_logger")
    handler = _ListHandler()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    s = StructuredLogger(logger)
    s.event(
        session_id="sess-1",
        event_type="test_event",
        message="hello",
        data={"foo": "bar"},
    )

    assert handler.messages, "No log messages captured"
    payload = json.loads(handler.messages[0])
    assert payload["session_id"] == "sess-1"
    assert payload["event_type"] == "test_event"
    assert payload["message"] == "hello"
    assert payload["data"]["foo"] == "bar"


def test_state_manager_latency_tracking_basic():
    """StateManager should accumulate latency samples per transition."""
    sm = StateManager("lat_test", redis_client=None)

    async def _run_transitions():
        await sm.transition(State.LISTENING, "client_connected")
        await sm.transition(State.THINKING, "stt_received", {"text": "hi"})
        await sm.transition(State.SPEAKING, "response_ready", {"response": "ok"})

    asyncio.run(_run_transitions())

    breakdown = asyncio.run(sm.get_latency_breakdown())
    # We expect at least some recorded transitions
    assert breakdown, "Latency breakdown should not be empty"
    for key, value in breakdown.items():
        assert value >= 0, f"Latency for {key} should be non-negative"



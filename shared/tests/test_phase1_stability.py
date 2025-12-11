import asyncio
import sys
from pathlib import Path

import pytest

# Ensure monorepo root (/home/prometheus/leibniz_agent) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /.../TARA-MICROSERVICE
REPO_ROOT = PROJECT_ROOT.parents[0]  # /home/prometheus/leibniz_agent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _alias_shared_modules():
    """
    Provide compatibility aliases for absolute imports used in orchestrator code:
    leibniz_agent.services.shared.events / event_broker -> shared.*
    """
    import importlib.util
    import types

    shared_dir = PROJECT_ROOT / "shared"
    events_path = shared_dir / "events.py"
    broker_path = shared_dir / "event_broker.py"

    # Create package shells
    sys.modules.setdefault("leibniz_agent", types.ModuleType("leibniz_agent"))
    sys.modules.setdefault("leibniz_agent.services", types.ModuleType("leibniz_agent.services"))
    sys.modules.setdefault("leibniz_agent.services.shared", types.ModuleType("leibniz_agent.services.shared"))

    # Load events module
    spec_events = importlib.util.spec_from_file_location(
        "leibniz_agent.services.shared.events", events_path
    )
    events_module = importlib.util.module_from_spec(spec_events)
    spec_events.loader.exec_module(events_module)  # type: ignore
    sys.modules["leibniz_agent.services.shared.events"] = events_module

    # Load event_broker module
    spec_broker = importlib.util.spec_from_file_location(
        "leibniz_agent.services.shared.event_broker", broker_path
    )
    broker_module = importlib.util.module_from_spec(spec_broker)
    spec_broker.loader.exec_module(broker_module)  # type: ignore
    sys.modules["leibniz_agent.services.shared.event_broker"] = broker_module

    # Load redis_client module
    redis_client_path = shared_dir / "redis_client.py"
    spec_redis = importlib.util.spec_from_file_location(
        "leibniz_agent.services.shared.redis_client", redis_client_path
    )
    redis_module = importlib.util.module_from_spec(spec_redis)
    spec_redis.loader.exec_module(redis_module)  # type: ignore
    sys.modules["leibniz_agent.services.shared.redis_client"] = redis_module

    # Stub livekit imports required by orchestrator.app during import
    livekit_mod = types.ModuleType("livekit")
    livekit_api_mod = types.ModuleType("livekit.api")

    class _DummyGrants:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _DummyAccessToken:
        def __init__(self, *args, **kwargs):
            pass

        def with_grants(self, grants):
            return self

        def with_identity(self, identity):
            return self

        def with_name(self, name):
            return self

        def to_jwt(self):
            return "dummy-jwt"

    livekit_api_mod.AccessToken = _DummyAccessToken
    livekit_api_mod.VideoGrants = _DummyGrants
    sys.modules["livekit"] = livekit_mod
    sys.modules["livekit.api"] = livekit_api_mod


_alias_shared_modules()

def _load_orchestrator_modules():
    """Load orchestrator modules and alias absolute import paths."""
    import importlib.util
    import types

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        sys.modules[name] = mod
        return mod

    # Ensure package shells exist
    sys.modules.setdefault("leibniz_agent", types.ModuleType("leibniz_agent"))
    sys.modules.setdefault("leibniz_agent.services", types.ModuleType("leibniz_agent.services"))
    sys.modules.setdefault("leibniz_agent.services.orchestrator", types.ModuleType("leibniz_agent.services.orchestrator"))

    module_map = {
        "config": PROJECT_ROOT / "orchestrator" / "config.py",
        "state_manager": PROJECT_ROOT / "orchestrator" / "state_manager.py",
        "orchestrator_fsm": PROJECT_ROOT / "orchestrator" / "orchestrator_fsm.py",
        "parallel_pipeline": PROJECT_ROOT / "orchestrator" / "parallel_pipeline.py",
        "interruption_handler": PROJECT_ROOT / "orchestrator" / "interruption_handler.py",
        "service_manager": PROJECT_ROOT / "orchestrator" / "service_manager.py",
        "dialogue_manager": PROJECT_ROOT / "orchestrator" / "dialogue_manager.py",
    }

    loaded = {}
    for name, path in module_map.items():
        full_name = f"leibniz_agent.services.orchestrator.{name}"
        loaded[name] = load(full_name, path)
        # Also register under local package for direct import
        sys.modules.setdefault(f"orchestrator.{name}", loaded[name])

    app_mod = load(
        "orchestrator.app",
        PROJECT_ROOT / "orchestrator" / "app.py",
    )
    return loaded["state_manager"], app_mod


state_mgr_mod, orchestrator_app = _load_orchestrator_modules()
StateManager, State = state_mgr_mod.StateManager, state_mgr_mod.State


@pytest.mark.asyncio
async def test_invalid_transition_rejected():
    """SPEAKING -> THINKING is invalid and should be rejected."""
    sm = StateManager("test_invalid_transition", redis_client=None)
    sm.state = State.SPEAKING

    await sm.transition(State.THINKING, "invalid_trigger")

    assert sm.state == State.SPEAKING


@pytest.mark.asyncio
async def test_speaking_requires_response():
    """Transition to SPEAKING must include response text."""
    sm = StateManager("test_missing_response", redis_client=None)
    await sm.transition(State.LISTENING, "client_connected")
    sm.state = State.THINKING

    # Missing response should be rejected
    await sm.transition(State.SPEAKING, "response_ready", {})
    assert sm.state == State.THINKING

    # Providing response should succeed
    await sm.transition(State.SPEAKING, "response_ready", {"response": "hello"})
    assert sm.state == State.SPEAKING


@pytest.mark.asyncio
async def test_replace_session_task_cancels_previous():
    """Session task helper should cancel the previous task before replacing it."""
    sid = "task_helper_session"
    orchestrator_app.active_sessions[sid] = {"current_task": None}

    async def sleeper(delay: float):
        await asyncio.sleep(delay)

    first = await orchestrator_app.replace_session_task(
        sid, sleeper(0.2), reason="first"
    )
    second = await orchestrator_app.replace_session_task(
        sid, sleeper(0.2), reason="second"
    )

    assert first.cancelled()
    assert orchestrator_app.active_sessions[sid]["current_task"] is second

    # Cleanup
    if second and not second.done():
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
    orchestrator_app.active_sessions.pop(sid, None)


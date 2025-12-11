import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import types
import importlib.util

# Ensure monorepo root (/home/prometheus/leibniz_agent) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /.../TARA-MICROSERVICE
REPO_ROOT = PROJECT_ROOT.parents[0]  # /home/prometheus/leibniz_agent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def _alias_shared_modules():
    """Alias shared modules."""
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

    # Stub livekit
    livekit_mod = types.ModuleType("livekit")
    livekit_api_mod = types.ModuleType("livekit.api")
    class _Dummy: 
        def __init__(self, *a, **k): pass
        def with_grants(self, g): return self
        def with_identity(self, i): return self
        def with_name(self, n): return self
        def to_jwt(self): return "jwt"
    livekit_api_mod.AccessToken = _Dummy
    livekit_api_mod.VideoGrants = _Dummy
    sys.modules["livekit"] = livekit_mod
    sys.modules["livekit.api"] = livekit_api_mod

def _load_orchestrator_modules():
    """Load orchestrator modules."""
    sys.modules.setdefault("leibniz_agent.services.orchestrator", types.ModuleType("leibniz_agent.services.orchestrator"))
    
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    # Register app early so OrchestratorFSM can import it
    # We create a dummy module first to allow circular import
    app_module = types.ModuleType("leibniz_agent.services.orchestrator.app")
    sys.modules["leibniz_agent.services.orchestrator.app"] = app_module
    # Also register under local package
    sys.modules["orchestrator.app"] = app_module

    # Now load modules in order
    module_map = {
        "config": PROJECT_ROOT / "orchestrator" / "config.py",
        "state_manager": PROJECT_ROOT / "orchestrator" / "state_manager.py",
        "parallel_pipeline": PROJECT_ROOT / "orchestrator" / "parallel_pipeline.py",
        "dialogue_manager": PROJECT_ROOT / "orchestrator" / "dialogue_manager.py",
        "stt_event_handler": PROJECT_ROOT / "orchestrator" / "stt_event_handler.py",
        "orchestrator_fsm": PROJECT_ROOT / "orchestrator" / "orchestrator_fsm.py",
        "interruption_handler": PROJECT_ROOT / "orchestrator" / "interruption_handler.py",
        "service_manager": PROJECT_ROOT / "orchestrator" / "service_manager.py",
    }

    loaded = {}
    for name, path in module_map.items():
        full_name = f"leibniz_agent.services.orchestrator.{name}"
        loaded[name] = load(full_name, path)
        # Register local alias
        sys.modules.setdefault(f"orchestrator.{name}", loaded[name])

    # Now actually load app.py into the existing module shell
    app_spec = importlib.util.spec_from_file_location("leibniz_agent.services.orchestrator.app", PROJECT_ROOT / "orchestrator" / "app.py")
    app_spec.loader.exec_module(app_module)
    
    # Also register 'app' under the package namespace so 'from ... import app' works
    sys.modules["leibniz_agent.services.orchestrator"].app = app_module

    return loaded["state_manager"], loaded["stt_event_handler"], loaded["orchestrator_fsm"], app_module

# Run setup
_alias_shared_modules()
state_mgr_mod, stt_handler_mod, fsm_mod, orchestrator_app = _load_orchestrator_modules()
StateManager = state_mgr_mod.StateManager
State = state_mgr_mod.State
STTEventHandler = stt_handler_mod.STTEventHandler
OrchestratorFSM = fsm_mod.OrchestratorFSM

@pytest.mark.asyncio
async def test_stt_event_handler_happy_path():
    """Test standard flow: Validate -> Thinking -> Pipeline -> Speaking."""
    session_id = "test_sess"
    state_mgr = MagicMock(spec=StateManager)
    state_mgr.state = State.LISTENING
    state_mgr.transition = AsyncMock()
    
    config = MagicMock()
    config.tara_mode = False
    config.skip_intent_service = False
    config.rag_service_url = "http://rag"
    config.intent_service_url = "http://intent"
    
    handler = STTEventHandler(session_id, state_mgr, config)
    
    # Mock pipeline processing to return a dummy generator
    with patch.object(handler, "_process_pipeline", new=AsyncMock(return_value="generator")) as mock_pipeline:
        result = await handler.handle_stt_final("hello", is_final=True)
        
        assert result == "generator"
        
        # Verify state transitions
        state_mgr.transition.assert_any_call(State.THINKING, "stt_received", {"text": "hello"})
        # Should now transition to SPEAKING in handle_stt_final
        state_mgr.transition.assert_any_call(State.SPEAKING, "response_ready", {"text": "hello"})

@pytest.mark.asyncio
async def test_state_contract_mic_gating():
    """Test that StateManager enforces mic gating via contract."""
    state_mgr = StateManager("mic_test", redis_client=None)
    state_mgr._control_mic = AsyncMock()
    
    # LISTENING -> Mic OPEN
    await state_mgr.transition(State.LISTENING, "trigger")
    state_mgr._control_mic.assert_called_with(gate_off=True)
    
    # THINKING -> Mic GATED
    await state_mgr.transition(State.THINKING, "trigger")
    state_mgr._control_mic.assert_called_with(gate_off=False)
    
    # SPEAKING -> Mic GATED
    await state_mgr.transition(State.SPEAKING, "trigger", {"response": "foo"})
    state_mgr._control_mic.assert_called_with(gate_off=False)

@pytest.mark.asyncio
async def test_fsm_uses_unified_handler():
    """Test that OrchestratorFSM uses STTEventHandler for STT processing."""
    fsm = OrchestratorFSM(session_id="fsm_test", redis_client=AsyncMock(), broker=AsyncMock())
    
    # Mock dependencies
    fsm.config = MagicMock()
    fsm.state_mgr = MagicMock()
    fsm.dialogue_manager = MagicMock()
    
    # Mock event
    event = MagicMock()
    event.payload = {"text": "hello", "is_final": True}
    
    # Spy on STTEventHandler.handle_stt_final
    with patch("leibniz_agent.services.orchestrator.stt_event_handler.STTEventHandler.handle_stt_final", new_callable=AsyncMock) as mock_handle:
        await fsm.on_stt_final(event)
        
        # Verify handler was called with correct args
        mock_handle.assert_called_once()
        args, kwargs = mock_handle.call_args
        assert args[0] == "hello"  # text
        assert args[1] is True     # is_final
        assert kwargs["source"] == "redis_stream"


# SINDH Orchestra Agent - Complete Standalone Package

## 🎯 Overview
This is a complete standalone package for the SINDH Orchestra Agent with all necessary dependencies, configurations, and data files. The package now includes the advanced TARA_v2 orchestrator with parallel multi-agent processing capabilities.

## 📁 Package Contents

### Core Files
- `orchestra_agent_backup.py` - Main orchestration agent (Legacy)
- `tara_config.py` - TARA configuration system
- `simple_rag.py` - RAG (Retrieval Augmented Generation) system
- `rag_memory_system.py` - RAG memory management
- `stt.py` - Speech-to-text functionality
- `tts.py` - Text-to-speech functionality
- `fast_intent_router.py` - Fast intent routing
- `sindh_intent_parser.py` - SINDH-specific intent parser
- `dialogue_manager.py` - Dialogue flow management
- `config.py` - General configuration

### TARA_v2 System (NEW)
- `tara_v2/tara_v2_orchestrator.py` - Advanced parallel orchestrator
- `tara_v2/mcp_event_bus.py` - MCP Event Bus for agent coordination
- `tara_v2/fusion_agent.py` - Response synthesis and TTS coordination
- `tara_v2/config/orchestrator_config.json` - Orchestrator configuration
- `tara_v2/examples/orchestrator_example.py` - Usage examples
- `tara_v2/tests/test_orchestrator.py` - Comprehensive test suite

### Optional Components
- `personal_info_browser.py` - Personal information browser
- `applied_jobs_checker.py` - Applied jobs checker
- `available_jobs_browser.py` - Available jobs browser
- `V2/` - V2 Intent Parser and FSM components

### Configuration Files
- `.env` - Environment variables (contains API keys)
- `tara_config.json` - TARA personality and voice settings
- `requirements.txt` - Python dependencies

### Data Folders
- `dialogue_database/` - JSON conversation templates and prompts
- `knowledge_base/` - RAG knowledge base documents
- `voices/` - TTS audio cache (created automatically)
- `background/` - Background audio files

## 🚀 Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Edit `.env` file with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URI=your_mongodb_connection_string_here
# Add other required environment variables
```

### 3. Configure TARA Settings
Edit `tara_config.json` to customize:
- Voice settings (speed, pitch, volume)
- Personality traits
- Language preferences
- Retry attempts and timeouts

### 4. Run the Application

#### Legacy Orchestra Agent
```bash
python orchestra_agent_backup.py
```

#### TARA_v2 Orchestrator (Recommended)
```bash
python -m tara_v2.examples.orchestrator_example
```

## 🎛️ Key Features

### 1. Natural Conversation Flow
- Human-like greeting and introduction
- Intelligent name and phone collection
- Context-aware responses
- RAG-powered question answering

### 2. TARA Integration
- Configurable voice settings
- Personality-driven responses
- Multi-language support
- Emotion expression

### 3. Intent Classification
- Fast pattern matching
- Context-aware routing
- Confidence scoring
- Fallback handling

### 4. Data Collection
- Phase-based registration
- Intelligent retry mechanisms
- Validation and verification
- MongoDB integration

### 5. Audio Processing
- Real-time speech recognition
- High-quality text-to-speech
- Background audio support
- Audio caching for performance

### 6. TARA_v2 Advanced Features (NEW)
- **Parallel Multi-Agent Processing**: Simultaneous execution of retrieval, extraction, and QA agents
- **MCP Event Bus**: Event-driven agent coordination for better scalability
- **Session Management**: Persistent conversation context across turns
- **Health Monitoring**: Real-time monitoring of agent and system health
- **Streaming Responses**: Real-time response generation and streaming
- **Error Recovery**: Comprehensive error handling and graceful degradation
- **Performance Metrics**: Detailed performance tracking and optimization

## 🔧 TARA_v2 Orchestrator Usage

### Basic Usage
```python
from tara_v2 import TARA_v2_Orchestrator, OrchestratorConfig

# Initialize orchestrator
config = OrchestratorConfig()
orchestrator = TARA_v2_Orchestrator(config)

# Start the system
await orchestrator.start()

# Create a session
session_id = await orchestrator.create_session()

# Process user input
response = await orchestrator.process_transcript(
    session_id=session_id,
    transcript="I need help finding a software engineering job"
)

# Clean up
await orchestrator.end_session(session_id)
await orchestrator.shutdown()
```

### Advanced Configuration
```python
from tara_v2 import OrchestratorConfig, AudioConfig, AgentConfig

# Custom configuration
config = OrchestratorConfig(
    max_concurrent_sessions=100,
    session_timeout=3600,  # 1 hour
    enable_metrics=True,
    enable_health_checks=True,
    audio_config=AudioConfig(
        sample_rate=16000,
        chunk_size=1024,
        format="pcm",
        channels=1
    )
)

orchestrator = TARA_v2_Orchestrator(config)
```

### Streaming Response
```python
# Stream responses in real-time
async for chunk in orchestrator.stream_response(
    session_id=session_id,
    transcript="Tell me about available jobs"
):
    print(chunk, end="", flush=True)
```

## 🔧 Customization

### Dialogue Templates
Edit files in `dialogue_database/` to customize:
- Greeting messages
- Data collection prompts
- Success/failure responses
- Retry messages

### Knowledge Base
Add/edit files in `knowledge_base/` to expand TARA's knowledge:
- Platform information
- FAQ responses
- Policy documents
- User guides

### Voice Settings
Modify `tara_config.json` for voice customization:
- Speed and pitch
- Voice selection
- Language preferences
- Emotional expressions

### TARA_v2 Orchestrator Configuration
Edit `tara_v2/config/orchestrator_config.json`:
- Agent settings and timeouts
- Parallel processing parameters
- Session management options
- Health monitoring configuration

## 🛠️ Troubleshooting

### Common Issues
1. **Audio not working**: Check microphone permissions and audio drivers
2. **API errors**: Verify API keys in `.env` file
3. **Database connection**: Check MongoDB URI and network connectivity
4. **Missing dependencies**: Run `pip install -r requirements.txt`
5. **TARA_v2 import errors**: Ensure all dependencies are installed and paths are correct

### Debug Mode
Set environment variable for detailed logging:
```bash
export DEBUG=true
```

### TARA_v2 Health Monitoring
```python
# Check system health
health = await orchestrator.get_health()
print(f"System status: {health['status']}")

# Get performance metrics
metrics = await orchestrator.get_metrics()
print(f"Active sessions: {metrics['active_sessions']}")
```

## 📋 Testing

### Run Legacy Tests
```bash
python test_conversation_flow.py
python test_sindh_parser.py
```

### Run TARA_v2 Tests
```bash
# Install pytest if not already installed
pip install pytest pytest-asyncio

# Run orchestrator tests
python -m pytest tara_v2/tests/test_orchestrator.py -v

# Run specific test categories
python -m pytest tara_v2/tests/test_orchestrator.py::TestOrchestratorCore -v
python -m pytest tara_v2/tests/test_orchestrator.py::TestPerformance -v
```

### Manual Testing
1. Start the application
2. Use spacebar to talk
3. Test different conversation scenarios
4. Verify RAG responses
5. Check data collection flow

### TARA_v2 Example Testing
```bash
# Run the comprehensive example
python -m tara_v2.examples.orchestrator_example

# Test specific features
python -c "
import asyncio
from tara_v2.examples.orchestrator_example import demonstrate_parallel_processing, load_orchestrator_config
from tara_v2 import TARA_v2_Orchestrator

async def test():
    config = await load_orchestrator_config('tara_v2/config/orchestrator_config.json')
    orchestrator = TARA_v2_Orchestrator(config)
    await orchestrator.start()
    await demonstrate_parallel_processing(orchestrator)
    await orchestrator.shutdown()

asyncio.run(test())
"
```

## 🔄 Updates and Maintenance

### Regular Updates
- Update knowledge base documents
- Refresh dialogue templates
- Monitor conversation logs
- Tune intent classification
- Update TARA_v2 agent configurations

### Performance Optimization
- Clear voice cache regularly
- Monitor memory usage with TARA_v2 metrics
- Update dependencies
- Optimize RAG retrieval
- Tune parallel processing parameters

### TARA_v2 Monitoring
```python
# Regular health checks
health = await orchestrator.get_health()
if health['status'] != 'healthy':
    print(f"System issues detected: {health}")

# Performance monitoring
metrics = await orchestrator.get_metrics()
print(f"Sessions: {metrics['active_sessions']}/{metrics['total_sessions']}")
print(f"Average response time: {metrics['avg_response_time']:.2f}s")
```

## � HTTP MCP Integration

### Overview
TARA now supports three MCP transport protocols for flexible deployment:
- **Stdio** (default): Subprocess-based communication, proven stable for single-machine deployments
- **HTTP JSON-RPC**: Request/response over HTTP POST for distributed deployments and easier debugging
- **HTTP SSE**: Server-Sent Events streaming for real-time updates (future enhancement)

### Architecture
- **Server**: FastMCP HTTP server with dual protocol support (SSE + JSON-RPC)
- **Client**: HTTP client with connection pooling, retry logic, and health monitoring
- **Transport Selection**: Environment variable `TARA_MCP_TRANSPORT` (stdio/http)
- **Fallback Mechanism**: Automatic degradation to stdio or direct functions if HTTP unavailable

### Quick Start - HTTP MCP

#### Step 1: Start HTTP MCP Server
```bash
python tara_mcp_http_server.py
```
Server endpoints:
- JSON-RPC & Health: `http://localhost:8000` (port 8000)
  - POST `/message` - JSON-RPC endpoint
  - GET `/health` - Health check endpoint
- SSE: `http://localhost:8001/sse` (port 8001)

**Note:** The server runs two separate HTTP servers:
- Port 8000: JSON-RPC + Health check (aiohttp)
- Port 8001: SSE streaming (FastMCP)

#### Step 2: Configure HTTP Transport
```bash
export TARA_MCP_TRANSPORT=http
export MCP_HTTP_SERVER_URL=http://localhost:8000
```

#### Step 3: Run TARA Application
```bash
python tara_pro_functional.py
```
- Automatically uses HTTP transport
- Falls back to stdio if HTTP unavailable
- Falls back to direct functions if both transports fail

#### Step 4: Verify Integration
```bash
python test_http_mcp_integration_complete.py
```
- Runs comprehensive test suites
- Validates all tools (STT, Intent, RAG, TTS)
- Tests end-to-end conversation flows
- Generates JSON test report

### Configuration - HTTP MCP

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TARA_MCP_TRANSPORT` | `stdio` | Transport protocol: `stdio` or `http` |
| `TARA_MCP_ENABLED` | `true` | Enable/disable MCP (all transports) |
| `MCP_HTTP_SERVER_URL` | `http://localhost:8000` | HTTP server base URL |
| `MCP_HTTP_TIMEOUT` | `30.0` | Request timeout in seconds |
| `MCP_HTTP_RECONNECT_ATTEMPTS` | `3` | Max reconnection attempts |
| `MCP_HTTP_HEALTH_CHECK_INTERVAL` | `60.0` | Health check interval in seconds |
| `SINDH_MCP_HTTP_DEBUG` | `false` | Enable debug logging |
| `SINDH_MCP_HTTP_METRICS` | `true` | Enable metrics collection |

### API Reference - HTTP MCP

**JSON-RPC Endpoint: POST /message**

Request Format:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "<method_name>",
  "params": {<parameters>}
}
```

**Supported Methods:**

1. **tools/list** - List available tools
   - Params: None
   - Returns: `{"tools": [{"name": "...", "description": "..."}]}`

2. **tools/call** - Execute a tool
   - Params: `{"name": "<tool_name>", "arguments": {<tool_args>}}`
   - Returns: `{"content": [{"text": "<json_result>"}]}`

3. **resources/read** - Read session state
   - Params: `{"uri": "session://current"}`
   - Returns: `{"contents": [{"text": "<json_session_data>"}]}`

**Available Tools:**
- `transcribe_audio` - **Live audio capture only** (audio_path parameter not supported, must be null)
- `classify_user_intent` - Intent classification with context support
- `query_knowledge_base` - RAG-powered knowledge retrieval
- `synthesize_speech` - Text-to-speech synthesis

**Tool Limitations:**
- `transcribe_audio`: Only supports live microphone capture. The `audio_path` parameter must be null or omitted. File-based audio transcription is not supported via HTTP server and will return an error.

### Examples - HTTP MCP

**Example 1: Basic Tool Call**
```python
import asyncio
from tara_mcp_http_client import get_mcp_http_client, close_mcp_http_client

async def main():
    client = await get_mcp_http_client()
    
    # Classify intent
    intent = await client.classify_user_intent("मुझे नौकरी चाहिए")
    print(f"Intent: {intent.intent}, Confidence: {intent.confidence}")
    
    # Query knowledge base
    rag = await client.query_knowledge_base("What is TARA?")
    print(f"Answer: {rag.answer}")
    
    await close_mcp_http_client()

asyncio.run(main())
```

**Example 2: Context Manager**
```python
from tara_mcp_http_client import MCPHTTPClientContext

async def main():
    async with MCPHTTPClientContext() as client:
        result = await client.query_knowledge_base("How to register?")
        print(result.answer)
    # Client automatically closed

asyncio.run(main())
```

**Example 3: Performance Monitoring**
```python
from tara_mcp_http_client import get_mcp_http_client

async def main():
    client = await get_mcp_http_client()
    
    # Make multiple requests
    for i in range(10):
        await client.classify_user_intent(f"test {i}")
    
    # Export metrics
    client.export_metrics("metrics.json")
    client.print_performance_dashboard()

asyncio.run(main())
```

### Troubleshooting - HTTP MCP

**Common Issues:**

1. **"Connection refused" Error**
   - Cause: HTTP server not running
   - Solution: Start server with `python tara_mcp_http_server.py`
   - Verify: `curl http://localhost:8000/health`
   - Verify: `curl http://localhost:8000/health`

2. **"Protocol mismatch" Error**
   - Cause: Client using wrong endpoint
   - Solution: Verify `MCP_HTTP_SERVER_URL` configuration
   - Test: `curl -X POST http://localhost:8000/message -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'`

3. **"Timeout" Error**
   - Cause: Tool execution exceeds timeout
   - Solution: Increase `MCP_HTTP_TIMEOUT` environment variable
   - Check: Server logs for slow tool execution

**Debug Mode:**
```bash
export SINDH_MCP_HTTP_DEBUG=true
python tara_pro_functional.py
```
- Enables verbose logging
- Shows request/response details
- Logs connection pool statistics

**Performance Metrics:**
```python
from tara_mcp_http_client import get_mcp_http_client

client = await get_mcp_http_client()
client.export_metrics("mcp_http_metrics.json")
client.print_performance_dashboard()
```

### Migration Guide - Stdio to HTTP

**Step 1: Verify Stdio Transport**
```bash
export TARA_MCP_TRANSPORT=stdio
python tara_pro_functional.py
```

**Step 2: Start HTTP Server**
```bash
python tara_mcp_http_server.py
```

**Step 3: Test HTTP Transport**
```bash
export TARA_MCP_TRANSPORT=http
python test_http_mcp_integration_complete.py
```

**Step 4: Gradual Rollout**
- Start with development environment
- Monitor metrics and error rates
- Gradually migrate staging, then production
- Keep stdio as fallback option

**Rollback Plan:**
- Set `TARA_MCP_TRANSPORT=stdio` to revert
- No code changes needed
- Fallback mechanism handles failures automatically

### FAQ - HTTP MCP

**Q: Should I use stdio or HTTP transport?**
A: Use stdio for single-machine deployments (simpler, proven stable). Use HTTP for distributed deployments, easier debugging, or network monitoring needs.

**Q: Can I use both protocols simultaneously?**
A: Yes, the server supports both SSE and JSON-RPC simultaneously. Configure clients independently.

**Q: What happens if HTTP server crashes?**
A: The client automatically detects failure via health checks, attempts reconnection, and falls back to stdio or direct functions.

**Q: How do I monitor HTTP MCP performance?**
A: Use `client.export_metrics()` or `client.print_performance_dashboard()`. Enable debug logging with `SINDH_MCP_HTTP_DEBUG=true`.

**Q: Can I run HTTP server on a different machine?**
A: Yes, set `MCP_HTTP_SERVER_URL=http://<server_ip>:8000` on client machine. Ensure firewall allows port 8000.

**Q: What's the performance overhead of HTTP vs stdio?**
A: HTTP adds ~10-50ms latency per request (network + serialization). For local deployments, stdio is faster. For distributed deployments, HTTP is necessary.

## �🏗️ Architecture Overview

### Legacy System
- Monolithic orchestrator with sequential processing
- Direct function calls between components
- Simple state management

### TARA_v2 System
- **Event-Driven Architecture**: MCP Event Bus coordinates all agents
- **Parallel Processing**: Multiple agents work simultaneously
- **Microservices Style**: Each agent is independent and specialized
- **Session Management**: Persistent context and conversation history
- **Health Monitoring**: Real-time system health and performance tracking
- **Scalable Design**: Easy to add new agents and scale horizontally

### Component Integration
```
User Input → Speech Recognition → TARA_v2 Orchestrator
    ↓
Event Bus → [Query Planning Agent] → [Retrieval Agents] → [Extraction Agents]
    ↓                                       ↓                    ↓
[QA Agents] ← [Fusion Agent] ← [Response Synthesis]
    ↓
TTS Agent → Audio Output
```

## 📞 Support
For issues and support, refer to the main SINDH platform documentation or the TARA_v2 specific documentation in the `tara_v2/` directory.

---
**Version**: 2.0 (TARA_v2 Orchestrator)
**Last Updated**: January 2025
**Status**: Production Ready ✅

**New Features in v2.0:**
- ✅ Parallel multi-agent processing
- ✅ MCP Event Bus architecture
- ✅ Advanced session management
- ✅ Real-time health monitoring
- ✅ Streaming response generation
- ✅ Comprehensive error handling
- ✅ Performance metrics and optimization
- ✅ Extensive test suite and examples

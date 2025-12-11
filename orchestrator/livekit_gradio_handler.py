"""
LiveKit Gradio Handler - Replaces FastRTC with LiveKit
Provides the same Gradio UI as FastRTC but uses LiveKit for WebRTC
"""
import os
import gradio as gr
import json
import logging

logger = logging.getLogger(__name__)

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
TOKEN_ENDPOINT = f"{ORCHESTRATOR_URL}/token"

def create_livekit_gradio_ui():
    """
    Create Gradio UI that looks exactly like FastRTC but uses LiveKit.
    This replaces FastRTC's Stream with a custom Gradio Blocks interface.
    """
    
    # Get hostname for WebSocket URL
    hostname = os.getenv("LIVEKIT_HOST", "localhost")
    port = os.getenv("LIVEKIT_PORT", "7880")
    livekit_ws_url = f"ws://{hostname}:{port}"
    
    # HTML with LiveKit JavaScript SDK - styled to match FastRTC's appearance
    livekit_html = f"""
    <div id="livekit-container" style="width: 100%; max-width: 800px; margin: 0 auto; padding: 20px;">
        <div id="livekit-status" style="padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 15px; font-family: 'Courier New', monospace; font-size: 13px; border-left: 4px solid #2196F3;">
            <strong>Status:</strong> <span id="status-text">Initializing...</span>
        </div>
        
        <div id="livekit-transcript" style="padding: 15px; background: #ffffff; border-radius: 8px; min-height: 150px; margin-bottom: 15px; border: 1px solid #e0e0e0;">
            <strong style="color: #1976d2;">Transcript:</strong><br><br>
            <div id="transcript-text" style="min-height: 100px; color: #424242; line-height: 1.6;">
                Waiting for speech... Speak into your microphone.
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <button id="connect-btn" onclick="connectLiveKit()" style="padding: 12px 24px; font-size: 16px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                Connect & Start
            </button>
            <button id="disconnect-btn" onclick="disconnectLiveKit()" disabled style="padding: 12px 24px; font-size: 16px; background: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Disconnect
            </button>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"></script>
    <script>
        (function() {{
            let room = null;
            let isConnected = false;
            const livekitUrl = "{livekit_ws_url}";
            const tokenUrl = "/token";
            let transcriptBuffer = "";
            
            function updateStatus(message, isError = false) {{
                const statusText = document.getElementById('status-text');
                const timestamp = new Date().toLocaleTimeString();
                statusText.textContent = `[${{timestamp}}] ${{message}}`;
                statusText.style.color = isError ? '#d32f2f' : '#1976d2';
            }}
            
            function updateTranscript(text, isFinal = false) {{
                const transcriptDiv = document.getElementById('transcript-text');
                if (isFinal) {{
                    transcriptBuffer += text + " ";
                    transcriptDiv.innerHTML = transcriptBuffer + '<span style="color: #999;">...</span>';
                }} else {{
                    transcriptDiv.innerHTML = transcriptBuffer + '<span style="color: #1976d2;">' + text + '</span><span style="color: #999;">...</span>';
                }}
            }}
            
            window.connectLiveKit = async function() {{
                if (isConnected) return;
                
                const connectBtn = document.getElementById('connect-btn');
                const disconnectBtn = document.getElementById('disconnect-btn');
                connectBtn.disabled = true;
                
                try {{
                    updateStatus("Fetching access token...");
                    const response = await fetch(tokenUrl);
                    if (!response.ok) {{
                        throw new Error(`Token request failed: ${{response.statusText}}`);
                    }}
                    const data = await response.json();
                    const token = data.token;
                    
                    updateStatus(`Connecting to LiveKit at ${{livekitUrl}}...`);
                    
                    // Create LiveKit room
                    room = new LivekitClient.Room({{
                        adaptiveStream: true,
                        dynacast: true,
                    }});
                    
                    // Event handlers
                    room.on(LivekitClient.RoomEvent.SignalConnected, () => {{
                        updateStatus("Signal connection established.");
                    }});
                    
                    room.on(LivekitClient.RoomEvent.Connected, () => {{
                        updateStatus("Room connected successfully!");
                        isConnected = true;
                        connectBtn.disabled = true;
                        disconnectBtn.disabled = false;
                    }});
                    
                    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {{
                        if (track.kind === LivekitClient.Track.Kind.Audio) {{
                            updateStatus(`Audio track subscribed from: ${{participant.identity}}`);
                            const element = track.attach();
                            element.style.display = 'none';
                            document.body.appendChild(element);
                            element.play().catch(e => updateStatus(`Autoplay failed: ${{e.message}}`, true));
                            updateStatus("Audio playback started - you should hear TARA's responses!");
                        }}
                    }});
                    
                    room.on(LivekitClient.RoomEvent.LocalTrackPublished, (publication, participant) => {{
                        updateStatus(`Microphone enabled and published! Speak now...`);
                    }});
                    
                    room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {{
                        updateStatus(`Disconnected: ${{reason}}`);
                        isConnected = false;
                        connectBtn.disabled = false;
                        disconnectBtn.disabled = true;
                    }});
                    
                    // Connect to room
                    await room.connect(livekitUrl, token);
                    
                    // Enable microphone (matching FastRTC auto-start behavior)
                    updateStatus("Requesting microphone access...");
                    await room.localParticipant.setMicrophoneEnabled(true);
                    updateStatus("Microphone enabled! Speak now...");
                    
                }} catch (error) {{
                    console.error("Connection error:", error);
                    updateStatus(`Error: ${{error.message}}`, true);
                    isConnected = false;
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                }}
            }};
            
            window.disconnectLiveKit = async function() {{
                if (room) {{
                    updateStatus("Disconnecting...");
                    await room.disconnect();
                    room = null;
                    isConnected = false;
                    document.getElementById('connect-btn').disabled = false;
                    document.getElementById('disconnect-btn').disabled = true;
                }}
            }};
            
            // Auto-connect on page load (matching FastRTC behavior)
            window.addEventListener('load', () => {{
                setTimeout(() => {{
                    connectLiveKit();
                }}, 1000);
            }});
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', async () => {{
                if (room) {{
                    await room.disconnect();
                }}
            }});
        }})();
    </script>
    """
    
    # Create Gradio Blocks interface matching FastRTC's appearance exactly
    with gr.Blocks(
        title="Leibniz STT/VAD Transcription Service",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("## Leibniz STT/VAD Transcription Service")
        gr.Markdown(
            "Speak into your browser microphone. Audio streams directly to the STT/VAD service for real-time transcription. "
            "Check Docker console logs for pipeline progress, speech detection, and transcript fragments."
        )
        
        # LiveKit HTML component (this replaces FastRTC's Stream component)
        livekit_component = gr.HTML(livekit_html)
    
    return demo




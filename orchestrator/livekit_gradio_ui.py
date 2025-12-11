"""
Gradio UI for LiveKit Connection
Creates a simple Gradio interface that connects to LiveKit via JavaScript SDK
"""
import os
import gradio as gr

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")

def create_livekit_ui():
    """Create Gradio UI with LiveKit JavaScript integration"""
    
    # HTML template with LiveKit JavaScript SDK
    livekit_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"></script>
        <style>
            body {
                font-family: sans-serif;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 10px 5px;
            }
            button:hover:not(:disabled) {
                background: #5568d3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            #status {
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                border-radius: 5px;
                font-family: monospace;
                font-size: 12px;
                white-space: pre-wrap;
                max-height: 200px;
                overflow-y: auto;
            }
            #transcript {
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
                min-height: 50px;
            }
            .error {
                border-left-color: #dc3545 !important;
                color: #dc3545;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 TARA LiveKit Voice Agent</h1>
            <button id="connectBtn" onclick="connect()">Connect & Speak</button>
            <button id="disconnectBtn" onclick="disconnect()" disabled>Disconnect</button>
            <div id="status">Status: Disconnected</div>
            <div id="transcript" style="display: none;">
                <strong>Transcript:</strong><br>
                <span id="transcriptText">Waiting for speech...</span>
            </div>
        </div>

        <script>
            let room = null;
            const livekitUrl = "ws://" + window.location.hostname + ":7880";
            
            function log(message, isError = false) {
                const statusDiv = document.getElementById('status');
                const timestamp = new Date().toLocaleTimeString();
                const msg = `[${timestamp}] ${message}`;
                console.log(msg);
                statusDiv.textContent += msg + '\\n';
                statusDiv.scrollTop = statusDiv.scrollHeight;
                if (isError) {
                    statusDiv.className = 'error';
                }
            }
            
            async function connect() {
                const connectBtn = document.getElementById('connectBtn');
                const disconnectBtn = document.getElementById('disconnectBtn');
                const statusDiv = document.getElementById('status');
                const transcriptDiv = document.getElementById('transcript');
                
                connectBtn.disabled = true;
                statusDiv.textContent = '';
                statusDiv.className = '';
                
                try {
                    // 1. Get Token
                    log("Fetching access token...");
                    const response = await fetch('/token');
                    if (!response.ok) {
                        throw new Error(`Token request failed: ${response.statusText}`);
                    }
                    const data = await response.json();
                    const token = data.token;
                    
                    log(`Token received. Connecting to LiveKit at ${livekitUrl}...`);
                    
                    // 2. Connect to Room
                    room = new LivekitClient.Room({
                        adaptiveStream: true,
                        dynacast: true,
                    });
                    
                    // Event handlers
                    room.on(LivekitClient.RoomEvent.SignalConnected, () => {
                        log("Signal connection established.");
                    });
                    
                    room.on(LivekitClient.RoomEvent.Connected, () => {
                        log("Room connected successfully!");
                    });
                    
                    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
                        if (track.kind === LivekitClient.Track.Kind.Audio) {
                            log(`Audio track subscribed from: ${participant.identity}`);
                            const element = track.attach();
                            document.body.appendChild(element);
                            element.play().catch(e => log(`Autoplay failed: ${e.message}`, true));
                            log("Audio playback started.");
                        }
                    });
                    
                    room.on(LivekitClient.RoomEvent.LocalTrackPublished, (publication, participant) => {
                        log(`Local track published: ${publication.kind}`);
                    });
                    
                    room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
                        log(`Disconnected: ${reason}`);
                        connectBtn.disabled = false;
                        disconnectBtn.disabled = true;
                    });
                    
                    // Connect
                    await room.connect(livekitUrl, token);
                    log("Room connection completed.");
                    
                    // 3. Publish Microphone
                    log("Requesting microphone access...");
                    await room.localParticipant.setMicrophoneEnabled(true);
                    log("Microphone enabled and published! Speak now...");
                    transcriptDiv.style.display = 'block';
                    
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                    
                } catch (error) {
                    console.error("Connection error:", error);
                    log(`Error: ${error.message}`, true);
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                }
            }
            
            async function disconnect() {
                if (room) {
                    log("Disconnecting...");
                    await room.disconnect();
                    room = null;
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                }
            }
        </script>
    </body>
    </html>
    """
    
    # Create Gradio interface with HTML component
    with gr.Blocks(title="TARA LiveKit Voice Agent") as demo:
        gr.Markdown("# 🎤 TARA LiveKit Voice Agent")
        gr.Markdown("Connect to LiveKit and speak with TARA. Your audio will be transcribed and TARA will respond.")
        
        # HTML component with LiveKit integration
        gr.HTML(livekit_html)
        
        gr.Markdown("### Instructions:")
        gr.Markdown("""
        1. Click **"Connect & Speak"** button
        2. Allow microphone access when prompted
        3. Speak into your microphone
        4. Wait for TARA's response
        5. Click **"Disconnect"** when done
        """)
    
    return demo




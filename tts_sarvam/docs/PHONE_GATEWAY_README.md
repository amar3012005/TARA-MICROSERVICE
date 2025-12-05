# 📞 Leibniz AI Phone Gateway (FastRTC + FastPhone)

This project transforms the **Leibniz Agent** into a real-time voice AI that you can call via a standard phone number. It integrates **FastRTC** for low-latency streaming and **FastPhone** for telephony access.

---

## 🏗️ Architecture

The system uses a decoupled, event-driven architecture to handle full-duplex voice conversations:

1.  **Phone Call (User)** ↔ **STT/VAD Service** (via `.fastphone()`)
2.  **STT/VAD** sends text to **Orchestrator** (via Redis)
3.  **Orchestrator** sends text to **TTS Service** (via HTTP)
4.  **TTS Service** publishes audio to **Redis**
5.  **STT/VAD** picks up audio from **Redis** and plays it to the **Phone Call**

---

## 🚀 Features

*   **Real-Time Telephony**: Get a temporary phone number instantly via Hugging Face.
*   **Ultra-Low Latency**: Uses WebRTC for sub-500ms response times.
*   **Echo Cancellation**: Built-in WebRTC AEC prevents the agent from hearing itself.
*   **Barge-In Support**: The agent stops speaking immediately when you interrupt.
*   **Sarvam AI TTS**: High-quality Indian English/Regional voices.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
*   Docker Desktop installed and running.
*   Hugging Face Account (for FastPhone token).
*   Sarvam AI API Key.

### 2. Environment Configuration
Create or update your `.env` file in `TARA-MICROSERVICE/`:

```bash
# Hugging Face Token (Required for Phone Number)
HF_TOKEN=hf_your_token_here

# Sarvam AI Configuration
SARVAM_API_KEY=your_sarvam_key
SARVAM_TTS_MODEL=bulbul:v2
SARVAM_TTS_SPEAKER=anushka
LEIBNIZ_TTS_SAMPLE_RATE=16000
```

### 3. Build Services
Rebuild the containers to include the new phone logic:

```powershell
cd TARA-MICROSERVICE
docker build -t stt-vad-service -f stt_vad/Dockerfile .
docker build -t tts-streaming-sarvam -f tts_streaming/Dockerfile .
```

### 4. Run the Stack
Start the entire microservice ecosystem:

```powershell
docker-compose up -d
```

---

## 📱 How to Call Your Agent

1.  Check the logs of the **STT/VAD** service:
    ```powershell
    docker logs -f stt-vad-service
    ```
2.  Look for a message like:
    ```
    📱 FastPhone initialized! Call this number: +1-555-012-3456
    ```
3.  Dial the number from your mobile phone.
4.  Start speaking! The agent will respond in real-time.

---

## 🧩 Troubleshooting

| Issue | Solution |
|-------|----------|
| **No Phone Number** | Ensure `HF_TOKEN` is set in `.env` and valid. |
| **Echo / Feedback** | Use headphones if testing via browser. Phone calls handle this automatically. |
| **Latency** | Check your internet connection. FastRTC requires stable UDP ports. |
| **Silence** | Verify Redis is running (`docker ps`) as it bridges audio between services. |

---

## 📂 Key Files

*   `stt_vad/fastrtc_handler.py`: Handles incoming audio, VAD, and plays back TTS.
*   `stt_vad/app.py`: Initializes the FastPhone gateway.
*   `tts_streaming/app.py`: Publishes generated audio to Redis.
*   `tts_streaming/sarvam_provider.py`: Generates audio via Sarvam AI.

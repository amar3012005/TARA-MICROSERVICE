#!/bin/bash
# Local harness for running STT Local service with Whisper large-v3
# Handles venv setup, dependency checks, model warm-up, and service launch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/../venv"

# Make model configurable; default to medium (769M)
MODEL="${LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE:-medium}"

printf '\n==========================================\n'
printf 'Whisper large-v3 Local Runner\n'
printf '==========================================\n'

# Ensure virtual environment exists
if [[ ! -d "${VENV_DIR}" ]]; then
    printf 'Creating virtual environment at %s...\n' "${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# Activate venv
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Upgrade pip (quietly)
pip install --upgrade pip >/dev/null

cd "${SCRIPT_DIR}"

# Light deps check
if ! python -c "import fastapi" >/dev/null 2>&1; then
    printf 'Installing lightweight dependencies...\n'
    pip install -r requirements.txt
fi

# Heavy deps check (torch + faster-whisper)
if ! python -c "import torch" >/dev/null 2>&1 || ! python -c "import faster_whisper" >/dev/null 2>&1; then
    printf 'Installing heavy dependencies (this may take a while)...\n'
    pip install -r requirements_after.txt
fi

# Determine runtime device
DEVICE=$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)

if [[ "${DEVICE}" == "cuda" ]]; then
    COMPUTE_TYPE="float16"
    USE_GPU="true"
else
    COMPUTE_TYPE="float32"
    USE_GPU="false"
fi

printf '\nRuntime configuration:\n'
printf '  Whisper model  : %s\n' "${MODEL}"
printf '  Device         : %s\n' "${DEVICE}"
printf '  Compute type   : %s\n' "${COMPUTE_TYPE}"
printf '  Use GPU        : %s\n' "${USE_GPU}"

# Export environment variables for the service
export LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE="${MODEL}"
export LEIBNIZ_STT_LOCAL_WHISPER_DEVICE="${DEVICE}"
export LEIBNIZ_STT_LOCAL_WHISPER_COMPUTE_TYPE="${COMPUTE_TYPE}"
export LEIBNIZ_STT_LOCAL_USE_GPU="${USE_GPU}"
export LEIBNIZ_STT_LOCAL_PARTIAL_UPDATE_INTERVAL_MS="400"
export LEIBNIZ_STT_LOCAL_MIN_AUDIO_LENGTH="0.4"

# Optional warm start to ensure model is downloaded before service boots
if [[ "${1:-}" != "--skip-warmup" ]]; then
    printf '\nWarming up Whisper model (%s) (first run may take several minutes)...\n' "${MODEL}"
    python - <<'PY'
import os
from faster_whisper import WhisperModel

device = os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_DEVICE", "cpu")
compute_type = os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_COMPUTE_TYPE", "float32")
model_name = os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE", "medium")

model = WhisperModel(
    model_size_or_path=model_name,
    device=device,
    compute_type=compute_type,
    num_workers=1,
    download_root=None,
)
print(f"Whisper model ready on {device} using {model_name} with {compute_type}")
PY
fi

printf '\nLaunch instructions:\n'
printf '  FastRTC UI : http://localhost:7861/fastrtc\n'
printf '  API Root   : http://localhost:8006\n'
printf '  Health     : http://localhost:8006/health\n'
printf '\nPress Ctrl+C to stop the service.\n'
printf '==========================================\n\n'

exec python3 run_local_fastrtc.py

#!/bin/bash
# One-click harness for the local microphone-to-Whisper demo.
# Mirrors run_whisper_large_v3_local.sh but pipes live audio into the STT stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/../venv"

MODEL="${LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE:-large-v3}"
SKIP_WARMUP=false

if [[ "${1:-}" == "--skip-warmup" ]]; then
    SKIP_WARMUP=true
    shift
fi

PASS_THROUGH_ARGS=("$@")

printf '\n==========================================\n'
printf 'Whisper Microphone Demo\n'
printf '==========================================\n'

if [[ ! -d "${VENV_DIR}" ]]; then
    printf 'Creating virtual environment at %s...\n' "${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip >/dev/null

cd "${SCRIPT_DIR}"

if ! python -c "import fastapi" >/dev/null 2>&1; then
    printf 'Installing lightweight dependencies...\n'
    pip install -r requirements.txt
fi

if ! python -c "import torch" >/dev/null 2>&1 || \
   ! python -c "import faster_whisper" >/dev/null 2>&1 || \
   ! python -c "import sounddevice" >/dev/null 2>&1; then
    printf 'Installing heavy dependencies (torch, faster-whisper, sounddevice)...\n'
    pip install -r requirements_after.txt
fi

# Honor LEIBNIZ_STT_LOCAL_USE_GPU if already set, otherwise auto-detect
if [[ -n "${LEIBNIZ_STT_LOCAL_USE_GPU}" ]]; then
    USE_GPU="${LEIBNIZ_STT_LOCAL_USE_GPU}"
    if [[ "${USE_GPU}" == "true" ]]; then
        DEVICE="cuda"
        COMPUTE_TYPE="float16"
    else
        DEVICE="cpu"
        COMPUTE_TYPE="float32"
    fi
else
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
fi

printf '\nRuntime configuration:\n'
printf '  Whisper model  : %s\n' "${MODEL}"
printf '  Device         : %s\n' "${DEVICE}"
printf '  Compute type   : %s\n' "${COMPUTE_TYPE}"
printf '  Use GPU        : %s\n' "${USE_GPU}"

export LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE="${MODEL}"
export LEIBNIZ_STT_LOCAL_WHISPER_DEVICE="${DEVICE}"
export LEIBNIZ_STT_LOCAL_WHISPER_COMPUTE_TYPE="${COMPUTE_TYPE}"
export LEIBNIZ_STT_LOCAL_USE_GPU="${USE_GPU}"
export LEIBNIZ_STT_LOCAL_PARTIAL_UPDATE_INTERVAL_MS="250"
export LEIBNIZ_STT_LOCAL_MIN_AUDIO_LENGTH="0.3"

if [[ "${SKIP_WARMUP}" == "false" ]]; then
    printf '\nEnsuring Whisper model (%s) is ready (first run may take a few minutes)...\n' "${MODEL}"
    python - <<'PY'
import os
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size_or_path=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE", "medium"),
    device=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_DEVICE", "cpu"),
    compute_type=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_COMPUTE_TYPE", "float32"),
    num_workers=1,
)
print("Whisper model downloaded and ready for microphone streaming.")
PY
fi

printf '\nLaunch instructions:\n'
printf '  1. (Optional) list devices: python3 test_mic_realtime.py --list-devices\n'
printf '  2. Streaming will start immediately. Speak into your microphone.\n'
printf '  3. Press Ctrl+C to stop.\n'
printf '\nCommand being executed:\n'
if [[ ${#PASS_THROUGH_ARGS[@]} -eq 0 ]]; then
    PRETTY_ARGS=""
else
    PRETTY_ARGS="${PASS_THROUGH_ARGS[*]}"
fi
printf '  python3 test_mic_realtime.py %s\n' "${PRETTY_ARGS}"
printf '\nPress Ctrl+C to stop the stream.\n'
printf '==========================================\n\n'

exec python3 test_mic_realtime.py "${PASS_THROUGH_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CACHE_DIR="${VIDARAX_MODEL_CACHE_DIR:-$ROOT_DIR/.vidarax-models}"
VENV_DIR="${VIDARAX_AUDIO_VENV_DIR:-$ROOT_DIR/.venv-audio}"
PYTHON_BIN="${VIDARAX_AUDIO_PYTHON:-python3}"
PROFILE="${1:-core}"
EFFICIENTAT_COMMIT="${VIDARAX_EFFICIENTAT_COMMIT:-a425fdce92572e602a1d5634799bd9f1f2efa806}"
EFFICIENTAT_DIR="$MODEL_CACHE_DIR/source/EfficientAT"

case "$PROFILE" in
  core|sensevoice|moonshine|qwen|whisper|lfm|all) ;;
  *)
    echo "usage: $0 [core|sensevoice|moonshine|qwen|whisper|lfm|all]" >&2
    exit 2
    ;;
esac

if [[ "$PROFILE" == "moonshine" || "$PROFILE" == "qwen" || "$PROFILE" == "whisper" || "$PROFILE" == "lfm" || "$PROFILE" == "all" ]]; then
  if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "$PROFILE requires Python 3.10 or newer; set VIDARAX_AUDIO_PYTHON" >&2
    exit 2
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  "msgpack>=1.1,<2" \
  "numpy>=1.26,<3" \
  "torch>=2.5" \
  "torchaudio>=2.5" \
  "onnxruntime>=1.18,<2" \
  "silero-vad>=6,<7" \
  "librosa>=0.10,<1" \
  "timm>=1,<2"

mkdir -p "$(dirname "$EFFICIENTAT_DIR")"
if [[ ! -d "$EFFICIENTAT_DIR/.git" ]]; then
  git clone https://github.com/fschmid56/EfficientAT.git "$EFFICIENTAT_DIR"
fi
git -C "$EFFICIENTAT_DIR" fetch origin "$EFFICIENTAT_COMMIT"
git -C "$EFFICIENTAT_DIR" checkout --detach "$EFFICIENTAT_COMMIT"

if [[ "$PROFILE" == "sensevoice" || "$PROFILE" == "all" ]]; then
  "$VENV_DIR/bin/python" -m pip install "funasr>=1.2,<2" "modelscope>=1.27,<2"
fi

if [[ "$PROFILE" == "moonshine" || "$PROFILE" == "whisper" || "$PROFILE" == "all" ]]; then
  "$VENV_DIR/bin/python" -m pip install "transformers>=5.7,<6" "accelerate>=1.10"
fi

if [[ "$PROFILE" == "qwen" || "$PROFILE" == "all" ]]; then
  "$VENV_DIR/bin/python" -m pip install "transformers>=5.13,<6" "accelerate>=1.10"
fi

if [[ "$PROFILE" == "lfm" || "$PROFILE" == "all" ]]; then
  "$VENV_DIR/bin/python" -m pip install "liquid-audio"
fi

echo "audio environment: $VENV_DIR"
echo "EfficientAT source: $EFFICIENTAT_DIR"
echo "start: VIDARAX_EFFICIENTAT_REPO=$EFFICIENTAT_DIR $VENV_DIR/bin/python scripts/audio_perception_server.py"

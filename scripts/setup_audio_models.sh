#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${VIDARAX_AUDIO_PYTHON:-python3}"
PROFILE="${1:-whisper}"

case "$PROFILE" in
  core|sensevoice|moonshine|qwen|whisper|lfm|all) ;;
  *)
    echo "usage: $0 [core|sensevoice|moonshine|qwen|whisper|lfm|all]" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/audio_runtime.py" install --profile "$PROFILE"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[model-provision] python3 is required" >&2
  exit 1
fi

MODEL_CACHE_DIR="${VIDARAX_MODEL_CACHE_DIR:-$ROOT_DIR/.vidarax-models}"
MODEL_IDS="${VIDARAX_MODEL_IDS:-Qwen/Qwen3-VL-8B-Instruct,allenai/Molmo2-8B,Qwen/Qwen3-VL-4B-Instruct,OpenGVLab/InternVL3_5-4B,Qwen/Qwen3-VL-2B-Instruct,openbmb/MiniCPM-V-4_5}"
ALLOW_PATTERNS="${VIDARAX_MODEL_ALLOW_PATTERNS:-*.json,*.txt,*.model,*.safetensors,tokenizer*,special_tokens_map*,generation_config*}"
MAX_WORKERS="${VIDARAX_MODEL_MAX_WORKERS:-8}"
DRY_RUN="${VIDARAX_MODEL_DRY_RUN:-0}"
VENV_DIR="${VIDARAX_MODEL_VENV_DIR:-$ROOT_DIR/.venv-model-provision}"
PYTHON_BIN="python3"

mkdir -p "$MODEL_CACHE_DIR"

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[model-provision] preparing virtualenv at $VENV_DIR"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install --upgrade "huggingface_hub>=0.26,<1.0"
fi

echo "[model-provision] cache_dir=$MODEL_CACHE_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[model-provision] HF_TOKEN detected"
else
  echo "[model-provision] HF_TOKEN not set (public artifacts only)"
fi

VIDARAX_MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
VIDARAX_MODEL_IDS="$MODEL_IDS" \
VIDARAX_MODEL_ALLOW_PATTERNS="$ALLOW_PATTERNS" \
VIDARAX_MODEL_MAX_WORKERS="$MAX_WORKERS" \
VIDARAX_MODEL_DRY_RUN="$DRY_RUN" \
"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

cache_dir = Path(os.environ["VIDARAX_MODEL_CACHE_DIR"]).resolve()
models = [m.strip() for m in os.environ["VIDARAX_MODEL_IDS"].split(",") if m.strip()]
allow_patterns = [p.strip() for p in os.environ["VIDARAX_MODEL_ALLOW_PATTERNS"].split(",") if p.strip()]
max_workers = int(os.environ["VIDARAX_MODEL_MAX_WORKERS"])
dry_run = os.environ.get("VIDARAX_MODEL_DRY_RUN", "0") == "1"
token = os.environ.get("HF_TOKEN")

if not models:
    raise SystemExit("no models configured")

cache_dir.mkdir(parents=True, exist_ok=True)

for model_id in models:
    target = cache_dir / model_id.replace("/", "--")
    print(f"[model-provision] model={model_id} target={target}")
    if dry_run:
        continue
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns if allow_patterns else None,
            token=token,
            max_workers=max_workers,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[model-provision] failed model={model_id}: {exc}", file=sys.stderr)
        raise

print("[model-provision] done")
PY

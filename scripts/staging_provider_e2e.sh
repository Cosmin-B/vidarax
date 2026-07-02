#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIDARAX_STAGING_VLLM_BASE_URL:-}" || -z "${VIDARAX_STAGING_SGLANG_BASE_URL:-}" ]]; then
  echo "[staging-e2e] skipped: set VIDARAX_STAGING_VLLM_BASE_URL and VIDARAX_STAGING_SGLANG_BASE_URL"
  exit 0
fi

echo "[staging-e2e] running live provider integration test"
cargo test -p vidarax-api --features live-tests staging_live_provider_e2e_opt_in -- --nocapture

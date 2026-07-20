#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench] running perf probe"
metrics_json="$(cargo run -q -p vidarax-core --release --features perf-probe --bin perf_probe)"
echo "$metrics_json"

max_alloc_total="${VIDARAX_MAX_ALLOC_TOTAL:-0}"
alloc_total="$(jq -r '.allocations.total' <<<"$metrics_json")"

if (( alloc_total > max_alloc_total )); then
  echo "FAIL: observed allocations ${alloc_total} > ${max_alloc_total}" >&2
  exit 1
fi

echo "[bench] PASS"

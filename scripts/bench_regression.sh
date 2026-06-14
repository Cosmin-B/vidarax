#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench] running perf probe"
metrics_json="$(cargo run -q -p vidarax-core --release --features perf-probe --bin perf_probe)"
echo "$metrics_json"

max_alloc_per_frame="${VIDARAX_MAX_ALLOC_PER_FRAME:-0.0}"

alloc_per_frame="$(jq -r '.allocations.per_frame' <<<"$metrics_json")"

if ! awk -v a="$alloc_per_frame" -v b="$max_alloc_per_frame" 'BEGIN { exit(a<=b ? 0 : 1) }'; then
  echo "FAIL: allocations/frame ${alloc_per_frame} > ${max_alloc_per_frame}" >&2
  exit 1
fi

echo "[bench] PASS"

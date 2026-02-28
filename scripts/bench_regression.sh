#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench] running perf probe"
metrics_json="$(cargo run -q -p vidarax-core --release --bin perf_probe)"
echo "$metrics_json"

min_spsc="${VIDARAX_MIN_SPSC_OPS_SEC:-100000}"
max_alloc_per_frame="${VIDARAX_MAX_ALLOC_PER_FRAME:-0.0}"

spsc_ops="$(jq -r '.spsc.throughput_ops_sec' <<<"$metrics_json")"
alloc_per_frame="$(jq -r '.allocations.per_frame' <<<"$metrics_json")"

if (( spsc_ops < min_spsc )); then
  echo "FAIL: spsc throughput ${spsc_ops} < ${min_spsc}" >&2
  exit 1
fi

if ! awk -v a="$alloc_per_frame" -v b="$max_alloc_per_frame" 'BEGIN { exit(a<=b ? 0 : 1) }'; then
  echo "FAIL: allocations/frame ${alloc_per_frame} > ${max_alloc_per_frame}" >&2
  exit 1
fi

echo "[bench] PASS"

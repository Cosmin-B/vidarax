#!/usr/bin/env bash
# Remote HTTP bench: MacBook → vidarax endpoint (macOS-compatible)
# Usage: VIDARAX_BASE_URL=http://localhost:8080 ./scripts/bench_remote_http.sh
set -euo pipefail

BASE_URL="${VIDARAX_BASE_URL:-http://localhost:8080}"
ITERATIONS="${VIDARAX_BENCH_ITERATIONS:-50}"
WARMUP="${VIDARAX_BENCH_WARMUP:-5}"

for cmd in curl jq python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "missing: $cmd" >&2; exit 1; }
done

ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

echo "[remote-bench] target: $BASE_URL  iterations: $ITERATIONS  warmup: $WARMUP"
curl -fsS "$BASE_URL/v1/health" >/dev/null || { echo "FAIL: health check" >&2; exit 1; }

create_ms=()
ingest_ms=()
analyze_ms=()
query_ms=()
workflow_ms=()

total=$((WARMUP + ITERATIONS))
for ((i=0; i<total; i++)); do
  t_wf0=$(ms)

  t0=$(ms)
  create_body=$(curl -fsS -X POST "$BASE_URL/v1/runs" \
    -H "content-type: application/json" \
    -d '{"mode":"balanced"}')
  t1=$(ms)
  run_id=$(jq -r '.run_id' <<<"$create_body")

  t2=$(ms)
  curl -fsS -X POST "$BASE_URL/v1/runs/$run_id/ingest" \
    -H "content-type: application/json" \
    -d '{"frame_index":1}' >/dev/null
  t3=$(ms)

  t4=$(ms)
  curl -fsS -X POST "$BASE_URL/v1/runs/$run_id/analyze" \
    -H "content-type: application/json" \
    -d '{
      "model":"Qwen/Qwen3-VL-8B-Instruct",
      "frames":[
        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
        {"frame_index":1,"pts_ms":41,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
      ]
    }' >/dev/null
  t5=$(ms)

  t6=$(ms)
  curl -fsS -X POST "$BASE_URL/v1/query" \
    -H "content-type: application/json" \
    -d "{\"run_id\":\"$run_id\"}" >/dev/null
  t7=$(ms)

  curl -fsS -X POST "$BASE_URL/v1/runs/$run_id/stop" >/dev/null 2>&1 || true
  t_wf1=$(ms)

  if ((i >= WARMUP)); then
    create_ms+=($((t1 - t0)))
    ingest_ms+=($((t3 - t2)))
    analyze_ms+=($((t5 - t4)))
    query_ms+=($((t7 - t6)))
    workflow_ms+=($((t_wf1 - t_wf0)))
  fi
done

# percentile using python3 (macOS safe)
pct() {
  local p=$1; shift
  python3 -c "
import sys
vals = sorted([int(x) for x in '$*'.split()])
n = len(vals)
idx = int((n-1)*$p/100)
print(vals[idx])
"
}

p50_create=$(pct 50 "${create_ms[@]}")
p95_create=$(pct 95 "${create_ms[@]}")
p50_ingest=$(pct 50 "${ingest_ms[@]}")
p95_ingest=$(pct 95 "${ingest_ms[@]}")
p50_analyze=$(pct 50 "${analyze_ms[@]}")
p95_analyze=$(pct 95 "${analyze_ms[@]}")
p50_query=$(pct 50 "${query_ms[@]}")
p95_query=$(pct 95 "${query_ms[@]}")
p50_wf=$(pct 50 "${workflow_ms[@]}")
p95_wf=$(pct 95 "${workflow_ms[@]}")

total_elapsed=0
for v in "${workflow_ms[@]}"; do total_elapsed=$((total_elapsed + v)); done
wf_per_sec=$(python3 -c "print(round($ITERATIONS * 1000 / max($total_elapsed, 1), 2))")

jq -n \
  --arg url "$BASE_URL" \
  --argjson iters "$ITERATIONS" \
  --argjson wf_per_sec "$wf_per_sec" \
  --argjson p50_create "$p50_create" --argjson p95_create "$p95_create" \
  --argjson p50_ingest "$p50_ingest" --argjson p95_ingest "$p95_ingest" \
  --argjson p50_analyze "$p50_analyze" --argjson p95_analyze "$p95_analyze" \
  --argjson p50_query "$p50_query" --argjson p95_query "$p95_query" \
  --argjson p50_wf "$p50_wf" --argjson p95_wf "$p95_wf" \
  '{
    target: $url,
    iterations: $iters,
    workflows_per_sec: $wf_per_sec,
    create_run:     { p50_ms: $p50_create,   p95_ms: $p95_create },
    ingest_run:     { p50_ms: $p50_ingest,   p95_ms: $p95_ingest },
    analyze_run:    { p50_ms: $p50_analyze,  p95_ms: $p95_analyze },
    query_run:      { p50_ms: $p50_query,    p95_ms: $p95_query },
    workflow_total: { p50_ms: $p50_wf,       p95_ms: $p95_wf }
  }'

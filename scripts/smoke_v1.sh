#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

api_bin="cargo run -q -p vidarax-api --bin vidarax-api"
api_addr="${VIDARAX_BIND_ADDR:-127.0.0.1:18080}"
export VIDARAX_BIND_ADDR="$api_addr"
export VIDARAX_DATA_DIR="${VIDARAX_DATA_DIR:-$ROOT_DIR/.smoke-data}"
export VIDARAX_REQUIRE_API_KEY="${VIDARAX_REQUIRE_API_KEY:-true}"
if [[ -n "${VIDARAX_API_KEYS:-}" ]]; then
  smoke_api_key="${VIDARAX_SMOKE_API_KEY:-${VIDARAX_API_KEYS%%,*}}"
else
  smoke_api_key="${VIDARAX_SMOKE_API_KEY:-smoke-key}"
  export VIDARAX_API_KEYS="$smoke_api_key"
fi

mkdir -p "$VIDARAX_DATA_DIR"
rm -f "$VIDARAX_DATA_DIR/timeline.wal"

echo "[smoke] starting API on $api_addr"
$api_bin >/tmp/vidarax-smoke-api.log 2>&1 &
api_pid=$!
trap 'kill "$api_pid" >/dev/null 2>&1 || true' EXIT

for _ in {1..200}; do
  if curl -fsS "http://$api_addr/v1/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$api_pid" >/dev/null 2>&1; then
    echo "FAIL: API process exited before health check became ready" >&2
    cat /tmp/vidarax-smoke-api.log >&2 || true
    exit 1
  fi
  sleep 0.1
done

if ! curl -fsS "http://$api_addr/v1/health" >/dev/null 2>&1; then
  echo "FAIL: API health check did not become ready in time" >&2
  cat /tmp/vidarax-smoke-api.log >&2 || true
  exit 1
fi

echo "[smoke] create run"
create_resp="$(curl -fsS -X POST "http://$api_addr/v1/runs" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d '{"mode":"balanced","model":"Qwen/Qwen3-VL-4B-Instruct"}')"
run_id="$(jq -r '.run_id' <<<"$create_resp")"

echo "[smoke] ingest frame payload"
curl -fsS -X POST "http://$api_addr/v1/runs/$run_id/ingest" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d '{"frame_index":1,"pts_ms":33}' >/dev/null

echo "[smoke] query timeline"
query_resp="$(curl -fsS -X POST "http://$api_addr/v1/query" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d "{\"run_id\":\"$run_id\",\"kind\":\"ingest_received\"}")"
match_count="$(jq -r '.matches | length' <<<"$query_resp")"
if [[ "$match_count" -lt 1 ]]; then
  echo "FAIL: expected at least 1 query match, got $match_count" >&2
  exit 1
fi

echo "[smoke] keepalive run"
curl -fsS -X POST "http://$api_addr/v1/runs/$run_id/keepalive" \
  -H "x-api-key: $smoke_api_key" >/dev/null

echo "[smoke] model catalog"
models_resp="$(curl -fsS "http://$api_addr/v1/models" \
  -H "x-api-key: $smoke_api_key")"
models_count="$(jq -r '.models | length' <<<"$models_resp")"
if [[ "$models_count" -lt 1 ]]; then
  echo "FAIL: expected at least one model in catalog, got $models_count" >&2
  exit 1
fi

echo "[smoke] stop run"
curl -fsS -X POST "http://$api_addr/v1/runs/$run_id/stop" \
  -H "x-api-key: $smoke_api_key" >/dev/null
state_resp="$(curl -fsS "http://$api_addr/v1/runs/$run_id/state" \
  -H "x-api-key: $smoke_api_key")"
state="$(jq -r '.state' <<<"$state_resp")"
if [[ "$state" != "cancelled" ]]; then
  echo "FAIL: expected state=cancelled, got $state" >&2
  exit 1
fi

events_resp="$(curl -fsS "http://$api_addr/v1/runs/$run_id/events" \
  -H "x-api-key: $smoke_api_key")"
events_count="$(jq -r '.events | length' <<<"$events_resp")"
if [[ "$events_count" -lt 3 ]]; then
  echo "FAIL: expected >=3 events, got $events_count" >&2
  exit 1
fi

echo "[smoke] PASS run_id=$run_id events=$events_count"

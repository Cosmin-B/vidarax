#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for cmd in curl jq ffmpeg; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[mp4-smoke] missing required command: $cmd" >&2
    exit 1
  fi
done

api_bin="cargo run -q -p vidarax-api --bin vidarax-api"
api_addr="${VIDARAX_BIND_ADDR:-127.0.0.1:18081}"
export VIDARAX_BIND_ADDR="$api_addr"
export VIDARAX_DATA_DIR="${VIDARAX_DATA_DIR:-$ROOT_DIR/.smoke-mp4-data}"
export VIDARAX_REQUIRE_API_KEY="${VIDARAX_REQUIRE_API_KEY:-true}"
if [[ -n "${VIDARAX_API_KEYS:-}" ]]; then
  smoke_api_key="${VIDARAX_SMOKE_API_KEY:-${VIDARAX_API_KEYS%%,*}}"
else
  smoke_api_key="${VIDARAX_SMOKE_API_KEY:-smoke-key}"
  export VIDARAX_API_KEYS="$smoke_api_key"
fi

mkdir -p "$VIDARAX_DATA_DIR"
rm -f "$VIDARAX_DATA_DIR/timeline.wal"

fixture_mp4="$(mktemp -t vidarax-smoke-XXXXXX).mp4"
cleanup() {
  if [[ -n "${api_pid:-}" ]]; then
    if kill -0 "$api_pid" >/dev/null 2>&1; then
      kill "$api_pid" >/dev/null 2>&1 || true
      wait "$api_pid" 2>/dev/null || true
    fi
  fi
  rm -f "$fixture_mp4"
}

trap cleanup EXIT

ffmpeg -v error -f lavfi -i "testsrc=size=160x120:rate=12" -t 1.2 -pix_fmt yuv420p -an -y "$fixture_mp4"

echo "[mp4-smoke] starting API on $api_addr"
$api_bin >/tmp/vidarax-smoke-mp4-api.log 2>&1 &
api_pid=$!

for _ in {1..300}; do
  if curl -fsS "http://$api_addr/v1/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$api_pid" >/dev/null 2>&1; then
    echo "FAIL: API process exited before health check became ready" >&2
    cat /tmp/vidarax-smoke-mp4-api.log >&2 || true
    exit 1
  fi
  sleep 0.1
done

if ! curl -fsS "http://$api_addr/v1/health" >/dev/null 2>&1; then
  echo "FAIL: API health check did not become ready in time" >&2
  cat /tmp/vidarax-smoke-mp4-api.log >&2 || true
  exit 1
fi

echo "[mp4-smoke] create run"
create_resp="$(curl -fsS -X POST "http://$api_addr/v1/runs" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d '{"mode":"balanced","model":"Qwen/Qwen3-VL-4B-Instruct"}')"
run_id="$(jq -r '.run_id' <<<"$create_resp")"

echo "[mp4-smoke] ingest mp4 source"
ingest_resp="$(curl -fsS -X POST "http://$api_addr/v1/runs/$run_id/ingest" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d "{\"source_uri\":\"$fixture_mp4\",\"sample_fps\":2.0,\"max_frames\":64}")"
decoded_frames="$(jq -r '.decoded_frames' <<<"$ingest_resp")"
if [[ "${decoded_frames:-0}" -lt 1 ]]; then
  echo "FAIL: expected decoded_frames >= 1, got ${decoded_frames:-0}" >&2
  exit 1
fi

echo "[mp4-smoke] analyze without explicit frames"
analyze_resp="$(curl -fsS -X POST "http://$api_addr/v1/runs/$run_id/analyze" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-4B-Instruct","window_size":8}')"
generated="$(jq -r '.generated' <<<"$analyze_resp")"
if [[ "${generated:-0}" -lt 1 ]]; then
  echo "FAIL: expected generated >= 1, got ${generated:-0}" >&2
  exit 1
fi

query_resp="$(curl -fsS -X POST "http://$api_addr/v1/query" \
  -H "x-api-key: $smoke_api_key" \
  -H "content-type: application/json" \
  -d "{\"run_id\":\"$run_id\",\"kind\":\"frames_decoded\"}")"
match_count="$(jq -r '.matches | length' <<<"$query_resp")"
if [[ "$match_count" -lt 1 ]]; then
  echo "FAIL: expected at least one frames_decoded event" >&2
  exit 1
fi

echo "[mp4-smoke] PASS run_id=$run_id decoded=$decoded_frames generated=$generated"

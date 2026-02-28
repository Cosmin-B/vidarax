#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

max_cli_size="${VIDARAX_MAX_CLI_SIZE_BYTES:-25000000}"
max_api_size="${VIDARAX_MAX_API_SIZE_BYTES:-45000000}"
max_gate_p95_ns="${VIDARAX_MAX_GATE_P95_NS:-50000}"

"$ROOT_DIR/scripts/validate_replay_and_schema.sh"
"$ROOT_DIR/scripts/bench_regression.sh"

echo "[gate] building release artifacts"
cargo build --release -p vidarax-cli -p vidarax-api

cli_size="$(wc -c < "$ROOT_DIR/target/release/vidarax-cli" | tr -d ' ')"
api_size="$(wc -c < "$ROOT_DIR/target/release/vidarax-api" | tr -d ' ')"

echo "[gate] binary sizes cli=${cli_size} api=${api_size}"

if (( cli_size > max_cli_size )); then
  echo "FAIL: vidarax-cli binary size ${cli_size} > ${max_cli_size}" >&2
  exit 1
fi

if (( api_size > max_api_size )); then
  echo "FAIL: vidarax-api binary size ${api_size} > ${max_api_size}" >&2
  exit 1
fi

probe_json="$(cargo run -q -p vidarax-core --release --bin perf_probe)"
gate_p95_ns="$(jq -r '.gate_process.p95_ns' <<<"$probe_json")"

echo "[gate] gate p95 ns=${gate_p95_ns}"
if (( gate_p95_ns > max_gate_p95_ns )); then
  echo "FAIL: gate p95 ${gate_p95_ns}ns > ${max_gate_p95_ns}ns" >&2
  exit 1
fi

echo "[gate] PASS"

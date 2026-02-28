#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench-provider] running provider transport benchmark"
bench_json="$(cargo run -q -p vidarax-api --bin provider_transport_bench)"
echo "$bench_json"

blocking_rps="$(jq -r '.blocking_spawn.throughput_rps' <<<"$bench_json")"
blocking_p95="$(jq -r '.blocking_spawn.p95_ms' <<<"$bench_json")"
async_rps="$(jq -r '.async_reqwest.throughput_rps' <<<"$bench_json")"
async_p95="$(jq -r '.async_reqwest.p95_ms' <<<"$bench_json")"
recommendation="$(jq -r '.recommendation' <<<"$bench_json")"

mkdir -p "$ROOT_DIR/docs"
cat > "$ROOT_DIR/docs/provider-transport-decision.md" <<EOF
# Provider Transport Decision

- Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- Benchmark source: \`crates/vidarax-api/src/bin/provider_transport_bench.rs\`

## Measured Results

- Blocking (spawn_blocking + current provider path):
  - throughput_rps: ${blocking_rps}
  - p95_ms: ${blocking_p95}
- Async (reqwest async client):
  - throughput_rps: ${async_rps}
  - p95_ms: ${async_p95}

## Decision

\`${recommendation}\`
EOF

echo "[bench-provider] decision written to docs/provider-transport-decision.md"

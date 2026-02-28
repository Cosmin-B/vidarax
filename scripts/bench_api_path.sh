#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench-api] running API hot-path benchmark"
cargo run -q -p vidarax-api --release --bin api_path_bench

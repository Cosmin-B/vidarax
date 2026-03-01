#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
VIDARAX_API="${VIDARAX_API:-http://localhost:8080}"
SPACETIMEDB_URL="${SPACETIMEDB_URL:-http://localhost:3000}"
SPACETIMEDB_DB="${SPACETIMEDB_DB:-vidarax}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
FIRST_PASS="${FIRST_PASS:-Qwen/Qwen3-VL-2B-Instruct}"
SECOND_PASS="${SECOND_PASS:-Qwen/Qwen3-VL-8B-Instruct}"
THRESHOLD="${THRESHOLD:-0.7}"

# Resolve script directory so video generator works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Vidarax E2E Integration Test ==="
echo "API:    $VIDARAX_API"
echo "STDB:   $SPACETIMEDB_URL"
echo "Model:  $MODEL (tiered: $FIRST_PASS → $SECOND_PASS @ $THRESHOLD)"
echo ""

# --- Step 1: Health checks ---
echo "--- Step 1: Health checks ---"

echo -n "vidarax API... "
health=$(curl -sS "$VIDARAX_API/v1/health" 2>&1) || { echo "FAIL: $health"; exit 1; }
echo "OK: $health"

echo -n "vLLM models... "
models=$(curl -sS "$VIDARAX_API/v1/models" 2>&1) || { echo "FAIL: $models"; exit 1; }
model_count=$(echo "$models" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("models",[])))' 2>/dev/null || echo "?")
echo "OK: $model_count models"

echo -n "SpacetimeDB... "
stdb_health=$(curl -sS --max-time 5 "$SPACETIMEDB_URL/v1/database/$SPACETIMEDB_DB/info" 2>&1) && echo "OK" || echo "WARN (continuing anyway)"
echo ""

# --- Step 2: Generate test video ---
echo "--- Step 2: Generate test video ---"
python3 "$SCRIPT_DIR/e2e_test_video.py" /tmp/vidarax-e2e-test.mp4

# Copy to Hetzner if vidarax runs remotely
if [[ "$VIDARAX_API" == *"100.125"* ]]; then
    echo "Copying video to Hetzner..."
    scp -i ~/.ssh/hetzner_linux_new /tmp/vidarax-e2e-test.mp4 ${HETZNER_HOST}:/tmp/vidarax-e2e-test.mp4
fi
echo ""

# --- Step 3: Create run ---
echo "--- Step 3: Create run ---"
create_resp=$(curl -sS -X POST "$VIDARAX_API/v1/runs" \
    -H "Content-Type: application/json" \
    -d "{\"mode\": \"balanced\", \"model\": \"$MODEL\"}")
run_id=$(echo "$create_resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
echo "run_id: $run_id"
echo ""

# --- Step 4: Run /reason ---
echo "--- Step 4: Run /reason (model=$MODEL, tiered: $FIRST_PASS → $SECOND_PASS) ---"
start_ms=$(python3 -c "import time; print(int(time.time() * 1000))")

# Include tiered fields in payload — ignored by current API, honoured once x12.3 lands.
reason_resp=$(curl -sS -X POST "$VIDARAX_API/v1/runs/$run_id/reason" \
    -H "Content-Type: application/json" \
    -d "{
        \"source_uri\": \"file:///tmp/vidarax-e2e-test.mp4\",
        \"model\": \"$MODEL\",
        \"mode\": \"balanced\",
        \"semantic_inference\": true,
        \"semantic_frames_per_chunk\": 2,
        \"chunk_size\": 25,
        \"first_pass_model\": \"$FIRST_PASS\",
        \"second_pass_model\": \"$SECOND_PASS\",
        \"second_pass_threshold\": $THRESHOLD
    }")

end_ms=$(python3 -c "import time; print(int(time.time() * 1000))")
elapsed=$((end_ms - start_ms))

echo "$reason_resp" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(f"  decoded_frames: {d.get(\"decoded_frames\", \"?\")}")
    print(f"  generated:      {d.get(\"generated\", \"?\")}")
    print(f"  markers_emitted:{d.get(\"markers_emitted\", \"?\")}")
    print(f"  lag_p95_ms:     {d.get(\"lag_p95_ms\", \"?\")}")
    print(f"  lag_p99_ms:     {d.get(\"lag_p99_ms\", \"?\")}")
    markers = d.get("markers", [])
    for m in markers:
        conf = m.get("confidence", 0)
        print(f"  marker: {m[\"event_type\"]} [{m.get(\"status\",\"?\")}] confidence={conf:.2f} frames={m.get(\"start_frame\",\"?\")}-{m.get(\"end_frame\",\"?\")}")
except Exception as e:
    print(f"  (parse error: {e})")
    print(sys.stdin.read() if hasattr(sys.stdin, "read") else "")
' 2>/dev/null || echo "  (could not parse /reason response)"
echo "  wall_time_ms: $elapsed"
echo ""

# --- Step 5: Verify markers ---
echo "--- Step 5: Verify markers ---"
markers_resp=$(curl -sS "$VIDARAX_API/v1/runs/$run_id/markers")
marker_count=$(echo "$markers_resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("markers",[])))' 2>/dev/null || echo "0")
echo "  marker_count: $marker_count"

scene_cuts=$(echo "$markers_resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d.get("markers",[]) if m.get("event_type")=="scene_cut"))' 2>/dev/null || echo "0")
echo "  scene_cuts: $scene_cuts"

if [ "${marker_count:-0}" -gt 0 ]; then
    echo "PASS: markers detected"
else
    echo "FAIL: no markers detected"
    exit 1
fi

# --- Step 6: Check SpacetimeDB events ---
echo ""
echo "--- Step 6: Check SpacetimeDB events ---"
stdb_events=$(curl -sS --max-time 10 -X POST "$SPACETIMEDB_URL/v1/database/$SPACETIMEDB_DB/sql" \
    -H "Content-Type: text/plain" \
    -d "SELECT COUNT(*) FROM agent_event" 2>&1) || stdb_events="WARN: query failed"
echo "  SpacetimeDB events: $stdb_events"

stdb_kf=$(curl -sS --max-time 10 -X POST "$SPACETIMEDB_URL/v1/database/$SPACETIMEDB_DB/sql" \
    -H "Content-Type: text/plain" \
    -d "SELECT COUNT(*) FROM keyframe_store" 2>&1) || stdb_kf="WARN: query failed"
echo "  SpacetimeDB keyframes: $stdb_kf"
echo ""

# --- Summary ---
echo "=== E2E Integration Test Complete ==="
echo "  Run ID:      $run_id"
echo "  Wall time:   ${elapsed}ms"
echo "  Markers:     ${marker_count:-0}"
echo "  Scene cuts:  ${scene_cuts:-0}"
echo "  Tiered:      $FIRST_PASS → $SECOND_PASS @ $THRESHOLD"
echo ""
echo "To view in dashboard: open Vue app → /runs/$run_id"

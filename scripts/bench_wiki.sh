#!/usr/bin/env bash
# Benchmark: Wiki.mp4 through vidarax pipeline
set -euo pipefail

API="${VIDARAX_API:-http://localhost:8080}"
VIDEO="${1:-/tmp/vidarax-uploads/Wiki.mp4}"
PAYLOAD_FILE=$(mktemp /tmp/vidarax-bench-XXXXXX.json)

trap "rm -f $PAYLOAD_FILE" EXIT

echo "=== Vidarax Wiki.mp4 Benchmark ==="
echo "Video: $VIDEO ($(du -h "$VIDEO" | cut -f1))"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Stage 1: Create Run
echo "--- Stage 1: Create Run ---"
T0=$(python3 -c "import time; print(time.time())")
RUN_RESP=$(curl -s -X POST "$API/v1/runs" \
  -H "Content-Type: application/json" \
  -d '{"mode": "detailed"}')
T1=$(python3 -c "import time; print(time.time())")
CREATE_MS=$(python3 -c "print(f'{($T1 - $T0)*1000:.0f}')")
RUN_ID=$(echo "$RUN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('run_id', d.get('data',{}).get('run_id','')))")
echo "Run ID: $RUN_ID"
echo "Create time: ${CREATE_MS}ms"
echo ""

if [ -z "$RUN_ID" ]; then
  echo "ERROR: Failed to create run. Response: $RUN_RESP"
  exit 1
fi

# Build reason payload
python3 -c "
import json
payload = {
    'source_uri': 'file://$VIDEO',
    'mode': 'detailed',
    'model': 'auto',
    'sampling_policy': 'fixed',
    'fixed_fps': 2,
    'max_frames': 512,
    'window_size': 16,
    'segment_ms': 500,
    'semantic_inference': True,
    'semantic_frames_per_chunk': 4,
    'semantic_timeout_ms': 30000,
    'semantic_prompt': 'You are analyzing a screen recording of someone browsing a wiki/documentation site. For each moment where the user clicks a button or link, output a JSON object with: timestamp (MM:SS format), button (the text or label of the button/link clicked), hover_duration_seconds (estimated seconds the cursor hovered over the element before clicking). Return ONLY a JSON array of all click events.',
    'output_schema': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'timestamp': {'type': 'string'},
                'button': {'type': 'string'},
                'hover_duration_seconds': {'type': 'number'}
            },
            'required': ['timestamp', 'button', 'hover_duration_seconds']
        }
    }
}
with open('$PAYLOAD_FILE', 'w') as f:
    json.dump(payload, f)
"

# Stage 2: Reason (ingest + gate + VLM all-in-one)
echo "--- Stage 2: Reason (decode + gate + VLM) ---"
T2=$(python3 -c "import time; print(time.time())")
REASON_RESP=$(curl -s --max-time 600 -X POST "$API/v1/runs/$RUN_ID/reason" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE")
T3=$(python3 -c "import time; print(time.time())")
REASON_MS=$(python3 -c "print(f'{($T3 - $T2)*1000:.0f}')")
REASON_S=$(python3 -c "print(f'{($T3 - $T2):.1f}')")
echo "Reason time: ${REASON_MS}ms (${REASON_S}s)"
echo ""

# Stage 3: Get events
echo "--- Stage 3: Retrieve Events ---"
T4=$(python3 -c "import time; print(time.time())")
EVENTS_RESP=$(curl -s "$API/v1/runs/$RUN_ID/events")
T5=$(python3 -c "import time; print(time.time())")
EVENTS_MS=$(python3 -c "print(f'{($T5 - $T4)*1000:.0f}')")
EVENT_COUNT=$(echo "$EVENTS_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
evts=d if isinstance(d,list) else d.get('data',d.get('events',[]))
print(len(evts))
" 2>/dev/null || echo "?")
echo "Events: $EVENT_COUNT"
echo "Events fetch: ${EVENTS_MS}ms"
echo ""

# Summary
TOTAL_MS=$(python3 -c "print(f'{($T5 - $T0)*1000:.0f}')")
TOTAL_S=$(python3 -c "print(f'{($T5 - $T0):.1f}')")
echo "========================================="
echo "BENCHMARK RESULTS"
echo "========================================="
echo "Create Run:  ${CREATE_MS}ms"
echo "Reason:      ${REASON_MS}ms (decode+gate+VLM)"
echo "Events:      ${EVENTS_MS}ms ($EVENT_COUNT events)"
echo "-----------------------------------------"
echo "TOTAL:       ${TOTAL_MS}ms (${TOTAL_S}s)"
echo "========================================="
echo ""

# Print reason response
echo "--- Reason Response (summary) ---"
echo "$REASON_RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    data = d.get('data', d)
    if isinstance(data, dict):
        for k in sorted(data.keys()):
            v = data[k]
            if isinstance(v, list):
                print(f'  {k}: {len(v)} items')
            elif isinstance(v, dict):
                print(f'  {k}: {{...}}')
            else:
                print(f'  {k}: {v}')
except Exception as e:
    print(f'Parse error: {e}')
" 2>/dev/null
echo ""

# Extract structured click events
echo "--- Structured Click Events ---"
echo "$EVENTS_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
evts = d if isinstance(d, list) else d.get('data', d.get('events', []))
clicks = []
for e in evts:
    payload = e.get('payload', {})
    if isinstance(payload, str):
        try: payload = json.loads(payload)
        except: continue
    for key in ['semantic_result', 'vlm_result', 'result', 'semantic_results']:
        sr = payload.get(key)
        if sr:
            if isinstance(sr, str):
                try: sr = json.loads(sr)
                except: pass
            if isinstance(sr, list):
                for item in sr:
                    if isinstance(item, dict) and 'timestamp' in item:
                        clicks.append(item)
                    elif isinstance(item, dict):
                        # Nested: might have text with JSON
                        txt = item.get('text', '')
                        if isinstance(txt, str) and '[' in txt:
                            try:
                                parsed = json.loads(txt[txt.index('['):txt.rindex(']')+1])
                                clicks.extend(parsed)
                            except: pass
            elif isinstance(sr, dict) and 'timestamp' in sr:
                clicks.append(sr)

if clicks:
    print(json.dumps(clicks, indent=2))
else:
    sem = [e for e in evts if 'semantic' in str(e.get('event_type',''))]
    if sem:
        print(f'Found {len(sem)} semantic events (no click structure extracted)')
        for s in sem[:3]:
            print(json.dumps(s, indent=2)[:500])
    else:
        types = list(set(e.get('event_type','?') for e in evts))
        print(f'No click events in {len(evts)} events. Types: {types}')
" 2>/dev/null

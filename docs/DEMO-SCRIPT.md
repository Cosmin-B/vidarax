# Demo Script (3 minutes)

Target audience: hackathon judges, engineers, investors.
Tone: direct, no fluff, show the product working.

---

## 0:00 -- 0:20 | The problem

> "Video agents are blind. They parse DOM, scrape screenshots, miss what they actually see. We built vidarax -- a real-time video intelligence engine that gives any agent eyes. Any stream in, semantic events out."

Screen: vidarax landing page or logo. Keep it brief.

---

## 0:20 -- 0:50 | Dashboard

Open the Dashboard (`/`).

> "This is the command center. Every video stream gets deterministic frame analysis -- scene cuts, artifacts, motion -- at O(1) per frame. No ML needed for pass one. Just math."

Point out:
- Active runs list
- Metrics counters (frames processed, markers emitted)
- Dark UI with real-time updates

---

## 0:50 -- 1:30 | Upload and analyze

Navigate to Upload (`/upload`). Upload a 10-second screen recording.

> "Watch this. Ten-second video going in."

Wait for processing. Navigate to Run Detail (`/runs/:runId`).

> "1.5 seconds. 242 frames decoded, 20 markers extracted. Scene cuts detected at exactly the right transitions."

Click a scene cut marker. Show the keyframe JPEG and the VLM description.

> "Every marker has a keyframe and a semantic description from the VLM. Click any marker, see exactly what the model saw."

Emphasize:
- Wall time: 1.5s for 10s video (6.7x real-time)
- Marker types: scene cuts, keyframe keeps, artifact suspected
- VLM descriptions are generated per-chunk, not per-frame

---

## 1:30 -- 2:10 | Pipeline tracing

Navigate to Tracing (`/tracing`).

> "Every frame flows through a pipeline: WebRTC or file ingest, decode, gate engine, VLM, SpacetimeDB. All observable. All measurable."

Point out:
- Pipeline flow diagram
- Per-stage latency
- Gate engine: 42 nanoseconds p95, zero allocations

> "The gate engine is lock-free. Zero-copy. We run a two-pass sliding window: first pass builds deterministic metadata, second pass refines markers. No frames dropped, no backpressure."

---

## 2:10 -- 2:40 | Tiered model routing

Navigate to Settings (`/settings`).

> "Model selector. We run Qwen 2B for the first pass -- 200 millisecond response times. If confidence drops below threshold, we route to 8B for a second opinion. Intelligent tiered routing."

Show:
- Model dropdown (Qwen3-VL-2B, Qwen3-VL-8B)
- Confidence threshold slider
- Primary/fallback provider config (vLLM / SGLang)

> "Both models run on your own GPU. Self-hosted. Open-source VLMs. No data leaves your infrastructure."

---

## 2:40 -- 3:00 | Closing

Switch back to Dashboard.

> "Self-hosted. Open-source models. Runs on your own hardware. 27 performance optimizations from gate engine to transport layer. Vidarax -- the Kafka of video. Smart routing between your streams and your models."

---

## Pre-demo checklist

- [ ] API server running (`cargo run --release -p vidarax-api`)
- [ ] vLLM running with Qwen3-VL-2B + 8B loaded
- [ ] Frontend running (`cd ui && npm run dev`)
- [ ] 10-second test video ready (3-scene color transitions)
- [ ] Browser open to Dashboard, tabs for each page pre-loaded
- [ ] Screen recording off (no UI lag)
- [ ] Terminal visible for server logs (shows tracing spans in real time)

## Backup talking points

If something breaks during demo:

- **VLM timeout**: "The deterministic pipeline still works without the VLM. Markers are gate-engine generated. VLM adds semantic descriptions on top."
- **Upload fails**: Switch to curl: `curl -X POST http://localhost:8080/v1/runs` then `curl -X POST http://localhost:8080/v1/runs/{id}/ingest -d '{"source_uri": "test.mp4"}'`
- **WebRTC not working**: "WebRTC is for live streams. The upload path uses the same pipeline, just file-based ingest instead of real-time."

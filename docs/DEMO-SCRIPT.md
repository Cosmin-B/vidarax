# Demo runbook (3 minutes)

Audience: engineers, investors. Tone: direct, technical, no fluff.

## 0:00 -- 0:15 | The Problem

> "Video agents often see isolated frames instead of video context. Vidarax is a real-time video intelligence engine -- video in, semantic events out."

Screen: Dashboard page at `/`.

## 0:15 -- 0:40 | Dashboard

Point out the Command Center:
- Recent runs list with status indicators
- Stats row: total runs, processing, events, completed
- Recent events read from the run timeline

## 0:40 -- 1:30 | Upload and Analyze

Navigate to Upload (`/upload`). Drag in a 10-second screen recording.
Select model (Qwen3-VL 4B), leave semantic inference on. Click Start Analysis.
Show the processing visualization: frame strip, chunk progress, live event feed.

> "Done. The gate engine marked scene cuts and keyframes."

Click View Results to open Run Detail (`/runs/:runId`). Show:
- Video player with timeline scrubbing
- Color-coded event timeline (scene cuts, keyframes, VLM descriptions)
- Keyframe gallery with JPEG thumbnails and semantic descriptions

## 1:30 -- 2:10 | Pipeline Tracing

Navigate to Tracing (`/tracing`).

> "The pipeline exposes decode, gate, model, WAL, and optional mirror latency."

Point out: pipeline counters, measured p50/p95/p99 latency, semantic reuse and forced-refresh counts, and binary keyframe-sidecar health. If `/v1/metrics` is unavailable, the page says so instead of showing sample data.

> "The deterministic gate is allocation-free in the measured hot path. The semantic gate fails open, bounds reuse by time and cumulative drift, and exposes shadow-calibration counters."

## 2:10 -- 2:45 | Settings and Model Routing

Navigate to Settings (`/settings`). Show:
- Connection panel with test-connection button
- Tiered model routing: first pass (2B), second pass (4B/8B)
- Stream controls: FPS, chunk size, semantic frames per chunk
- Gate engine tuning: Hamming threshold, luma shift, loop detection

> "Point it at self-hosted models on your own hardware, and video stays inside your infrastructure."

## 2:45 -- 3:00 | Close

> "Vidarax filters repeat work, keeps the event record local, and lets each deployment prove its own latency and quality targets."

## Pre-demo Checklist

- [ ] API server running (`cargo run --release -p vidarax-api`)
- [ ] vLLM running with target models loaded
- [ ] Embedding sidecar running if semantic novelty is part of the demo
- [ ] Novelty threshold calibrated for the selected video and provider
- [ ] Frontend running (`cd ui && npm run dev`)
- [ ] 10-second test video ready
- [ ] Browser tabs pre-loaded: Dashboard, Upload, Tracing, Settings

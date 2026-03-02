# Demo Script (3 minutes)

Audience: engineers, investors. Tone: direct, technical, no fluff.

## 0:00 -- 0:15 | The Problem

> "Video agents are blind. They scrape screenshots and miss context. Vidarax is a real-time video intelligence engine -- any video in, semantic events out."

Screen: Dashboard page at `/`.

## 0:15 -- 0:40 | Dashboard

Point out the Command Center:
- Recent runs list with status indicators
- Stats row: total runs, processing, events, completed
- Live events feed streaming from SpacetimeDB

## 0:40 -- 1:30 | Upload and Analyze

Navigate to Upload (`/upload`). Drag in a 10-second screen recording.
Select model (Qwen3-VL 4B), leave semantic inference on. Click Start Analysis.
Show the processing visualization: frame strip, chunk progress, live event feed.

> "Done. Scene cuts detected at exactly the right transitions."

Click View Results to open Run Detail (`/runs/:runId`). Show:
- Video player with timeline scrubbing
- Color-coded event timeline (scene cuts, keyframes, VLM descriptions)
- Keyframe gallery with JPEG thumbnails and semantic descriptions

## 1:30 -- 2:10 | Pipeline Tracing

Navigate to Tracing (`/tracing`).

> "Every frame is observable. Ingest, decode, gate engine, VLM, SpacetimeDB."

Point out: pipeline flow diagram, live metrics grid (2s refresh), trace timeline waterfall.

> "Gate engine is lock-free Rust. Two-pass sliding window -- deterministic metadata first, refined markers second."

## 2:10 -- 2:45 | Settings and Model Routing

Navigate to Settings (`/settings`). Show:
- Connection panel with test-connection button
- Tiered model routing: first pass (2B), second pass (4B/8B)
- Stream controls: FPS, chunk size, semantic frames per chunk
- Gate engine tuning: Hamming threshold, luma shift, loop detection

> "All models run on your GPU. Self-hosted. No data leaves your infrastructure."

## 2:45 -- 3:00 | Close

> "Open VLMs. Deterministic gate engine. Tiered inference. Vidarax -- the intelligent routing layer between your video and your models."

## Pre-demo Checklist

- [ ] API server running (`cargo run --release -p vidarax-api`)
- [ ] vLLM running with target models loaded
- [ ] Frontend running (`cd ui && npm run dev`)
- [ ] 10-second test video ready
- [ ] Browser tabs pre-loaded: Dashboard, Upload, Tracing, Settings

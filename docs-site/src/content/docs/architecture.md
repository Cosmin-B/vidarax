---
title: Architecture
description: The control plane on tokio, the media plane on OS threads, event sinks, and WAL persistence.
---

Vidarax separates request handling from media processing. HTTP and control logic run on the tokio async runtime. Decode, analysis, and VLM work runs on dedicated OS threads connected by bounded queues. Worker output commits to the local write-ahead log. When SpacetimeDB is configured, blocking WHIP description events are mirrored only after that local commit.

```
  Sources                        vidarax                         Consumers
┌──────────┐   ┌──────────────────────────────────────────┐   ┌──────────────┐
│ MP4/File │──>│                                          │──>│ REST API     │
│ WebRTC   │──>│  Decode ──> Gate Engine ──> VLM Tiering  │──>│ TypeScript   │
│ RTSP/HLS │──>│               │                │         │──>│   SDK        │
│ Upload   │──>│          Markers +         Semantic      │──>│ Vue 3 UI     │
│          │   │          Keyframes         Events        │   │ Prometheus   │
│          │   │               │                │         │   │ Optional     │
│          │   │            WAL event log      │          │   │ SpacetimeDB  │
└──────────┘   └──────────────────────────────────────────┘   └──────────────┘
```

## The control plane

The HTTP server is Axum over Hyper, on tokio, speaking HTTP/1.1 and HTTP/2 with optional HTTP/3 behind the `h3-experimental` build feature. It owns:

- Routing, request validation, and the shared error envelope.
- API-key authentication and principal resolution. Ownership of runs and files derives from the authenticated principal.
- Rate limiting, globally and per resolved principal.
- Run lifecycle: create, list, stop, keepalive, delete, and reads of events, markers, and derived state.
- The provider chain for inference backends: OpenAI-compatible vLLM and SGLang endpoints tried in priority order with fallback, or backends declared in a TOML config file.

## The media plane

The media plane splits by workload. WebRTC ingress is async: the session event loop and the per-track tasks that receive, depacketize, and enqueue RTP frames run as tokio tasks. The processing stages are blocking OS threads: each WebRTC session gets a decode worker and, depending on the mode, either VLM workers (keyframe mode) or an analysis worker, a clip accumulator, and clip VLM workers (clip mode). The stages are connected by bounded `kanal` queues, so every handoff has explicit backpressure. Closing an upstream sender propagates shutdown to the downstream threads.

One ordered stream uses one stateful decoder, and the analysis and VLM stages own stream-order state, so the per-stream worker count for each stage is clamped to one. Parallelism comes from running many sessions, not from splitting one ordered stream.

Decoding for file and URL sources goes through a pluggable backend registry with two phases: a frame-signal pass that computes per-frame statistics for the gate engine, then selective JPEG extraction for only the frames the gate keeps. See [Ingest](/docs/ingest/) for the decode paths and [The gate](/docs/gate/) for what happens to each frame.

## Event sinks

Worker threads report results through an `EventSink` trait rather than writing storage directly. Live sessions use the WAL-backed implementation:

- It bridges worker events into the API timeline, so live VLM results appear in `GET /v1/runs/{id}/events` without an external database. Appends funnel through a bounded channel into the single timeline-writer thread, which assigns sequence numbers and swaps the registry snapshot.
- `store_keyframe_sync` writes raw JPEG bytes to the content-addressed blob sidecar before appending a `keyframe_stored` metadata event. The WAL never carries JSON-encoded or base64 image bytes.
- When `VIDARAX_SPACETIMEDB_URL` is set, successful blocking description events are mirrored after the WAL commit and feedback endpoints are enabled. Mirror failure is logged and does not roll back local durability. Nonblocking events and raw keyframes remain local.

## How state is persisted

The durable store is a write-ahead log at `${VIDARAX_DATA_DIR}/timeline.wal` (data directory default: `.vidarax-data`). Its properties:

- Append-only plain text, one event per line, tab-separated with escaped fields. JPEG bytes live under `${VIDARAX_DATA_DIR}/keyframes/blobs/`; the WAL stores their relative reference, media type, size, and SHA-256.
- Each event carries a monotonic sequence number, a run ID, a stream ID, a presentation timestamp, a kind, and a JSON payload.
- The file is created with owner-only read and write permissions on Unix.

Run state is not stored as a mutable row anywhere. `GET /v1/runs/{id}/state` derives the current state by replaying the run's persisted events, and deletion is soft: `DELETE /v1/runs/{id}` appends a `run_deleted` event. Recently appended runs keep an in-memory tail of their events, so those reads are served from memory; when a run falls out of that set, reads fall back to WAL replay with the same cursor order.

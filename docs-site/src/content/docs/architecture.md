---
title: Architecture
description: The control plane on tokio, the media plane on OS threads, event sinks, and WAL persistence.
---

Vidarax keeps request handling away from blocking media work. HTTP and control
logic run on the tokio async runtime. Decode, analysis, and VLM work run on
dedicated OS threads connected by bounded queues. Each live session is one
typed pipeline generation supervised as a unit. Worker output commits to the
local write-ahead log before any optional mirror or delivery path observes it.

```
 Sources          Per-session generation          Durable state       Delivery
┌──────────┐   ┌────────────────────────────────┐   ┌─────────────────┐   ┌─────────────┐
│ MP4/File │──>│ Decode -> Frame filter -> VLM  │──>│ WAL event log   │──>│ REST / SSE  │
│ WebRTC   │──>│              |                 │   │                 │   │ Webhooks    │
│ RTSP/HLS │──>│          Trigger VM            │   │ Binary media    │   │ TypeScript  │
│ Upload   │──>│              |                 │   │ sidecar         │   │ SDK / UI    │
│          │   │        supervised generation     │   │                 │   │ Prometheus  │
└──────────┘   └────────────────────────────────┘   └─────────────────┘   └─────────────┘

 Signed release manifest -> edge updater -> shadow -> canary -> active model
```

## The control plane

The HTTP server is Axum over Hyper, on tokio, speaking HTTP/1.1 and HTTP/2 with optional HTTP/3 behind the `h3-experimental` build feature. It owns:

- Routing, request validation, and the shared error envelope.
- API-key authentication and principal resolution. Ownership of runs and files derives from the authenticated principal.
- Rate limiting, globally and per resolved principal.
- Run lifecycle: create, list, stop, keepalive, delete, and reads of events, markers, and derived state.
- The provider chain for inference backends: OpenAI-compatible vLLM and SGLang endpoints tried in priority order with fallback, or backends declared in a TOML config file.
- One process-wide inference scheduler that fair-queues streams and principals, prioritizes urgent live work over ordinary live and offline work, rejects work that cannot fit its token/encoded-media budget, and stops waiting once a request can no longer meet its absolute deadline. Provider servers still own token-level batching.

## The media plane

The media plane splits by workload. WebRTC ingress is async: the session event loop and the per-track tasks that receive, depacketize, and enqueue RTP frames run as tokio tasks. The processing stages are blocking OS threads: each WebRTC session gets a decode worker and, depending on the mode, either VLM workers (keyframe mode) or an analysis worker, a clip accumulator, and clip VLM workers (clip mode). The stages are connected by bounded `kanal` queues, so every handoff has explicit backpressure. Closing an upstream sender propagates shutdown to the downstream threads.

One ordered stream uses one stateful decoder, and the analysis and VLM stages own stream-order state, so the per-stream worker count for each stage is clamped to one. Parallelism comes from running many sessions, not from splitting one ordered stream.

Decoding for file and URL sources goes through a pluggable backend registry with two phases: a frame-signal pass that computes statistics for the per-frame filter, then selective JPEG extraction for only the frames the filter keeps. See [Ingest](/docs/ingest/) for the decode paths and [The per-frame filter](/docs/gate/) for what happens to each frame.

## Session generations and control

The stages of a live stream do not fail or restart independently. `PipelineRuntime` owns a stage-tagged join handle for every worker in one process-unique `PipelineGeneration`. The first unexpected exit faults that generation, raises its monotonic stop signal, closes the WebRTC peer, and gives every sibling a join deadline derived from the VLM request timeouts, so ordinary teardown during an in-flight inference call is not misjudged. A generation that exceeds that deadline is reported as a forced shutdown. Its stragglers keep running detached and are counted in `vidarax_pipeline_detached_workers_total`, and the session's media reservation is kept because those threads still hold that memory. Vidarax never restarts a decoder or VLM worker (including its inline novelty check) underneath temporal state from the old generation.

```
                        generation N
                  ┌────────────────────┐
 stage exits ─────>│ supervisor         │─────> close peer
                  │ stop + join set    │─────> fault metrics
                  └────────────────────┘

 PATCH prompt ────> bounded command[N] ──────> VLM owner
       200 <────── worker acknowledgement <───┘
```

The VLM worker owns the live prompt and output schema. `PATCH
/v1/stream/whip/:sess/prompt` sends an eight-slot, generation-tagged command and
waits up to two seconds for the worker acknowledgement. A closed or replaced
generation returns 409. An acknowledgement timeout returns 503, and the worker
drops that cancelled command.

Before `run_created` is appended, process-wide admission reserves a conservative byte and worker-thread envelope for the negotiated generation. The calculation includes bounded RTP input, decoded and JPEG pools, decode and provider scratch space, and a 64 MiB allowance for an ffmpeg sidecar when used. `VIDARAX_MEDIA_MEMORY_BUDGET_BYTES` and `VIDARAX_MEDIA_WORKER_THREAD_BUDGET` cap the sum across sessions. If either reservation cannot fit, creation returns 503 without leaving a durable run behind.

Provider calls have a second process-wide budget for concurrency, queued
callers, output tokens, and encoded media bytes. Each request carries its stream
identity, latency class, deadline, and service estimate. The scheduler preserves
order within a stream and finds parallelism across streams.

H.264 and H.265 use an ffmpeg child process so a native decoder crash cannot
abort the API process. The supervisor owns the Rust stages and decoder teardown.
An OS thread cannot safely force-kill another OS thread. A native child that
outlives normal teardown becomes a measured join-deadline fault. See [Media
plane](/docs/internals/media-plane/) and [Decode
sidecar](/docs/internals/decode-sidecar/) for the detailed behavior.

## Event sinks

Worker threads report results through an `EventSink` trait. The WAL-backed
implementation owns storage writes for live sessions:

- It bridges worker events into the API timeline, so live VLM results appear in `GET /v1/runs/{id}/events` without an external database. Appends funnel through a bounded channel into the single timeline-writer thread, which assigns sequence numbers and swaps the registry snapshot.
- `store_keyframe_sync` writes raw JPEG bytes to the content-addressed blob sidecar before appending a `keyframe_stored` metadata event. The WAL never carries JSON-encoded or base64 image bytes.
- Frame and keyframe events carry `coordinate_schema: "vidarax.image.v1"` plus source dimensions, the requested normalized crop, the exact resolved pixel region, and the analyzed extent. The contract describes image coordinates, not camera extrinsics or a robot/world transform.
- Operator feedback, policy revisions, deployments, rollbacks, and replay
  evaluations commit to the same local WAL as media events. Immutable events
  reconstruct their current state.
- When `VIDARAX_SPACETIMEDB_URL` is set, successful blocking description events and feedback are mirrored after the WAL commit. Mirror failure is logged and does not roll back local durability. Nonblocking events and raw keyframes remain local.

## Edge update loop

The first edge package runs this same pipeline beside a local model server. An
enrolled device pins an Ed25519 public key, a hardware cohort, and an activation
hook. It streams a signed binary artifact to private local storage, verifies the
declared length and SHA-256, evaluates shadow and canary health reports, and
changes the current model only after the serving hook acknowledges that exact
release. Each staged transition is journaled and acknowledged, and a failed
candidate is removed only after the hook acknowledges rollback. Network loss
stops updates while the active pipeline keeps running. See [Edge
deployment](/docs/edge/).

## How state is persisted

The durable store is a write-ahead log at `${VIDARAX_DATA_DIR}/timeline.wal` (data directory default: `.vidarax-data`). Its properties:

- Append-only plain text, one event per line, tab-separated with escaped fields. JPEG bytes live under `${VIDARAX_DATA_DIR}/keyframes/blobs/`. The WAL stores their relative reference, media type, size, and SHA-256.
- Each event carries a monotonic sequence number, a run ID, a stream ID, a presentation timestamp, a kind, and a JSON payload.
- The file is created with owner-only read and write permissions on Unix.

Blob creation happens before the referencing WAL append. A crash between those steps can leave an unreferenced blob. Reads remain consistent because no event points at missing bytes, but automatic orphan reconciliation is not implemented yet.

Run state is not stored as a mutable row anywhere. `GET /v1/runs/{id}/state` derives the current state by replaying the run's persisted events, and deletion is soft: `DELETE /v1/runs/{id}` appends a `run_deleted` event. Recently appended runs keep an in-memory tail of their events, so those reads are served from memory. When a run falls out of that set, reads fall back to WAL replay with the same cursor order.

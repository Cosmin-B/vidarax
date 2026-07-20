---
title: Events and SDK
description: Event shapes and kinds, per-kind payloads, markers, query and search, and the TypeScript SDK.
---

Everything Vidarax learns about a video becomes an event on a run's timeline. The local WAL is authoritative and the API serves its current state. When SpacetimeDB is configured, blocking WHIP description events are mirrored after the local commit; mirror failures do not remove events from the WAL. Nonblocking events and raw keyframes remain local.

## Event shape

A timeline event has a fixed envelope with a kind-specific payload:

```json
{
  "seq": 17,
  "pts_ms": 1764316800123,
  "kind": "marker_emitted",
  "payload": { }
}
```

- `seq` is a monotonic sequence number, unique per server, usable as a cursor.
- `pts_ms` on the envelope is the wall-clock time the event was appended, in epoch milliseconds. It is stamped by the timeline writer, not taken from the media. Media-relative timestamps live inside payloads: worker events carry their own `pts_ms` payload field, and markers carry `start_pts_ms` and `end_pts_ms`.
- `kind` names the event type.
- `payload` is a JSON object whose fields depend on the kind.

`GET /v1/runs/{id}/events` returns `{ request_id, run_id, events }` with events in sequence order, and accepts `?index=<name>` to filter to events whose payload carries a matching `index_name`.

## Event kinds and payloads

Run lifecycle and analysis kinds written by the API handlers, with the payload fields each one carries:

| Kind | Meaning | Payload fields |
|------|---------|----------------|
| `run_created` | The run was created. | `request_id`, `mode`, `model`, `principal_key`, `tenant_id`; WHIP sessions write `principal_key`, `session_id`, `source: "whip"` instead |
| `ingest_received` | An ingest request was accepted for the run. | `request_id`, `ingest` (echo of the request body); decoded sources add `decoded_frames`, `source_uri`, `sampling_policy`, `sample_fps` |
| `frames_decoded` | A decode pass finished and reported its frames. | `request_id`, `source_uri`, `stream_id`, `sampling_policy`, `source_fps`, `sample_fps`, `decoded_frames`, `width`, `height`, `pixel_format`, `coordinate_schema`, `coordinates`, `signals` (per-frame signal array) |
| `marker_emitted` | The analysis pass produced a marker; the payload is the marker object. | `marker_id`, `run_id`, `stream_id`, `event_type`, `status`, `start_frame`, `end_frame`, `start_pts_ms`, `end_pts_ms`, `confidence`, `supersedes_marker_id` |
| `analysis_generated` | A deterministic analysis pass produced its result. | `request_id`, `stream_id`, `frames`, `window_size`, `segment_ms`, `sampling_policy`, `sample_fps`, `mode`, `model`, `markers` |
| `semantic_chunk_inferred` | A chunk finished tiered VLM inference. | `request_id`, `stream_id`, `chunk_index`, `provider`, `provider_fallback_used`, `semantic_fallback_used`, `semantic_error`, `finish_reason`, `response_chars`, `event_type`, `object_label`, `summary`, `description`, `confidence`, `raw_output`, token counts (`prompt_tokens`, `completion_tokens`, `thinking_tokens`, `total_tokens`), `inference_latency_ms`, optional `index_name` |
| `semantic_chunk_generated` | A semantic result for a chunk was recorded. | `request_id`, `stream_id`, `chunk_index`, `chunk_frames`, `process_ms`, `source_span_ms`, `lag_ms`, `index_name`, token counts, `inference_latency_ms` |
| `semantic_fallback_activated` | The semantic path fell back (for example, no provider). | `request_id`, `stream_id`, `reason` |
| `inference_completed` | A direct inference request completed. | `request_id`, `provider`, `model`, `fallback_used`, `prompt_bytes`, `output_bytes` |
| `run_completed` | The run reached a terminal state. | `request_id`, `stream_id`, `frames`, `markers`, `index_name` |
| `stop_requested` | A graceful stop was requested. | `request_id` |
| `keepalive_refreshed` | The run's idle TTL was refreshed. | `request_id` |
| `run_deleted` | The run was soft-deleted. | `request_id` (WHIP reclaim and tombstone paths carry reclaim metadata instead) |

With concurrent semantic inference, `semantic_chunk_inferred` records are appended as chunks finish and can therefore arrive out of `chunk_index` order. Use the WAL sequence for observation order and `chunk_index` for source order.

Live sessions add streaming kinds through the event sink. The worker's `event_type` string becomes the WAL `kind`, and all of them share one payload shape, `{ session_id, frame_index, pts_ms, coordinate_schema, coordinates, confidence, description }`, where this `pts_ms` is media time:

| Kind | Emitted by |
|------|------------|
| `vlm` / `vlm_tiered` | Keyframe VLM worker; the tiered suffix means the second pass answered |
| `clip_vlm` / `clip_vlm_tiered` | Clip VLM worker |
| `state_transition` | VLM worker, when consecutive descriptions diverge past the word-overlap threshold |
| `loop_detected` | Gate or analysis worker, once per loop entry |
| `keyframe_stored` | The sink's keyframe path. The payload includes `frame_index`, `pts_ms`, `coordinate_schema`, `coordinates`, `event_type`, `description`, `image_ref`, `image_media_type`, `image_bytes`, and `image_sha256`. Raw JPEG bytes live in the content-addressed sidecar, not in JSON or the WAL. |

The outer payload remains add-only and has no schema negotiation. Spatial metadata is different: `coordinate_schema: "vidarax.image.v1"` versions the meaning of the nested `coordinates` object. Consumers should still tolerate unknown fields.

## Image coordinate contract

`vidarax.image.v1` describes the transform from the source video frame to the pixels Vidarax analyzed. Pixel coordinates start at the source image's top-left; `x` increases right and `y` increases down. Normalized coordinates use the same origin and axes in `[0, 1]`.

```json
{
  "coordinate_schema": "vidarax.image.v1",
  "coordinates": {
    "source_extent": { "width": 1920, "height": 1080 },
    "requested_region": { "x": 0.25, "y": 0.1, "width": 0.5, "height": 0.5 },
    "resolved_region": { "x": 480, "y": 108, "width": 960, "height": 540 },
    "analysis_extent": { "width": 960, "height": 540 }
  }
}
```

- `source_extent` is the decoded extent before crop or resize.
- `requested_region` preserves the caller's normalized crop.
- `resolved_region` is the exact even-aligned source-pixel rectangle used by the 4:2:0 pipeline.
- `analysis_extent` is the post-crop extent inspected by the deterministic filter. `semantic_frame_max_edge` may preserve that region while resizing the JPEG sent to a model; this schema does not claim the model transport's pixel dimensions.

The value is carried as fixed-size frame metadata. It does not own image bytes or allocate when copied between stages. This is image-space provenance, not a camera-extrinsics or robot-world transform; embodied consumers must attach those calibration transforms downstream.

## Markers

Markers are frame-range annotations derived from the analysis pass, exposed as their own timeline at `GET /v1/runs/{id}/markers` with `status`, `event_type`, `from_frame`, and `to_frame` filters. They are not raw gate decisions; the server derives them in two steps:

1. Each analyzed frame gets an `event_type` from `compose_frame_metadata`: `scene_cut` (hard transition), `artifact_suspected` (elevated temporal artifact signal), `keyframe_keep` (frame retained by the deterministic gate), or `context_observation` (no hard trigger). When semantic inference ran, the model's normalized event type takes precedence over the deterministic one, and tenant label maps can rename the final label.
2. Consecutive frames with the same event type are merged into segments, and each segment becomes a marker with a confidence averaged over its frames (clamped to [0, 1]).

Each marker has a `status` with three values:

- `exact`: the segment's confidence met the threshold (default 0.7), or its event type is `scene_cut`, which is always exact.
- `provisional`: the segment's confidence was below the threshold.
- `finalized`: a correction marker. When a provisional segment is followed by another segment of the same event type within the correction window (default 3 frames, settable per request with `marker_correction_window_frames`), the server emits an additional marker spanning both segments, with the averaged confidence and `supersedes_marker_id` pointing at the provisional marker it replaces.

The filters compose as range overlap: `from_frame` matches markers whose `end_frame` is at or past it, `to_frame` matches markers whose `start_frame` is at or before it, and `status` and `event_type` are exact matches. Results are sorted by `start_frame`, then `end_frame`, then `marker_id`.

The reference fixtures for frame metadata and processing configuration are validated against the published JSON Schemas, `schemas/frame-metadata.schema.json` and `schemas/processing-config.schema.json`, by the replay release check (`scripts/validate_replay_and_schema.sh`). That test validates the checked-in fixtures; it does not validate the server's live output against the schemas.

## Query and search

Two endpoints read across runs:

- `POST /v1/query` filters events by `run_id` (required, ownership-checked), optional `kind`, and a `from_seq` cursor, which is the polling primitive for consumers that track their position. It returns `{ request_id, query, matches }`.
- `POST /v1/search` runs a substring search over stored VLM descriptions. Its exact contract: the query is trimmed and must be 1 to 1,024 bytes; `limit` defaults to 50 and must be in [1, 500]; matching is case-insensitive and looks only at the `description` field of each event payload, falling back to `summary`; without a `run_id` the scan covers only runs owned by the calling principal; with a `run_id` the run must exist, be owned, and not be deleted. Hits come back with their sequence numbers, run IDs, media timestamps, kinds, and optional `index_name`, ordered by sequence, plus `scanned` and `total_hits` counts.

## The TypeScript SDK

The SDK is the `vidarax` package in `packages/vidarax-sdk/` in the repository (not published to a registry as of this writing; build it from the checkout). It requires Node.js 18 or newer, or a modern browser.

```typescript
import { Vidarax } from 'vidarax'

const v = new Vidarax('http://localhost:8080', { apiKey: 'your-key' })

const run = await v.analyze('/srv/vidarax-media/demo.mp4')

for (const event of await v.getEvents(run.runId)) {
  console.log(event.kind, event.payload)
}
```

Constructor options: `apiKey` (sent as `x-api-key` on every request), `maxRetries` and `retryBaseDelayMs` (retry policy for transient failures, with a growing back-off delay), and `timeoutMs` (per-request timeout).

The full public surface:

| Method | Description |
|---|---|
| `analyze(source, opts?)` | High-level: upload if given a `File`, create a run, ingest, analyze, return a handle with `events()` and `markers()` iterators. |
| `createRun(opts?)` / `listRuns()` | Create or list runs. |
| `getRun(id)` / `deleteRun(id)` | Fetch or soft-delete a run. |
| `stopRun(id)` / `keepaliveRun(id)` | Request a graceful stop; refresh the idle TTL. |
| `getRunState(id)` | Derived run state as a string. |
| `ingestRun(id, opts)` | Attach a source and decode frames. |
| `analyzeRun(id, opts)` | Run analysis on ingested frames. |
| `reason(id, opts)` | Realtime semantic reasoning over a source, including `semantic_prompt`. |
| `getEvents(id, index?)` / `getMarkers(id, query?)` | Fetch the current event list or filtered marker list. |
| `getInteractions(id, index?)` | Fetch guided semantic interactions derived from chunk events. |
| `getKeyframe(id, sha256)` | Fetch a referenced keyframe as a raw JPEG `Blob`. |
| `streamEvents(id, index?)` / `streamMarkers(id, query?)` | Async iterators over one fetched snapshot. |
| `query(request)` | Cross-run event query with a `from_seq` cursor. |
| `search(query, opts?)` | Substring search over VLM descriptions, with optional `run_id` and `limit`. |
| `infer(opts)` / `inferBatch(requests, opts?)` | Single or batch inference. |
| `uploadFile(file, onProgress?)` | Upload a video file; returns the server-side path. |
| `submitFeedback(runId, feedback)` / `listFeedback()` | Feedback endpoints (require the SpacetimeDB integration server-side). |
| `whipOffer(sdp, opts?)` | WebRTC WHIP session setup (browser). |
| `whipIce(sessionId, candidate)` | Trickle a single ICE candidate. |
| `whipUpdatePrompt(sessionId, config)` | Update a live session's prompt and output schema. |
| `whipTerminate(sessionId)` | End a WebRTC session. |
| `listModels()` / `health()` / `waitUntilHealthy(opts?)` | Model catalog and health checks. |

`streamEvents` and `streamMarkers` are convenience iterators over a single fetched snapshot, not push streams. Live consumers poll `query()` with a `from_seq` cursor. Structured inference and live prompt updates accept `output_schema` as a JSON Schema object; callers do not stringify it first.

All SDK errors extend `VidaraxError`, with subclasses `HttpError`, `NetworkError`, `RetryExhaustedError`, `UploadError`, and `ParseError`.

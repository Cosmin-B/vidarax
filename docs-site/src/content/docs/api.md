---
title: API reference
description: Per-route contracts, the error envelope, and server configuration reference.
---

The API is served over HTTP/1.1 and HTTP/2, with optional HTTP/3 behind the `h3-experimental` build feature. All routes sit under `/v1`.

## Conventions

- Auth. When API keys are enabled (the default), every route except `GET /v1/health` requires an `x-api-key` header; `GET /v1/metrics` has its own toggle (`VIDARAX_METRICS_REQUIRE_API_KEY`). Missing or invalid keys return 401 `unauthorized`.
- Ownership. Runs and uploaded files belong to the authenticated principal. A run owned by a different principal returns 404 `not_found`, indistinguishable from a missing run. `x-tenant-id` is metadata, not an authorization boundary.
- Rate limits. The global limiter (when configured) applies to every request, including health checks; the per-principal limiter applies to authenticated routes. Both return 429 `rate_limited`.
- Request IDs. Handler-generated JSON bodies usually carry a string `request_id` (format `req-` plus 16 hex digits). The health check, run list, upload response, interaction response, file serving, and WHIP routes are exceptions documented below.
- Errors use the JSON envelope described [below](#error-envelope), except for the routes explicitly marked as returning raw bodies.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/runs` | Create a new analysis run |
| `GET` | `/v1/runs` | List runs |
| `GET` | `/v1/runs/:id` | Get run details |
| `DELETE` | `/v1/runs/:id` | Delete a run |
| `POST` | `/v1/runs/:id/ingest` | Ingest and decode a video source |
| `POST` | `/v1/runs/:id/analyze` | Deterministic frame analysis |
| `POST` | `/v1/runs/:id/reason` | Realtime semantic reasoning (tiered VLM) |
| `POST` | `/v1/runs/:id/stop` | Stop a run |
| `POST` | `/v1/runs/:id/keepalive` | Refresh active run TTL |
| `GET` | `/v1/runs/:id/events` | Read run events |
| `GET` | `/v1/runs/:id/markers` | Marker timeline (filterable) |
| `GET` | `/v1/runs/:id/state` | Derived run state |
| `GET` | `/v1/runs/:id/interactions` | Interaction timeline |
| `POST` | `/v1/runs/:id/feedback` | Submit feedback for a run |
| `GET` | `/v1/feedback` | List feedback |
| `POST` | `/v1/query` | Query events across runs |
| `POST` | `/v1/search` | Search VLM descriptions |
| `POST` | `/v1/infer` | Single VLM inference |
| `POST` | `/v1/infer/batch` | Batch inference (bounded parallelism) |
| `GET` | `/v1/models` | Model catalog with availability |
| `POST` | `/v1/stream/whip` | WHIP WebRTC offer (RFC 9725) |
| `PATCH` | `/v1/stream/whip/:sess` | ICE trickle candidate |
| `DELETE` | `/v1/stream/whip/:sess` | Terminate WebRTC session |
| `PATCH` | `/v1/stream/whip/:sess/prompt` | Update live-session prompt |
| `POST` | `/v1/upload` | Upload a file for processing |
| `GET` | `/v1/files/:filename` | Serve an uploaded or allowed-root file |
| `GET` | `/v1/runs/:id/keyframes/:sha256` | Serve a run-owned keyframe as raw JPEG |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/metrics` | Prometheus-compatible metrics |

## Route contracts

### Run lifecycle

| Route | Request | Success (200) | Failures | Side effects |
|---|---|---|---|---|
| `POST /v1/runs` | `{ mode?, model? }`. `mode` is one of `balanced`, `detailed`, `efficiency`, `custom` (default `balanced`); `model` must be in the supported model contract. | `{ run_id, request_id, status: "pending", mode, model }` | 409 active stream limit, 422 validation, 500 | Appends `run_created` |
| `GET /v1/runs` | none | JSON array of `{ run_id, status, mode, model, source_uri, created_at, updated_at }`, caller-owned runs only, deleted runs excluded, ordered by creation time | 500 | none |
| `GET /v1/runs/:id` | none | The same run summary object | 404, 422, 500 | none |
| `DELETE /v1/runs/:id` | none | `{ request_id, run_id }`; repeat deletes stay 200 while the deletion record is retained | 404, 422, 500 | Appends `run_deleted` once per run via the single-winner claim |
| `POST /v1/runs/:id/stop` | none | `{ request_id, run_id, status: "cancelled" }` | 404, 409 already terminal, 422, 500 | Appends `stop_requested` |
| `POST /v1/runs/:id/keepalive` | none | `{ request_id, run_id, state: "processing" }` | 404, 409 terminal run, 422, 500 | Appends `keepalive_refreshed` |
| `GET /v1/runs/:id/state` | none | `{ request_id, run_id, state }` (state derived by replaying events) | 404, 422, 500 | none |

### Ingest and analysis

| Route | Request | Success (200) | Failures | Side effects |
|---|---|---|---|---|
| `POST /v1/runs/:id/ingest` | `{ source_uri, sampling_policy?, fixed_fps?, sample_fps?, max_frames?, stream_id? }`. `source_uri` is required and must resolve under an ingest root or the upload root. `sampling_policy` is `source_fps_adaptive` (default) or `fixed`; `fixed_fps` must be in [0.2, 120] and is required for `fixed`; `max_frames` is in [1, 500000], default 512. Unknown fields are rejected. | `{ request_id, run_id, status: "processing", decoded_frames, source_uri, sampling_policy, source_fps, sample_fps }` | 404, 409 terminal run, 422 (including source validation), 500 | Appends `ingest_received` and `frames_decoded` |
| `POST /v1/runs/:id/analyze` | `{ model, mode?, stream_id?, sampling_policy?, fixed_fps?, frames?, window_size?, segment_ms?, trace_id? }`. Supply 1 to 4096 frames with normalized scores, or omit `frames` to reuse signals from the run's latest `frames_decoded` event. `window_size` is in [2, 256]; `segment_ms` is in [50, 60000]. | `{ request_id, run_id, generated, metadata[], markers[] }` | 404, 409 terminal run, 422, 500 | Appends one `marker_emitted` per marker, then `analysis_generated` |
| `POST /v1/runs/:id/reason` | `{ source_uri, model, ... }` with `chunk_size` in [5, 500], `window_size` in [2, 256], `segment_ms >= 1`, `max_frames` in [1, 500000], `semantic_inference?`, `semantic_frames_per_chunk` in [1, 4], `semantic_frame_max_edge` in [64, 4096], `crop?: { x, y, width, height }` as normalized fractions, `semantic_timeout_ms` in [100, 120000], `semantic_prompt` up to 4096 bytes, `output_schema?`, `first_pass_model?`, `second_pass_model?`, `second_pass_threshold?`, `index_name?`, `temporal_chain?`, `visual_diff?`, `video_clip_mode?`, `video_clip_duration_s` in (0, 60], `vlm_concurrency?` clamped to [1, 64] | `{ request_id, run_id, generated, markers_emitted, decoded_frames, sample_fps, lag_p95_ms, lag_p99_ms, tokens, metadata[], markers[] }` | 404, 409 terminal run, 422, 500 | Appends `semantic_chunk_inferred` and `semantic_chunk_generated` per chunk, `marker_emitted` per marker, `semantic_fallback_activated` when semantic inference was requested with no provider, and `run_completed` |

### Reading events and markers

| Route | Request | Success (200) | Failures |
|---|---|---|---|
| `GET /v1/runs/:id/events` | `?index=<name>` optional payload filter | `{ request_id, run_id, events[] }` in sequence order | 404, 422, 500 |
| `GET /v1/runs/:id/markers` | `?status`, `?event_type`, `?from_frame`, `?to_frame` | `{ request_id, run_id, markers[] }` sorted by frame range | 404, 422, 500 |
| `GET /v1/runs/:id/interactions` | `?index=<name>` optional | `{ run_id, count, interactions[] }` derived from semantic chunk events | 404, 422, 500 |
| `POST /v1/query` | `{ run_id, kind?, from_seq? }`; `run_id` is required and ownership-checked | `{ request_id, query, matches[] }` | 404, 422, 500 |
| `POST /v1/search` | `{ query, run_id?, limit? }`; query trimmed, 1 to 1024 bytes; limit in [1, 500], default 50 | `{ request_id, scanned, total_hits, hits[] }`; case-insensitive substring over payload `description` (fallback `summary`); scoped to owned runs when `run_id` is absent | 404, 422, 500 |

### Inference

| Route | Request | Success (200) | Failures | Side effects |
|---|---|---|---|---|
| `POST /v1/infer` | `{ model, prompt, run_id?, max_tokens?, temperature?, timeout_ms?, allow_fallback?, primary_provider?, output_schema? }`. Prompt 1 to 32768 bytes; `max_tokens` in [1, 4096]; `temperature` in [0, 2]; `timeout_ms` in [1, 120000]; `primary_provider` one of `vllm`, `sglang`, `gemini`, `mlx`. | `{ request_id, run_id, provider, model, fallback_used, output_text, finish_reason, inference_latency_ms, tokens }` | 422, 500 (including no provider configured; provider failures are sanitized) | Appends `inference_completed` when `run_id` is set |
| `POST /v1/infer/batch` | `{ requests[], max_parallel? }`; requests length in [1, 256]; `max_parallel` in [1, 64], default 8 | `{ request_id, processed, succeeded, failed, results[] }` with per-item `{ index, ok, result?, error? }` | 422, 500 | Same per-item event behavior as `/v1/infer` |
| `GET /v1/models` | none | `{ request_id, models[] }` with `{ id, tier, availability, providers_available, fallback_candidates }` | 500 | none |

### Feedback

Both feedback routes require the optional SpacetimeDB integration; without `VIDARAX_SPACETIMEDB_URL` they return 500 with a "spacetimedb client not configured" message.

| Route | Request | Success (200) | Failures |
|---|---|---|---|
| `POST /v1/runs/:id/feedback` | `{ rating, category, feedback? }`; rating in [0, 10], category non-empty | `{ request_id, run_id, status: "submitted" }` | 404, 422, 500 |
| `GET /v1/feedback` | none | `{ request_id, feedback[] }`, filtered to caller-owned runs | 500 |

### Files

| Route | Request | Success | Failures | Notes |
|---|---|---|---|---|
| `POST /v1/upload` | multipart form with a `file` field; body capped at 200 MiB by the route's body limit | 200 `{ file_path }`, the server-side path to use as `source_uri` | 422 unsupported type or invalid media container, 500 | Filenames are sanitized and prefixed per principal; the content must validate as a media container, not a playlist |
| `GET /v1/files/:filename` | bare filename only | 200 file bytes with a video content type and `Accept-Ranges: bytes` | 400, 404 | Errors are raw text bodies, not the JSON envelope. Uploads are only visible to the uploading principal; operator-configured roots are shared |

### Keyframe blobs

`GET /v1/runs/:id/keyframes/:sha256` returns `image/jpeg` bytes for a hash referenced by a `keyframe_stored` event on that run. The caller must own the run; knowing a blob hash from another run is not enough to retrieve it. Responses use a private immutable cache policy and an ETag equal to the content hash. Invalid hashes return the JSON validation envelope, and missing references or files return the JSON not-found envelope. Image bytes are never placed in JSON or base64-encoded by this API.

### WebRTC (WHIP)

Success and failure statuses for the four WHIP routes are covered in [WebRTC ingest](/docs/internals/webrtc-ingest/#endpoint-contract). Two things differ from the rest of the API: `POST /v1/stream/whip` answers with raw SDP (plus `Location` and `x-vidarax-run-id` headers) rather than JSON, and WHIP failures return bare status codes or plain-text bodies rather than the JSON envelope. The offer accepts an optional `x-attach-config` header (base64url-encoded JSON, no padding, size-capped) whose `prompt`, `max_output_tokens_per_second`, `clip_mode`, and normalized `crop` fields apply before workers start. Unknown attach fields are rejected. `PATCH /v1/stream/whip/:sess/prompt` accepts `{ prompt, output_schema? }`, where `output_schema` is a JSON Schema object, and returns the applied values. Token caps, crop, and clip mode cannot be changed after start.

### Health and metrics

| Route | Success (200) | Notes |
|---|---|---|
| `GET /v1/health` | `{ "status": "ok" }` | No API key required. Reports the HTTP server only, not model backend availability |
| `GET /v1/metrics` | Prometheus text format | Requires an API key by default; returns 503 `metrics_unavailable` if metrics auth is enabled with no keys configured |

## Error envelope

Handler errors share one JSON shape. The `request_id` is a string and lives inside the `error` object:

```json
{
  "error": {
    "code": "validation_error",
    "message": "invalid ingest request",
    "request_id": "req-000000000000002a",
    "details": [
      { "field": "source_uri", "message": "..." }
    ]
  }
}
```

| Status | Code | When |
|--------|------|------|
| 400 | `validation_error` | CORS preflight without an `Origin` header. |
| 401 | `unauthorized` | Missing or invalid `x-api-key`; missing `x-tenant-id` when required. |
| 403 | `cors_forbidden` | Preflight from an origin outside the allowlist. |
| 404 | `not_found` | Unknown, deleted, or other-principal `run_id`. |
| 409 | `conflict` | Action on a terminal run; active stream limit exceeded. |
| 422 | `validation_error` | Field-level validation failure; `details` lists the fields. |
| 429 | `rate_limited` | Global or per-principal rate limit exceeded. |
| 500 | `internal_error` | Internal failure; the message is sanitized and details are logged server-side. |
| 503 | `metrics_unavailable` | Metrics auth enabled with no API keys configured. |

Not everything uses the envelope. WHIP routes return raw SDP on success and bare status codes or plain text on failure; `GET /v1/files` failures are plain text; and requests rejected before a handler runs (malformed JSON bodies, unknown routes, oversized uploads) get the framework's default plain responses.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDARAX_VLLM_BASE_URL` | unset | vLLM inference endpoint |
| `VIDARAX_SGLANG_BASE_URL` | unset | SGLang inference endpoint (fallback) |
| `VIDARAX_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind address |
| `VIDARAX_REQUIRE_API_KEY` | `true` | Require `x-api-key` header |
| `VIDARAX_API_KEYS` | unset | Comma-separated accepted API keys |
| `VIDARAX_TRANSPORT` | `h1h2` | Transport mode (`h1h2` or `h3`) |
| `VIDARAX_DATA_DIR` | `.vidarax-data` | WAL and runtime data directory |
| `VIDARAX_INGEST_FILE_ROOTS` | unset | Directories local `source_uri` paths may come from |
| `VIDARAX_ACTIVE_STREAM_LIMIT` | `5` | Max active runs per resolved principal |
| `VIDARAX_STREAM_TTL_SECS` | `3600` | Run idle TTL |
| `VIDARAX_NOVELTY_EMBEDDING_ADDR` | unset | Binary TCP embedding sidecar; setting it enables live semantic novelty |
| `VIDARAX_NOVELTY_REUSE_THRESHOLD` | `0.01` | Conservative embedding-distance ceiling for description reuse; calibrate it on labelled deployment traffic |

When neither backend URL is set, the server reads a TOML config file (`VIDARAX_CONFIG`, default `vidarax.toml`) that declares backends in priority order; the parser supports `openai_compat` and `gemini` backend types, and string fields interpolate `${ENV_VAR}` references. When either explicit URL is set, the TOML file is not read.

The full configuration reference, including decode backend selection, CORS, rate limits, WebRTC and TURN settings, and SpacetimeDB, lives in `docs/deployment.md` in the repository. The hardening-relevant variables are summarized in [Operations](/docs/operations/#security-and-hardening).

# API Surface (v1 Foundation)

Implemented endpoints in `vidarax-api`:

- `POST /v1/runs`
- `POST /v1/runs/:run_id/ingest`
- `POST /v1/runs/:run_id/analyze`
- `POST /v1/runs/:run_id/reason`
- `POST /v1/runs/:run_id/stop`
- `POST /v1/runs/:run_id/keepalive`
- `GET /v1/runs/:run_id/events`
- `GET /v1/runs/:run_id/markers`
- `GET /v1/runs/:run_id/state`
- `POST /v1/query`
- `POST /v1/infer`
- `POST /v1/infer/batch`
- `GET /v1/models`
- `GET /v1/health`
- `GET /v1/metrics`

## Current behavior

- Deterministic JSON response shapes.
- Run IDs use random 128-bit hex identifiers.
- Request IDs from a monotonic atomic counter.
- WAL-backed lifecycle:
  - `POST /v1/runs` appends `run_created`
  - `POST /v1/runs/:run_id/ingest` appends `ingest_received`
  - `POST /v1/runs/:run_id/ingest` with `source_uri` + `max_frames` decodes MP4 via ffmpeg and appends `frames_decoded`
  - ingest fixed fps supports `fixed_fps` up to `120.0`
  - ingest `max_frames` supports up to `500000` for whole-video processing
  - ingest sampling supports `sampling_policy=source_fps_adaptive` (default) and `sampling_policy=fixed` (`fixed_fps` required)
  - `POST /v1/runs/:run_id/analyze` runs two-pass sliding-window metadata generation, emits marker lifecycle events, and appends `analysis_generated`
  - `POST /v1/runs/:run_id/analyze` can omit `frames` and reuse the latest `frames_decoded` payload from ingest
  - analyze supports `sampling_policy` (`source_fps_adaptive` default, `fixed` with `fixed_fps`)
  - `POST /v1/runs/:run_id/reason` executes whole-video semantic reasoning in chunked realtime mode and emits:
    - `semantic_chunk_generated`
    - `semantic_chunk_inferred`
    - `marker_emitted`
    - `analysis_generated` (with lag and marker summary)
  - realtime reason can enrich deterministic gate metadata with provider semantics using sampled raw JPEG frames from each chunk
  - realtime reason semantic controls:
    - `semantic_inference` (default `true`)
    - `semantic_frames_per_chunk` (range `[1, 4]`, default `2`)
    - `semantic_timeout_ms` (range `[100, 120000]`, default `1500`)
    - `primary_provider` (`vllm` or `sglang`)
    - `semantic_prompt` (optional override, max 4096 bytes)
  - marker lifecycle statuses support `exact`, `provisional`, and `finalized`
  - `GET /v1/runs/:run_id/markers` returns marker timeline and supports filtering by `status`, `event_type`, `from_frame`, `to_frame`
  - `POST /v1/runs/:run_id/stop` appends `stop_requested`
  - `POST /v1/runs/:run_id/keepalive` appends `keepalive_refreshed`
  - `GET /v1/runs/:run_id/state` derives state from persisted events
  - state derivation applies idle TTL expiry and can return `expired`
  - `GET /v1/runs/:run_id/events` returns decoded event list
  - `POST /v1/query` filters by `run_id`, optional `kind`, and `from_seq`
- WAL lives at `VIDARAX_DATA_DIR/timeline.wal` (default `.vidarax-data/timeline.wal`).
- Active-run cap is enforced per principal (`x-tenant-id` when present, otherwise API key hash, otherwise `public`).

## Validation and error envelopes

- Validation failures are stable and structured:
  - HTTP `422`
  - `error.code = validation_error`
  - deterministic `request_id`
  - field-level `details[]`
- Additional failure classes:
  - HTTP `404` with `error.code = not_found` for unknown `run_id`
  - HTTP `409` with `error.code = conflict` for terminal-run ingest/stop conflicts

## External inference behavior

- `POST /v1/infer` validates model/prompt/provider contract.
- `POST /v1/infer` and realtime reason semantic inference both support multimodal payloads (text + image_url parts) when images are provided.
- Calls configured vLLM/SGLang endpoints.
- Uses retryable fallback routing (`vllm` <-> `sglang`) when allowed.
- Appends `inference_completed` when a `run_id` is provided.
- `POST /v1/infer/batch` runs validated infer requests with bounded in-flight parallelism (`max_parallel`, default `8`, range `[1, 64]`) and returns per-item success/error entries in request order.
- `GET /v1/models` returns required-model catalog with runtime availability (`ready`, `degraded`, `unavailable`) and fallback candidates.

## Security and tenant boundaries

- Optional API key enforcement (`x-api-key`).
- Optional tenant enforcement (`x-tenant-id`).
- Lock-free global and tenant fixed-window rate limiting.
- Tenant slot table reuses stale-window slots to avoid permanent churn lockout.
- Metrics endpoint can require API key independently (`VIDARAX_METRICS_REQUIRE_API_KEY`, default `true`).
- CORS allowlist support via `VIDARAX_CORS_ALLOWED_ORIGINS` (comma-separated exact origins, `*` allowed).
- API responses include security headers: `x-content-type-options`, `x-frame-options`, `referrer-policy`, `cache-control`.
- Stream TTL is configurable via `VIDARAX_STREAM_TTL_SECS` (default `3600`).
- Active-run cap is configurable via `VIDARAX_ACTIVE_STREAM_LIMIT` (default `5`).
- Tenant label maps are configurable via `VIDARAX_TENANT_LABEL_MAPS_PATH` (JSON file).
- Label-map fallback is emitted in frame metadata (`fallback.used=true`) when tenant-specific mapping is unavailable.

## Error handling

- Internal failures return a sanitized `internal server error` message while logging details server-side with `request_id`.

## Provider observability

- Per-provider success/error totals.
- Per-provider latency histogram buckets.
- Fallback totals plus SLO/error-budget helper series via `/v1/metrics`.
- Realtime reason responses include lag summaries (`lag_p95_ms`, `lag_p99_ms`) to track bounded-lag targets.

## Transport

- Main runtime: Axum/Hyper over HTTP/1.1 + HTTP/2.
- Optional HTTP/3 path:
  - `VIDARAX_TRANSPORT=h3`
  - `cargo run -p vidarax-api --bin vidarax-api --features h3-experimental`
  - UDP bind: `VIDARAX_H3_BIND_ADDR` (default `127.0.0.1:8443`)
  - TLS paths:
    - `VIDARAX_H3_TLS_CERT_PATH` (default `deploy/certs/dev.crt`)
    - `VIDARAX_H3_TLS_KEY_PATH` (default `deploy/certs/dev.key`)
    - local generation helper: `make dev-cert`
  - h3 requests are translated into standard HTTP requests and dispatched through the same router as h1/h2.

## Ingest source constraints

- `source_uri` accepts `http://`, `https://`, and local file paths (`file://` or plain path).
- URL inputs reject localhost/local domains and direct private/link-local IP targets.
- Local file paths are canonicalized and must be under `VIDARAX_INGEST_FILE_ROOTS` (default: process cwd + system temp dir).
- ffmpeg/ffprobe decode/probe calls run with protocol allowlist: `file,http,https,tcp,tls`.

## Route parity guard

Startup computes a deterministic route manifest fingerprint and fails fast on mismatch.

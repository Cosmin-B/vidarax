# API Surface

Complete endpoint reference for `vidarax-api`. All routes are defined in
`crates/vidarax-api/src/router.rs`. Transport: Axum/Hyper over HTTP/1.1 + HTTP/2
(optional HTTP/3 via `--features h3-experimental`).

---

## Run Lifecycle

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `GET`    | `/v1/runs`                        | List runs for the current principal.           |
| `POST`   | `/v1/runs`                        | Create a new run.                              |
| `GET`    | `/v1/runs/{run_id}`               | Get a single run by ID.                        |
| `DELETE` | `/v1/runs/{run_id}`               | Soft-delete a run (appends `run_deleted`).     |
| `POST`   | `/v1/runs/{run_id}/ingest`        | Ingest frames from a video source URI.         |
| `POST`   | `/v1/runs/{run_id}/analyze`       | Two-pass sliding-window deterministic analysis.|
| `POST`   | `/v1/runs/{run_id}/reason`        | Chunked real-time semantic reasoning pipeline. |
| `POST`   | `/v1/runs/{run_id}/stop`          | Request graceful stop (appends `stop_requested`).|
| `POST`   | `/v1/runs/{run_id}/keepalive`     | Refresh idle TTL for an active run.            |
| `GET`    | `/v1/runs/{run_id}/events`        | List timeline events. Supports `?index=<name>`.|
| `GET`    | `/v1/runs/{run_id}/markers`       | List markers. Filters: `?status`, `?event_type`, `?from_frame`, `?to_frame`.|
| `GET`    | `/v1/runs/{run_id}/state`         | Derive current run state from persisted events.|
| `POST`   | `/v1/runs/{run_id}/feedback`      | Submit feedback (rating, category, text).      |

## File Upload and Serving

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `POST`   | `/v1/upload`                      | Multipart file upload (200 MB limit). Returns `{ file_path }`. |
| `GET`    | `/v1/files/{filename}`            | Serve an uploaded file by filename from allowed ingest roots. |

## Inference

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `POST`   | `/v1/infer`                       | Single VLM inference request. Supports multimodal payloads (text + image_url). |
| `POST`   | `/v1/infer/batch`                 | Batch inference with bounded parallelism (`max_parallel`, default 8, range [1, 64]). |
| `GET`    | `/v1/models`                      | Model catalog with runtime availability (`ready`, `degraded`, `unavailable`). |

## Query and Search

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `POST`   | `/v1/query`                       | Filter events by `run_id`, optional `kind`, and `from_seq`. |
| `POST`   | `/v1/search`                      | Substring search over VLM descriptions. Accepts `query`, optional `run_id`, optional `limit` (default 50, max 500). |

## Feedback

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `GET`    | `/v1/feedback`                    | List all feedback entries from SpacetimeDB.    |

## Observability

| Method   | Path                              | Description                                    |
|----------|-----------------------------------|------------------------------------------------|
| `GET`    | `/v1/health`                      | Returns `{ "status": "ok" }`.                  |
| `GET`    | `/v1/metrics`                     | Prometheus-format metrics (runs, events, inference latency, pipeline counters). |

## WHIP WebRTC Ingestion (RFC 9725)

| Method   | Path                                          | Description                                    |
|----------|-----------------------------------------------|------------------------------------------------|
| `POST`   | `/v1/stream/whip`                             | SDP offer/answer exchange. Returns 201 + Location header. |
| `PATCH`  | `/v1/stream/whip/{sess_id}`                   | Trickle ICE candidate.                         |
| `DELETE` | `/v1/stream/whip/{sess_id}`                   | Terminate WebRTC session.                      |
| `PATCH`  | `/v1/stream/whip/{sess_id}/prompt`            | Update the analysis prompt for a live session. |

---

## Key Request/Response Examples

### Create a run

```
POST /v1/runs
Content-Type: application/json

{ "mode": "batch", "model": "Qwen/Qwen3-VL-4B-Instruct" }
```

```json
{
  "run_id": "a1b2c3d4e5f6...",
  "request_id": 1,
  "status": "pending",
  "mode": "batch",
  "model": "Qwen/Qwen3-VL-4B-Instruct"
}
```

### Ingest from file

```
POST /v1/runs/{run_id}/ingest
Content-Type: application/json

{
  "source_uri": "/tmp/demo.mp4",
  "sampling_policy": "fixed",
  "fixed_fps": 5.0,
  "max_frames": 512
}
```

### Run semantic reasoning

```
POST /v1/runs/{run_id}/reason
Content-Type: application/json

{
  "source_uri": "/tmp/demo.mp4",
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "semantic_inference": true,
  "semantic_frames_per_chunk": 2,
  "chunk_size": 30,
  "fixed_fps": 5.0,
  "sampling_policy": "fixed"
}
```

---

## Error Envelope

All errors follow a consistent shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "invalid ingest request",
    "details": [
      { "field": "source_uri", "message": "..." }
    ]
  },
  "request_id": 42
}
```

| Status | Code               | When                                              |
|--------|--------------------|----------------------------------------------------|
| 404    | `not_found`        | Unknown `run_id`.                                  |
| 409    | `conflict`         | Action on a terminal run; active stream limit exceeded. |
| 422    | `validation_error` | Field-level validation failure.                    |
| 500    | (sanitized)        | Internal failure; details logged server-side.      |

## Security

- Optional API key enforcement via `x-api-key` header.
- Data ownership is based on the authenticated API-key principal. `x-tenant-id`
  is not a security or authorization boundary unless a trusted upstream auth
  layer enforces that mapping before requests reach Vidarax.
- Open/no-key mode uses a shared `public` principal with no isolation and is
  intended for development only.
- Rate limiting is global and per authenticated principal. `x-tenant-id` is not
  used as the quota identity.
- Uploaded files are private to the uploader principal via the stored filename
  prefix. Operator-configured non-upload ingest roots are admin-trusted shared
  media roots. Legacy unprefixed files in the upload temp root are not
  auto-claimed by authenticated callers.
- Remote ingest has application-level SSRF mitigations, but untrusted-source
  deployments should enforce network-level egress controls; see
  `docs/security.md`.
- CORS allowlist via `VIDARAX_CORS_ALLOWED_ORIGINS`.
- Security headers on all responses: `x-content-type-options`, `x-frame-options`, `referrer-policy`, `cache-control`.

## Route Parity Guard

Startup computes a deterministic route manifest fingerprint and fails fast on mismatch.

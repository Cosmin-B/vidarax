# vidarax

Real-time video intelligence engine -- any stream in, semantic events out.

## What it does

- **Deterministic frame analysis**: Scene cuts, artifacts, motion markers at O(1) per frame via a lock-free gate engine. No ML needed for pass one.
- **Tiered VLM reasoning**: First pass with Qwen 2B in ~200ms. If confidence is low, second pass with 8B. Intelligent routing between your streams and your models.
- **Observable end to end**: WAL-backed run lifecycle, per-provider latency histograms, structured tracing with OpenTelemetry export. Every frame is traceable from ingest to event.

## Architecture

```
                         vidarax
  ┌─────────┐    ┌──────────────────────────┐    ┌──────────┐
  │ Sources  │    │                          │    │ Clients  │
  │          │───>│  WebRTC ─> Decode ─>     │    │          │
  │ WebRTC   │    │              │           │    │ Vue 3 UI │
  │ MP4      │    │         Gate Engine      │    │ REST API │
  │ Upload   │    │          │       │       │───>│ SSE      │
  │          │    │     Markers    VLM       │    │          │
  │          │    │          │       │       │    │          │
  │          │    │      SpacetimeDB         │    │          │
  └─────────┘    └──────────────────────────┘    └──────────┘
```

## Performance

Tested on Hetzner (Qwen3-VL-2B + 8B tiered):

- **1.5s** wall time for a 10-second video (6.7x real-time)
- **242 frames** decoded at 24fps
- **20 markers** emitted (scene cuts, keyframes, artifacts)
- Gate processing: **42ns p95** per frame, zero allocations
- API workflow (create + ingest + analyze + query): **2.99ms p95**

## Tech stack

Rust | Vue 3 | SpacetimeDB | vLLM | WebRTC (WHIP)

## Quick start

```bash
git clone https://github.com/vidarax/vidarax && cd vidarax
cargo build --release -p vidarax-api
VIDARAX_VLLM_BASE_URL=http://localhost:8000 cargo run --release -p vidarax-api
```

Frontend (separate terminal):

```bash
cd ui && npm install && npm run dev
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/runs` | Create a new run |
| `GET` | `/v1/runs` | List runs |
| `GET` | `/v1/runs/:run_id` | Get run details |
| `DELETE` | `/v1/runs/:run_id` | Delete a run |
| `POST` | `/v1/runs/:run_id/ingest` | Ingest video (MP4 URI or raw frames) |
| `POST` | `/v1/runs/:run_id/analyze` | Run deterministic frame analysis |
| `POST` | `/v1/runs/:run_id/reason` | Realtime semantic reasoning (tiered VLM) |
| `POST` | `/v1/runs/:run_id/stop` | Stop a run |
| `POST` | `/v1/runs/:run_id/keepalive` | Refresh run TTL |
| `GET` | `/v1/runs/:run_id/events` | Get run events |
| `GET` | `/v1/runs/:run_id/markers` | Get markers (filterable by status/type/frame range) |
| `GET` | `/v1/runs/:run_id/state` | Derive current run state |
| `POST` | `/v1/runs/:run_id/feedback` | Submit feedback on a run |
| `GET` | `/v1/feedback` | List all feedback |
| `POST` | `/v1/upload` | Upload a file for processing |
| `POST` | `/v1/query` | Query events across runs |
| `POST` | `/v1/infer` | Single inference request |
| `POST` | `/v1/infer/batch` | Batch inference (bounded parallelism) |
| `GET` | `/v1/models` | Model catalog with availability status |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/metrics` | Prometheus-compatible metrics |
| `POST` | `/v1/stream/whip` | WHIP WebRTC offer (RFC 9725) |
| `PATCH` | `/v1/stream/whip/:sess_id` | ICE trickle candidate |
| `DELETE` | `/v1/stream/whip/:sess_id` | Terminate WebRTC session |
| `PATCH` | `/v1/stream/whip/:sess_id/prompt` | Update live stream prompt |

## Frontend pages

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Command center -- active runs, metrics overview |
| Stream | `/stream` | Live WebRTC stream viewer and control |
| Active Stream | `/stream/:sessionId` | Single stream session view |
| Run Detail | `/runs/:runId` | Frame-by-frame markers, keyframes, VLM descriptions |
| Upload | `/upload` | Upload video files for processing |
| Tracing | `/tracing` | Pipeline flow visualization and observability |
| Settings | `/settings` | Model selector, tiered routing config |

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDARAX_VLLM_BASE_URL` | -- | vLLM inference endpoint |
| `VIDARAX_SGLANG_BASE_URL` | -- | SGLang inference endpoint (fallback) |
| `VIDARAX_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind address |
| `VIDARAX_REQUIRE_API_KEY` | `true` | Require `x-api-key` header |
| `VIDARAX_TRANSPORT` | `h1h2` | Transport mode (`h1h2` or `h3`) |
| `VIDARAX_DATA_DIR` | `.vidarax-data` | WAL and runtime data directory |
| `VIDARAX_ACTIVE_STREAM_LIMIT` | `5` | Max concurrent runs per principal |
| `VIDARAX_STREAM_TTL_SECS` | `3600` | Run idle TTL |

See the full list in the [deployment docs](docs/deployment.md).

## Workspace layout

```
crates/
  vidarax-core/       Lock-free primitives, gate engine, ingest pipeline
  vidarax-contracts/  Shared model contracts and error mapping
  vidarax-api/        Axum HTTP server, handlers, WHIP, security middleware
  vidarax-cli/        CLI tooling for contract checks
ui/                   Vue 3 frontend
spacetime-module/     SpacetimeDB server module
docs/                 Architecture docs, runbooks, specs
deploy/               Docker, compose, certificates
scripts/              Benchmarks, smoke tests, release gates
```

## License

MIT

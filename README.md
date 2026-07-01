# vidarax

Real-time video intelligence engine. Any stream in, structured semantic events out.

Vidarax decodes live or file-based video, runs a deterministic gate engine (scene cuts, flicker, ghosting, exposure shifts, loop detection) to mark every frame, samples a subset of frames through tiered Vision Language Models for semantic analysis, and emits structured events in real time. Self-hosted, open-source VLMs, no external API dependencies.

## Architecture

```
 Sources                      vidarax                           Consumers
┌──────────┐   ┌──────────────────────────────────────────┐   ┌──────────────┐
│ MP4/File  │──>│                                          │──>│ REST / SSE   │
│ WebRTC    │──>│  Decode ──> Gate Engine ──> VLM Tiering  │──>│ TypeScript   │
│ RTSP/HLS  │──>│               │                │         │──>│   SDK        │
│ Upload    │──>│          Markers +         Semantic       │──>│ Vue 3 UI     │
│           │   │          Keyframes        Events          │   │ Prometheus   │
│           │   │               │                │         │   │ Optional     │
│           │   │            WAL event log       │         │   │ SpacetimeDB  │
└──────────┘   └──────────────────────────────────────────┘   └──────────────┘
```

## Performance

Throughput depends on your hardware, the models you run, and the input video, so
there is no single headline number worth quoting. Measure on your own setup: the
Python harnesses in `benchmarks/`, the bench binaries under `crates/*/src/bin`,
and the scripts in `scripts/` cover the gate engine, provider transport, and the
end-to-end API path. The gate engine is the cheap stage that runs on every frame;
tiered routing keeps the small model on the common case and escalates to a larger
one only when it is uncertain.

## Quick start

### Local

```bash
git clone https://github.com/vidarax/vidarax && cd vidarax
cargo build --release -p vidarax-api
VIDARAX_API_KEYS=dev-key VIDARAX_VLLM_BASE_URL=http://localhost:8000 cargo run --release -p vidarax-api
```

Frontend (separate terminal):

```bash
cd ui && npm install && npm run dev
```

### SDK

```bash
npm install vidarax
```

```typescript
import { Vidarax } from 'vidarax'

const v = new Vidarax('http://localhost:8080', { apiKey: 'your-key' })

const run = await v.analyze('video.mp4', {
  prompt: 'Describe what happens in each scene',
})

for await (const event of run.events()) {
  console.log(event.kind, event.payload)
}
```

The SDK also supports WebRTC streaming, batch inference, structured JSON output via `output_schema`, and typed async iterators over events and markers.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/runs` | Create a new analysis run |
| `GET` | `/v1/runs` | List runs |
| `GET` | `/v1/runs/:id` | Get run details |
| `DELETE` | `/v1/runs/:id` | Delete a run |
| `POST` | `/v1/runs/:id/ingest` | Ingest video (URI, file path, raw frames) |
| `POST` | `/v1/runs/:id/analyze` | Deterministic frame analysis |
| `POST` | `/v1/runs/:id/reason` | Realtime semantic reasoning (tiered VLM) |
| `POST` | `/v1/runs/:id/stop` | Stop a run |
| `POST` | `/v1/runs/:id/keepalive` | Refresh active run TTL |
| `GET` | `/v1/runs/:id/events` | Stream run events |
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
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/metrics` | Prometheus-compatible metrics |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDARAX_VLLM_BASE_URL` | — | vLLM inference endpoint |
| `VIDARAX_SGLANG_BASE_URL` | — | SGLang inference endpoint (fallback) |
| `VIDARAX_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind address |
| `VIDARAX_REQUIRE_API_KEY` | `true` | Require `x-api-key` header |
| `VIDARAX_API_KEYS` | — | Comma-separated accepted API keys |
| `VIDARAX_TRANSPORT` | `h1h2` | Transport mode (`h1h2` or `h3`) |
| `VIDARAX_DATA_DIR` | `.vidarax-data` | WAL and runtime data directory |
| `VIDARAX_ACTIVE_STREAM_LIMIT` | `5` | Max active runs per resolved principal |
| `VIDARAX_STREAM_TTL_SECS` | `3600` | Run idle TTL |

Full configuration reference in [docs/deployment.md](docs/deployment.md).

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | Rust, Axum, Hyper (HTTP/1.1 + H2, optional H3) |
| Gate engine | Deterministic frame analysis on a single-threaded hot path |
| Inference | vLLM and SGLang through OpenAI-compatible backends with fallback |
| Decode | ffmpeg CPU and NVDEC; the registered MLX decode path currently falls back to CPU ffmpeg |
| Persistence | WAL-backed event log; optional SpacetimeDB client and module are present but not wired into the production server path |
| Frontend | Vue 3, dark command-center UI |
| Streaming | WebRTC via WHIP (RFC 9725) |
| SDK | TypeScript (`vidarax` on npm) |
| Observability | OpenTelemetry, Prometheus metrics |

## Workspace layout

```
crates/
  vidarax-core/         Lock-free primitives, gate engine, ingest pipeline
  vidarax-contracts/    Shared model contracts and error mapping
  vidarax-api/          Axum HTTP server, handlers, WHIP, security
  vidarax-cli/          CLI tooling
ui/                     Vue 3 frontend
packages/vidarax-sdk/   TypeScript SDK
spacetime-module/       SpacetimeDB server module
docs/                   Architecture docs, runbooks, specs
deploy/                 Docker, compose, certificates
scripts/                Benchmarks, smoke tests, release gates
```

## License

MIT

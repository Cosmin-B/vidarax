# vidarax

Self-hosted video intelligence for live streams and recorded files.

Vidarax decodes video, applies a deterministic per-frame filter, and sends selected
frames to a configured vision-language model. Live capture can also use an
embedding sidecar to reuse recent descriptions while a scene remains
semantically stable. Events commit to a local write-ahead log; selected JPEGs
are stored as content-addressed blobs and referenced by event metadata.

## Architecture

```
 Sources                         vidarax                          Consumers
┌──────────┐   ┌──────────────────────────────────────────┐   ┌──────────────┐
│ MP4/File │──>│                                          │──>│ REST API     │
│ WebRTC   │──>│ Decode ──> Frame Filter ──> VLM Tiering   │──>│ TypeScript   │
│ RTSP/HLS │──>│              │                │          │──>│ SDK          │
│ Upload   │──>│              v                v          │──>│ Vue 3 UI     │
│          │   │        Markers +         Semantic       │   │ Prometheus   │
│          │   │        Keyframes          Events        │   │ Optional     │
│          │   │              └───────┬────────┘          │   │ SpacetimeDB  │
│          │   │                      v                   │   │              │
│          │   │                WAL event log             │   │              │
└──────────┘   └──────────────────────────────────────────┘   └──────────────┘
```

## Performance

Throughput depends on your hardware, the models you run, and the input video, so
there is no single headline number worth quoting. Measure on your own setup: the
Python harnesses in `benchmarks/`, the bench binaries under `crates/*/src/bin`,
and the scripts in `scripts/` cover the per-frame filter, provider transport, and the
end-to-end API path. The deterministic filter is the cheap per-frame stage. The
optional semantic novelty filter reduces repeat model calls, while bounded tiering
can escalate an uncertain first pass to a second model. Calibration and
provider/hardware measurements are deployment-specific; see
[deployment and evidence](docs/deployment.md#live-semantic-novelty-and-evidence).

## Quick start

### Local

```bash
git clone https://github.com/Cosmin-B/vidarax && cd vidarax
cargo build --release -p vidarax-api
VIDARAX_API_KEYS=dev-key VIDARAX_VLLM_BASE_URL=http://localhost:8000 cargo run --release -p vidarax-api
```

Frontend (separate terminal):

```bash
cd ui && npm install && npm run dev
```

### SDK

```bash
cd packages/vidarax-sdk
npm install
npm run build
npm link
```

Until the first npm release, install the SDK from this workspace and link it
into your application with `npm link vidarax`.

```typescript
import { Vidarax } from 'vidarax'

const v = new Vidarax('http://localhost:8080', { apiKey: 'your-key' })

// analyze() runs the deterministic frame-signal pipeline; it takes no prompt.
const run = await v.analyze('/srv/vidarax-media/video.mp4', { mode: 'balanced' })

for (const event of await v.getEvents(run.runId)) {
  console.log(event.kind, event.payload)
}
```

For prompt-driven semantic analysis, create a run and call `reason()` with a
`semantic_prompt` instead:

```typescript
const { run_id } = await v.createRun()
const result = await v.reason(run_id, {
  source_uri: 'video.mp4',
  model: 'your-model',
  semantic_prompt: 'Describe what happens in each scene',
})
```

The SDK also supports WHIP/WebRTC, batch inference, structured JSON output via
`output_schema`, interactions, and snapshot reads of events and markers.

## API endpoints

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
| `GET` | `/v1/runs/:id/events` | List run events (poll; the SDK does not push-stream this) |
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

### Local first-pass VLM on Apple Silicon (mlx-vlm)

[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) runs a Vision Language Model on-device
on Apple Silicon and serves it behind the same OpenAI-compatible
`/v1/chat/completions` API as vLLM/SGLang, so it plugs into vidarax as an
ordinary `openai_compat` backend, with no new provider code needed on the
vidarax side:

```bash
pip install mlx-vlm
# Serve a supported VL family. vidarax accepts only its curated model ids, so the
# id you hand back to vidarax has to be one of them (see the note below).
mlx_vlm.server --model mlx-community/Qwen3-VL-4B-Instruct-4bit --port 8080
```

Point a backend at it with `openai_kind = "mlx"` so its telemetry (`/v1/metrics`)
and its side of the tiering split are labelled `mlx` instead of `vllm`:

```toml
[[backends]]
name = "mlx"
type = "openai_compat"
openai_kind = "mlx"
base_url = "http://127.0.0.1:8080"
model = "Qwen/Qwen3-VL-4B-Instruct"
upstream_model = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
priority = 1
```

To escalate uncertain first-pass results to a hosted Gemini second pass, add a
`gemini` backend (see the commented example in `vidarax.toml`) and set the
`reason()`/WHIP second-pass model to that backend's exact `model` id: the
model-routing provider dispatches on that id, so nothing else needs to change.

vidarax checks every request's `model` id against its supported-model contract
and rejects unknown ids before it opens a connection, so the first-pass and
second-pass models you configure must be supported ids, for example
`Qwen/Qwen3-VL-4B-Instruct`, not a raw `mlx-community/...-4bit` conversion name.
`model` declares the curated Vidarax id this backend serves, while
`upstream_model` maps it to the conversion id mlx-vlm actually loads. The model
catalog reports only that curated id as available on this backend.

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | Rust, Axum, Hyper (HTTP/1.1 + H2, optional H3) |
| Gate engine | Deterministic frame analysis on a single-threaded hot path |
| Inference | vLLM and SGLang through OpenAI-compatible backends with fallback |
| Decode | ffmpeg CPU, NVDEC, and Apple VideoToolbox; VideoToolbox may fall back to software decode inside ffmpeg when the input or host cannot initialise hardware |
| Persistence | Local WAL plus content-addressed JPEG blobs; optional SpacetimeDB mirror for blocking WHIP description events |
| Frontend | Vue 3, dark command-center UI |
| Streaming | WebRTC via WHIP (RFC 9725) |
| SDK | TypeScript (`vidarax` on npm) |
| Observability | OpenTelemetry, Prometheus metrics |

## Workspace layout

```
crates/
  vidarax-core/         Frame filter, media primitives, ingest pipeline
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

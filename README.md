# vidarax

Self-hosted video intelligence for live streams and recorded files.

Vidarax turns continuous video into actionable, replayable assertions. It
decodes each stream, runs a deterministic per-frame filter, and spends model
calls when visual or semantic change warrants them. Bounded trigger programs
turn perception signals into namespaced events, signed webhooks, or
metadata-only local outputs.

Events commit to a local write-ahead log. Selected JPEGs live in a
content-addressed binary store and event metadata carries their references.
For fixed cameras, a restricted-zone policy can turn sustained motion inside a
normalized image region into a durable assertion. A signed edge updater moves
model releases through shadow and canary checks while the last active model
keeps running through network loss or a rejected candidate.

## Architecture

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

Each live session is admitted as one typed pipeline generation. The process
reserves that generation's bounded memory and worker-thread envelope before it
creates a durable run. A supervisor then owns every stage handle: if one
stateful worker exits unexpectedly, Vidarax stops and joins the sibling stages,
closes the peer, and reports the fixed stage and reason through metrics. It never
restarts one worker underneath older temporal state. Live prompt and schema
changes use a bounded, generation-tagged command and return only after the VLM
worker acknowledges the new configuration.

Provider calls from all streams pass through one deadline-aware scheduler. It
fair-queues principals and streams across urgent live, normal live, and offline
classes while enforcing process-wide concurrency, output-token, and encoded
media-byte reservations. Per-stream temporal order remains owned by that
stream's single VLM worker. Provider runtimes remain responsible for
token-level batching.

## Performance

Throughput depends on your hardware, the models you run, and the input video, so
there is no single headline number worth quoting. Measure on your own setup: the
Python harnesses in `benchmarks/`, the bench binaries under `crates/*/src/bin`,
and the scripts in `scripts/` cover the per-frame filter, provider transport, and the
end-to-end API path. The deterministic filter is the cheap per-frame stage. The
optional semantic novelty filter reduces repeat model calls, while bounded tiering
can escalate an uncertain first pass to a second model. Calibration and
provider/hardware measurements are deployment-specific. See
[deployment and calibration](docs/deployment.md#live-semantic-novelty-calibration).

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
`semantic_prompt`:

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
| `POST` | `/v1/runs/:id/reason` | Real-time semantic analysis (tiered VLM) |
| `POST` | `/v1/runs/:id/stop` | Stop a run |
| `POST` | `/v1/runs/:id/keepalive` | Refresh active run TTL |
| `GET` | `/v1/runs/:id/events` | List the current run-event snapshot |
| `GET` | `/v1/runs/:id/events/stream` | Replay and follow run events over cursor-based SSE |
| `GET`, `POST` | `/v1/runs/:id/webhooks` | List or register signed event webhooks |
| `DELETE` | `/v1/runs/:id/webhooks/:webhook_id` | Remove a webhook |
| `GET` | `/v1/runs/:id/markers` | Marker timeline (filterable) |
| `GET` | `/v1/runs/:id/state` | Derived run state |
| `GET` | `/v1/runs/:id/interactions` | Interaction timeline |
| `POST` | `/v1/runs/:id/feedback` | Submit feedback for a run |
| `GET` | `/v1/feedback` | List feedback |
| `GET/POST` | `/v1/runs/:id/policies` | List or create immutable policy revisions |
| `GET` | `/v1/runs/:id/policies/:revision` | Read reconstructed policy state |
| `POST` | `/v1/runs/:id/policies/:revision/activate` | Promote through shadow, canary, and active |
| `POST` | `/v1/runs/:id/policies/:revision/rollback` | Restore a previously active revision |
| `POST` | `/v1/runs/:id/policies/:revision/replay` | Re-evaluate persisted restricted-zone candidates |
| `POST` | `/v1/triggers/compile` | Compile bounded trigger source into the current ISA |
| `POST` | `/v1/triggers/validate` | Validate a compiled trigger program |
| `POST` | `/v1/triggers/evaluate` | Deterministically replay trigger samples |
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
| `VIDARAX_MEDIA_MEMORY_BUDGET_BYTES` | `8589934592` | Process-wide reservation budget for live media payloads |
| `VIDARAX_MEDIA_WORKER_THREAD_BUDGET` | `64` | Process-wide reservation budget for live pipeline OS threads |
| `VIDARAX_INFERENCE_TOKEN_BUDGET` | `32768` | Aggregate output-token reservation across active provider calls |
| `VIDARAX_INFERENCE_BYTE_BUDGET` | `268435456` | Aggregate encoded-media byte reservation across active provider calls |
| `VIDARAX_STREAM_TTL_SECS` | `3600` | Run idle TTL |
| `VIDARAX_WEBHOOK_SECRET` | — | Server-side root for per-webhook signing keys. 32 bytes minimum |
| `VIDARAX_TRIGGER_LOCAL_OUTPUT_SOCKET` | — | Absolute Unix datagram socket for metadata-only local trigger actions |
| `VIDARAX_CONFIG` | `vidarax.toml` | Backend configuration and optional device-level restricted-zone policy |

Full configuration reference in [docs/deployment.md](docs/deployment.md).

Trigger programs are bounded, forward-only bytecode. Compile and replay a
source file before attaching the resulting program to a WHIP session:

```bash
vidarax triggers compile loading-bay.vxt --output loading-bay.json
vidarax triggers validate loading-bay.json
vidarax triggers evaluate loading-bay.json samples.json
```

The first edge package verifies signed binary model artifacts and advances a
candidate through shadow and canary health checks without placing model bytes
in JSON. See [edge deployment](docs/edge-deployment.md).

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

Point a backend at it with `openai_kind = "mlx"`. Its telemetry
(`/v1/metrics`) and tiering role are then labelled `mlx`:

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
| Frame filter | Deterministic frame analysis on a single-threaded hot path |
| Inference | vLLM and SGLang through OpenAI-compatible backends with fallback |
| Decode | ffmpeg CPU, NVDEC, and Apple VideoToolbox. VideoToolbox may fall back to software decode inside ffmpeg when the input or host cannot initialise hardware |
| Persistence | Local WAL plus content-addressed JPEG blobs. Optional SpacetimeDB mirror for blocking WHIP description events |
| Frontend | Vue 3, dark command-center UI |
| Streaming | WebRTC via WHIP (RFC 9725) |
| SDK | TypeScript workspace package (`vidarax`, pending its first npm release) |
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

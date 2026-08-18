
# vidarax

Vidarax is a self-hosted media runtime for applications that need ordered,
timestamped records from live video or recorded media. It accepts a stream or
file, narrows the media before inference, and gives the application an event
cursor plus references to the selected binary media.

It is built for developers operating camera, fleet, facility, or robotics
software who want to run their own media plane and choose where inference
runs.

[Product site](https://vidarax.cosminbararu.com/) ·
[Documentation](https://vidarax.cosminbararu.com/docs/) ·
[System tour](https://cosminbararu.com/blog/video-stream-through-vidarax)

## The system model

```
 source                bounded media work                 ordered commit           consumers
┌────────────┐    ┌──────────────────────────────┐    ┌──────────────────┐    ┌──────────────┐
│ file/upload│───>│ decode -> frame filter       │───>│ timeline writer  │───>│ REST / SSE   │
│ RTSP / HLS │    │             -> novelty -> VLM│    │ WAL + blob refs  │    │ webhooks     │
│ WHIP video │    │ audio windows -> local audio │    │ JPEG / MP4 / WAV │    │ SDK / UI     │
└────────────┘    └──────────────────────────────┘    └──────────────────┘    └──────────────┘
```

One timeline writer assigns a monotonic sequence number before an event becomes
visible to readers. That sequence is global to one data directory and is the
cursor used by REST, SSE, and webhook delivery. The envelope timestamp records
when the writer appended the event. Frame, clip, and audio timestamps remain in
the payload as source-relative media time.

Live work stays ordered inside one stream. A live session reserves its queue,
buffer-pool, scratch-space, sidecar, and worker allowance before `run_created`
commits. Bounded queues connect the stages. Provider calls from every stream go
through a second process-wide scheduler with concurrency, token, encoded-media,
waiter, and deadline limits. Parallelism comes from separate streams, while one
stateful decoder and one stateful VLM worker preserve order within a stream.

## Inputs and inference

| Input | Analysis path | Provider boundary |
|---|---|---|
| Local file, upload, HTTP(S), HLS, or RTSP | Deterministic frame signals, selected JPEGs, optional clip windows | OpenAI-compatible vLLM, SGLang, or MLX endpoints, with optional Gemini routing |
| Recorded `video` or `audio_video` mode | Bounded source-time MP4 windows with synchronized audio | Gemini File API |
| WHIP/WebRTC video | Ordered live keyframes or clips | The configured image provider chain |
| WHIP Opus audio | Bounded four-second windows | Optional local audio sidecar |

File and URL decoding supports CPU ffmpeg, NVIDIA NVDEC, and an Apple
VideoToolbox selective-JPEG path. WHIP video accepts H.264 and H.265. VP8 is
available only in builds that enable the `vp8` feature.

The first decision is deterministic and runs on decoded frame signals. It
selects scene changes, keepalives, and suspected visual artifacts before JPEG
encoding. Live WHIP capture can add an opt-in SigLIP2 novelty sidecar after
that filter. The live novelty gate compares each selected JPEG with the last
frame that produced a usable description. Reuse is bounded by capture time and
cumulative drift. Timeouts, malformed embeddings, and reconnect backoff run the
VLM. A shadow sample checks some reuse decisions without changing state or
emitting an event.

The shipped novelty threshold is a conservative starting value. It must be
calibrated on ordered examples from the deployment. File analysis does not pass
through this live gate. The reusable novelty library also contains OCR and
perceptual-hash signals, but the live image path uses embeddings only.

## Retention and delivery

Metadata commits to `${VIDARAX_DATA_DIR}/timeline.wal`. Selected JPEGs,
retained MP4 windows, and generated WAV files are written to content-addressed
stores before an event references them. JSON, SSE, webhooks, and the WAL carry
the reference, media type, byte count, and SHA-256 instead of binary bytes.
Vidarax does not retain the full source automatically.

The WAL is flushed after each append but is not fsynced. It is designed to
survive a process crash, not sudden power loss or a kernel failure. A crash
after a blob write and before its event append can leave an unreferenced blob,
and automatic orphan cleanup is not implemented. Older run reads can also fall
back to a WAL scan once their in-memory tail has been evicted.

## Current boundary

- Vidarax is self-hosted. The operator supplies storage, ffmpeg, and each model
  service or API credential.
- Capacity and semantic reuse depend on codec, resolution, scene content,
  provider, and hardware. The repository includes probes instead of a universal
  throughput claim.
- Native synchronized audio-video reasoning currently uses Gemini. The
  OpenAI-compatible path handles selected images, but it is not the recorded
  native-media route.
- The event log has one writer per process and no built-in retention policy or
  blob garbage collector.
- The TypeScript SDK is built from this checkout until its first registry
  release.

For deployment settings and calibration commands, see
[docs/deployment.md](docs/deployment.md). For the complete event contract, see
[Events and SDK](https://vidarax.cosminbararu.com/docs/events/).

## Quick start

### Local

```bash
git clone https://github.com/Cosmin-B/vidarax && cd vidarax
cargo build --release -p vidarax-api
VIDARAX_API_KEYS=dev-key \
VIDARAX_VLLM_BASE_URL=http://localhost:8000 \
VIDARAX_INGEST_FILE_ROOTS=/srv/vidarax-media \
cargo run --release -p vidarax-api
```

The example expects a video under `/srv/vidarax-media` and an
OpenAI-compatible VLM at `http://localhost:8000`. API keys are enabled by
default.

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

const v = new Vidarax('http://localhost:8080', { apiKey: 'dev-key' })

// analyze() runs the deterministic frame-signal pipeline. It takes no prompt.
const run = await v.analyze('/srv/vidarax-media/video.mp4', { mode: 'balanced' })

for (const event of await v.getEvents(run.runId)) {
  console.log(event.kind, event.payload)
}
```

This example runs the deterministic frame path. Prompt-driven analysis,
synchronized audio/video windows, live WHIP, and the full CLI flow are in the
[quickstart](https://vidarax.cosminbararu.com/docs/quickstart/). The
[TypeScript SDK reference](packages/vidarax-sdk/README.md) documents the public
client surface.

## Agent skill

The repository ships the platform-neutral
[`vidarax-review-media`](.agents/skills/vidarax-review-media/SKILL.md) Agent
Skill. Compatible agent harnesses can use it to create an isolated current-build
runtime, install the local audio path on first use, review a recording, and
separate supported moments from uncertain candidates. Its reporting rules keep
binary media out of JSON and treat local transcripts as the source for exact
speech wording.

## Where to go next

- [Quickstart](https://vidarax.cosminbararu.com/docs/quickstart/) covers the
  server, SDK, CLI, and raw HTTP path.
- [Architecture](https://vidarax.cosminbararu.com/docs/architecture/) explains
  queue ownership, ordering, cancellation, and persistence.
- [API reference](https://vidarax.cosminbararu.com/docs/api/) is the complete
  route contract. [Deployment](docs/deployment.md) is the complete environment
  and backend configuration reference.
- [Local audio](docs-site/src/content/docs/audio.md), [trigger
  programs](https://vidarax.cosminbararu.com/docs/triggers/), and [edge
  deployment](docs/edge-deployment.md) cover the optional paths.

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
scripts/                Benchmarks, smoke tests, release checks
```

## License

MIT

# vidarax

TypeScript SDK for the Vidarax video analysis API.

## Quick start

```bash
npm install vidarax
```

```typescript
import { Vidarax } from 'vidarax'

const v = new Vidarax('http://192.0.2.11:8080')

const run = await v.analyze('video.mp4', { prompt: 'Describe what happens' })
for await (const event of run.events()) {
  console.log(event.kind, event.payload)
}
```

Three lines, full type safety, automatic retries — that is the whole idea.

---

## Installation

```bash
npm install vidarax       # npm
yarn add vidarax          # yarn
pnpm add vidarax          # pnpm
```

Requires Node.js 18+ (native `fetch`) or any modern browser.

---

## Constructor

```typescript
const v = new Vidarax(baseUrl, options?)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseUrl` | `string` | Base URL of your Vidarax deployment, e.g. `"http://10.0.0.1:8080"`. |
| `options.apiKey` | `string` | Optional `x-api-key` header value sent with every request. |
| `options.maxRetries` | `number` | Maximum automatic retry count for transient failures (default: `3`). |
| `options.retryBaseDelayMs` | `number` | Starting back-off delay in ms (default: `200`). |
| `options.timeoutMs` | `number` | Per-request timeout in ms (default: `30 000`). |

---

## `analyze()` — the main entry point

```typescript
const result = await v.analyze(source, options?)
```

`source` can be:
- A **file path or URI** understood by the server (`"video.mp4"`, `"file:///data/clip.mp4"`, an HTTP URL).
- A **`File` / `Blob`** object — the SDK uploads it first via `uploadFile()`.

Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | `"llama3.2-vision:11b"` | Model to use for analysis. |
| `mode` | `string` | server default | Analysis mode (`"analysis"`, `"realtime"`, etc.). |
| `samplingPolicy` | `"source_fps_adaptive" \| "fixed"` | `"source_fps_adaptive"` | Frame sampling strategy. |
| `fixedFps` | `number` | — | Required when `samplingPolicy` is `"fixed"`. |
| `maxFrames` | `number` | `512` | Upper bound on frames decoded. |
| `windowSize` | `number` | `16` | Analysis window size (2–256). |
| `segmentMs` | `number` | `250` | Segment duration in ms (50–60 000). |

Returns an `AnalysisResult` handle:

```typescript
interface AnalysisResult {
  runId: string
  analyzeResponse: AnalyzeFramesResponse
  events(): AsyncGenerator<AgentEvent>
  markers(): AsyncGenerator<Marker>
}
```

---

## Runs

```typescript
// Create
const run = await v.createRun({ mode: 'analysis', model: 'llama3.2-vision:11b' })

// List (ordered by creation time)
const runs = await v.listRuns()

// Get one
const run = await v.getRun(runId)

// Delete (soft-delete)
await v.deleteRun(runId)

// Stop an active run
await v.stopRun(runId)

// Keepalive for long-running ingests
await v.keepaliveRun(runId)
```

---

## Ingest

Attach a source URI to an existing run and decode frames server-side:

```typescript
const ingest = await v.ingestRun(runId, {
  source_uri: 'file:///data/clip.mp4',
  sampling_policy: 'fixed',
  fixed_fps: 2,
  max_frames: 1000,
})
console.log(`Decoded ${ingest.decoded_frames} frames`)
```

---

## Analysis (low-level)

Run the two-pass analysis pipeline over pre-ingested frames:

```typescript
const result = await v.analyzeRun(runId, {
  model: 'llama3.2-vision:11b',
  window_size: 16,
  segment_ms: 250,
})
console.log(`Generated ${result.generated} frame annotations`)
console.log(`Emitted ${result.markers.length} markers`)
```

---

## Realtime reason

Process a live or file-based stream end-to-end with semantic inference:

```typescript
const result = await v.reason(runId, {
  source_uri: 'rtsp://camera.local/stream',
  model: 'llama3.2-vision:11b',
  semantic_inference: true,
  semantic_prompt: 'Alert when a person enters the frame',
  clip_mode: {
    target_fps: 6,
    clip_length_seconds: 0.5,
  },
})
```

---

## Streaming iterables

Events and markers are exposed as `AsyncGenerator` streams:

```typescript
// Events
for await (const event of v.streamEvents(runId)) {
  console.log(event.seq, event.kind, event.payload)
}

// Markers with filtering
for await (const marker of v.streamMarkers(runId, { status: 'confirmed' })) {
  console.log(marker.event_type, marker.confidence, marker.start_pts_ms)
}
```

---

## Inference

Single prompt:

```typescript
const result = await v.infer('What objects are visible?', {
  model: 'llama3.2-vision:11b',
  max_tokens: 512,
  temperature: 0.2,
})
console.log(result.output_text)
```

Batch (results ordered to match input, failures are per-item rather than throwing):

```typescript
const batch = await v.inferBatch([
  { model: 'llama3.2-vision:11b', prompt: 'Describe scene A' },
  { model: 'llama3.2-vision:11b', prompt: 'Describe scene B' },
], { max_parallel: 4 })

for (const item of batch.results) {
  if (item.ok) {
    console.log(item.result?.output_text)
  } else {
    console.error(item.error?.message)
  }
}
```

---

## Models and health

```typescript
const models = await v.listModels()
models.forEach(m => console.log(m.id, m.tier, m.availability))

const { status } = await v.health()     // { status: 'ok' }

// Poll until healthy (e.g. at application startup)
const healthy = await v.waitUntilHealthy(1_000, 30_000)
```

---

## File upload

```typescript
const file = new File([buffer], 'clip.mp4', { type: 'video/mp4' })
const { file_path } = await v.uploadFile(file, (loaded, total) => {
  process.stdout.write(`\r${Math.round(loaded / total * 100)}%`)
})

// file_path can be used as source_uri
const result = await v.analyze(file_path, { model: 'llama3.2-vision:11b' })
```

---

## WHIP WebRTC (browser only)

```typescript
const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
pc.addTransceiver('video', { direction: 'sendonly' })

const offer = await pc.createOffer()
await pc.setLocalDescription(offer)

const session = await v.whipOffer(offer.sdp!, {
  prompt: 'Watch for unusual motion',
  clip_mode: { target_fps: 6, clip_length_seconds: 0.5 },
})

await pc.setRemoteDescription({ type: 'answer', sdp: session.answerSdp })

// Trickle-ICE
pc.addEventListener('icecandidate', async ({ candidate }) => {
  if (candidate?.candidate) {
    await v.whipIce(session.sessionId, candidate.candidate)
  }
})

// Update prompt mid-stream
await v.whipUpdatePrompt(session.sessionId, { prompt: 'Count the people' })

// Teardown
await v.whipTerminate(session.sessionId)
```

---

## Error handling

All SDK errors extend `VidaraxError` so you can handle them with a single `catch`:

```typescript
import { HttpError, NetworkError, isVidaraxError } from 'vidarax'

try {
  await v.getRun('unknown-run')
} catch (err) {
  if (err instanceof HttpError) {
    console.error(err.status, err.apiError?.code, err.message)
    if (err.isNotFound) { /* 404 */ }
    if (err.isValidationError) { /* 422 */ }
  } else if (err instanceof NetworkError) {
    console.error('Connectivity problem:', err.message)
  } else if (isVidaraxError(err)) {
    console.error('SDK error:', err.code, err.message)
  } else {
    throw err   // re-throw unexpected errors
  }
}
```

### Error hierarchy

```
VidaraxError
  HttpError           — non-2xx HTTP response (.status, .apiError)
  NetworkError        — fetch/connectivity failure
  RetryExhaustedError — all retry attempts failed (.lastError, .attempts)
  UploadError         — multipart upload failure
  ParseError          — unparseable server response (.raw)
```

---

## Retry behaviour

The SDK retries `NetworkError` and server errors (5xx, 429) with exponential
back-off plus jitter.  Client errors (4xx except 429) are surfaced immediately
without retrying.

```
Attempt 1 — immediate
Attempt 2 — ~200 ms
Attempt 3 — ~400 ms
Attempt 4 — ~800 ms   (if maxRetries = 3)
```

Override defaults in the constructor:

```typescript
const v = new Vidarax(url, {
  maxRetries: 5,
  retryBaseDelayMs: 100,
})
```

---

## TypeScript notes

The SDK is written in strict TypeScript with `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes` enabled.  All public types are exported from the
root entry point:

```typescript
import type { Run, Marker, AgentEvent, AnalysisResult } from 'vidarax'
```

---

## Building from source

```bash
cd packages/vidarax-sdk
npm install
npm run build        # compiles to dist/
npm run typecheck    # type-check without emitting
```

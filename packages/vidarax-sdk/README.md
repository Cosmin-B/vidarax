# vidarax

TypeScript SDK for the Vidarax video analysis API.

## Installation

```bash
npm install vidarax
```

Requires Node.js 18+ or any modern browser.

## Quick start

```typescript
import { Vidarax } from 'vidarax'
const v = new Vidarax('http://localhost:8080')
const run = await v.analyze('video.mp4', { prompt: 'Describe what happens' })
for await (const event of run.events()) {
  console.log(event.kind, event.payload)
}
```

## Constructor

```typescript
const v = new Vidarax(baseUrl, options?)
```

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | -- | `x-api-key` header sent with every request. |
| `maxRetries` | `number` | `3` | Retry count for transient failures. |
| `retryBaseDelayMs` | `number` | `200` | Starting back-off delay in ms. |
| `timeoutMs` | `number` | `30000` | Per-request timeout in ms. |

## API summary

| Method | Description |
|---|---|
| `analyze(source, opts?)` | High-level: ingest + analyze + stream results. |
| `createRun(opts)` / `listRuns()` | Create or list runs. |
| `getRun(id)` / `deleteRun(id)` / `stopRun(id)` | Manage individual runs. |
| `ingestRun(id, opts)` | Attach a source and decode frames. |
| `analyzeRun(id, opts)` | Run analysis on ingested frames. |
| `reason(id, opts)` | Realtime inference over a stream. |
| `streamEvents(id)` / `streamMarkers(id)` | Async iterators for results. |
| `infer(opts)` / `inferBatch(items)` | Single or batch inference. |
| `uploadFile(file, onProgress?)` | Upload a video file. |
| `whipOffer(sdp, opts)` | WebRTC WHIP session (browser). |
| `listModels()` / `health()` | Models and health checks. |

## Error handling

All errors extend `VidaraxError`. Subclasses: `HttpError`, `NetworkError`,
`RetryExhaustedError`, `UploadError`, `ParseError`.

## Building from source

```bash
cd packages/vidarax-sdk && npm install && npm run build
```

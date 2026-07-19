# vidarax

TypeScript SDK for the Vidarax video analysis API.

## Installation

The package has not had its first npm release. Build and link it from this
workspace:

```bash
cd packages/vidarax-sdk
npm install
npm run build
npm link
# In your application:
npm link vidarax
```

Requires Node.js 18+ or any modern browser.

## Quick start

```typescript
import { Vidarax } from 'vidarax'
const v = new Vidarax('http://localhost:8080', { apiKey: 'dev-key' })
// analyze() runs the deterministic frame-signal pipeline; it takes no prompt.
const run = await v.analyze('/srv/vidarax-media/video.mp4')
for (const event of await v.getEvents(run.runId)) {
  console.log(event.kind, event.payload)
}
```

For prompt-driven semantic analysis, use `reason(id, opts)` with a
`semantic_prompt` field instead of `analyze()`.

## Constructor

```typescript
const v = new Vidarax(baseUrl, options?)
```

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | -- | `x-api-key` header sent with every request. |
| `tenantId` | `string` | -- | `x-tenant-id` header sent with every request. Servers started with `VIDARAX_REQUIRE_TENANT_ID` reject requests without it. |
| `maxRetries` | `number` | `3` | Retry count for transient failures. |
| `retryBaseDelayMs` | `number` | `200` | Starting back-off delay in ms. |
| `timeoutMs` | `number` | `30000` | Per-request timeout in ms. |

## API summary

| Method | Description |
|---|---|
| `analyze(source, opts?)` | High-level: ingest and analyze, then return a run handle. |
| `createRun(opts)` / `listRuns()` | Create or list runs. |
| `getRun(id)` / `deleteRun(id)` / `stopRun(id)` | Manage individual runs. |
| `ingestRun(id, opts)` | Attach a source and decode frames. |
| `analyzeRun(id, opts)` | Run analysis on ingested frames. |
| `reason(id, opts)` | Realtime inference over a stream. |
| `getEvents(id, index?)` / `getMarkers(id, query?)` | Fetch one snapshot of events or markers. |
| `getInteractions(id, index?)` | Fetch guided semantic interactions. |
| `getKeyframe(id, sha256)` | Fetch a run-owned keyframe as a raw JPEG `Blob`. |
| `streamEvents(id)` / `streamMarkers(id)` | Async-iterate a one-time snapshot of results (not a live stream; the server has no SSE endpoint). |
| `infer(opts)` / `inferBatch(items)` | Single or batch inference. |
| `uploadFile(file, onProgress?)` | Upload a video file. |
| `whipOffer(sdp, opts)` | WebRTC WHIP session (browser). |
| `whipUpdatePrompt(id, request)` | Update the prompt and optional JSON Schema object. |
| `listModels()` / `health()` | Models and health checks. |

## Error handling

All errors extend `VidaraxError`. Subclasses: `HttpError`, `NetworkError`,
`RetryExhaustedError`, `UploadError`, `ParseError`.

## Building from source

```bash
cd packages/vidarax-sdk && npm install && npm run build
```

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
`semantic_prompt` field. `analyze()` runs the deterministic frame-signal path.

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
| `getRun(id)` / `deleteRun(id)` / `stopRun(id)` / `keepaliveRun(id)` | Manage individual runs. |
| `getRunState(id)` | Read the state derived from the run timeline. |
| `ingestRun(id, opts)` | Attach a source and decode frames. |
| `analyzeRun(id, opts)` | Run analysis on ingested frames. |
| `reason(id, opts)` | Realtime inference over a stream. |
| `getEvents(id, index?)` / `getMarkers(id, query?)` | Fetch one snapshot of events or markers. |
| `getInteractions(id, index?)` | Fetch guided semantic interactions. |
| `getKeyframe(id, sha256)` | Fetch a run-owned keyframe as a raw JPEG `Blob`. |
| `streamEvents(id)` / `streamMarkers(id)` | Async-iterate a one-time compatibility snapshot. |
| `subscribeEvents(id, options?)` | Replay and follow events over SSE with `Last-Event-ID` reconnect. |
| `createWebhook(id, request)` / `listWebhooks(id)` / `deleteWebhook(id, webhookId)` | Manage signed action hooks and inspect dead-letter state. Save the per-hook signing secret returned only at creation. |
| `query(request)` / `search(query, opts?)` | Read timeline events by cursor or search stored VLM descriptions. |
| `infer(opts)` / `inferBatch(items)` | Single or batch inference. |
| `uploadFile(file, onProgress?)` | Upload a video file. |
| `whipOffer(sdp, opts)` | WebRTC WHIP session (browser). |
| `whipUpdatePrompt(id, request)` | Update the prompt and optional JSON Schema object. |
| `submitFeedback(id, feedback)` / `listFeedback()` | Store and list durable operator feedback. |
| `createPolicy(id, request)` / `listPolicies(id)` / `getPolicy(id, revision)` | Create and inspect immutable policy revisions. |
| `activatePolicy` / `rollbackPolicy` / `replayPolicy` | Exercise the staged policy control loop. |
| `compileTrigger(source)` / `validateTrigger(program)` / `evaluateTrigger(program, samples)` | Compile and replay bounded trigger programs before live attachment. |
| `whipIce(id, candidate)` / `whipTerminate(id)` | Trickle ICE or end a live WHIP resource. |
| `listModels()` / `health()` / `waitUntilHealthy(opts?)` | Models and health checks. |

## Trigger programs

```typescript
const compiled = await v.compileTrigger(`
trigger loading-bay-entry version 1
when motion_score >= 0.40 for 2 frames
edge rising
cooldown 5000ms
emit loading_bay_entry
capture keyframe
notify webhook
end
`)

await v.validateTrigger(compiled.program)
const replay = await v.evaluateTrigger(compiled.program, [
  { pts_ms: 0, motion_score: 0.1 },
  { pts_ms: 33, motion_score: 0.6 },
  { pts_ms: 66, motion_score: 0.7 },
])
```

Pass `compiled.program` as `trigger_program` in `whipOffer`. Trigger event kinds
are namespaced as `trigger.<event_type>`. Binary keyframes stay in the server's
content-addressed store. Timeline and webhook payloads carry references only.

## Error handling

All errors extend `VidaraxError`. Subclasses: `HttpError`, `NetworkError`,
`RetryExhaustedError`, `UploadError`, `ParseError`.

## Building from source

```bash
cd packages/vidarax-sdk && npm install && npm run build
```

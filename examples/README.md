# Examples

TypeScript demos that run against a local server with `npx tsx`. Each one
imports the SDK straight from `packages/vidarax-sdk/src`, so no build step is
needed. Start the server first, then:

```bash
VIDARAX_API_KEY=your-key npx tsx examples/<name>.ts
```

- `sdk-demo.ts`: the happy path, from health check through analysis, events, markers, inference, and search.
- `whip-live-demo.ts`: the WHIP live-session signalling flow (offer, trickle ICE, live prompt update, terminate), with marked placeholders where a browser or GStreamer attaches real media and handling for the prompt update's 409/503 outcomes.
- `error-handling-demo.ts`: the typed error surface, catching a real 404 as `HttpError` and an unreachable server as `RetryExhaustedError` wrapping a `NetworkError`.

## CLI walkthrough

The same flows are available from the `vidarax` CLI:

```bash
# Check local config and API readiness before anything else.
vidarax doctor

# Upload a local video and run the full analysis pipeline. Uses the default
# model (Qwen/Qwen3-VL-2B-Instruct) and skips the separate ingest pass
# unless you add --with-ingest.
vidarax analyze video.mp4

# Analyze a source the server can reach directly (http(s), rtsp, hls, or a
# server-local path) and skip the upload step.
vidarax analyze --source-uri rtsp://camera.local/stream

# Stop a run without deleting its history. This also closes the run's live
# WHIP session, if it has one.
vidarax runs stop <run_id>
```

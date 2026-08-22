# Examples


These TypeScript examples run against a local server with `npx tsx`. Each one
imports the SDK straight from `packages/vidarax-sdk/src`, so no build step is
needed. Start the server first, then:

```bash
VIDARAX_API_URL=http://127.0.0.1:8080 \
VIDARAX_API_KEY=dev-key \
npx tsx examples/<name>.ts
```

- `sdk-demo.ts` checks health, starts analysis, and reads events, markers, inference results, and search results from the server.
- `whip-live-demo.ts` sends an SDP offer, trickles ICE, updates the live prompt,
  and terminates the session. A browser or GStreamer must attach the media track
  where the file marks that integration point. The example handles 409 and 503
  prompt-update responses.
- `error-handling-demo.ts` catches a server 404 as `HttpError` and an unreachable server as `RetryExhaustedError` wrapping a `NetworkError`.

## CLI walkthrough

The same flows are available from the `vidarax` CLI:

```bash
export VIDARAX_API_URL=http://127.0.0.1:8080
export VIDARAX_API_KEY=dev-key

# Check local config and API readiness before anything else.
vidarax doctor

# Upload a local video and run the full analysis pipeline. Uses the default
# model (Qwen/Qwen3-VL-2B-Instruct) and skips the separate ingest pass
# unless you add --with-ingest.
vidarax analyze video.mp4

# Analyze a source the server can reach directly and skip the upload step.
# Local paths need an allowed ingest root. Remote HLS and unencrypted HTTP or
# RTSP remain disabled until the corresponding server setting enables them.
vidarax analyze --source-uri rtsps://camera.example.com/stream

# Stop a run without deleting its history. This also closes the run's live
# WHIP session, if it has one.
vidarax runs stop <run_id>
```

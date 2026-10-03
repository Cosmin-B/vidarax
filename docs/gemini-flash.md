# Gemini Flash media review

Implementation status: the compatibility and dense review changes are available
on the `gemini-3-8-flash` review branch, pending PR review. Validation used offline
fixtures and mock inference. Publishing this documentation does not deploy the
Rust API or demonstrate live Gemini billing, sampling, or review quality.

Verified on 2026-10-03 against Google's [model catalog](https://ai.google.dev/gemini-api/docs/models)
and [model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash):
the latest stable full Flash model is `gemini-3.8-flash`. It supports image,
video, audio and structured JSON. `gemini-flash-latest` resolves locally to
this ID. Flash-Lite aliases and low-cost backend/UI/CLI defaults are unchanged.

Configure the existing Gemini backend with `model = "gemini-3.8-flash"`.
The commented backend in `vidarax.toml` illustrates the shape. Use existing
credential management. Explicit model routing recognizes aliases in both
inference tiers and does not silently reroute a full Flash failure to a local
model. An omitted Gemini backend model still defaults to Flash-Lite.

Image input uses inline JPEG/PNG; native video uses source-time MP4 clips and
the Gemini File API. Responses join visible text parts while excluding thought
summaries, preserve token usage, and retain the thinking-budget retry. Vidarax
leaves thinking level to Google's default; it does not request unsupported
`minimal` thinking. The benchmark's illustrative standard rate is $0.75 input /
$3.75 output per million tokens through 2026-12-31, then $1.50 / $7.50 from
2027-01-01. Check the [rate card](https://ai.google.dev/gemini-api/docs/pricing)
and use benchmark overrides for current billing.

## Caller-directed recorded review

`semantic_prompt` and `output_schema` are preserved in both inference tiers.
General default prompts remain unchanged. Supply your own visible-evidence
criteria and schema when the review needs more than a generic event description.
Custom output is available as `raw_output`; a custom schema without `moments`
does not automatically become timestamped moment events.

For dense frame review, a `/v1/runs/{run_id}/reason` body can include:

```json
{
  "source_uri": "/path/to/video.mp4",
  "model": "gemini-3.8-flash",
  "sampling_policy": "fixed",
  "fixed_fps": 60,
  "max_frames": 3600,
  "chunk_size": 60,
  "semantic_frames_per_chunk": 120,
  "semantic_context_frames": 30,
  "source_start_ms": 10000,
  "source_end_ms": 14000,
  "semantic_prompt": "Inspect this interval for the caller's visible criteria; distinguish observation from uncertainty."
}
```

Frame mode allows 1–256 images per chunk (default 2), selected from the owned
chunk plus up to 128 context samples on each side. Increasing the count alone
does not increase local decoding FPS. To inspect every sample, choose a count
at least as large as chunk plus context. Large windows still select uniformly
when the count is smaller. Each supplied image has an actual sampled source
frame index and PTS in the prompt and `input_image_timestamps`; context reuse
does not duplicate stateful analysis metadata. A visual-diff previous image is
identified separately. Context is bounded by the selected source interval.

For native media, use `media` instead of frame context:

```json
{
  "mode": "audio_video",
  "window_ms": 2000,
  "overlap_ms": 1000,
  "video_fps": 24,
  "resolution": "high",
  "persist_evidence": true
}
```

Attach this object under `media` in the same reason request. `video` omits audio;
`audio_video` requires an audio track. Clips use exact source-time boundaries
and retain normal-speed source frames. Overlap may be at most half the window;
windows are 100–60000 ms. Both tiers receive the same clip, prompt, schema and
FPS. Overlapping identical moments are deduplicated by absolute source times;
adjacent repeated actions remain separate. Semantic events are journaled in
chunk order even when inference completes out of order.

CLI equivalents include `--semantic-frames-per-chunk`,
`--semantic-context-frames`, `--source-start-ms`, `--source-end-ms`,
`--media-window-ms`, `--media-overlap-ms`, and `--video-fps`. CLI mode spelling is
`audio-video`; JSON uses `audio_video`. TypeScript request types expose the same
optional controls.

## Capabilities, budgets and evidence limits

Both tier providers must declare the required capacity. Gemini accepts up to
256 images and explicit video FPS in `(0,24]`. An OpenAI-compatible backend is
conservatively limited to 5 images by default; set its `max_input_images` to a
verified value (1–256) for a backend supporting more. Native FPS requires a
provider declaring that capability. Dense and native requests disable fallback
that could discard controls.

Requests reject invalid intervals, unsupported capabilities, excessive overlap,
more than 2048 review chunks/windows, more than 10000 unique JPEGs, or more than
20000 image submissions counting context reuse and both possible tiers. JPEG
pipe output is bounded at 256 MiB and per-request base64 image data at 32 MiB.
Each recorded JPEG/clip/audio child has a 120-second deadline with kill,
reap and bounded pipe draining. A cancelled blocking task can finish after the
caller leaves; audio-video plus local WAV preparation can use two sequential
child deadlines. Immediate cancellation is not promised.
Native clips are limited to 64 MiB each and aggregate retained native evidence
to 256 MiB; a runtime extraction/input budget failure is recorded as an explicit
semantic error rather than silently thinning evidence. Split or downscale large
reviews. Native overlap also increases provider work and cost.

Explicit dense/target/FPS requests reject `max_frames` truncation. Targeting
currently filters after local source decoding: the full-source scan budget must
cover the source, even for a late narrow interval. A range selects available
samples; it cannot recover motion between them. Provider video FPS is separate
from local `fixed_fps` and does not guarantee observation of every source frame.

The [generateContent reference](https://ai.google.dev/api/generate-content#VideoMetadata)
and current [v1beta discovery schema](https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta)
still list `Part.videoMetadata.fps` (default 1, range `(0,24]`) and
`Part.mediaProcessing: STATIC`. These are Vidarax's serialized controls. Both
mark `VideoMetadata` deprecated and name `GenerateContentRequest.processing_options`
as its successor, but neither defines that request field or its JSON shape.
Schema presence does not prove that the live service honors it for a given model.
Vidarax retains the documented field rather than inventing a replacement payload.

The [current video guide](https://ai.google.dev/gemini-api/docs/video-understanding#set-a-custom-frame-rate)
documents `input[].processing = {"type":"static","fps":...}` for the separate
`/v1beta/interactions` API; this is not a documented drop-in generateContent
request field. Interactions/agentic processing is not implemented in Vidarax.
The guide warns default 1 FPS sampling may miss rapid motion.

Provider capability checks report ability to serialize a documented FPS request,
not measured or guaranteed effective density. `provider_sampling_interval_ms`
is the requested interval; `provider_sampling_status` is `requested_unverified`
when FPS is specified. No service acknowledgment/effective-rate measurement is
available in this path. `timestamp_resolution_ms` remains conservative (1000 ms
for Gemini). None of these fields guarantees observation of every source frame
or localization accuracy. Live acceptance and effective sampling still require
an authorized evaluation.

## Validation and visual-review quality

Validation is offline: catalog/aliases, routing, image/video payloads, visible
response parsing, custom schema in both tiers, source timing and resource
rejections. Synthetic 60 FPS footage includes a 200 ms pulse and audio; tests
verify dense ordered images and complete action preservation across overlapping
video and audio-video clips. Inference is mocked. No credentials were configured
and no paid request was made. Actual model access, upload processing, deprecated
FPS-control acceptance, latency, billing and visual judgment require an
authorized live evaluation.

Model support alone does not establish reliable visual reviews. Telemetry such
as state changes or animation triggers cannot prove visible contact, path
quality, sliding, or IK attachment. Caller-specific prompts, sufficient temporal
and spatial evidence, replayable findings, and evaluations against known flaws
remain necessary. This code review does not establish which footage, prompts or
controls Dreamlit uses; Dreamlit was not modified.

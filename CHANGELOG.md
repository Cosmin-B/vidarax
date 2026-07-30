# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project has not tagged a
release yet.

## [Unreleased]

### Added

- Recorded audio-video reasoning with bounded source-time windows, deterministic
  audio-track mixing, Gemini File API uploads, timestamped `multimodal_moment`
  events, and content-addressed MP4 evidence.
- Selective Whisper large-v3-turbo transcription behind Silero VAD, grounded
  provider output, and boundary-safe 20-second local-audio windows.
- A locked install-on-first-use audio runtime plus a repository Agent Skill for
  isolated recorded-media reviews.
- Live WHIP Opus analysis through bounded queues, in-memory Ogg framing,
  ffmpeg PCM extraction, the local binary audio sidecar, durable moments, and
  queue-drop telemetry.
- Native media responses can omit per-frame metadata while durable chunk and
  moment events retain the useful results.
- Gemini 3.5 Flash-Lite and Gemini 3.6 Flash in the model catalog, with
  request-level media resolution and cleanup of temporary Gemini File API
  uploads.
- `vidarax analyze --media audio-video` plus matching API, TypeScript SDK, Vue
  interface, and Prometheus telemetry.
- Supervised live media pipeline generations: every worker of a session joins
  one stage-tagged generation, and the first unexpected exit faults the whole
  set, closes the peer, and joins the siblings.
- Process-wide media admission budgets, reserved before `run_created`, via
  `VIDARAX_MEDIA_MEMORY_BUDGET_BYTES` and
  `VIDARAX_MEDIA_WORKER_THREAD_BUDGET`.
- A deadline-aware provider scheduler with fair queues across principals,
  streams, and urgent live, live, and offline work.
- Cursor-based SSE with WAL replay, stable CloudEvent IDs, bounded subscriber
  queues, and SDK reconnect support.
- Signed event webhooks with filters, retries, dead-letter state, and delivery
  metrics.
- A bounded trigger language exposed through the API, CLI, and TypeScript SDK,
  with live WHIP execution and metadata-only local actions.
- A content-addressed binary store for selected JPEGs. Timeline events carry
  hashes and authenticated references, never JSON/base64 image payloads.
- A signed edge update controller with enrollment, anti-replay sequence checks,
  shadow and canary health checks, activation hooks, and rollback.
- `vidarax_pipeline_detached_workers_total` metric counting worker threads
  left running past the join deadline of a forced shutdown.
- CLI verbs `vidarax runs stop` and `vidarax runs keepalive`.
- CLI `vidarax analyze --source-uri` for sources the server can reach
  directly, without uploading a local file.
- SDK `tenantId` client option, sent as the `x-tenant-id` header, and the
  `expired` run status.
- Frame-metadata schema fields `sampling_policy`, `sample_fps`, and
  `finish_reason`, held in sync with the live response type by a schema-sync
  test.
- Axum API for run lifecycle, ingest, analysis, search, inference, feedback,
  file upload, health, and Prometheus metrics.
- WebRTC ingest over WHIP, including offer exchange, ICE trickle, session
  termination, and prompt updates for live sessions.
- TypeScript SDK with run creation, ingest and analysis helpers, streaming
  iterators, WebRTC attach support, and batch inference helpers.
- Vue 3 UI for local operation and inspection.
- SpacetimeDB module and optional API client code. The production server path
  currently persists through the WAL.
- WAL-backed run timeline under the configured data directory.

### Changed

- The default EfficientAT model is `dymn10_as`. Gameplay sound mapping now
  includes engines, whooshes, hisses, wind, mechanisms, and scrape cues.
- Provider sound wording requires a matching local sound observation whenever
  local audio is requested. This also applies when the provider marks the
  moment as video-only or the local sidecar fails.
- The generation join deadline is derived from the VLM pass timeouts, the
  configured backend fallback count, the admission wait, and the novelty
  embedding timeout. Teardown during an in-flight call is no longer measured
  against a flat five-second deadline.
- A forced shutdown keeps the session's media reservation, because detached
  worker threads still hold that memory until process exit.
- REST run stop and delete now close a live WHIP session after recording the
  intent. Stop preserves the run's history, so the session reclaimer skips the
  tombstone for that close.
- Deleted runs reject further event appends, so a worker that outlives its
  run cannot write past the tombstone.
- The CLI default analyze model is `Qwen/Qwen3-VL-2B-Instruct`.
- The CLI config file is read from `VIDARAX_CLI_CONFIG`, because the server
  already owns `VIDARAX_CONFIG` for its backend TOML path.
- `vidarax analyze` skips the ingest pass by default, since reason decodes
  the source itself. `--with-ingest` opts back in.
- The CLI retries transient request failures with the same status
  classification the TypeScript SDK uses.
- SDK `whipUpdatePrompt` documents its 409 (generation closed or replaced)
  and 503 (acknowledgement timeout, command discarded, retry) outcomes.
- Feedback commits to the local WAL before returning success. A configured
  SpacetimeDB service receives a best-effort mirror after the local commit.
- `GET /v1/files/{filename}` failures now use the structured JSON error
  envelope.
- Inference uses a provider chain with priority order and fallback for
  OpenAI-compatible vLLM and SGLang backends.
- Active stream limits apply per resolved principal, derived from the API key
  when authentication is enabled.
- Remote media ingest validates source URLs before decode and prefetches
  downloadable HTTP(S) media to a bounded local file.

### Security

- API-key authentication is enabled by default, and metrics can require the
  same key set.
- Ownership for runs and uploaded files derives from the authenticated
  principal, not the caller-controlled `x-tenant-id` header.
- Remote fetch rejects embedded credentials, localhost names, private and
  link-local IP literals, blocked DNS resolutions, unsafe redirects, and
  content-sniffed HLS playlists.

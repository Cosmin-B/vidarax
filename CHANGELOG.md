# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project has not tagged a
release yet.

## [Unreleased]

### Added

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

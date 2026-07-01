# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project has not tagged a
release yet.

## [Unreleased]

### Added

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

---
title: What is vidarax
description: A self-hosted video intelligence engine that turns video streams into structured semantic events.
---

[← Vidarax](/)

Vidarax is a self-hosted video intelligence engine. Video streams go in, structured semantic events come out.

It decodes live or file-based video, runs a deterministic gate engine over every frame (scene cuts, flicker, ghosting, exposure shifts, loop detection), samples a subset of frames through tiered vision language models for semantic analysis, and emits structured events in real time. The server, storage, and event pipeline run on infrastructure you operate. Inference goes to a VLM backend you configure: a self-hosted OpenAI-compatible endpoint (vLLM or SGLang), or optionally Google's Gemini API. No SaaS dependency is required, but the Gemini integration exists for deployments that want it.

The engine is a Rust workspace. Axum serves the HTTP API. Events commit to a local write-ahead log; selected keyframe JPEGs live in a content-addressed blob sidecar. An optional SpacetimeDB client mirrors blocking WHIP description events after the local commit. A TypeScript SDK, a Vue 3 UI, and Prometheus-format metrics sit on the consumer side.

## Who it is for

Vidarax is for teams that run their own inference and need machine-readable answers about what is happening in video:

- Operators who point the server at an OpenAI-compatible VLM backend (vLLM or SGLang) they already run, or at Gemini through the TOML backend config.
- Applications that analyze recorded files: upload or reference a video, receive a timeline of markers and semantic events.
- Applications that watch live streams: WebRTC via WHIP, RTSP cameras, and HLS sources, with a prompt that can be updated while the session runs.
- Consumers who want events as data: a REST API with sequence-numbered events, cross-run query and search, and a typed SDK.

## What it is not

- It is not a hosted service. You deploy the server, the model backend, and the storage.
- It does not ship or serve models. A deployment needs a configured inference backend, either an OpenAI-compatible endpoint or Gemini; without one, inference routes fail.
- It is not a source-video archive. The WAL holds event metadata and selected keyframes are stored as JPEG blobs, but source video is not retained by the event store.
- It is not an authorization layer for multi-tenant products on its own. Ownership derives from the authenticated API-key principal; the `x-tenant-id` header is metadata, not a security boundary.

## Where to go next

- [Quickstart](/docs/quickstart/): run the server and get events from a video.
- [Architecture](/docs/architecture/): the control plane, the media plane, and how state persists.
- [API reference](/docs/api/): endpoints and configuration.

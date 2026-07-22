---
title: What is vidarax
description: A self-hosted video intelligence engine that turns video streams into structured semantic events.
---

[← Vidarax](/)

Vidarax turns continuous video into actionable, replayable assertions while spending expensive model calls only where meaning changes.

It decodes live or file-based video, runs a deterministic per-frame filter, sends selected frames through tiered vision-language models, and emits structured events. The server, storage, and event pipeline run on infrastructure you operate. Inference goes to a self-hosted OpenAI-compatible endpoint such as vLLM or SGLang, or to Google's Gemini API through the backend configuration.

The engine is a Rust workspace with an Axum HTTP API. Events commit to a local write-ahead log. Selected JPEGs live in a content-addressed binary store and events carry their references. An optional SpacetimeDB client mirrors blocking WHIP descriptions after the local commit. Consumers can use the TypeScript SDK, the Vue 3 interface, cursor-based SSE, signed webhooks, or the REST API.

## Who it is for

Vidarax is for teams that run their own inference and need machine-readable answers about what is happening in video:

- Operators who already run an OpenAI-compatible VLM backend or want to call Gemini through the TOML backend configuration.
- Applications that analyze recorded files and need a durable timeline of markers and semantic events.
- Camera applications that ingest WebRTC through WHIP, RTSP, or HLS and need acknowledged configuration changes while a session runs.
- Event consumers that need sequence cursors, WAL replay, filtering, signed delivery, cross-run search, and a typed SDK.

## Current operating boundary

Vidarax is a self-hosted engine, so the operator supplies the server, model backend, and storage. The event store retains metadata and selected JPEGs, not the source stream. Run ownership derives from the authenticated API-key principal. `x-tenant-id` is descriptive metadata and must not be used as an authorization boundary.

## Where to go next

- [Quickstart](/docs/quickstart/): run the server and get events from a video.
- [Architecture](/docs/architecture/): the control plane, the media plane, and how state persists.
- [API reference](/docs/api/): endpoints and configuration.

---
title: What is vidarax
description: A self-hosted media intelligence engine that turns video and synchronized sound into structured events.
---

[← Vidarax](/)

Vidarax turns continuous video and synchronized sound into actionable,
replayable assertions while spending expensive model calls only where meaning
changes.

It decodes live or file-based video, runs a deterministic per-frame filter,
sends selected frames through tiered vision-language models, and emits
structured events. Recorded media can also be analyzed as synchronized audio
and video windows. Gemini receives raw MP4 files and returns timestamped
moments for speech intent, non-speech sound, visible actions, and their
relationship.

The engine is a Rust workspace with an Axum HTTP API. Events commit to a local
write-ahead log. Selected JPEGs, retained MP4 windows, and generated WAV
feedback live in content-addressed binary stores while events carry references.
Media bytes never enter JSON, SSE, webhooks, or the WAL. Consumers can use the
TypeScript SDK, the Vue 3 interface, cursor-based SSE, signed webhooks, or the
REST API.

## Who it is for

Vidarax is for teams that run their own inference and need machine-readable answers about what is happening in video:

- Operators who already run an OpenAI-compatible VLM backend or want to call Gemini through the TOML backend configuration.
- Applications that analyze recorded files and need a durable timeline of
  visual, audible, and combined moments.
- Camera applications that ingest WebRTC through WHIP, RTSP, or HLS and need acknowledged configuration changes while a session runs.
- Event consumers that need sequence cursors, WAL replay, filtering, signed delivery, cross-run search, and a typed SDK.

## Current operating boundary

Vidarax is a self-hosted engine, so the operator supplies the server, model
backend, and storage. The event store retains metadata, selected JPEGs, MP4
windows, and generated WAV feedback when requested. It never retains the full
source automatically.
Run ownership derives from the authenticated API-key principal. `x-tenant-id`
is descriptive metadata and must not be used as an authorization boundary.

## Where to go next

- [Quickstart](/docs/quickstart/): run the server and get events from a video.
- [Architecture](/docs/architecture/): the control plane, the media plane, and how state persists.
- [API reference](/docs/api/): endpoints and configuration.
- [Local audio perception](/docs/audio/): sound events, selective ASR, and spoken feedback.
- [Mage-VL debug modes](/docs/mage-vl/): compare visual tokens and inspect proactive streaming.

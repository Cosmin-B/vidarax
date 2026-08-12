---
title: What is vidarax
description: A self-hosted media runtime that turns live streams and recordings into an ordered event timeline.
---

<!-- status: draft, needs Cosmin's rewrite pass before publication -->

[Product site](/) ·
[GitHub](https://github.com/Cosmin-B/vidarax) ·
[A video stream through Vidarax](https://cosminbararu.com/blog/video-stream-through-vidarax)

Vidarax is a self-hosted media runtime for applications that need ordered,
timestamped records from live video or recorded media. The application sends a
source and a prompt or deterministic analysis request. Vidarax owns decoding,
selection, model routing, event persistence, and delivery.

The primary output is an append-only timeline. One writer assigns each event a
monotonic sequence number, which becomes the recovery cursor for REST, SSE,
and signed webhooks. The envelope timestamp is commit time. Source-relative
frame, clip, and audio timestamps live inside event payloads.

## Who it is for

Vidarax is for developers building camera, fleet, facility, and robotics
software who already operate part of their inference stack. It fits when the
application needs to:

- consume a durable cursor instead of polling a model call
- retain selected media without placing image or clip bytes in JSON
- keep live stream work ordered while bounding queues and process-wide
  inference admission
- choose between local OpenAI-compatible providers and an explicitly configured
  Gemini backend

## What goes through the system

Files, uploads, HTTP(S), HLS, and RTSP sources enter the recorded path. WHIP
accepts live WebRTC video and optional Opus audio. The deterministic frame path
supports CPU ffmpeg, NVIDIA NVDEC, and Apple VideoToolbox for selective JPEG
extraction. Selected images can reach vLLM, SGLang, MLX, or Gemini through the
configured provider chain.

Recorded `video` and `audio_video` modes are narrower. They extract bounded MP4
windows and use Gemini File API for native synchronized media. The optional
local audio sidecar handles sound events and selective speech recognition for
recorded windows and live WHIP Opus.

Every live session reserves a media and worker envelope before its durable run
is created. Each handoff is bounded. A process-wide inference scheduler also
limits concurrency, queued callers, output tokens, encoded media bytes, and
deadline misses across streams.

The deterministic frame filter runs before image inference. Live WHIP capture
can add an opt-in embedding-only novelty gate that reuses the last successful
description inside time and cumulative-drift bounds. A sidecar failure runs the
VLM. File analysis does not traverse this gate, and its threshold must be
calibrated on representative ordered frames.

## Current operating boundary

The operator supplies the server, storage, ffmpeg, and model services or API
credentials. Vidarax retains timeline metadata plus selected JPEGs, requested
MP4 windows, and generated WAV files in content-addressed stores. It does not
retain the full source automatically.

The timeline WAL is flushed on append but is not fsynced, so it is process-crash
safe rather than power-loss safe. Automatic retention and orphaned-blob cleanup
are not implemented. Older run reads may scan the WAL after their in-memory
tail is evicted.

Run ownership derives from the authenticated API-key principal. `x-tenant-id`
is descriptive metadata and is not an authorization boundary.

## Where to go next

- [Quickstart](/docs/quickstart/): run the server and get events from a video.
- [Agent workflows](/docs/agents/): install on first use and review media from a compatible agent harness.
- [Architecture](/docs/architecture/): ordering, bounded media work, persistence, and delivery.
- [Ingest](/docs/ingest/): accepted sources, codecs, and decode backends.
- [The per-frame filter](/docs/gate/): deterministic selection and live semantic novelty.
- [Events and SDK](/docs/events/): cursor semantics, media references, SSE, and webhooks.
- [API reference](/docs/api/): endpoints and configuration.
- [Local audio perception](/docs/audio/): sound events, selective ASR, and spoken feedback.
- [Deployment](/docs/operations/): process configuration and operational checks.

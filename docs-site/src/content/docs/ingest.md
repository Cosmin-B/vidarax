---
title: Ingest
description: Supported sources, the ffmpeg decode sidecar, the drain-before-write rule, and decoder warm-up.
---

Ingest turns a source into decoded frames the per-frame filter can score. There are two decode paths: an ffmpeg subprocess path for files and URLs, and a long-lived decoder pipeline for live WebRTC streams.

## Supported sources

- Local files: video files (for example MP4), referenced by path or `file://` URI. Paths must sit under a directory listed in `VIDARAX_INGEST_FILE_ROOTS`; the list defaults to empty and paths are canonicalized at startup.
- Uploads: `POST /v1/upload` stores a file under a dedicated upload root, owned by the uploading principal.
- HTTP(S) URLs: downloadable media is validated and prefetched to a bounded local file before decode. Plain `http://` sources require `VIDARAX_ALLOW_INSECURE_HTTP=true`.
- HLS: `.m3u8` manifests, handled natively by ffmpeg's `hls` demuxer. Remote manifests require `VIDARAX_ALLOW_REMOTE_HLS=true`.
- RTSP cameras: `rtsps://` is accepted; unencrypted `rtsp://` requires `VIDARAX_ALLOW_UNENCRYPTED_RTSP=true`.
- WebRTC: live streams over WHIP (RFC 9725), processed through per-session worker pools rather than ffmpeg's demuxer.

Remote sources pass application-level SSRF checks before decode: embedded credentials, localhost names, private and link-local IP literals, blocked DNS resolutions, and unsafe redirects are rejected. On the downloadable HTTP(S) path, a response that content-sniffs as an HLS playlist is also rejected; an explicitly selected HLS source is accepted when remote HLS is enabled. Each source kind also gets the narrowest useful ffmpeg protocol whitelist. See the [security notes](/docs/operations/#security-and-hardening) for the residual that remains and the recommended egress control.

## Decode backends for files and URLs

File and URL decode goes through a backend registry with two phases: a signal-extraction pass that computes per-frame statistics for the gate, then selective JPEG extraction for only the frames the gate keeps. Callers swap backends without touching handler code.

| Backend | Selection | Behavior |
|---------|-----------|----------|
| `cpu-ffmpeg` | `cpu`, `ffmpeg`, `cpu-ffmpeg` | ffmpeg subprocess, CPU decode, CPU JPEG encode. Works everywhere. |
| `nvdec-cuda` | `nvdec`, `cuda`, `nvdec-cuda`, `gpu` | ffmpeg with `-hwaccel nvdec` decodes on the GPU; frames are downloaded and JPEG-encoded on the CPU. Requires an NVIDIA GPU. |
| `videotoolbox` | `mlx`, `apple`, `metal`, `videotoolbox` | ffmpeg uses VideoToolbox for the selective JPEG phase on Apple Silicon; the signal pass remains on CPU. If hardware initialization fails for an input, ffmpeg may fall back to software decode. |

Any other `VIDARAX_DECODE_BACKEND` value fails at startup with an unknown-backend error.

`VIDARAX_DECODE_BACKEND=auto` (the default) probes for NVIDIA hardware with `nvidia-smi` and picks `nvdec-cuda` when it is present, otherwise `cpu-ffmpeg`. The decode backend is independent of the VLM backend.

Ingest requests control sampling: a `sampling_policy` (for example `fixed` with a `fixed_fps` value) and a `max_frames` cap bound how much of the source is decoded.

## The ffmpeg decode sidecar for live streams

Live WebRTC video needs a stateful decoder that survives across frames. GPU H.264 and H.265 use a long-lived ffmpeg sidecar process; software H.264 uses openh264 in-process; software H.265 uses the ffmpeg sidecar; VP8 uses libvpx in-process when the `vp8` build feature is enabled.

The sidecar paths are built around one hazard: a subprocess connected by two pipes can deadlock. If the parent blocks writing encoded input while ffmpeg blocks writing decoded output, neither side makes progress. Vidarax narrows that interleaving window with a dedicated reader thread and a strict ordering rule.

### The reader thread and the bounded channel

A reader thread owns ffmpeg's stdout. It continuously reads complete YUV frames and hands them to the decode side through a bounded channel using blocking sends. The handoff is lossless: the reader never drops or evicts a decoded frame; when the channel is full, the reader blocks.

Each frame is read directly into buffers acquired from a bounded pool. Recycled buffers keep their capacity, so once the pool has been through its first cycles the steady-state read loop does not allocate. The pool is sized to cover every place a frame can legally exist at once: the full reader channel, the decoder's small pending FIFO, the frame the reader is currently assembling, and the frame the consumer currently holds.

### The drain-before-write rule

The decode side owns ffmpeg's stdin, and it follows one ownership rule: drain before write. Every `decode()` call first drains all currently ready YUV frames out of the bounded channel into a decoder-local FIFO, and only then writes the next encoded input to ffmpeg.

The channel holds 16 frames and the reader uses blocking sends. Draining it before each write makes room for output already produced, but the reader can still block when a new output burst fills the handoff. The current implementation does not establish a maximum decoded-output burst per input write, so this ordering reduces the deadlock window rather than proving it absent. No decoded output is dropped by the handoff itself.

Under real-time backlog the decoder-local FIFO sheds older decoded frames and returns the freshest ready frame, so downstream labels stay close to the current stream position. Shedding happens after the lossless handoff, as an explicit freshness policy, not as a side effect of a full pipe.

### Decoder warm-up

A live H.264 decoder commonly cannot emit a frame from its first input: it needs the SPS and PPS parameter sets and an IDR frame before it can produce output, and those often arrive across several access units. (A first access unit that already carries all three can produce output immediately; the pipeline may simply buffer until the decoder has what it needs.) During warm-up the parent must keep writing input while nothing is coming back, which is exactly the phase where a naive write-then-read loop stalls. The reader thread makes writes and reads concurrent, so warm-up completes without special-casing.

Warm-up also applies to memory: pooled frame buffers are grown to their working size during the first frames, and because recycled buffers keep their capacity, later resizes do not reallocate.

Codec detection is lazy. The decoder is created on the first frame, when the codec is known, and rebuilt if a later session negotiates a different codec on the same worker. The codec and the decoder travel together in a single slot so they cannot drift apart.

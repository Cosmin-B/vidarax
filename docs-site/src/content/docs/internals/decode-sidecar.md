---
title: Decode sidecar
description: The long-lived ffmpeg subprocess, its reader thread and bounded handoff, and the limits of the drain-before-write rule.
slug: internals/decode-sidecar
---

The decode sidecar is the long-lived ffmpeg subprocess that turns encoded live video into raw YUV frames. Its implementation lives in `crates/vidarax-core/src/webrtc/decode.rs`. A dedicated reader and a drain-before-write rule reduce the classic two-pipe deadlock window between parent and child. The decoded-frame handoff is lossless, and the steady-state decode loop draws from bounded pools without allocating a fresh buffer per frame. [Media plane](/internals/media-plane/) follows the frames downstream. [Ingest](/ingest/) shows the higher-level source paths.

## Backend selection

`Decoder::new` selects a backend from `DecoderConfig` via `DecoderBackend::select(gpu_available, codec)`:

| `gpu_available` | Codec | Backend |
|---|---|---|
| true | H.264 | `NvDec` (ffmpeg sidecar, `-hwaccel auto`) |
| true | H.265 / HEVC | `NvDec` |
| false | H.264 | `FfmpegSw` (ffmpeg sidecar, CPU) |
| false | H.265 / HEVC | `FfmpegSw` (ffmpeg sidecar, CPU) |
| any | VP8 | `Vp8` (libvpx in-process) with the `vp8` feature, else `Unsupported` |

All live H.264 and H.265 selections use `NvDec` or `FfmpegSw`, so native decoder faults are contained to a child process. The direct `Software` openh264 variant remains available inside the decoder module but is not selected for a live session. `Unsupported` is a real variant whose `decode` returns `DecodeError::UnsupportedCodec`, which lets the caller fail the stream with a precise message and avoids a panic.

## Process spawn

`new_nvdec` and `new_ffmpeg_sw` spawn one ffmpeg process per decoder with piped stdin and stdout and a null stderr. The GPU variant looks like this. The CPU variant is identical minus the `-hwaccel auto` pair:

```rust
let mut child = Command::new(crate::ingest::ffmpeg_path())
    .args([
        "-hwaccel", "auto",
        "-threads", "1",
        "-filter_threads", "1",
        "-f", input_fmt,          // "h264" or "hevc"
        "-i", "pipe:0",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", &format!("{width}x{height}"),
        "pipe:1",
    ])
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null())
    .spawn()
```

The input format comes from `VideoCodec::ffmpeg_input_format`, which returns `Some("h264")` or `Some("hevc")` and `None` for VP8 (VP8 never takes the sidecar path). Decoder and filter thread counts are fixed at one so session admission is not undermined by unbounded native thread creation. The output contract is fixed: packed planar I420 at the configured `width` x `height`, with no row padding and no per-frame metadata, so a frame on stdout is exactly `w*h + 2*(w/2)*(h/2)` bytes and can be parsed by length alone. `stdin` is taken from the child and owned by the decode side. `stdout` is wrapped in a `BufReader` and moved into the reader thread.

## Decoder warm-up

An H.264 decoder commonly cannot emit a frame from its first input. Before any output exists, it needs the SPS and PPS parameter sets (which describe resolution, profile, and reference structure) and an IDR frame to decode against, and those usually arrive across several access units. A first access unit that already carries all three can produce output immediately, though the pipeline may still buffer it. On the WHIP path these arrive as ordinary access units at the head of the stream, forwarded with Annex B framing like everything else (see [WebRTC ingest](/internals/webrtc-ingest/)). The sidecar needs no special casing for them, only tolerance for input that produces no output yet.

Warm-up is exactly the phase that breaks a naive write-then-read loop: the parent must keep writing input while nothing is coming back, and a blocking read after each write would stall forever on the SPS. The sidecar handles this two ways at once. First, output reading happens on a dedicated thread, so writes never wait for reads. Second, "no frame yet" is a first-class result: `decode()` returns `DecodeError::Buffered`, and the decode worker just moves to the next access unit.

## The reader thread and its bounded channel

`spawn_frame_reader` owns ffmpeg stdout. It loops reading exactly one plane at a time into pooled buffers and sends the assembled frame over a bounded `std::sync::mpsc::sync_channel`:

```rust
let (tx, rx) = mpsc::sync_channel(FFMPEG_YUV_READER_QUEUE_CAPACITY);
// per frame:
let mut y = pools.y.acquire();
y.resize(y_size, 0);
if stdout.read_exact(&mut y).is_err() { break; }
// ... u, v the same way ...
if !send_yuv_frame_lossless(&tx, frame) { break; }
```

The channel holds `FFMPEG_YUV_READER_QUEUE_CAPACITY` (16) frames. The send is blocking and the handoff is lossless: the reader never drops or evicts a decoded frame. When the channel is full the reader parks, stops reading stdout, and can let the pipe fill. The decode side drains before its next write, but the code does not yet prove a bound on how much output one input can produce before that drain. Reading directly into pooled buffers means the frame handed downstream is the one ffmpeg wrote, with no intermediate copy, and recycled buffers keep their capacity so the `resize` stops allocating after warm-up.

## The drain-before-write rule

The decode side owns stdin. Each call receives at most 16 ready frames,
retains the newest received frame, then writes and flushes the encoded payload.
Each replacement recycles the older frame and increments the dropped-frame
count. A reader that refills the channel cannot extend the drain indefinitely.

```rust
let (reader_exited, shed) = drain_ready_yuv_frames(frame_rx, pending);
if let Some(metrics) = metrics {
    metrics.inc_frames_dropped_by(shed as u64);
}
stdin.write_all(payload).map_err(DecodeError::WriteError)?;
stdin.flush().map_err(DecodeError::FlushError)?;
if let Some(frame) = pending.take() {
    return Ok(frame);
}
```

`pending` is an `Option<YuvFrame>`. While a new frame replaces an old frame,
the decoder briefly owns both. The pool reserves those two positions. The
reader channel retains its blocking, lossless handoff.

The drain makes room for output already waiting before the parent writes
stdin. This ordering does not prove that the two pipes cannot deadlock. One
input could produce more output than the channel and pipe can hold before
ffmpeg reads more input. The maximum output burst per input remains unknown.

The returned frame uses the current RTP access unit's label. Raw YUV output
has no timestamp channel, so that label remains an approximation.

On destruction, the decoder disconnects the channel, kills and reaps ffmpeg,
then joins the reader. Disconnecting first releases a reader blocked on send.
A failed child or reader startup returns an error to the worker, which faults
the generation through the existing supervisor.

## Return contract

| `decode()` result | Meaning | Decode worker response |
|---|---|---|
| `Ok(YuvFrame)` | Freshest decoded frame | Process it |
| `Err(Buffered)` | Input accepted, no output ready (warm-up, B-frame delay) | Continue with next access unit |
| `Err(ReaderExited)` | Reader thread gone: ffmpeg exited or pipe closed | End decode stage. Supervisor faults the generation |
| `Err(WriteError)` / `Err(FlushError)` | stdin write failed | End decode stage. Supervisor faults the generation |
| `Err(SoftwareDecode)` | openh264 hard error (software path only) | Skip to next RTP frame, decoder kept |
| `Err(Vp8Decode)` | libvpx hard error (`vp8` feature only) | Skip to next RTP frame, decoder kept |
| `Err(UnsupportedCodec)` | No live decoder for the negotiated codec | End the worker with an error log |

Buffered and per-frame native decode errors remain recoverable. A broken pipe or exited reader is not: the decode worker returns, and the generation supervisor closes the peer and stops every sibling. Vidarax does not restart an individual stateful decoder or temporal worker inside the same generation.

## Frame pool interaction

Plane buffers come from `YuvPlanePools`, three `VecPool` free-lists (Y, U, V) sized together. The reader-path pool minimum is `FFMPEG_YUV_READER_POOL_MIN_SLOTS` (22): a full reader channel (16), the steady-state pending allowance (4), one frame under construction, one held by the consumer. This is the same sum `decode_output_pool_slots` computes in `workers.rs`. See [Media plane](/internals/media-plane/#the-yuv-decode-output-pool) for the derivation.

Capacity per slot is bucketed, not exact. `required_y_capacity(width, height)` takes the larger of the luma requirement and the chroma requirement expressed in luma terms. This covers odd dimensions where truncated `(w/2)*(h/2)` math would under-provision. It clamps to `MAX_POOL_Y_CAPACITY` and rounds up to a power of two. A sender that ramps or oscillates resolution rebuilds the free-lists once per bucket crossing, not once per distinct size. The cap prevents an enormous declared resolution from forcing a giant speculative pre-allocation. A genuinely larger frame still decodes, and the copy path grows that one buffer.

`ensure_dims` applies a first-frame-then-grow policy. The WebRTC path opens the pool at a guessed default resolution. The first real decoded frame may resize the pool in either direction, and after that the pool only grows. When a rebuild replaces the free-lists, buffers still in flight hold the old list's sender. Its receiver is gone, so those buffers free on drop. The one-time cost is bounded by the slot count.

## Teardown

`Decoder` implements `Drop`: the sidecar variants call `child.kill()` best-effort. Killing ffmpeg closes stdout and wakes the reader's `read_exact`. `Drop` then joins the owned reader handle and waits for the child. In the live pipeline the decode worker owns the `Decoder` on its stack, so the sidecar and reader are reaped before the stage handle completes.

## Edge cases and limits

- Output resolution is fixed at spawn by `-s {width}x{height}`. Ffmpeg scales whatever the stream carries to that size. The pool reconciliation logic exists mainly for the in-process backends, which emit the stream's true resolution.
- PTS labeling across the pipe is approximate by design. Anything needing exact PTS-to-frame mapping must not use the raw-pipe sidecar.
- `DecoderConfig::auto_detect` probes `nvidia-smi` for GPU availability and defaults to 1920x1080 and H.264. Callers are expected to override the codec from the SDP offer.
- Sidecar construction panics if ffmpeg is missing (`expect` on spawn). The process fails at session start because this is not a recoverable per-frame error.
- The retained direct `Software` backend and optional `Vp8` backend bypass everything on this page except the pools. Live H.264 does not select `Software`. VP8 remains an explicit in-process exception when that feature is enabled.

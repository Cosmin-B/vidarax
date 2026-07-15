---
title: Decode sidecar
description: The long-lived ffmpeg subprocess, its reader thread and bounded handoff, and the drain-before-write rule that removes the two-pipe interleaving deadlock.
---

The decode sidecar is the long-lived ffmpeg subprocess that turns encoded live video into raw YUV frames, implemented in `crates/vidarax-core/src/webrtc/decode.rs`. Its design removes the classic two-pipe interleaving deadlock between parent and child, keeps the handoff of decoded frames lossless, and makes the steady-state decode loop allocate from bounded pools rather than the heap. This page walks the process spawn, decoder warm-up, the reader thread and its bounded channel, the drain-before-write rule, pool interaction, and teardown. Where these frames go next is covered in [Media plane](/internals/media-plane/); the higher-level view of ingest paths is in [Ingest](/ingest/).

## Backend selection

`Decoder::new` selects a backend from `DecoderConfig` via `DecoderBackend::select(gpu_available, codec)`:

| `gpu_available` | Codec | Backend |
|---|---|---|
| true | H.264 | `NvDec` (ffmpeg sidecar, `-hwaccel auto`) |
| true | H.265 / HEVC | `NvDec` |
| false | H.264 | `Software` (openh264 in-process) |
| false | H.265 / HEVC | `FfmpegSw` (ffmpeg sidecar, CPU) |
| any | VP8 | `Vp8` (libvpx in-process) with the `vp8` feature, else `Unsupported` |

Only `NvDec` and `FfmpegSw` are sidecar paths; they share all of the machinery below. `Unsupported` is a real variant whose `decode` returns `DecodeError::UnsupportedCodec` so the caller can fail the stream with a precise message instead of panicking.

## Process spawn

`new_nvdec` and `new_ffmpeg_sw` spawn one ffmpeg process per decoder with piped stdin and stdout and a null stderr. The GPU variant looks like this; the CPU variant is identical minus the `-hwaccel auto` pair:

```rust
let mut child = Command::new(crate::ingest::ffmpeg_path())
    .args([
        "-hwaccel", "auto",
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

The input format comes from `VideoCodec::ffmpeg_input_format`, which returns `Some("h264")` or `Some("hevc")` and `None` for VP8 (VP8 never takes the sidecar path). The output contract is fixed: packed planar I420 at the configured `width` x `height`, with no row padding and no per-frame metadata, so a frame on stdout is exactly `w*h + 2*(w/2)*(h/2)` bytes and can be parsed by length alone. `stdin` is taken from the child and owned by the decode side; `stdout` is wrapped in a `BufReader` and moved into the reader thread.

## Decoder warm-up

An H.264 decoder commonly cannot emit a frame from its first input. Before any output exists, it needs the SPS and PPS parameter sets (which describe resolution, profile, and reference structure) and an IDR frame to decode against, and those usually arrive across several access units; a first access unit that already carries all three can produce output immediately, though the pipeline may still buffer it. On the WHIP path these arrive as ordinary access units at the head of the stream, forwarded with Annex B framing like everything else (see [WebRTC ingest](/internals/webrtc-ingest/)); the sidecar needs no special casing for them, only tolerance for input that produces no output yet.

Warm-up is exactly the phase that breaks a naive write-then-read loop: the parent must keep writing input while nothing is coming back, and a blocking read after each write would stall forever on the SPS. The sidecar handles this two ways at once. First, output reading happens on a dedicated thread, so writes never wait for reads. Second, "no frame yet" is a first-class result: `decode()` returns `DecodeError::Buffered`, and the decode worker just moves to the next access unit. The in-process openh264 path expresses the same state when its decoder returns no output for an SPS/PPS-only payload.

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

The channel holds `FFMPEG_YUV_READER_QUEUE_CAPACITY` (16) frames. The send is blocking and the handoff is lossless: the reader never drops or evicts a decoded frame. When the channel is full the reader parks, which in turn stops it reading stdout, which lets the pipe fill, which is fine, because the decode side is guaranteed to come back and drain (next section). Reading directly into pooled buffers means the frame handed downstream is the one ffmpeg wrote, with no intermediate copy, and recycled buffers keep their capacity so the `resize` stops allocating after warm-up.

The reader exits when `read_exact` fails (ffmpeg closed stdout) or when the receiver side is gone (send fails). There is no separate shutdown signal.

## The drain-before-write rule

The decode side owns stdin and follows one ordering rule, implemented in `decode_ffmpeg_pipe`: drain everything ready, then write, then answer from the pending FIFO.

```rust
let reader_exited = drain_ready_yuv_frames(frame_rx, pending);
observe_pending_depth(pending.len(), metrics, pending_warned, codec, width, height);

stdin.write_all(payload).map_err(DecodeError::WriteError)?;
stdin.flush().map_err(DecodeError::FlushError)?;

if let Some(frame) = pending.pop_back() {
    let shed = pending.len();
    if shed != 0 {
        metrics.inc_frames_dropped_by(shed as u64);
        pending.clear();
    }
    return Ok(frame);
}
```

`drain_ready_yuv_frames` is a non-blocking `try_recv` loop that moves every ready frame from the channel into the decoder-local `pending: VecDeque<YuvFrame>` and reports whether the reader has exited.

The deadlock argument is short, and it targets one specific hazard: the interleaving deadlock where the parent blocks writing stdin while ffmpeg blocks writing stdout. The parent can only block writing stdin when ffmpeg's stdin pipe is full; ffmpeg only stops consuming stdin when its stdout pipe is full; stdout only stays full when the reader thread is parked on a full channel; and the parent emptied that channel immediately before writing. So a full handoff channel can never be the standing reason both sides are stuck: the reader always has room for at least one blocking send after each drain, which unblocks ffmpeg's stdout, which unblocks its stdin consumption. No decoded output is ever dropped by the handoff itself. The argument assumes healthy pipes and that one written input produces a bounded amount of output before the next drain; it removes the known interleaving deadlock rather than proving every hang impossible.

Dropping does happen, but as policy rather than pipe pressure: after the write, `pop_back` returns the freshest pending frame and everything older is shed and counted through `inc_frames_dropped_by`. Under real-time backlog this keeps downstream labels close to the current RTP timestamp instead of replaying a growing latency queue. The label itself is best-effort: the raw pipe has no metadata channel, so a returned frame is attributed to the current access unit.

Two bounds watch the pending FIFO. `FFMPEG_YUV_PENDING_FIFO_CAPACITY` (reader queue capacity plus `FFMPEG_YUV_PENDING_POOL_ALLOWANCE`, so 20) pre-sizes the `VecDeque`. `FFMPEG_YUV_PENDING_SANITY_BOUND` (four times the reader queue capacity, 64) is a diagnostic ceiling: exceeding it fires a `debug_assert`, a one-shot warning, and the `inc_decode_pending_sanity_violations` metric, but never evicts: if the bound is exceeded, an invariant broke somewhere upstream, and the frames are kept for diagnosis rather than dropped.

## Return contract

| `decode()` result | Meaning | Decode worker response |
|---|---|---|
| `Ok(YuvFrame)` | Freshest decoded frame | Process it |
| `Err(Buffered)` | Input accepted, no output ready (warm-up, B-frame delay) | Continue with next access unit |
| `Err(ReaderExited)` | Reader thread gone: ffmpeg exited or pipe closed | Skip to next RTP frame, decoder kept |
| `Err(WriteError)` / `Err(FlushError)` | stdin write failed | Skip to next RTP frame, decoder kept |
| `Err(SoftwareDecode)` | openh264 hard error (software path only) | Skip to next RTP frame, decoder kept |
| `Err(Vp8Decode)` | libvpx hard error (`vp8` feature only) | Skip to next RTP frame, decoder kept |
| `Err(UnsupportedCodec)` | No live decoder for the negotiated codec | End the worker with an error log |

Note the asymmetry: only `UnsupportedCodec` ends the worker. Every other error, including the pipe failures that leave an ffmpeg sidecar effectively dead, makes the worker `continue` to the next RTP frame with the same decoder still in its slot; a dead sidecar is not detected and rebuilt, it just keeps returning errors. The decoder is replaced only when a later frame arrives with a different codec.

## Frame pool interaction

Plane buffers come from `YuvPlanePools`, three `VecPool` free-lists (Y, U, V) sized together. The reader-path pool minimum is `FFMPEG_YUV_READER_POOL_MIN_SLOTS` (22): a full reader channel (16), the steady-state pending allowance (4), one frame under construction, one held by the consumer. This is the same sum `decode_output_pool_slots` computes in `workers.rs`; see [Media plane](/internals/media-plane/#the-yuv-decode-output-pool) for the derivation.

Capacity per slot is bucketed, not exact. `required_y_capacity(width, height)` takes the larger of the luma requirement and the chroma requirement expressed in luma terms (covering odd dimensions where truncated `(w/2)*(h/2)` math would under-provision), clamps to `MAX_POOL_Y_CAPACITY` (`1 << 25` bytes), and rounds up to a power of two. Bucketing means an untrusted sender that ramps or oscillates resolution rebuilds the free-lists a bounded number of times, once per bucket crossing, instead of once per distinct size; the cap means a hostile stream declaring an enormous resolution cannot force a giant speculative pre-allocation (a genuinely larger frame still decodes; the copy path grows that one buffer).

`ensure_dims` applies a first-frame-then-grow policy: the WebRTC path opens the pool at a guessed default resolution, the first real decoded frame may resize the pool in either direction to match reality, and after that the pool only grows. When a rebuild replaces the free-lists, buffers still in flight hold the old list's sender; its receiver is gone, so on drop they free instead of recycling, a one-time cost bounded by the slot count.

## Teardown

`Decoder` implements `Drop`: the sidecar variants call `child.kill()` best-effort and ignore errors. Killing ffmpeg closes its stdout, which fails the reader thread's `read_exact`, which ends the reader loop and drops the channel sender; any later `decode` call then observes `ReaderExited`. The reader thread is detached and needs no join. In the live pipeline the decode worker owns the `Decoder` on its stack, so the sidecar dies when the worker's RTP channel closes and the thread returns, and also when the worker replaces the decoder after a codec change on the same stream.

## Edge cases and limits

- Output resolution is fixed at spawn by `-s {width}x{height}`; ffmpeg scales whatever the stream carries to that size. The pool reconciliation logic exists mainly for the in-process backends, which emit the stream's true resolution.
- PTS labeling across the pipe is approximate by design; anything needing exact PTS-to-frame mapping must not use the raw-pipe sidecar.
- `DecoderConfig::auto_detect` probes `nvidia-smi` for GPU availability and defaults to 1920x1080 and H.264; callers are expected to override the codec from the SDP offer.
- Sidecar construction panics if ffmpeg is missing (`expect` on spawn). This is a deliberate fail-fast at session start rather than a recoverable per-frame error.
- The `Software` and `Vp8` backends bypass everything on this page except the pools: they decode synchronously in-process and de-stride planes straight into pooled buffers.

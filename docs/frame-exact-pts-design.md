# Frame-exact PTS for WebRTC ffmpeg decode paths

## Problem

The current WebRTC decode pipeline has exact decoded pixels and gate signals, but
the ffmpeg-backed paths do not have exact per-frame labels.

The input side starts in
`crates/vidarax-core/src/webrtc/session.rs`. `WebRtcSession::run` receives a
`MediaSample::Video`, converts the 90 kHz RTP timestamp to milliseconds, assigns
a per-session `seq`, and sends an `RtpFrame` with `nals`, `pts_ms`, `seq`, and
`codec`. H.264 payloads are converted by `frame_payload_to_nals` by prepending
the Annex B start code. VP8 payloads are passed through unchanged.

The decode side is in `crates/vidarax-core/src/webrtc/decode.rs`. `Decoder`
has three variants:

- `Software`, the in-process openh264 H.264 path.
- `FfmpegSw`, the CPU ffmpeg subprocess path.
- `NvDec`, the ffmpeg subprocess path started with `-hwaccel auto`.

`DecoderBackend::select` maps GPU-enabled sessions to `NvDec`, non-GPU H.264
to `Software`, and non-GPU VP8 to `FfmpegSw`. `Decoder::decode` passes
`NvDec` and `FfmpegSw` through `decode_ffmpeg_pipe`. Those variants write raw
Annex B H.264 or raw VP8 bytes to ffmpeg stdin and read raw `yuv420p` frames
from ffmpeg stdout through `spawn_frame_reader`.

That raw pipe cannot carry exact per-frame PTS today. The input is a raw codec
bitstream, not a timestamped packet stream. The output is raw YUV420 plane data,
not a framed stream with metadata. There is no PTS, RTP timestamp, or frame
index on stdout for `spawn_frame_reader` to attach to `YuvFrame`. Input and
output also are not guaranteed to be 1:1: startup parameter sets may buffer
without output, pre-sync input may be discarded, inter-frames may be
undecodable after loss, and the decoder may reorder frames before output.

`crates/vidarax-core/src/webrtc/workers.rs` therefore labels decoded frames at
the worker boundary. In `spawn_decode_workers`, after `dec.decode(&frame.nals)`
returns a `YuvFrame`, `build_stream_frame_from_yuv` is called with the current
`RtpFrame`'s `frame.seq` and `frame.pts_ms`. On the ffmpeg subprocess paths,
that is a best-effort current-RTP approximation, not a proof that the returned
YUV frame came from that access unit.

What is already exact:

- The decoded YUV pixels returned by ffmpeg are the decoder output.
- `yuv_to_frame_signal` in `crates/vidarax-core/src/webrtc/signals.rs`
  computes `FrameSignal` from the actual `YuvFrame` luma plane it receives.
- The openh264 `Software` H.264 path is frame-exact for labels in the current
  worker model because `Decoder::decode` calls `decode_software` synchronously
  on the access unit passed to that call. When openh264 returns output, the
  worker attaches the same access unit's `seq` and `pts_ms`.

This document extends `docs/webrtc-decode-limitations.md`; it does not change
the current limitation statement.

## Approach A: container-framed input with PTS passthrough

This approach keeps the long-lived ffmpeg subprocess model but stops feeding
ffmpeg raw codec bytes with no timestamps. Instead, the decode path would mux
incoming access units into a timestamp-carrying container before writing them to
the subprocess.

A concrete version would:

1. Convert each incoming `RtpFrame` into a packet with the access unit bytes and
   its existing `pts_ms`.
2. Write those packets into a streaming container that ffmpeg can demux from
   stdin, such as `nut` or Matroska. The muxer must preserve packet PTS and must
   support the required H.264 and VP8 bitstreams.
3. Start ffmpeg with the container as input instead of `-f h264` or `-f vp8`.
4. Produce decoded YUV frames plus their output PTS. Rawvideo stdout is not
   sufficient by itself, so the implementation would need either a
   timestamp-carrying output format or a second side channel that emits one PTS
   per decoded output frame.
5. Correlate each decoded YUV frame from stdout with the matching output PTS
   from the metadata stream before returning it to the worker.

The hard part is not assigning PTS to input packets. The hard part is preserving
the relationship between stdout YUV frames and output PTS across a subprocess
boundary. `spawn_frame_reader` currently reads fixed-size Y, U, and V planes and
sends a plain `YuvFrame` over an `mpsc::sync_channel`. It has no metadata stream
and no framing other than byte counts. A future design would need the YUV reader
and PTS reader to advance in lockstep without losing either side.

The edge cases are the same ones that make the raw-pipe label approximate:

- Parameter-set or header packets may be accepted but produce no frame.
- Pre-sync packets may be dropped by the decoder.
- Undecodable inter-frames may not produce output.
- Reordered codecs may output a frame whose PTS is older than the most recent
  input packet.
- Startup and flush behavior must not leave a stale PTS paired with the next
  YUV frame.

Pros:

- Keeps the current subprocess isolation used by `NvDec` and `FfmpegSw`.
- Keeps most of the existing ffmpeg stdout reader handoff and YUV pool shape.
- Avoids linking libavcodec into the Rust process.
- Can continue to use the configured ffmpeg CLI path from
  `crate::ingest::ffmpeg_path`.

Cons:

- Adds an in-memory muxer on the hot path.
- Requires a reliable output PTS channel and lockstep correlation with raw YUV
  frames.
- Still crosses a subprocess stdin/stdout boundary.
- Still needs careful handling when ffmpeg drops, buffers, or reorders input.
- The current `YuvFrame` type has no PTS field, so metadata still has to be
  added to the decode return path.

## Approach B: in-process libavcodec decoder

This approach replaces the raw ffmpeg subprocess for the timestamp-exact mode
with an in-process libavcodec decoder. The worker would still receive the same
`RtpFrame` values from `WebRtcSession::run`, but the ffmpeg-backed decoder mode
would feed libavcodec `AVPacket`s instead of writing bytes to stdin.

A concrete version would:

1. Build one decoder instance per ordered stream, matching the current
   `spawn_decode_workers` model.
2. For each `RtpFrame`, create or reuse an `AVPacket` whose data points at the
   access unit bytes and whose `pts` is derived from `RtpFrame::pts_ms` in the
   decoder time base.
3. Call the libavcodec send/receive API: send the packet, then receive all
   currently available `AVFrame`s.
4. For each `AVFrame`, read `best_effort_timestamp` first and `pts` where
   appropriate, convert it back to milliseconds, and attach that value to the
   returned decoded frame.
5. Copy the active Y, U, and V planes into the existing pooled `RecycledBytes`
   buffers used by `YuvFrame`.

This eliminates the raw pipe's correlation problem. The same decoder that
handles parameter sets, reordering, and dropped frames also returns the decoded
frame timestamp. If an input packet produces no frame, no output timestamp is
returned. If a later receive call produces a reordered frame, the frame carries
its own timestamp.

The Rust crate should be selected for explicit access to packet and frame
timestamps rather than for high-level media convenience. `rsmpeg` is the better
fit for this use case because the implementation needs direct control over
`AVPacket` PTS, decoder send/receive, `AVFrame` timestamps, pixel format, and
frame plane copying. `ffmpeg-next` may work, but it is a broader wrapper and
should only be chosen if a prototype shows it exposes the required timestamp
and lifetime controls without extra allocation or hidden buffering.

Pros:

- Exact PTS comes from `AVFrame` metadata, not from a side-channel guess.
- Removes the subprocess stdin/stdout handoff for the exact mode.
- Removes the need to correlate raw YUV bytes with a separate PTS stream.
- Keeps the one ordered decoder per stream model already used by
  `spawn_decode_workers`.

Cons:

- Adds a native dependency that links libavcodec and related libav libraries.
- Build and CI requirements change. The repository already uses the ffmpeg CLI
  in tests and runtime paths, but linking libav libraries is a different
  toolchain requirement.
- Introduces an `unsafe` FFI surface that must be isolated behind a small
  decoder module.
- Hardware decode parity with the current `NvDec` subprocess path needs a
  separate prototype. The design should first prove timestamp correctness on
  CPU decode, then decide whether to wire libavcodec hardware contexts for the
  GPU path or keep `NvDec` on the current subprocess path until that is ready.
- License and distribution implications must be reviewed before enabling the
  linked decoder by default.

## Recommendation

Use Approach B as the primary implementation path for frame-exact PTS.

The main reason is that this pipeline needs the decoded frame and its timestamp
to be one object at the decode boundary. The current bug is not that the worker
lacks an RTP timestamp; `RtpFrame::pts_ms` already exists. The bug is that
`decode_ffmpeg_pipe` returns a raw `YuvFrame` with no proof that it corresponds
to the current `RtpFrame`. Container-framing the subprocess input fixes only
half of that problem. It still requires a second metadata path and exact
correlation with rawvideo stdout.

An in-process libavcodec decoder is more invasive at build time, but it matches
the correctness requirement better: packet PTS goes into the decoder, frame PTS
comes out with the decoded frame. It also lets the implementation keep the
existing ordered worker topology and bounded downstream queues.

This should not be flipped on by default in the first patch. The first
implementation should be a gated decoder mode that proves the timestamp
contract against deterministic fixtures. The existing subprocess paths should
remain available while build, CI, licensing, and hardware-decode implications
are worked through.

## Hot-path constraints the implementation must honor

The decode hot path must not introduce per-frame heap allocation, mutexes, or
CAS loops. The existing code already has the shape the new mode should follow:
one stateful decoder per stream, bounded queues, and preallocated reusable
buffers.

Current boundedness:

- `FFMPEG_YUV_READER_QUEUE_CAPACITY` is 16 in
  `crates/vidarax-core/src/webrtc/decode.rs`.
- `FFMPEG_YUV_PENDING_POOL_ALLOWANCE` is 4.
- `FFMPEG_YUV_READER_POOL_MIN_SLOTS` is
  `FFMPEG_YUV_READER_QUEUE_CAPACITY + FFMPEG_YUV_PENDING_POOL_ALLOWANCE + 2`,
  which is 22 slots.
- `decode_output_pool_slots` in
  `crates/vidarax-core/src/webrtc/workers.rs` mirrors that sizing for `NvDec`
  and `FfmpegSw`: one reader-constructed frame, 16 reader handoff frames, four
  decoder-pending frames, and one decode-consumer frame.
- The synchronous openh264 path uses `SOFTWARE_YUV_POOL_MIN_SLOTS`, which is
  2 slots.

The new in-process exact-PTS mode should reuse the 22-slot ffmpeg-backed pool
budget at first, even if it no longer has a stdout reader thread. That preserves
the current bound while allowing decoded frames that become ready in a burst
after packet send. Once the deterministic harness exists, the pool size can be
reduced only with evidence.

The implementation should store exact PTS on the decoded frame object, not in a
side map. Today `YuvFrame` in `decode.rs` contains `y`, `u`, `v`, `width`, and
`height`; it has no timestamp field. The concrete change should be to carry the
decoded PTS with the `YuvFrame` returned by `Decoder::decode`, either by adding
a `pts_ms: u64` field to `YuvFrame` or by returning a small decoded-frame
wrapper that owns a `YuvFrame` and `pts_ms`. A side map keyed by pointer, frame
index, or queue position would recreate the correlation problem and likely add
allocation or synchronization.

Downstream, `spawn_decode_workers` should stop passing the current
`RtpFrame::pts_ms` into `build_stream_frame_from_yuv` for exact-PTS decoder
modes. It should pass the decoded frame's attached PTS. `StreamFrame::pts_ms`
and `FrameSignal::pts_ms` can remain the downstream storage locations because
they already carry presentation time to analysis, VLM work, and sink events.

The freshness policy should stay the same. `decode_ffmpeg_pipe` currently
returns the newest pending decoded frame and sheds older pending decoded output,
incrementing `PipelineMetrics::inc_frames_dropped_by`, which is exported as
`vidarax_pipeline_frames_dropped_total` in
`crates/vidarax-core/src/metrics.rs`. The exact-PTS mode should shed decoded
outputs, not encoded inputs, under overload. When a decoded output is shed, its
pooled buffers and attached scalar PTS should be dropped together.

The downstream backpressure should also stay the same. `spawn_decode_workers`
sends `StreamFrame` through the bounded `frame_tx`; when that send blocks, the
worker stops receiving RTP frames and stops feeding the decoder. The new mode
must not add an unbounded packet queue between `RtpFrame` receive and decoder
send.

## Test and verification strategy

Frame-exact PTS cannot be proven by the existing unit tests alone. The current
tests cover queue sizing, ffmpeg reader behavior, drop accounting, and RTP
timestamp conversion, but they do not provide a deterministic media stream with
known ground-truth per-frame PTS and expected decoded outputs.

The implementation needs a new fixture harness:

- Generate or store a deterministic H.264 stream with known per-access-unit PTS.
- Generate or store a deterministic VP8 stream with known per-frame PTS.
- Include startup parameter-set cases where packets produce no decoded output.
- Include reorder cases for H.264 if the selected encoder settings can produce
  them.
- Include controlled drop or undecodable inter-frame cases where the decoder
  should skip output and later recover.
- Feed the stream through the WebRTC decode path as `RtpFrame` values, not
  through a separate ingest path.
- Assert that every emitted `StreamFrame::pts_ms` equals the known PTS for the
  decoded frame with 0 frames of error.
- Assert that skipped inputs do not shift labels onto later decoded frames.
- Assert that shed decoded frames increment the existing drop metric while the
  retained frame keeps its own exact PTS.

Captured RTP can be used if it includes a checked-in manifest of expected
decoded-frame PTS values. A synthetic stream is preferable for regression tests
because it can cover parameter sets, reorder, and drop cases without relying on
network capture behavior.

## Rollout

The rollout should be additive and reversible.

1. Add a feature or config flag for the exact-PTS libavcodec decoder mode.
2. Keep the current `FfmpegSw` and `NvDec` best-effort labeling as the default.
3. Land the deterministic fixture harness and require 0-frame timestamp error
   before enabling the new mode outside targeted tests.
4. Enable the exact-PTS mode for one ffmpeg-backed CPU path first, most likely
   VP8 `FfmpegSw`, because it avoids solving hardware decode at the same time.
5. Extend to H.264 ffmpeg-backed decode after the H.264 fixture covers
   parameter sets and reorder behavior.
6. Decide separately whether `NvDec` moves to libavcodec hardware decode or
   remains a subprocess path until hardware timestamp correctness is proven.
7. After the exact mode is verified and selected as the default, update
   `docs/webrtc-decode-limitations.md` to describe the old raw-pipe limitation
   as historical or as the fallback-path behavior.


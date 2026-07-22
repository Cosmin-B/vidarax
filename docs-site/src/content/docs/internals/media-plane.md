---
title: Media plane
description: The per-session worker threads, the bounded channels between them, and how every buffer pool is sized as a sum over in-flight positions.
---

The media plane is the execution graph that turns RTP frames into VLM events for one live session. It lives in `crates/vidarax-core/src/webrtc/workers.rs`. Decode, filtering, and inference run on blocking OS threads outside the tokio async runtime. WebRTC ingress stays async. Every stage handoff is a bounded queue with a defined full-queue behavior, and every byte buffer that crosses a stage boundary comes from a pool with a computed slot count. The decoder is covered in [Decode sidecar](/docs/internals/decode-sidecar/). The keyframe decision is covered in [Filter internals](/docs/internals/gate-internals/).

## The task and thread topology

The control plane and WebRTC ingress run on tokio. The processing stages are owned `std::thread` workers spawned by `spawn_pipeline` in `workers.rs`, bridged from async context through one `tokio::task::spawn_blocking` call in `whip.rs`. `PipelineRuntime` retains a stage tag and join handle for every worker. One supervisor owns the session generation. An unexpected exit closes the peer, raises the shared stop signal, and bounded-joins every sibling. Stateful stages are never restarted alone.

The complete per-session inventory, grouped by runtime and mode:

| Execution unit | Runtime | Mode | Spawned by | Role | Shutdown |
|---|---|---|---|---|---|
| Session event loop | tokio task | both | `tokio::spawn(session.run(...))` in `whip.rs` | Drives the rustrtc peer connection | Peer close ends the future |
| Track receive task (per video track) | tokio task | both | `session.run` on each `Track` event | Receives depacketized samples, frames them as `RtpFrame`, lossless enqueue | Track receive error or channel close |
| `vx-decode-0` | OS thread | both | `spawn_decode_workers` | Decodes RTP access units to YUV, computes frame signals, runs the gate inline (keyframe mode) or PTS sampling (clip mode) | RTP channel closes |
| ffmpeg stdout reader (unnamed) | OS thread | both | `spawn_frame_reader` per sidecar decoder | Reads YUV planes from ffmpeg stdout into pooled buffers, blocking sends into the reader channel | Sidecar exit or receiver close. `Decoder::drop` kills the child and joins the reader |
| `vx-vlm-{i}` | OS thread | keyframe only | `spawn_vlm_workers` | Semantic-novelty check, tiered VLM inference, dedup, temporal context | VLM work channel closes |
| `vx-event-writer` | OS thread | keyframe only | `spawn_vlm_workers` | Drains `SinkEvent`s and calls the blocking `EventSink` methods so inference threads do not own storage writes | Sink event channel closes |
| `vx-trigger-writer` | OS thread | keyframe trigger | `spawn_pipeline` | Stores the selected JPEG, commits the namespaced assertion, dispatches requested local output, then forwards the same buffer to semantic inference | Trigger queue closes |
| `vx-analysis-{i}` | OS thread | clip only | `spawn_analysis_workers` | Loop detection, forwards accepted frames to the clip accumulator | Stream frame channel closes |
| `vx-clip-acc` | OS thread | clip only | `spawn_clip_accumulator` | Batches sampled frames into `ClipWork` windows | Clip frame channel closes |
| `vx-clip-vlm-{i}` | OS thread | clip only | `spawn_clip_vlm_workers` | Multi-image VLM inference over a clip window, calls the sink directly | Clip work channel closes |

One process-wide thread sits behind all sessions: `vidarax-timeline-writer`, which owns the WAL writer and is described in [WAL and events](/docs/internals/wal-and-events/#who-appends-each-event-family).

`PipelineGeneration`, `PipelineStage`, and `PipelineHealth` make lifecycle state
explicit and typed. Live prompt/schema changes travel over an eight-slot
`SessionCommand` channel with the generation attached. The PATCH request returns
success only after the VLM worker acknowledges ownership of the new values. A
closed, stale, or timed-out command cannot mutate a replacement generation.

The `{i}` suffixes are cosmetic. `per_stream_decode_workers`, `per_stream_analysis_workers`, and `per_stream_vlm_workers` all clamp the configured count to 1:

```rust
pub fn per_stream_analysis_workers(_configured: usize) -> usize {
    // Analysis owns stream-order gate and loop-detector state. Parallelism is
    // across sessions; splitting one ordered stream would race that state.
    1
}
```

One ordered stream means one stateful decoder, one gate, one loop detector, and one VLM worker carrying temporal context (`last_description`, dedup state). The requested counts in `WorkerPoolConfig` are accepted for API compatibility and clamped at spawn time. Parallelism comes from running many sessions.

## Channel topology

All inter-stage channels are bounded `kanal` MPMC channels. Capacities are named constants at the top of `workers.rs` (plus one in `session.rs`):

| Channel | Capacity constant | Value | Producer | Consumer | When full |
|---|---|---|---|---|---|
| RTP frames | `RTP_FRAME_QUEUE_CAPACITY` | 128 | `session.run` (tokio task) | decode worker | Sender awaits capacity. Backpressure reaches the WebRTC media layer |
| Stream frames | `STREAM_FRAME_QUEUE_CAPACITY` | 64 | decode worker (clip mode) | analysis worker | Blocking `send`. Decode worker waits |
| VLM work | `VLM_WORK_QUEUE_CAPACITY` | 32 | decode worker (keyframe mode) | VLM worker | `try_send`. Keyframe is dropped and counted |
| Clip frames | `CLIP_FRAME_QUEUE_CAPACITY` | 64 | analysis worker | clip accumulator | `try_send`. Frame is dropped |
| Clip work | `CLIP_WORK_QUEUE_CAPACITY` | 0 | clip accumulator | clip VLM worker | Rendezvous. Accumulator blocks in `send` |
| Sink events | `SINK_EVENT_QUEUE_CAPACITY` | 512 | VLM workers | `vx-event-writer` | Blocking `send`. Inference thread waits |
| Trigger binary writes | `TRIGGER_BINARY_QUEUE_CAPACITY` | 8 | decode worker | `vx-trigger-writer` | `try_send`. Assertion is dropped and counted without blocking decode |

Two behaviors are deliberate and worth internalizing before changing anything:

- While a session is active, the RTP handoff is lossless. `enqueue_rtp_frame_lossless` awaits channel capacity and never drops an active-session frame. Sustained overload reaches WebRTC jitter buffering, NACKs, and keyframe requests without introducing a gap into the ordered decoder stream. Session teardown is the explicit exception. A monotonic stop signal cancels an in-flight receive or enqueue, the track task drops that final frame, and the owner joins every track task before closing the downstream channel.
- The decode-to-VLM handoff is lossy on purpose. In the decode worker, `vlm_tx.try_send(work)` treats a full queue the same as a closed channel: the keyframe is dropped and `inc_keyframes_dropped` is recorded. A stalled VLM must cost keyframes, not stall decoding for the whole stream. Note the kanal detail encoded in the match: `try_send` returns `Ok(false)` on a full queue, so only `Ok(true)` counts as a kept keyframe.

## The two wiring modes

`spawn_pipeline` builds one of two topologies from `PipelineWiring`, selected by whether `clip_config` is set:

- Keyframe mode (default). The per-frame filter and optional trigger VM run inline inside the decode worker (`DecodeSink::Keyframe`). A firing trigger can select a frame even when the ordinary filter would not. Its bounded writer stores the pooled JPEG and commits the assertion before forwarding that same buffer to the VLM queue. Ordinary selected frames go directly to the VLM queue. There is no analysis stage. The `stream_tx`/`stream_rx` pair the caller allocated goes unused.
- Clip mode. The decode worker samples frames by PTS before encoding (`DecodeSink::Stream` with a `ClipRateGate`), the analysis worker runs loop detection, the accumulator builds windows of up to `MAX_CLIP_FRAMES_PER_REQUEST` (64) frames, and clip VLM workers make multi-image inference calls.

## The zero-capacity clip work queue

`CLIP_WORK_QUEUE_CAPACITY` is 0, which makes the accumulator-to-VLM channel a rendezvous. A `ClipWork` is handed over only when a worker is ready to take it. Each `ClipWork` can carry up to 64 pooled JPEG buffers, so every queued clip would add 64 slots to the worst-case JPEG pool. With a zero-capacity queue, the number of clips in flight is exactly one per active worker plus one blocked in the accumulator's `send`. The pool bound stays small and provable. Upstream, the analysis worker uses `try_send` into the clip frame queue. A slow VLM sheds frames there and leaves decode running.

## Pool sizing as a sum over in-flight positions

Every pool in the media plane is sized by the same method: enumerate every position where a buffer can legally exist at one instant, and add them up. The channels are bounded and the worker counts are clamped, so the enumeration is finite and checkable.

### The YUV decode-output pool

For the ffmpeg sidecar backends, `decode_output_pool_slots` returns `DECODE_OUTPUT_POOL_SLOTS_PER_WORKER`, built from named constants:

```rust
const FFMPEG_READER_CONSTRUCTING_YUV_FRAMES: usize = 1;
const FFMPEG_DECODER_PENDING_YUV_FRAMES: usize = FFMPEG_YUV_PENDING_POOL_ALLOWANCE;
const DECODE_CONSUMER_YUV_FRAMES: usize = 1;
const DECODE_OUTPUT_POOL_SLOTS_PER_WORKER: usize = FFMPEG_READER_CONSTRUCTING_YUV_FRAMES
    + FFMPEG_YUV_READER_QUEUE_CAPACITY
    + FFMPEG_DECODER_PENDING_YUV_FRAMES
    + DECODE_CONSUMER_YUV_FRAMES;
```

Reading the sum as positions: one frame the reader thread is currently assembling from ffmpeg stdout, plus a full reader handoff channel (`FFMPEG_YUV_READER_QUEUE_CAPACITY`, 16), plus the decoder-local pending FIFO's steady-state allowance (`FFMPEG_YUV_PENDING_POOL_ALLOWANCE`, 4), plus one frame held by the decode consumer. Total: 22 slots. The same figure appears in `decode.rs` as `FFMPEG_YUV_READER_POOL_MIN_SLOTS`, and `spawn_frame_reader` clamps up to it, so a caller cannot under-provision the reader path. Live H.264 and H.265 both use the ffmpeg process boundary. The retained direct openh264 decoder path has no reader thread or pending FIFO and needs only `SOFTWARE_YUV_POOL_MIN_SLOTS` (2), but live backend selection does not choose it.

### The JPEG pool

`jpeg_pool_slots(analysis_workers, vlm_workers)` applies the same method to JPEG thumbnails, which travel further:

```rust
let decode_to_analysis = STREAM_FRAME_QUEUE_CAPACITY + analysis_workers + 1;
let normal_path = VLM_WORK_QUEUE_CAPACITY + vlm_workers + JPEG_SINK_EVENT_POOL_ALLOWANCE + 1;
let clip_path = CLIP_FRAME_QUEUE_CAPACITY
    + crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST
    + (CLIP_WORK_QUEUE_CAPACITY + active_clip_workers + blocked_clip_sender)
        * crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST;
decode_to_analysis + normal_path + clip_path
```

With the per-stream clamps applied, the doc comment at `workers.rs:40` itemizes the result: 484 slots total, as 66 on the decode-to-analysis leg (a full stream-frame queue, one frame in the analysis worker, one in the sender), 162 on the normal VLM path (a full VLM queue, one in the worker, the 128-slot sink backlog allowance, one in the sender), 64 in the clip-frame queue, 64 held by the accumulator's current window, and 128 for clip work in flight (one active worker plus one blocked sender, each holding a full 64-frame clip. The queued term is zero because the queue has no capacity). A unit test, `jpeg_pool_covers_full_clip_path_and_bounded_sink_backlog_without_heap_growth`, re-derives the sum and pins it to 484 and to `JPEG_POOL_SLOT_CEILING` (512), so a change to any capacity constant fails the test until the derivation is updated deliberately.

Undersizing a pool affects allocation behavior, not correctness. `VecPool::acquire` returns a fresh `Vec` when the free-list is empty. `RecycledBytes::drop` frees a buffer when the free-list is full. The sizing keeps pool-covered positions allocation-free in steady state. Clip inference still allocates outside the pool. The clip VLM worker clones the window's last frame for its metadata event, `RecycledBytes::clone` deep-copies into an unpooled `Vec`, and the multi-image request encodes each JPEG into a fresh string. See [Allocation discipline](/docs/internals/allocation-discipline/) for the checks that enforce the pooled-path property.

### The sink backlog permit counter

The `vx-event-writer` channel holds 512 events, but only `JPEG_SINK_EVENT_POOL_ALLOWANCE` (128) of them may carry JPEG bytes. `JpegSinkBacklog` enforces this with a single atomic counter: `try_acquire` does one `fetch_add`, and a caller whose pre-increment count landed at or above the allowance subtracts itself back out and drops the keyframe (counted by `inc_sink_keyframes_dropped`). There is no compare-exchange loop. The counter can transiently overshoot while racing losers back out, but a live permit never backs out, so the number of held permits never exceeds the allowance. Ordering is `Relaxed` throughout because the counter guards no data. The JPEG bytes travel on the sink channel, which carries its own happens-before. The permit is an RAII struct (`SinkJpegPermit`) that rides inside the `SinkEvent::StoreKeyframe` and releases on drop, whichever path the event takes.

## The event sink boundary

Workers report results only through the `EventSink` trait (`emit_event_sync`, `emit_event_nonblocking`, `store_keyframe_sync`), never by touching storage directly. The trait is `Send + Sync` because worker threads share one `Arc<dyn EventSink>`. Keyframe-mode VLM workers enqueue `SinkEvent`s and let the dedicated writer thread absorb storage latency. Clip VLM workers call the blocking sink methods directly, which is acceptable because clip cadence is bounded by the accumulator's window and delay settings. The WAL-backed sink and its optional SpacetimeDB mirror are described in [WAL and events](/docs/internals/wal-and-events/).

## Edge cases and limits

- A malformed frame (planes shorter than the declared dimensions) is dropped by `check_frame` before it can update `prev_signal`, so one corrupt decode cannot poison the temporal deltas of every following frame.
- Compressed RTP access units larger than 2 MiB are dropped before the pipeline-owned queue copy, and JPEGs larger than 2 MiB are recycled before they can enter downstream work. Those payload limits make the process reservation a byte bound, not a queue-item estimate.
- The process reserves the full per-generation byte envelope and fixed worker count before workers start. `VIDARAX_MEDIA_MEMORY_BUDGET_BYTES` and `VIDARAX_MEDIA_WORKER_THREAD_BUDGET` bound total admitted generations even across principals.
- A frame the gate keeps but whose JPEG encode fails or comes back empty is dropped entirely. An empty payload would waste a VLM call.
- While the loop detector reports the stream stuck (`loop_active`), the VLM worker skips inference for kept keyframes and counts them as dropped. The `loop_detected` event was already emitted by the gate side.
- The per-session output token budget (`max_output_tokens_per_second`, zero disables) is enforced in the VLM worker with a one-second window per session. Over-budget keyframes are dropped, and token counts are approximated from output byte length.
- `SinkState` is a deliberately large enum on the decode worker's stack. Boxing the big variant would add a pointer chase per frame for no benefit (`#[allow(clippy::large_enum_variant)]` marks the decision).
- The decode worker builds its decoder lazily on the first frame and rebuilds it if a later frame arrives with a different codec. Codec and decoder live in one slot so they cannot drift apart.

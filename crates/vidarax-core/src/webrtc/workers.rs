//! WebRTC decode, analysis, and VLM worker pools.
//!
//! The pipeline is ordered through bounded `kanal` queues. Closing an upstream
//! sender propagates shutdown to downstream worker threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arc_swap::{ArcSwap, ArcSwapOption};
use base64::Engine as _;

use crate::coordinates::FrameCoordinates;
use crate::crop::{CropRegion, PixelCrop};
use crate::dedup::DedupFilter;
use crate::embedding_sidecar::EmbeddingSidecarClient;
use crate::gate::{FrameSignal, GateConfig, GateEventType};
use crate::loop_detector::LoopDetector;
use crate::metrics::PipelineMetrics;
use crate::novelty::{LiveNoveltyConfig, LiveNoveltyGate, LiveNoveltyOutcome};
use crate::pipeline::{TwoPassConfig, TwoPassPipeline};
use crate::provider::{InferenceImage, InferenceObserver, InferenceProvider, InferenceRequest};
use crate::tiered_vlm::{run_tiered, TieredVlmConfig};
use crate::webrtc::clip::{
    spawn_clip_accumulator, spawn_clip_vlm_workers, ClipConfig, ClipRateGate, ClipWork,
};
use crate::webrtc::decode::{
    DecodeError, Decoder, DecoderBackend, DecoderConfig, VideoCodec, YuvFrame,
    FFMPEG_YUV_PENDING_POOL_ALLOWANCE, FFMPEG_YUV_READER_QUEUE_CAPACITY,
};
use crate::webrtc::recycle::{RecycledBytes, VecPool};
use crate::webrtc::session::{RtpFrame, WebRtcConfig};
use crate::webrtc::signals::{
    check_frame, crop_yuv, yuv_to_frame_signal_unchecked, yuv_to_jpeg_unchecked, CropPool,
};

pub const STREAM_FRAME_QUEUE_CAPACITY: usize = 64;
pub const VLM_WORK_QUEUE_CAPACITY: usize = 32;
pub const CLIP_FRAME_QUEUE_CAPACITY: usize = STREAM_FRAME_QUEUE_CAPACITY;
/// Clip VLM backlog is intentionally backpressured with no queued clip work:
/// 1 active worker plus 1 accumulator-owned request that may block in send.
/// That keeps the full JPEG pool worst case at 484 slots: 66 decode→analysis
/// + 162 normal VLM/sink + 64 clip-frame queue + 64 accumulator + 128
///   active/blocked clip work.
pub const CLIP_WORK_QUEUE_CAPACITY: usize = 0;
pub const SINK_EVENT_QUEUE_CAPACITY: usize = 512;
pub const JPEG_POOL_SLOT_CEILING: usize = 512;
const FFMPEG_READER_CONSTRUCTING_YUV_FRAMES: usize = 1;
const FFMPEG_DECODER_PENDING_YUV_FRAMES: usize = FFMPEG_YUV_PENDING_POOL_ALLOWANCE;
const DECODE_CONSUMER_YUV_FRAMES: usize = 1;
const DECODE_OUTPUT_POOL_SLOTS_PER_WORKER: usize = FFMPEG_READER_CONSTRUCTING_YUV_FRAMES
    + FFMPEG_YUV_READER_QUEUE_CAPACITY
    + FFMPEG_DECODER_PENDING_YUV_FRAMES
    + DECODE_CONSUMER_YUV_FRAMES;
const JPEG_SINK_EVENT_POOL_ALLOWANCE: usize = 128;

pub fn jpeg_sink_event_backlog_capacity() -> usize {
    JPEG_SINK_EVENT_POOL_ALLOWANCE
}

pub fn per_stream_decode_workers(_configured: usize) -> usize {
    1
}

pub fn per_stream_vlm_workers(_configured: usize) -> usize {
    1
}

pub fn per_stream_analysis_workers(_configured: usize) -> usize {
    // Analysis owns stream-order gate and loop-detector state. Parallelism is
    // across sessions; splitting one ordered stream would race that state.
    1
}

pub fn decode_output_pool_slots(gpu_available: bool, codec: VideoCodec) -> usize {
    let backend = DecoderBackend::select(gpu_available, codec);
    match backend {
        DecoderBackend::NvDec | DecoderBackend::FfmpegSw => {
            // Full bounded ffmpeg reader queue, steady-state decoder pending
            // FIFO, one reader-constructed frame, and one decode-consumer frame.
            DECODE_OUTPUT_POOL_SLOTS_PER_WORKER
        }
        #[cfg(feature = "vp8")]
        DecoderBackend::Vp8 => crate::webrtc::decode::SOFTWARE_YUV_POOL_MIN_SLOTS,
        DecoderBackend::Software | DecoderBackend::Unsupported => backend.min_yuv_pool_slots(),
    }
}

pub fn jpeg_pool_slots(analysis_workers: usize, vlm_workers: usize) -> usize {
    let analysis_workers = per_stream_analysis_workers(analysis_workers);
    let vlm_workers = per_stream_vlm_workers(vlm_workers);
    let decode_to_analysis = STREAM_FRAME_QUEUE_CAPACITY + analysis_workers + 1;
    let normal_path = VLM_WORK_QUEUE_CAPACITY + vlm_workers + JPEG_SINK_EVENT_POOL_ALLOWANCE + 1;
    let active_clip_workers = vlm_workers;
    let blocked_clip_sender = 1;
    let clip_path = CLIP_FRAME_QUEUE_CAPACITY
        + crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST
        + (CLIP_WORK_QUEUE_CAPACITY + active_clip_workers + blocked_clip_sender)
            * crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST;

    decode_to_analysis + normal_path + clip_path
}

// ─── EventSink trait ──────────────────────────────────────────────────────────

/// Event writes used by worker pools.
///
/// # Thread-safety
///
/// The `Send + Sync` bounds are required because worker threads hold
/// `Arc<dyn EventSink>` and call these methods concurrently.  The
/// implementation must be safe to call from multiple threads simultaneously.
pub trait EventSink: Send + Sync {
    /// Emit a real-time agent event (blocking; must not hold locks indefinitely).
    // Event sinks receive distinct event fields from worker threads without an intermediate event object.
    #[allow(clippy::too_many_arguments)]
    fn emit_event_sync(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String>;

    /// Emit a real-time agent event without waiting for durable storage.
    #[allow(clippy::too_many_arguments)]
    fn emit_event_nonblocking(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String> {
        self.emit_event_sync(
            run_id,
            session_id,
            frame_index,
            pts_ms,
            coordinates,
            event_type,
            confidence,
            description,
        )
    }

    /// Persist a keyframe with its JPEG thumbnail (blocking).
    fn store_keyframe_sync(&self, event: KeyframeEvent<'_>) -> Result<(), String>;
}

/// One keyframe and the provenance required to persist it correctly.
#[derive(Debug, Clone, Copy)]
pub struct KeyframeEvent<'a> {
    pub run_id: &'a str,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub coordinates: FrameCoordinates,
    pub event_type: &'a str,
    pub description: &'a str,
    pub jpeg_data: &'a [u8],
}

// ─── Pipeline types ───────────────────────────────────────────────────────────

/// Decoded video frame ready for the analysis stage.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Gate-engine signal computed from the luma plane.
    pub signal: crate::gate::FrameSignal,
    /// JPEG thumbnail of the decoded frame (`Some` after successful decode).
    ///
    /// Stored as a recycled buffer so downstream workers can move ownership
    /// without copying the JPEG payload.
    pub jpeg: Option<RecycledBytes>,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Per-session monotonically increasing frame index (== `signal.frame_index`).
    pub seq: u64,
    /// Transform from source pixels to this frame's analyzed pixels.
    pub coordinates: FrameCoordinates,
}

/// Work item forwarded to VLM workers when a keyframe is decided upon.
#[derive(Debug, Clone)]
pub struct KeyframeWork {
    /// Session run identifier — shared via `Arc<str>` so cloning only updates a refcount.
    pub run_id: Arc<str>,
    /// Session identifier — shared via `Arc<str>` so cloning only updates a refcount.
    pub session_id: Arc<str>,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub coordinates: FrameCoordinates,
    /// Gate reason code: `"scene_cut"` | `"periodic_keepalive"` | `"initial_frame"`.
    pub event_type: &'static str,
    /// Gate confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Gate-derived novelty score (0=familiar, 1=novel).
    pub novelty_score: f32,
    /// Gate-derived motion score (0=static, 1=high motion).
    pub motion_score: f32,
    /// Raw JPEG bytes — base64-encoded only at provider boundaries and stored raw
    /// in the local content-addressed sidecar.
    ///
    /// Recycled buffer moved through VLM and storage without copying the payload.
    pub jpeg_bytes: RecycledBytes,
    /// Semantic prompt to pass to the VLM.
    pub prompt: Arc<str>,
    /// When `true` the analysis worker detected an active visual loop at the
    /// time this work item was queued.  VLM workers use this flag to skip
    /// inference entirely (the scene has not changed), avoiding redundant
    /// calls and their associated cost.
    pub loop_active: bool,
}

#[cfg(test)]
fn build_stream_frame_from_yuv(
    yuv: &YuvFrame,
    seq: u64,
    pts_ms: u64,
    prev_signal: &mut Option<crate::gate::FrameSignal>,
    ycbcr_scratch: &mut Vec<u8>,
    jpeg_pool: &VecPool,
) -> Option<StreamFrame> {
    // A frame whose planes don't match its dimensions can't be read safely, and
    // its statistics would poison the temporal deltas of every frame after it.
    // Drop it before it touches prev_signal; the next good frame recovers.
    if let Err(err) = check_frame(yuv) {
        tracing::warn!(seq, %err, "dropping malformed frame");
        return None;
    }

    let signal = yuv_to_frame_signal_unchecked(yuv, seq, pts_ms, prev_signal.as_ref());
    // A frame the encoder rejects costs this one thumbnail, not the stream. Drop
    // it to None and keep decoding; the signal still flows to the gate engine.
    let jpeg = match yuv_to_jpeg_unchecked(yuv, 75, ycbcr_scratch, jpeg_pool) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            tracing::warn!(seq, %err, "dropping thumbnail for unencodable frame");
            None
        }
    };
    *prev_signal = Some(signal);

    Some(StreamFrame {
        signal,
        jpeg,
        pts_ms,
        seq,
        coordinates: FrameCoordinates::full_frame(yuv.width, yuv.height),
    })
}

fn build_clip_stream_frame_from_yuv(
    yuv: &YuvFrame,
    seq: u64,
    pts_ms: u64,
    coordinates: FrameCoordinates,
    prev_signal: &mut Option<crate::gate::FrameSignal>,
    clip_rate_gate: &mut ClipRateGate,
    encode: impl FnOnce() -> Option<RecycledBytes>,
) -> Option<StreamFrame> {
    // Invalid planes must not update temporal deltas or clip sampling state.
    if let Err(err) = check_frame(yuv) {
        tracing::warn!(seq, %err, "dropping malformed frame");
        return None;
    }

    let signal = yuv_to_frame_signal_unchecked(yuv, seq, pts_ms, prev_signal.as_ref());
    *prev_signal = Some(signal);

    let jpeg = if clip_rate_gate.should_keep(pts_ms) {
        let jpeg = encode().filter(|b| !b.is_empty());
        if jpeg.is_some() {
            clip_rate_gate.commit(pts_ms);
        }
        jpeg
    } else {
        None
    };

    Some(StreamFrame {
        signal,
        jpeg,
        pts_ms,
        seq,
        coordinates,
    })
}

/// Borrowed per-frame context the gate needs to emit events and build work.
///
/// Grouped so the gate signature stays readable and callers pass one struct
/// instead of six loose arguments.
struct KeyframeContext<'a> {
    run_id: &'a Arc<str>,
    session_id: &'a Arc<str>,
    prompt: &'a Arc<ArcSwap<Arc<str>>>,
    stdb: &'a Arc<dyn EventSink>,
    metrics: &'a PipelineMetrics,
}

struct LoopDetectedEvent {
    frame_index: u64,
    pts_ms: u64,
    confidence: f32,
    description: &'static str,
}

/// Per-stream loop-detection and gate state, driven once per decoded frame.
///
/// Owns the two-pass gate and the loop detector for one ordered stream. The
/// keyframe decision is split out here, away from any live decoder, so it stays
/// unit-testable: [`on_frame`](GateStreamState::on_frame) takes the cheap frame
/// signal plus a lazy `encode` closure and only runs the encoder when the gate
/// actually keeps the frame. That is what lets the decode worker skip the JPEG
/// for the ~95% of frames the gate drops.
struct GateStreamState {
    pipeline: TwoPassPipeline,
    loop_det: LoopDetector,
    /// True while the loop detector considers the stream stuck; cleared when the
    /// detector no longer sees enough repeated hashes in its window.
    loop_active: bool,
}

impl GateStreamState {
    fn new(
        gate_config: GateConfig,
        loop_hamming_threshold: u32,
        loop_repeat_threshold: usize,
    ) -> Self {
        Self {
            pipeline: TwoPassPipeline::new(TwoPassConfig::default(), gate_config),
            loop_det: LoopDetector::new(loop_hamming_threshold, loop_repeat_threshold),
            loop_active: false,
        }
    }

    /// Feed one frame's perceptual hash to the loop detector, returning the
    /// `loop_detected` event once on entry so the caller can emit it off the
    /// decode thread's blocking path.
    fn observe_loop(
        &mut self,
        signal: &FrameSignal,
        pts_ms: u64,
        metrics: &PipelineMetrics,
    ) -> Option<LoopDetectedEvent> {
        if self.loop_det.check(signal.perceptual_hash) {
            if !self.loop_active {
                metrics.inc_loop_detected();
                self.loop_active = true;
                return Some(LoopDetectedEvent {
                    frame_index: signal.frame_index,
                    pts_ms,
                    confidence: 0.9,
                    description: "loop detected via perceptual-hash ring buffer",
                });
            }
            self.loop_active = true;
        } else {
            self.loop_active = false;
        }
        None
    }

    /// Run loop detection and the gate on one frame, returning the keyframe work
    /// to dispatch or `None` when the frame is dropped.
    ///
    /// `encode` is called at most once, only after the gate decides to keep the
    /// frame, so a dropped frame never pays for a JPEG. A kept frame whose
    /// `encode` yields nothing usable is also dropped: an empty payload would
    /// waste a VLM call.
    fn on_frame(
        &mut self,
        signal: FrameSignal,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        ctx: &KeyframeContext<'_>,
        encode: impl FnOnce() -> Option<RecycledBytes>,
    ) -> Option<KeyframeWork> {
        if let Some(event) = self.observe_loop(&signal, pts_ms, ctx.metrics) {
            let _ = ctx.stdb.emit_event_nonblocking(
                ctx.run_id,
                ctx.session_id,
                event.frame_index,
                event.pts_ms,
                coordinates,
                "loop_detected",
                event.confidence,
                event.description,
            );
        }

        let gate_start = std::time::Instant::now();
        let metas = self.pipeline.analyze_batch_defer_gate_commit(&[signal]);
        ctx.metrics
            .gate_latency_us
            .record(gate_start.elapsed().as_micros() as u64);
        ctx.metrics.inc_gate_frame_analyzed();
        let meta = *metas.first()?;

        if meta.gate_event != GateEventType::KeepKeyframe {
            return None;
        }
        ctx.metrics.inc_gate_keyframe_selected();

        let jpeg_bytes = encode().filter(|b| !b.is_empty())?;
        self.pipeline.commit_gate_keyframe(signal);

        let event_type: &'static str = if meta.scene_cut {
            "scene_cut"
        } else {
            "periodic_keepalive"
        };

        Some(KeyframeWork {
            run_id: Arc::clone(ctx.run_id),
            session_id: Arc::clone(ctx.session_id),
            frame_index: signal.frame_index,
            pts_ms,
            coordinates,
            event_type,
            confidence: meta.confidence,
            novelty_score: meta.novelty_score,
            motion_score: meta.motion_score,
            jpeg_bytes,
            prompt: Arc::clone(&*ctx.prompt.load_full()),
            loop_active: self.loop_active,
        })
    }
}

/// Live decode-worker state derived from a [`DecodeSink`].
///
/// `Keyframe` carries the running gate plus the dispatch handles. `Stream`
/// carries clip sampling state so over-rate frames forward signals without JPEGs.
//
// One value of this lives per decode worker, on its stack, for the worker's
// whole life, and the large `Keyframe` variant is the common case. Boxing it to
// even out the variants would only add a pointer chase to every frame in the hot
// loop for no gain, so the size gap is deliberate.
#[allow(clippy::large_enum_variant)]
enum SinkState {
    Keyframe {
        gate: GateStreamState,
        vlm_tx: kanal::Sender<KeyframeWork>,
        stdb: Arc<dyn EventSink>,
        run_id: Arc<str>,
        session_id: Arc<str>,
        prompt: Arc<ArcSwap<Arc<str>>>,
    },
    Stream {
        frame_tx: kanal::Sender<StreamFrame>,
        clip_rate_gate: ClipRateGate,
    },
}

// ─── Worker pool configuration ────────────────────────────────────────────────

/// Per-session pipeline topology and tunables, derived from [`WebRtcConfig`].
///
/// Built once on the control path when a session starts, so cloning the gate
/// config here never touches the frame hot path.
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    /// Requested decode worker threads (clamped to one per stream at spawn).
    pub decode_workers: usize,
    /// Requested analysis worker threads (clamped to one per stream at spawn).
    pub analysis_workers: usize,
    /// Requested VLM worker threads (clamped to one per stream at spawn).
    pub vlm_workers: usize,
    /// Decode target width in pixels.
    pub decode_width: u32,
    /// Decode target height in pixels.
    pub decode_height: u32,
    /// Whether the decoder may use a hardware backend when available.
    pub gpu_available: bool,
    /// Gate-engine thresholds applied by analysis workers.
    pub gate_config: GateConfig,
    /// Perceptual-hash Hamming-distance threshold for treating frames as the same screen.
    pub loop_hamming_threshold: u32,
    /// Repeat count within the window that marks a stream as looping.
    pub loop_repeat_threshold: usize,
    /// VLM output token-rate cap; zero disables the limiter.
    pub max_output_tokens_per_second: u32,
    /// Optional region of interest cropped from each decoded frame before the
    /// gate and JPEG encoder run. `None` analyzes the whole frame.
    pub crop: Option<CropRegion>,
}

impl From<&WebRtcConfig> for WorkerPoolConfig {
    fn from(cfg: &WebRtcConfig) -> Self {
        Self {
            decode_workers: cfg.decode_workers,
            analysis_workers: cfg.analysis_workers,
            vlm_workers: cfg.vlm_workers,
            decode_width: cfg.decode_width,
            decode_height: cfg.decode_height,
            gpu_available: cfg.gpu_available,
            gate_config: cfg.gate_config.clone(),
            loop_hamming_threshold: cfg.loop_hamming_threshold,
            loop_repeat_threshold: cfg.loop_repeat_threshold,
            max_output_tokens_per_second: cfg.max_output_tokens_per_second,
            crop: cfg.crop,
        }
    }
}

// ─── Decode workers ───────────────────────────────────────────────────────────

/// Inputs for the decode worker pool: channels, pool sizing, decode target,
/// and telemetry handles.
pub struct DecodeWorkerParams {
    /// Requested worker count; clamped to one stateful decoder per stream.
    pub workers: usize,
    /// Whether the decoder may use a hardware backend.
    pub gpu_available: bool,
    /// Decode target width in pixels.
    pub decode_width: u32,
    /// Decode target height in pixels.
    pub decode_height: u32,
    /// YUV output pool slots reserved for the decoder.
    pub output_pool_slots: usize,
    /// JPEG encode pool slots reserved for this worker.
    pub jpeg_pool_slots: usize,
    /// RTP frames in.
    pub rtp_rx: kanal::Receiver<RtpFrame>,
    /// Where decoded frames go, and whether the gate runs inline.
    pub sink: DecodeSink,
    /// Optional region of interest cropped from each decoded frame before the
    /// gate and JPEG encoder run.
    pub crop: Option<CropRegion>,
    pub metrics: Arc<PipelineMetrics>,
    pub session_span: tracing::Span,
}

/// Downstream wiring for a decode worker.
///
/// `Keyframe` is the default path: the worker runs the gate inline and only
/// JPEG-encodes the frames the gate keeps, dispatching them straight to the VLM
/// queue. `Stream` is clip mode: the worker samples by PTS first, then encodes
/// only frames the accumulator would keep.
pub enum DecodeSink {
    /// Gate inline and dispatch kept keyframes to the VLM queue.
    Keyframe(KeyframeSink),
    /// Forward clip signals, attaching JPEGs only to sampled frames.
    Stream {
        frame_tx: kanal::Sender<StreamFrame>,
        clip_config: ClipConfig,
    },
}

/// Everything a decode worker needs to run the gate and dispatch keyframes.
pub struct KeyframeSink {
    /// Gate-engine thresholds applied to each frame.
    pub gate_config: GateConfig,
    /// Perceptual-hash Hamming-distance threshold for treating frames as the same screen.
    pub loop_hamming_threshold: u32,
    /// Repeat count within the window that marks a stream as looping.
    pub loop_repeat_threshold: usize,
    /// Kept keyframes out to the VLM queue.
    pub vlm_tx: kanal::Sender<KeyframeWork>,
    /// Event sink for the loop-detected notice.
    pub stdb: Arc<dyn EventSink>,
    pub run_id: Arc<str>,
    pub session_id: Arc<str>,
    /// Live prompt handle, reloaded per keyframe.
    pub prompt: Arc<ArcSwap<Arc<str>>>,
}

/// Spawn decode workers for one ordered media stream.
///
/// One stream uses one stateful decoder. Codec detection is lazy on the first
/// frame, and the decoder is rebuilt if a later session uses another codec on
/// the same worker. `params.workers` is API-compatible only; parallelism is
/// across sessions, not within one ordered stream.
pub fn spawn_decode_workers(params: DecodeWorkerParams) -> std::io::Result<()> {
    let DecodeWorkerParams {
        workers,
        gpu_available,
        decode_width,
        decode_height,
        output_pool_slots,
        jpeg_pool_slots,
        rtp_rx,
        sink,
        crop,
        metrics,
        session_span,
    } = params;
    // One ordered stream is one stateful decoder, so this pool is always a single
    // thread; the requested count only scales parallelism across sessions.
    debug_assert_eq!(per_stream_decode_workers(workers), 1);

    // Resolve the wiring into the worker's running state once, up front. In the
    // keyframe path this is where the gate lives, so the hot loop encodes a JPEG
    // only for the frames the gate keeps instead of for every decoded frame.
    let mut sink_state = match sink {
        DecodeSink::Keyframe(KeyframeSink {
            gate_config,
            loop_hamming_threshold,
            loop_repeat_threshold,
            vlm_tx,
            stdb,
            run_id,
            session_id,
            prompt,
        }) => SinkState::Keyframe {
            gate: GateStreamState::new(gate_config, loop_hamming_threshold, loop_repeat_threshold),
            vlm_tx,
            stdb,
            run_id,
            session_id,
            prompt,
        },
        DecodeSink::Stream {
            frame_tx,
            clip_config,
        } => SinkState::Stream {
            frame_tx,
            clip_rate_gate: ClipRateGate::new(clip_config.target_fps),
        },
    };

    std::thread::Builder::new()
        .name("vx-decode-0".to_string())
        .spawn(move || {
            use crate::webrtc::decode::VideoCodec;

            // Lazy decoder: created on the first frame so we know the codec.
            // The codec travels with the decoder in one slot so "we have a
            // decoder for this codec" is a single state the type enforces,
            // rather than two fields that could drift apart.
            let mut decoder: Option<(VideoCodec, Decoder)> = None;
            let mut prev_signal: Option<FrameSignal> = None;
            // Warn only once if the crop cannot be represented at the live size.
            let mut warned_undersized_crop = false;
            // Reused across frames to avoid per-frame 6MB YCbCr allocation.
            let mut ycbcr_scratch: Vec<u8> =
                Vec::with_capacity(decode_width as usize * decode_height as usize * 3);
            let jpeg_pool = VecPool::with_slots(jpeg_pool_slots);
            // Only built when a crop is configured; keeps the re-pack off the
            // per-frame allocator, matching the JPEG pool above.
            let crop_pool = crop.map(|_| CropPool::new(output_pool_slots));

            'rtp_frames: while let Ok(frame) = rtp_rx.recv() {
                let _guard = session_span.enter();
                metrics.inc_rtp_received();
                let seq = frame.seq;
                let pts_ms = frame.pts_ms;

                // Re-initialise the decoder when the codec changes (e.g. a
                // new session with a different codec arrives on the same worker).
                // Matching on the slot yields the decoder directly, so there is
                // no separate "it must be present" step that could fail.
                let dec = match &mut decoder {
                    Some((codec, dec)) if *codec == frame.codec => dec,
                    slot => {
                        let config = DecoderConfig {
                            gpu_available,
                            codec: frame.codec,
                            width: decode_width,
                            height: decode_height,
                            output_pool_slots,
                        };
                        let entry = (
                            frame.codec,
                            Decoder::new_with_metrics(&config, Arc::clone(&metrics)),
                        );
                        &mut slot.insert(entry).1
                    }
                };

                let decode_start = std::time::Instant::now();
                let yuv = match dec.decode(&frame.nals) {
                    Ok(yuv) => yuv,
                    Err(DecodeError::UnsupportedCodec(codec)) => {
                        tracing::error!(
                            ?codec,
                            "no supported WebRTC decoder for negotiated codec; VP8 software decode is unsupported in the live zero-dependency pipeline; configure the client to offer H.264"
                        );
                        return;
                    }
                    Err(_) => continue 'rtp_frames,
                };
                metrics
                    .decode_latency_us
                    .record(decode_start.elapsed().as_micros() as u64);
                metrics.inc_frames_decoded();

                // Resolve the crop once, then carry both the requested and exact
                // pixel region with the frame. Durable events can map a model's
                // coordinates back to the source without guessing which resize
                // or crop was active.
                let coordinates = match FrameCoordinates::resolve(yuv.width, yuv.height, crop) {
                    Some(coordinates) => coordinates,
                    None => {
                        // The requested region is too small to represent at this
                        // resolution. Dropping is deliberate: widening it back to
                        // the whole frame would analyze more of the screen than
                        // the caller asked for. Warn once so the operator sees why
                        // no analysis is coming out, without per-frame spam.
                        if !warned_undersized_crop {
                            warned_undersized_crop = true;
                            tracing::warn!(
                                width = yuv.width,
                                height = yuv.height,
                                "configured crop is smaller than 2x2 at this resolution; dropping frames rather than analyzing the whole screen"
                            );
                        }
                        continue 'rtp_frames;
                    }
                };
                // Restrict analysis to the configured region of interest. Cropping
                // here, before either sink branch, means the frame filter and VLM
                // JPEG both see the same pixels.
                let selected = coordinates.resolved_region;
                let yuv = if selected.x == 0
                    && selected.y == 0
                    && selected.width == coordinates.source_extent.width
                    && selected.height == coordinates.source_extent.height
                {
                    yuv
                } else {
                    let rect = PixelCrop {
                        x: selected.x,
                        y: selected.y,
                        width: selected.width,
                        height: selected.height,
                    };
                    // A malformed source re-pack falls back to the original,
                    // which the per-branch check_frame then drops.
                    match &crop_pool {
                        Some(pool) => crop_yuv(&yuv, rect, pool).unwrap_or(yuv),
                        None => yuv,
                    }
                };

                match &mut sink_state {
                    SinkState::Keyframe {
                        gate,
                        vlm_tx,
                        stdb,
                        run_id,
                        session_id,
                        prompt,
                    } => {
                        // A frame whose planes don't match its dimensions can't be
                        // read safely, and its statistics would poison the temporal
                        // deltas of every frame after it. Drop it before it touches
                        // prev_signal; the next good frame recovers.
                        if let Err(err) = check_frame(&yuv) {
                            tracing::warn!(seq, %err, "dropping malformed frame");
                            continue 'rtp_frames;
                        }
                        let signal = yuv_to_frame_signal_unchecked(
                            &yuv,
                            seq,
                            pts_ms,
                            prev_signal.as_ref(),
                        );
                        prev_signal = Some(signal);

                        let ctx = KeyframeContext {
                            run_id,
                            session_id,
                            prompt,
                            stdb,
                            metrics: &metrics,
                        };
                        // The encoder only runs if the gate keeps the frame. A
                        // rejected thumbnail costs this one frame, not the stream.
                        let work = gate.on_frame(signal, pts_ms, coordinates, &ctx, || {
                            match yuv_to_jpeg_unchecked(&yuv, 75, &mut ycbcr_scratch, &jpeg_pool) {
                                Ok(bytes) => Some(bytes),
                                Err(err) => {
                                    tracing::warn!(
                                        seq,
                                        %err,
                                        "dropping thumbnail for unencodable frame"
                                    );
                                    None
                                }
                            }
                        });
                        let Some(work) = work else {
                            continue 'rtp_frames;
                        };

                        // Non-blocking: drop if the VLM queue is full rather than
                        // stall the decode loop for the whole stream.
                        // kanal try_send yields Ok(false) when the queue is full,
                        // so only a real enqueue (Ok(true)) is a kept keyframe; a
                        // full queue or a closed channel is a drop.
                        if matches!(vlm_tx.try_send(work), Ok(true)) {
                            metrics.inc_keyframes();
                        } else {
                            metrics.inc_keyframes_dropped();
                        }
                    }
                    SinkState::Stream {
                        frame_tx,
                        clip_rate_gate,
                    } => {
                        // Clip mode uses the accumulator's PTS sampling rule
                        // before encoding so over-rate frames stay cheap.
                        let Some(sf) = build_clip_stream_frame_from_yuv(
                            &yuv,
                            seq,
                            pts_ms,
                            coordinates,
                            &mut prev_signal,
                            clip_rate_gate,
                            || match yuv_to_jpeg_unchecked(&yuv, 75, &mut ycbcr_scratch, &jpeg_pool) {
                                Ok(bytes) => Some(bytes),
                                Err(err) => {
                                    tracing::warn!(
                                        seq,
                                        %err,
                                        "dropping thumbnail for unencodable frame"
                                    );
                                    None
                                }
                            },
                        ) else {
                            continue 'rtp_frames;
                        };
                        if frame_tx.send(sf).is_err() {
                            return; // downstream dropped — shut down
                        }
                    }
                }
            }
        })?;

    Ok(())
}

// ─── Analysis workers ─────────────────────────────────────────────────────────

/// Spawn analysis workers.
///
/// Clip mode forwards accepted frames to the clip accumulator with `try_send`,
/// dropping if its queue is full.
/// Inputs for the analysis worker pool.
pub struct AnalysisWorkerParams {
    /// Requested worker count; clamped to one per ordered stream.
    pub workers: usize,
    /// Gate-engine thresholds applied to each frame.
    pub gate_config: GateConfig,
    /// Perceptual-hash Hamming-distance threshold for treating frames as the same screen.
    pub loop_hamming_threshold: u32,
    /// Repeat count within the window that marks a stream as looping.
    pub loop_repeat_threshold: usize,
    /// Decoded stream frames in.
    pub frame_rx: kanal::Receiver<StreamFrame>,
    /// Accepted clip frames out to the accumulator.
    pub clip_tx: kanal::Sender<StreamFrame>,
    pub stdb: Arc<dyn EventSink>,
    pub run_id: Arc<str>,
    pub session_id: Arc<str>,
    pub prompt: Arc<ArcSwap<Arc<str>>>,
    pub metrics: Arc<PipelineMetrics>,
    pub session_span: tracing::Span,
}

pub fn spawn_analysis_workers(params: AnalysisWorkerParams) -> std::io::Result<()> {
    let AnalysisWorkerParams {
        workers,
        gate_config,
        loop_hamming_threshold,
        loop_repeat_threshold,
        frame_rx,
        clip_tx,
        stdb,
        run_id,
        session_id,
        prompt,
        metrics,
        session_span,
    } = params;
    for i in 0..per_stream_analysis_workers(workers) {
        let frame_rx = frame_rx.clone();
        let clip_tx = clip_tx.clone();
        let stdb = Arc::clone(&stdb);
        let run_id = Arc::clone(&run_id);
        let session_id = Arc::clone(&session_id);
        let prompt = Arc::clone(&prompt);
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();
        let gate_config = gate_config.clone();

        std::thread::Builder::new()
            .name(format!("vx-analysis-{i}"))
            .spawn(move || {
                let mut gate = GateStreamState::new(
                    gate_config,
                    loop_hamming_threshold,
                    loop_repeat_threshold,
                );

                while let Ok(sf) = frame_rx.recv() {
                    let _guard = session_span.enter();

                    let ctx = KeyframeContext {
                        run_id: &run_id,
                        session_id: &session_id,
                        prompt: &prompt,
                        stdb: &stdb,
                        metrics: &metrics,
                    };

                    if let Some(event) = gate.observe_loop(&sf.signal, sf.pts_ms, ctx.metrics) {
                        let _ = ctx.stdb.emit_event_nonblocking(
                            ctx.run_id,
                            ctx.session_id,
                            event.frame_index,
                            event.pts_ms,
                            sf.coordinates,
                            "loop_detected",
                            event.confidence,
                            event.description,
                        );
                    }
                    let _ = clip_tx.try_send(sf);
                }
            })?;
    }

    Ok(())
}

// ─── VLM workers ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct JpegSinkBacklog {
    in_flight: Arc<AtomicUsize>,
}

impl JpegSinkBacklog {
    fn new() -> Self {
        Self {
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn try_acquire(&self) -> Option<SinkJpegPermit> {
        // Claim a slot with a single wait-free `fetch_add`, then hand it back if we
        // turned out to be over the cap. No compare-exchange retry loop: every
        // caller reads a distinct pre-increment count, and only the callers whose
        // count landed below the allowance keep their slot, so at most
        // `JPEG_SINK_EVENT_POOL_ALLOWANCE` permits are ever live at once.
        //
        // The counter can momentarily read above the allowance while racing
        // callers each add before the losers subtract back. That overshoot is
        // private to this function — nothing else reads `in_flight` — and it
        // cannot admit an extra permit, since a live permit never backs out, so
        // the count stays at or above the number of held permits. The flip side is
        // that a caller can be turned away while another is mid-back-out and a slot
        // is really free; for a backlog whose whole job is to shed keyframes under
        // pressure, dropping one slightly early is the right kind of wrong.
        //
        // Ordering is `Relaxed` throughout: this atomic only counts admissions. It
        // guards no shared data — the keyframe bytes ride the sink channel, which
        // carries its own happens-before — so there is nothing for an acquire/
        // release edge to publish, and the count's own modification order is all
        // the bound proof needs.
        let prev = self.in_flight.fetch_add(1, Ordering::Relaxed);
        if prev >= JPEG_SINK_EVENT_POOL_ALLOWANCE {
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(SinkJpegPermit {
            in_flight: Arc::clone(&self.in_flight),
        })
    }
}

struct SinkJpegPermit {
    in_flight: Arc<AtomicUsize>,
}

impl Drop for SinkJpegPermit {
    fn drop(&mut self) {
        // Relaxed for the same reason as `try_acquire`: releasing a slot publishes
        // no data, it just frees the count for the next admission.
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

// Builds sink events while preserving separate backlog, metric, metadata, and JPEG ownership inputs.
#[allow(clippy::too_many_arguments)]
fn store_keyframe_event_with_backlog(
    backlog: &JpegSinkBacklog,
    metrics: &PipelineMetrics,
    run_id: Arc<str>,
    frame_index: u64,
    pts_ms: u64,
    coordinates: FrameCoordinates,
    event_type: &'static str,
    description: Arc<str>,
    jpeg_bytes: RecycledBytes,
) -> Option<SinkEvent> {
    let jpeg_permit = match backlog.try_acquire() {
        Some(permit) => permit,
        None => {
            metrics.inc_sink_keyframes_dropped();
            return None;
        }
    };

    Some(SinkEvent::StoreKeyframe {
        run_id,
        frame_index,
        pts_ms,
        coordinates,
        event_type,
        description,
        jpeg_bytes,
        _jpeg_permit: jpeg_permit,
    })
}

/// Event routed from VLM worker threads to the dedicated SpacetimeDB writer.
enum SinkEvent {
    Emit {
        run_id: Arc<str>,
        session_id: Arc<str>,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &'static str,
        confidence: f32,
        description: Arc<str>,
    },
    StoreKeyframe {
        run_id: Arc<str>,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &'static str,
        description: Arc<str>,
        jpeg_bytes: RecycledBytes,
        _jpeg_permit: SinkJpegPermit,
    },
}

// ─── Pipeline assembly ────────────────────────────────────────────────────────

/// Channels, sinks, and provider that wire one session's worker pipeline
/// together. Everything here is constructed once on the control path.
pub struct PipelineWiring<I>
where
    I: InferenceProvider + 'static,
{
    /// RTP frames from the session feed into the decoder.
    pub rtp_rx: kanal::Receiver<RtpFrame>,
    /// Decoder to analysis.
    pub stream_tx: kanal::Sender<StreamFrame>,
    pub stream_rx: kanal::Receiver<StreamFrame>,
    /// Analysis to the keyframe VLM workers.
    pub vlm_tx: kanal::Sender<KeyframeWork>,
    pub vlm_rx: kanal::Receiver<KeyframeWork>,
    /// Event sink for emitted semantic events.
    pub event_sink: Arc<dyn EventSink>,
    /// Inference provider shared by the VLM workers.
    pub provider: Arc<I>,
    pub run_id: Arc<str>,
    pub session_id: Arc<str>,
    /// Live prompt handle; VLM workers reload it per keyframe.
    pub prompt: Arc<ArcSwap<Arc<str>>>,
    /// Optional guided-JSON schema handle.
    pub guided_json: Arc<ArcSwapOption<Arc<str>>>,
    pub vlm_config: TieredVlmConfig,
    pub novelty: LiveNoveltyConfig,
    /// When set, run the stream in clip mode instead of the keyframe path.
    pub clip_config: Option<ClipConfig>,
    /// Negotiated codec, used to size the decoder output pool.
    pub codec: VideoCodec,
    pub metrics: Arc<PipelineMetrics>,
    pub session_span: tracing::Span,
    /// Where tiered VLM inference outcomes are recorded for `/metrics`. `None`
    /// when the caller has no metrics sink wired up (e.g. tests).
    pub observer: Option<Arc<dyn InferenceObserver>>,
}

/// Spawn one session's full worker pipeline from a config and its wiring.
///
/// Decode workers always run. When `wiring.clip_config` is set the stream runs
/// in clip mode (analysis to clip accumulator to clip VLM); otherwise it runs
/// the keyframe path (analysis to VLM). Every worker thread is detached and
/// shuts down when its upstream channel closes.
pub fn spawn_pipeline<I>(cfg: &WorkerPoolConfig, wiring: PipelineWiring<I>) -> std::io::Result<()>
where
    I: InferenceProvider + 'static,
{
    let PipelineWiring {
        rtp_rx,
        stream_tx,
        stream_rx,
        vlm_tx,
        vlm_rx,
        event_sink,
        provider,
        run_id,
        session_id,
        prompt,
        guided_json,
        vlm_config,
        novelty,
        clip_config,
        codec,
        metrics,
        session_span,
        observer,
    } = wiring;

    let output_pool_slots = decode_output_pool_slots(cfg.gpu_available, codec);
    let jpeg_pool_slots = jpeg_pool_slots(cfg.analysis_workers, cfg.vlm_workers);

    if let Some(clip_config) = clip_config {
        // Clip mode: the decoder samples before encoding, the analysis worker
        // runs loop detection, and the accumulator builds clip windows.
        spawn_decode_workers(DecodeWorkerParams {
            workers: cfg.decode_workers,
            gpu_available: cfg.gpu_available,
            decode_width: cfg.decode_width,
            decode_height: cfg.decode_height,
            output_pool_slots,
            jpeg_pool_slots,
            rtp_rx,
            sink: DecodeSink::Stream {
                frame_tx: stream_tx,
                clip_config: clip_config.clone(),
            },
            crop: cfg.crop,
            metrics: Arc::clone(&metrics),
            session_span: session_span.clone(),
        })?;

        let (clip_frame_tx, clip_frame_rx) =
            kanal::bounded::<StreamFrame>(CLIP_FRAME_QUEUE_CAPACITY);
        let (clip_tx, clip_rx) = kanal::bounded::<ClipWork>(CLIP_WORK_QUEUE_CAPACITY);

        spawn_analysis_workers(AnalysisWorkerParams {
            workers: cfg.analysis_workers,
            gate_config: cfg.gate_config.clone(),
            loop_hamming_threshold: cfg.loop_hamming_threshold,
            loop_repeat_threshold: cfg.loop_repeat_threshold,
            frame_rx: stream_rx,
            clip_tx: clip_frame_tx,
            stdb: Arc::clone(&event_sink),
            run_id: Arc::clone(&run_id),
            session_id: Arc::clone(&session_id),
            prompt: Arc::clone(&prompt),
            metrics: Arc::clone(&metrics),
            session_span: session_span.clone(),
        })?;
        spawn_clip_accumulator(
            clip_frame_rx,
            clip_tx,
            clip_config,
            Arc::clone(&run_id),
            Arc::clone(&session_id),
            // load_full yields Arc<Arc<str>>; deref once so the accumulator gets the inner Arc<str>.
            Arc::clone(&*prompt.load_full()),
            session_span.clone(),
        )?;
        spawn_clip_vlm_workers(
            cfg.vlm_workers,
            clip_rx,
            provider,
            event_sink,
            vlm_config,
            metrics,
            session_span,
            cfg.max_output_tokens_per_second,
            guided_json,
            observer,
        )?;
    } else {
        // Keyframe mode: the gate runs inline in the decoder, so a JPEG is
        // encoded only for the frames it keeps and handed straight to the VLM
        // workers. There is no separate analysis stage, and the `stream_tx` /
        // `stream_rx` channel the caller allocated goes unused here.
        spawn_decode_workers(DecodeWorkerParams {
            workers: cfg.decode_workers,
            gpu_available: cfg.gpu_available,
            decode_width: cfg.decode_width,
            decode_height: cfg.decode_height,
            output_pool_slots,
            jpeg_pool_slots,
            rtp_rx,
            sink: DecodeSink::Keyframe(KeyframeSink {
                gate_config: cfg.gate_config.clone(),
                loop_hamming_threshold: cfg.loop_hamming_threshold,
                loop_repeat_threshold: cfg.loop_repeat_threshold,
                vlm_tx,
                stdb: Arc::clone(&event_sink),
                run_id,
                session_id,
                prompt,
            }),
            crop: cfg.crop,
            metrics: Arc::clone(&metrics),
            session_span: session_span.clone(),
        })?;
        let _ = (stream_tx, stream_rx);
        spawn_vlm_workers(VlmWorkerParams {
            workers: cfg.vlm_workers,
            vlm_rx,
            provider,
            stdb: event_sink,
            config: vlm_config,
            metrics,
            session_span,
            max_output_tokens_per_second: cfg.max_output_tokens_per_second,
            guided_json,
            novelty,
            observer,
        })?;
    }

    Ok(())
}

pub struct VlmWorkerParams<I>
where
    I: InferenceProvider + 'static,
{
    pub workers: usize,
    pub vlm_rx: kanal::Receiver<KeyframeWork>,
    pub provider: Arc<I>,
    pub stdb: Arc<dyn EventSink>,
    pub config: TieredVlmConfig,
    pub metrics: Arc<PipelineMetrics>,
    pub session_span: tracing::Span,
    pub max_output_tokens_per_second: u32,
    pub guided_json: Arc<ArcSwapOption<Arc<str>>>,
    pub novelty: LiveNoveltyConfig,
    /// Where tiered VLM inference outcomes are recorded for `/metrics`. `None`
    /// when the caller has no metrics sink wired up (e.g. tests).
    pub observer: Option<Arc<dyn InferenceObserver>>,
}

/// Remove token-budget windows outside the active one-second interval.
pub(super) fn prune_stale_token_budget_entries(
    budget: &mut std::collections::HashMap<Arc<str>, (std::time::Instant, u32)>,
    now: std::time::Instant,
) {
    budget.retain(|_, (ts, _)| now.duration_since(*ts).as_secs() < 1);
}

/// Returns the token-budget window for `session`, inserting it if absent.
pub(super) fn token_budget_entry<'a>(
    budget: &'a mut std::collections::HashMap<Arc<str>, (std::time::Instant, u32)>,
    session: &Arc<str>,
    now: std::time::Instant,
) -> &'a mut (std::time::Instant, u32) {
    // entry() reuses the existing window or inserts a fresh one; either way it
    // hands back a live reference without a second lookup or an unwrap.
    budget.entry(Arc::clone(session)).or_insert((now, 0))
}

/// Spawn VLM inference workers with live novelty reuse and optional second-pass routing.
///
/// The semantic-novelty gate can skip redundant frames before inference. Frames
/// that reach inference use `config.first_pass_model`, with the distinct
/// `config.second_pass_model` only when confidence falls below the configured
/// threshold.
///
/// SpacetimeDB writes are fire-and-forget: VLM threads send [`SinkEvent`]s to
/// a bounded kanal channel; a dedicated writer thread drains it sequentially,
/// so inference latency is never stalled by SpacetimeDB HTTP round-trips.
///
/// Threads exit when `params.vlm_rx` is closed.  Keyframe analysis is stateful:
/// dedup, previous description, and previous timestamp must advance in stream
/// order.  The worker count is therefore clamped to one per stream.  Future VLM
/// parallelism must be stateless or explicitly batched, not racing workers with
/// independent temporal context.
pub fn spawn_vlm_workers<I>(params: VlmWorkerParams<I>) -> std::io::Result<()>
where
    I: InferenceProvider + 'static,
{
    let VlmWorkerParams {
        workers,
        vlm_rx,
        provider,
        stdb,
        config,
        metrics,
        session_span,
        max_output_tokens_per_second,
        guided_json,
        novelty,
        observer,
    } = params;

    // FIFO channel between VLM workers and the SpacetimeDB writer.
    let (event_tx, event_rx) = kanal::bounded::<SinkEvent>(SINK_EVENT_QUEUE_CAPACITY);

    // The writer thread needs its own Arc<PipelineMetrics> clone so it can
    // record HTTP POST latency without contending with VLM worker threads.
    let writer_metrics = Arc::clone(&metrics);

    // Dedicated writer thread: drains the FIFO and calls blocking sink methods.
    std::thread::Builder::new()
        .name("vx-event-writer".to_string())
        .spawn(move || {
            while let Ok(event) = event_rx.recv() {
                let emit_start = std::time::Instant::now();
                match event {
                    SinkEvent::Emit {
                        run_id,
                        session_id,
                        frame_index,
                        pts_ms,
                        coordinates,
                        event_type,
                        confidence,
                        description,
                    } => {
                        let _ = stdb.emit_event_sync(
                            &run_id,
                            &session_id,
                            frame_index,
                            pts_ms,
                            coordinates,
                            event_type,
                            confidence,
                            &description,
                        );
                    }
                    SinkEvent::StoreKeyframe {
                        run_id,
                        frame_index,
                        pts_ms,
                        coordinates,
                        event_type,
                        description,
                        jpeg_bytes,
                        ..
                    } => {
                        let _ = stdb.store_keyframe_sync(KeyframeEvent {
                            run_id: &run_id,
                            frame_index,
                            pts_ms,
                            coordinates,
                            event_type,
                            description: &description,
                            jpeg_data: &jpeg_bytes,
                        });
                    }
                }
                let emit_ms = emit_start.elapsed().as_millis() as u64;
                writer_metrics.stdb_emit_latency_ms.record(emit_ms);
            }
        })?;

    let jpeg_sink_backlog = JpegSinkBacklog::new();

    for i in 0..per_stream_vlm_workers(workers) {
        let vlm_rx = vlm_rx.clone();
        let provider = Arc::clone(&provider);
        let event_tx = event_tx.clone();
        let jpeg_sink_backlog = jpeg_sink_backlog.clone();
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();
        let guided_json = Arc::clone(&guided_json);
        let observer = observer.clone();
        let novelty = novelty.clone();

        std::thread::Builder::new()
            .name(format!("vx-vlm-{i}"))
            .spawn(move || {
                // Per-session token budget: (window_start, tokens_emitted_in_window).
                // Cloning Arc<str> only updates a refcount; Hash/Eq compare string content.
                let mut token_budget: std::collections::HashMap<
                    Arc<str>,
                    (std::time::Instant, u32),
                > = std::collections::HashMap::new();

                let novelty_enabled = novelty.embedding_sidecar_addr.is_some();
                let mut embedding_client = if novelty_enabled {
                    novelty.embedding_sidecar_addr.as_deref().and_then(|address| {
                        match EmbeddingSidecarClient::new(
                            address,
                            novelty.embedding_timeout_ms,
                        ) {
                            Ok(client) => Some(client),
                            Err(err) => {
                                tracing::warn!(%err, "invalid embedding sidecar address; admitting all frames");
                                None
                            }
                        }
                    })
                } else {
                    None
                };
                let mut novelty_gate = if novelty_enabled {
                    match LiveNoveltyGate::try_new(&novelty) {
                        Ok(gate) => Some(gate),
                        Err(err) => {
                            tracing::warn!(%err, "invalid live novelty config; admitting all frames");
                            None
                        }
                    }
                } else {
                    None
                };

                // Per-worker dedup filter: suppresses emitting identical VLM
                // descriptions so that a stuck loop doesn't pollute the store
                // with repeated identical events.
                let mut dedup = DedupFilter::new();

                // Temporal context: carry the previous keyframe's VLM description
                // so the next inference knows what came before without sending a
                // second image.
                let default_prompt: Arc<str> =
                    Arc::from("Briefly describe what is happening in this video frame.");
                let mut last_description: Arc<str> = Arc::from("");
                let mut last_pts_ms: u64 = 0;
                let mut prompt_buf = String::with_capacity(512);
                let mut jpeg_b64 = String::new();
                let mut input_images: Vec<InferenceImage> = Vec::with_capacity(1);

                while let Ok(work) = vlm_rx.recv() {
                    let _guard = session_span.enter();

                    // The analysis worker already fired `loop_detected`; skip
                    // the expensive VLM inference call entirely while the scene
                    // is stuck in a loop.
                    if work.loop_active {
                        metrics.inc_keyframes_dropped();
                        continue;
                    }

                    // Token rate limit: skip this frame if the session has
                    // already exceeded the per-second output token budget.
                    if max_output_tokens_per_second > 0 {
                        let now = std::time::Instant::now();
                        prune_stale_token_budget_entries(&mut token_budget, now);
                        let entry = token_budget_entry(&mut token_budget, &work.session_id, now);
                        if now.duration_since(entry.0).as_secs() >= 1 {
                            *entry = (now, 0); // reset 1-second window
                        }
                        if entry.1 >= max_output_tokens_per_second {
                            metrics.inc_keyframes_dropped();
                            continue; // backpressure: drop this inference
                        }
                    }

                    let embedding_started = std::time::Instant::now();
                    let embedding = embedding_client
                        .as_mut()
                        .and_then(|client| client.embed(&work.jpeg_bytes).ok());
                    if novelty_enabled && embedding_client.is_some() {
                        metrics
                            .novelty_embedding_latency_ms
                            .record(embedding_started.elapsed().as_millis() as u64);
                    }

                    let mut commit_novelty_anchor = false;
                    let mut shadow_novelty_probe = false;
                    if let Some(gate) = novelty_gate.as_mut() {
                        if let Some(embedding) = embedding.as_ref() {
                            metrics.inc_novelty_evaluated();
                            match gate.evaluate(embedding, work.pts_ms) {
                                LiveNoveltyOutcome::Reuse => {
                                    if sample_novelty_shadow(work.frame_index, novelty.shadow_sample_rate) {
                                        shadow_novelty_probe = true;
                                        metrics.inc_novelty_shadow_sampled();
                                    } else {
                                        metrics.inc_novelty_reused();
                                        metrics.inc_keyframes_dropped();
                                        continue;
                                    }
                                }
                                LiveNoveltyOutcome::ForcedRefresh => {
                                    metrics.inc_novelty_forced_refresh();
                                    commit_novelty_anchor = true;
                                }
                                LiveNoveltyOutcome::Run => commit_novelty_anchor = true,
                            }
                        } else {
                            metrics.inc_novelty_embedding_unavailable();
                        }
                    }

                    metrics.inc_vlm_inferences();

                    jpeg_b64.clear();
                    base64::engine::general_purpose::STANDARD
                        .encode_string(&work.jpeg_bytes, &mut jpeg_b64);

                    // State diffs are computed server-side below, so the
                    // default prompt only asks for a scene description.
                    let prompt: Arc<str> = if work.prompt.is_empty() {
                        Arc::clone(&default_prompt)
                    } else {
                        Arc::clone(&work.prompt)
                    };

                    let base_prompt: &str = &prompt;

                    prompt_buf.clear();
                    if last_description.is_empty() {
                        use std::fmt::Write as _;
                        let _ = write!(
                            prompt_buf,
                            "{base_prompt}\n[gate: trigger={}, confidence={:.2}, novelty={:.2}, motion={:.2}, pts_ms={}]",
                            work.event_type, work.confidence, work.novelty_score, work.motion_score, work.pts_ms
                        );
                    } else {
                        use std::fmt::Write as _;
                        let _ = write!(
                            prompt_buf,
                            "{base_prompt}\n[previous_state ({last_pts_ms}ms): {last_description}]\n[gate: trigger={}, confidence={:.2}, novelty={:.2}, motion={:.2}, pts_ms={}]",
                            work.event_type, work.confidence, work.novelty_score, work.motion_score, work.pts_ms
                        );
                    }

                    // First-pass VLM with optional second-pass escalation.
                    let vlm_start = std::time::Instant::now();
                    let inference_outcome = {
                        // Snapshot the current guided_json schema once per inference.
                        let current_guided_json: Option<Arc<str>> =
                            guided_json.load_full().map(|schema| Arc::clone(&*schema));
                        let prompt_arc: Arc<str> = Arc::from(prompt_buf.as_str());
                        let mut request_images = std::mem::take(&mut input_images);
                        if request_images.is_empty() {
                            request_images.push(InferenceImage {
                                media_type: "image/jpeg",
                                data_base64: String::new(),
                            });
                        } else {
                            request_images.truncate(1);
                            request_images[0].media_type = "image/jpeg";
                        }
                        std::mem::swap(&mut request_images[0].data_base64, &mut jpeg_b64);

                        let request = InferenceRequest {
                            model: Arc::clone(&config.first_pass_model),
                            prompt: Arc::clone(&prompt_arc),
                            input_images: request_images,
                            input_videos: Vec::new(),
                            max_tokens: 128,
                            temperature: 0.0,
                            timeout_ms: 5_000,
                            allow_fallback: true,
                            guided_json: current_guided_json,
                        };

                        let tiered_call_start = std::time::Instant::now();
                        match run_tiered(
                            provider.as_ref(),
                            &config,
                            request,
                            1024,
                            10_000,
                            observer.as_deref(),
                        ) {
                            Ok(output) => {
                                input_images = output.request.input_images;
                                if let Some(image) = input_images.get_mut(0) {
                                    std::mem::swap(&mut image.data_base64, &mut jpeg_b64);
                                }
                                Some((output.result.output_text, output.used_second_pass))
                            }
                            Err(err) => {
                                if let Some(o) = observer.as_deref() {
                                    // First-pass failure: err.request.model is
                                    // the failed model, so attribute the error
                                    // to its backend rather than the router's
                                    // default kind.
                                    o.record_error(
                                        provider.kind_for_model(err.request.model.as_ref()),
                                        tiered_call_start.elapsed().as_millis() as u64,
                                    );
                                }
                                input_images = err.request.input_images;
                                if let Some(image) = input_images.get_mut(0) {
                                    std::mem::swap(&mut image.data_base64, &mut jpeg_b64);
                                }
                                tracing::warn!(error = ?err.error, "vlm inference failed");
                                None
                            }
                        }
                    };
                    let vlm_elapsed_ms = vlm_start.elapsed().as_millis() as u64;
                    metrics.vlm_latency_ms.record(vlm_elapsed_ms);
                    let Some((description_str, used_second_pass)) = inference_outcome else {
                        continue;
                    };

                    // Failed or empty descriptions must not become reuse anchors.
                    if commit_novelty_anchor && !description_str.trim().is_empty() {
                        if let (Some(gate), Some(embedding)) =
                            (novelty_gate.as_mut(), embedding.as_ref())
                        {
                            gate.commit(embedding, work.pts_ms);
                        }
                    }

                    // Charge output tokens against the session budget.
                    // Approximate: 4 bytes per token (UTF-8 average).
                    if max_output_tokens_per_second > 0 {
                        let token_count = (description_str.len() / 4).max(1) as u32;
                        if let Some(entry) = token_budget.get_mut(work.session_id.as_ref()) {
                            entry.1 = entry.1.saturating_add(token_count);
                        }
                    }

                    let event_type: &'static str =
                        if used_second_pass { "vlm_tiered" } else { "vlm" };

                    // Wrap description in Arc<str> so both SinkEvent sends share
                    // the same allocation without cloning the String content.
                    let description: Arc<str> = Arc::from(description_str.into_boxed_str());

                    // Compare current and previous descriptions to detect a
                    // coarse state transition without sending a second image.
                    // Jaccard word-overlap compares the *set* of words, so
                    // paraphrases or reordering don't fool the check. Computed on
                    // the stack (no per-keyframe HashSet). `last_description` is
                    // still the previous frame's text here — it's updated below.
                    let state_changed = if last_description.is_empty() {
                        // First frame — always emit as the initial state.
                        true
                    } else {
                        jaccard_word_overlap(&last_description, &description)
                            < STATE_CHANGE_JACCARD_MAX
                    };

                    // Shadow probes do not update state or emit events.
                    if shadow_novelty_probe {
                        if description.trim().is_empty() {
                            continue;
                        }
                        metrics.inc_novelty_shadow_completed();
                        if state_changed {
                            metrics.inc_novelty_shadow_changed();
                        }
                        continue;
                    }

                    // Update temporal context for the next keyframe, keeping the
                    // outgoing description for the transition event below.
                    // `mem::replace` moves the old Arc out — no clone, no bump.
                    let prev_description = advance_temporal_context(
                        Some(&description),
                        work.pts_ms,
                        &mut last_description,
                        &mut last_pts_ms,
                    )
                    .expect("successful VLM descriptions advance temporal context");

                    // Always emit the VLM description event.
                    if dedup.should_emit(&description) {
                        let _ = event_tx.send(SinkEvent::Emit {
                            run_id: Arc::clone(&work.run_id),
                            session_id: Arc::clone(&work.session_id),
                            frame_index: work.frame_index,
                            pts_ms: work.pts_ms,
                            coordinates: work.coordinates,
                            event_type,
                            confidence: work.confidence,
                            description: Arc::clone(&description),
                        });
                    }

                    // Emit state_transition when the scene changed.
                    if state_changed && !prev_description.is_empty() {
                        let transition_desc = format!(
                            "{{\"from_state\":{},\"to_state\":{},\"trigger\":\"{}\",\"confidence\":{:.2}}}",
                            serde_json::Value::String(prev_description.to_string()),
                            serde_json::Value::String(description.to_string()),
                            work.event_type,
                            work.confidence,
                        );
                        let _ = event_tx.send(SinkEvent::Emit {
                            run_id: Arc::clone(&work.run_id),
                            session_id: Arc::clone(&work.session_id),
                            frame_index: work.frame_index,
                            pts_ms: work.pts_ms,
                            coordinates: work.coordinates,
                            event_type: "state_transition",
                            confidence: work.confidence,
                            description: Arc::from(transition_desc),
                        });
                    }
                    if let Some(event) = store_keyframe_event_with_backlog(
                        &jpeg_sink_backlog,
                        &metrics,
                        Arc::clone(&work.run_id),
                        work.frame_index,
                        work.pts_ms,
                        work.coordinates,
                        work.event_type,
                        description,
                        work.jpeg_bytes,
                    ) {
                        let _ = event_tx.send(event);
                    }
                }
                // event_tx clone is dropped here; once all worker clones drop,
                // the outer event_tx also drops, closing the writer channel.
            })?;
    }

    Ok(())
}

// ─── VLM worker helpers ────────────────────────────────────────────────────────

fn advance_temporal_context(
    description: Option<&Arc<str>>,
    pts_ms: u64,
    last_description: &mut Arc<str>,
    last_pts_ms: &mut u64,
) -> Option<Arc<str>> {
    let description = description?;
    let previous = std::mem::replace(last_description, truncate_arc_str(description, 200));
    *last_pts_ms = pts_ms;
    Some(previous)
}

fn truncate_arc_str(text: &Arc<str>, max_bytes: usize) -> Arc<str> {
    if text.len() <= max_bytes {
        return Arc::clone(text);
    }

    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    Arc::from(&text[..end])
}

fn hash_word(word: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in word.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Below this word-overlap, two consecutive descriptions count as a state
/// change. 0.5 = "at least half the words turned over": tolerant of the VLM
/// rephrasing a stable scene (which shares most nouns) while still firing when
/// the subject actually changes. Coarse on purpose — it only gates whether we
/// emit a transition event, never whether we call the VLM.
const STATE_CHANGE_JACCARD_MAX: f32 = 0.5;

/// Word budget for the on-stack Jaccard. The VLM caps generation at 128 tokens
/// and one word is ≥ one token, so a real description never overflows this;
/// pathological input is clamped to the first `JACCARD_MAX_WORDS` words, which
/// only ever makes the overlap *look higher* (fewer distinct words), i.e. errs
/// toward "no change" — the safe direction for a coarse transition gate.
const JACCARD_MAX_WORDS: usize = 128;

/// Fill `buf` with the sorted, deduplicated hashes of the words in `text`,
/// returning the number of unique words. Models a `HashSet<u64>` of word hashes
/// entirely on the stack — no allocation.
fn word_hash_set(text: &str, buf: &mut [u64; JACCARD_MAX_WORDS]) -> usize {
    let mut n = 0;
    for word in text.split_whitespace() {
        if n == JACCARD_MAX_WORDS {
            break;
        }
        buf[n] = hash_word(word);
        n += 1;
    }
    buf[..n].sort_unstable();
    // Collapse runs of equal hashes so the count is a set cardinality.
    let mut unique = 0;
    for r in 0..n {
        if unique == 0 || buf[r] != buf[unique - 1] {
            buf[unique] = buf[r];
            unique += 1;
        }
    }
    unique
}

/// Jaccard similarity of the word *sets* of two descriptions, on the stack.
///
/// Set-identical to the previous `HashSet::intersection`/`union` version, but
/// with zero per-keyframe heap traffic: two fixed buffers, sorted, then a
/// single merge-walk. Returns 1.0 for two empty inputs (they are identical).
fn jaccard_word_overlap(prev: &str, curr: &str) -> f32 {
    let mut prev_buf = [0u64; JACCARD_MAX_WORDS];
    let mut curr_buf = [0u64; JACCARD_MAX_WORDS];
    let a = word_hash_set(prev, &mut prev_buf);
    let b = word_hash_set(curr, &mut curr_buf);

    let (mut i, mut j, mut intersection) = (0usize, 0usize, 0usize);
    while i < a && j < b {
        match prev_buf[i].cmp(&curr_buf[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a + b - intersection;
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn sample_novelty_shadow(frame_index: u64, rate: f32) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    const DENOMINATOR: u64 = 10_000;
    let mixed = frame_index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(17);
    mixed % DENOMINATOR < (rate * DENOMINATOR as f32) as u64
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        advance_temporal_context, build_clip_stream_frame_from_yuv, build_stream_frame_from_yuv,
        decode_output_pool_slots, hash_word, jaccard_word_overlap, jpeg_pool_slots,
        prune_stale_token_budget_entries, token_budget_entry, DecoderBackend, EventSink,
        GateStreamState, KeyframeContext, KeyframeEvent, KeyframeWork, StreamFrame,
        CLIP_FRAME_QUEUE_CAPACITY, CLIP_WORK_QUEUE_CAPACITY, FFMPEG_YUV_READER_QUEUE_CAPACITY,
        JPEG_POOL_SLOT_CEILING, JPEG_SINK_EVENT_POOL_ALLOWANCE, SINK_EVENT_QUEUE_CAPACITY,
        STREAM_FRAME_QUEUE_CAPACITY, VLM_WORK_QUEUE_CAPACITY,
    };
    use crate::coordinates::FrameCoordinates;
    use crate::gate::FrameSignal;
    use crate::webrtc::clip::ClipRateGate;
    use crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST;
    use crate::webrtc::decode::{
        VideoCodec, YuvFrame, FFMPEG_YUV_PENDING_POOL_ALLOWANCE, FFMPEG_YUV_READER_POOL_MIN_SLOTS,
        SOFTWARE_YUV_POOL_MIN_SLOTS,
    };
    use crate::webrtc::recycle::VecPool;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    struct MockSink {
        events: Mutex<Vec<String>>,
        keyframes: Mutex<Vec<String>>,
    }

    impl MockSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                events: Mutex::new(Vec::new()),
                keyframes: Mutex::new(Vec::new()),
            })
        }
    }

    impl EventSink for MockSink {
        fn emit_event_sync(
            &self,
            _run_id: &str,
            _session_id: &str,
            _frame_index: u64,
            _pts_ms: u64,
            _coordinates: FrameCoordinates,
            event_type: &str,
            _confidence: f32,
            _description: &str,
        ) -> Result<(), String> {
            self.events.lock().unwrap().push(event_type.to_string());
            Ok(())
        }

        fn store_keyframe_sync(&self, event: KeyframeEvent<'_>) -> Result<(), String> {
            self.keyframes
                .lock()
                .unwrap()
                .push(event.event_type.to_string());
            Ok(())
        }
    }

    fn make_stream_frame(seq: u64) -> StreamFrame {
        StreamFrame {
            signal: FrameSignal {
                frame_index: seq,
                pts_ms: seq * 33,
                perceptual_hash: seq.wrapping_mul(0xDEAD_BEEF_CAFE_0001),
                luma_mean: 0.4,
                flicker_score: 0.0,
                ghosting_score: 0.0,
                noise_variance_score: 0.0,
            },
            jpeg: Some([0xff_u8, 0xd8, 0xff, 0xd9].into()), // minimal JPEG markers
            pts_ms: seq * 33,
            seq,
            coordinates: FrameCoordinates::full_frame(640, 480),
        }
    }

    fn tiny_yuv_frame(width: u32, height: u32, luma: u8) -> YuvFrame {
        YuvFrame {
            y: vec![luma; (width * height) as usize].into(),
            u: vec![128; (width / 2 * height / 2) as usize].into(),
            v: vec![128; (width / 2 * height / 2) as usize].into(),
            width,
            height,
        }
    }

    #[test]
    fn stream_frame_and_keyframe_work_are_clone_debug() {
        let sf = make_stream_frame(0);
        let _ = sf.clone();
        let _ = format!("{sf:?}");

        let kw = KeyframeWork {
            run_id: "r1".into(),
            session_id: "s1".into(),
            frame_index: 0,
            pts_ms: 0,
            coordinates: FrameCoordinates::full_frame(640, 480),
            event_type: "scene_cut",
            confidence: 0.9,
            novelty_score: 0.8,
            motion_score: 0.5,
            jpeg_bytes: [0xFF_u8, 0xD8, 0xFF, 0xD9].into(),
            prompt: Arc::from(""),
            loop_active: false,
        };
        let _ = kw.clone();
        let _ = format!("{kw:?}");
    }

    #[test]
    fn mock_sink_records_calls() {
        let sink = MockSink::new();
        let coordinates = FrameCoordinates::full_frame(640, 480);
        sink.emit_event_sync("r", "s", 0, 0, coordinates, "vlm", 0.9, "hello")
            .unwrap();
        sink.store_keyframe_sync(KeyframeEvent {
            run_id: "r",
            frame_index: 0,
            pts_ms: 0,
            coordinates,
            event_type: "scene_cut",
            description: "hello",
            jpeg_data: b"",
        })
        .unwrap();
        assert_eq!(sink.events.lock().unwrap().as_slice(), ["vlm"]);
        assert_eq!(sink.keyframes.lock().unwrap().as_slice(), ["scene_cut"]);
    }

    #[test]
    fn token_budget_prunes_sessions_outside_active_window() {
        let mut budget: HashMap<Arc<str>, (Instant, u32)> = HashMap::new();
        let now = Instant::now();
        let stale_session: Arc<str> = Arc::from("stale");
        let active_session: Arc<str> = Arc::from("active");

        budget.insert(stale_session.clone(), (now - Duration::from_secs(2), 3));
        budget.insert(
            active_session.clone(),
            (now - Duration::from_millis(500), 4),
        );

        prune_stale_token_budget_entries(&mut budget, now);

        assert!(!budget.contains_key(stale_session.as_ref()));
        assert_eq!(
            budget.get(active_session.as_ref()),
            Some(&(now - Duration::from_millis(500), 4))
        );
    }

    #[test]
    fn token_budget_window_still_enforces_per_second_cap() {
        let mut budget: HashMap<Arc<str>, (Instant, u32)> = HashMap::new();
        let now = Instant::now();
        let session: Arc<str> = Arc::from("session");
        let cap = 8;

        let entry = token_budget_entry(&mut budget, &session, now);
        entry.1 = entry.1.saturating_add(8);
        assert!(entry.1 >= cap);

        let next = now + Duration::from_millis(500);
        let entry = token_budget_entry(&mut budget, &session, next);
        assert!(entry.1 >= cap);

        let reset = now + Duration::from_secs(1);
        let entry = token_budget_entry(&mut budget, &session, reset);
        if reset.duration_since(entry.0).as_secs() >= 1 {
            *entry = (reset, 0);
        }
        assert_eq!(*entry, (reset, 0));
    }

    #[test]
    fn decode_output_pool_is_sized_to_ffmpeg_pipe_in_flight() {
        let expected = super::FFMPEG_READER_CONSTRUCTING_YUV_FRAMES
            + FFMPEG_YUV_READER_QUEUE_CAPACITY
            + FFMPEG_YUV_PENDING_POOL_ALLOWANCE
            + super::DECODE_CONSUMER_YUV_FRAMES;
        assert_eq!(expected, FFMPEG_YUV_READER_POOL_MIN_SLOTS);
        assert_eq!(
            expected,
            FFMPEG_YUV_READER_QUEUE_CAPACITY + FFMPEG_YUV_PENDING_POOL_ALLOWANCE + 2
        );
        assert_eq!(decode_output_pool_slots(true, VideoCodec::H264), expected);
    }

    #[test]
    fn decode_output_pool_is_small_for_synchronous_openh264() {
        assert_eq!(
            decode_output_pool_slots(false, VideoCodec::H264),
            SOFTWARE_YUV_POOL_MIN_SLOTS
        );
        const _: () = assert!(SOFTWARE_YUV_POOL_MIN_SLOTS < FFMPEG_YUV_READER_POOL_MIN_SLOTS);
    }

    #[test]
    #[cfg(not(feature = "vp8"))]
    fn decode_output_pool_is_empty_for_unsupported_vp8() {
        assert_eq!(
            DecoderBackend::select(false, VideoCodec::Vp8),
            DecoderBackend::Unsupported
        );
        assert_eq!(
            DecoderBackend::select(true, VideoCodec::Vp8),
            DecoderBackend::Unsupported
        );
        assert_eq!(decode_output_pool_slots(true, VideoCodec::Vp8), 0);
        assert_eq!(decode_output_pool_slots(false, VideoCodec::Vp8), 0);
    }

    #[test]
    #[cfg(feature = "vp8")]
    fn decode_output_pool_is_small_for_synchronous_vp8() {
        assert_eq!(
            DecoderBackend::select(false, VideoCodec::Vp8),
            DecoderBackend::Vp8
        );
        assert_eq!(
            DecoderBackend::select(true, VideoCodec::Vp8),
            DecoderBackend::Vp8
        );
        assert_eq!(
            decode_output_pool_slots(true, VideoCodec::Vp8),
            SOFTWARE_YUV_POOL_MIN_SLOTS
        );
        assert_eq!(
            decode_output_pool_slots(false, VideoCodec::Vp8),
            SOFTWARE_YUV_POOL_MIN_SLOTS
        );
    }

    #[test]
    fn decoded_stream_frames_use_current_rtp_label_once_per_frame() {
        let yuv = tiny_yuv_frame(64, 64, 128);
        let mut prev_signal = None;
        let mut ycbcr_scratch = Vec::with_capacity(64 * 64 * 3);
        let jpeg_pool = VecPool::with_slots(3);
        let labels = [(10, 330), (11, 363), (12, 396)];

        let emitted: Vec<_> = labels
            .iter()
            .map(|&(seq, pts_ms)| {
                build_stream_frame_from_yuv(
                    &yuv,
                    seq,
                    pts_ms,
                    &mut prev_signal,
                    &mut ycbcr_scratch,
                    &jpeg_pool,
                )
                .expect("a well-formed frame produces a stream frame")
            })
            .collect();

        let emitted_labels: Vec<_> = emitted.iter().map(|sf| (sf.seq, sf.pts_ms)).collect();
        let signal_labels: Vec<_> = emitted
            .iter()
            .map(|sf| (sf.signal.frame_index, sf.signal.pts_ms))
            .collect();
        assert_eq!(emitted_labels, labels);
        assert_eq!(signal_labels, labels);
        assert!(emitted_labels.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn jpeg_pool_covers_full_clip_path_and_bounded_sink_backlog_without_heap_growth() {
        let vlm_workers = 2;
        let decode_to_analysis = STREAM_FRAME_QUEUE_CAPACITY + 1 + 1;
        let normal_path = VLM_WORK_QUEUE_CAPACITY
            + super::per_stream_vlm_workers(vlm_workers)
            + JPEG_SINK_EVENT_POOL_ALLOWANCE
            + 1;
        let active_clip_workers = super::per_stream_vlm_workers(vlm_workers);
        assert_eq!(CLIP_WORK_QUEUE_CAPACITY, 0);
        let blocked_clip_sender = 1;
        let clip_path = CLIP_FRAME_QUEUE_CAPACITY
            + MAX_CLIP_FRAMES_PER_REQUEST
            + (CLIP_WORK_QUEUE_CAPACITY + active_clip_workers + blocked_clip_sender)
                * MAX_CLIP_FRAMES_PER_REQUEST;
        let expected = decode_to_analysis + normal_path + clip_path;

        assert_eq!(expected, 484);
        assert_eq!(
            super::jpeg_sink_event_backlog_capacity(),
            JPEG_SINK_EVENT_POOL_ALLOWANCE
        );
        const _: () = assert!(JPEG_SINK_EVENT_POOL_ALLOWANCE < SINK_EVENT_QUEUE_CAPACITY);
        assert_eq!(jpeg_pool_slots(1, vlm_workers), expected);
        assert!(expected <= JPEG_POOL_SLOT_CEILING);
    }

    #[test]
    fn sink_jpeg_backlog_drops_store_events_past_pool_allowance() {
        let backlog = super::JpegSinkBacklog::new();
        let metrics = crate::metrics::PipelineMetrics::new();
        let mut retained = Vec::with_capacity(JPEG_SINK_EVENT_POOL_ALLOWANCE);

        for frame_index in 0..JPEG_SINK_EVENT_POOL_ALLOWANCE {
            let event = super::store_keyframe_event_with_backlog(
                &backlog,
                &metrics,
                Arc::from("run"),
                frame_index as u64,
                0,
                FrameCoordinates::full_frame(640, 480),
                "scene_cut",
                Arc::from("description"),
                [0xff_u8, 0xd8, 0xff, 0xd9].into(),
            );
            assert!(event.is_some(), "permit {frame_index} should be admitted");
            retained.push(event.unwrap());
        }

        let dropped = super::store_keyframe_event_with_backlog(
            &backlog,
            &metrics,
            Arc::from("run"),
            JPEG_SINK_EVENT_POOL_ALLOWANCE as u64,
            0,
            FrameCoordinates::full_frame(640, 480),
            "scene_cut",
            Arc::from("description"),
            [0xff_u8, 0xd8, 0xff, 0xd9].into(),
        );

        assert!(dropped.is_none());
        assert_eq!(metrics.sink_keyframes_dropped_total(), 1);
        drop(retained.pop());
        assert!(super::store_keyframe_event_with_backlog(
            &backlog,
            &metrics,
            Arc::from("run"),
            999,
            0,
            FrameCoordinates::full_frame(640, 480),
            "scene_cut",
            Arc::from("description"),
            [0xff_u8, 0xd8, 0xff, 0xd9].into(),
        )
        .is_some());
    }

    #[test]
    fn sink_jpeg_backlog_bounds_permits_under_concurrent_acquire() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        // Oversubscribe hard: many threads each try far more acquisitions than the
        // allowance, and every granted permit is held (never dropped) until all
        // threads have joined. The wait-free fetch_add path must still hand out
        // exactly `JPEG_SINK_EVENT_POOL_ALLOWANCE` permits — no more (the cap), and
        // no fewer: total attempts far exceed the cap and no permit is released
        // before the join barrier, so the pool fills even under serialized
        // scheduling, not by luck of the interleaving.
        let backlog = StdArc::new(super::JpegSinkBacklog::new());
        let granted = StdArc::new(AtomicUsize::new(0));
        let threads = 8;
        let attempts_per_thread = JPEG_SINK_EVENT_POOL_ALLOWANCE;

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let backlog = StdArc::clone(&backlog);
                let granted = StdArc::clone(&granted);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    for _ in 0..attempts_per_thread {
                        if let Some(permit) = backlog.try_acquire() {
                            held.push(permit);
                        }
                    }
                    granted.fetch_add(held.len(), Ordering::Relaxed);
                    held // keep the permits alive past the join barrier
                })
            })
            .collect();

        let all_held: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            granted.load(Ordering::Relaxed),
            JPEG_SINK_EVENT_POOL_ALLOWANCE,
            "concurrent acquire must grant exactly the allowance while all permits are held"
        );

        // Once every permit drops, the pool is empty again and re-admits.
        drop(all_held);
        assert!(backlog.try_acquire().is_some());
    }

    #[test]
    fn per_stream_stateful_worker_counts_are_clamped_to_one() {
        assert_eq!(super::per_stream_decode_workers(0), 1);
        assert_eq!(super::per_stream_decode_workers(8), 1);
        assert_eq!(super::per_stream_vlm_workers(0), 1);
        assert_eq!(super::per_stream_vlm_workers(8), 1);
        assert_eq!(super::per_stream_analysis_workers(0), 1);
        assert_eq!(super::per_stream_analysis_workers(8), 1);
    }

    #[test]
    fn observe_loop_returns_event_without_emitting_inline() {
        let sink = MockSink::new();
        let metrics = crate::metrics::PipelineMetrics::new();
        let mut gate = GateStreamState::new(crate::gate::GateConfig::default(), 6, 3);

        let mut event = None;
        for i in 0..8u64 {
            let signal = FrameSignal {
                frame_index: i,
                pts_ms: i * 33,
                perceptual_hash: 0xAAAA_AAAA_AAAA_AAAA,
                luma_mean: 0.4,
                flicker_score: 0.0,
                ghosting_score: 0.0,
                noise_variance_score: 0.0,
            };
            event = event.or_else(|| gate.observe_loop(&signal, signal.pts_ms, &metrics));
        }

        let event = event.expect("loop event should be returned on loop entry");
        assert_eq!(event.frame_index, 3);
        assert_eq!(sink.events.lock().unwrap().len(), 0);
        assert!(metrics
            .render_prometheus()
            .contains("vidarax_pipeline_loop_detected_total 1"));
    }

    #[test]
    fn gate_stream_state_loads_latest_prompt_for_keyframes() {
        let sink = MockSink::new();
        let prompt = Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from(""))));
        let metrics = crate::metrics::PipelineMetrics::new();
        let stdb = sink as Arc<dyn EventSink>;
        let run_id: Arc<str> = "run-test".into();
        let session_id: Arc<str> = "sess-test".into();
        let mut gate = GateStreamState::new(crate::gate::GateConfig::default(), 6, 3);
        let ctx = KeyframeContext {
            run_id: &run_id,
            session_id: &session_id,
            prompt: &prompt,
            stdb: &stdb,
            metrics: &metrics,
        };

        let first = gate
            .on_frame(
                make_stream_frame(0).signal,
                0,
                FrameCoordinates::full_frame(640, 480),
                &ctx,
                || Some([0xff_u8, 0xd8, 0xff, 0xd9].into()),
            )
            .expect("initial keyframe work");
        assert_eq!(&*first.prompt, "");

        prompt.store(Arc::new(Arc::from("describe updated prompt")));

        let mut scene_cut = make_stream_frame(1);
        scene_cut.signal.perceptual_hash = !scene_cut.signal.perceptual_hash;
        let second = gate
            .on_frame(
                scene_cut.signal,
                scene_cut.pts_ms,
                scene_cut.coordinates,
                &ctx,
                || Some([0xff_u8, 0xd8, 0xff, 0xd9].into()),
            )
            .expect("scene-cut keyframe work");
        assert_eq!(&*second.prompt, "describe updated prompt");
    }

    #[test]
    fn failed_keyframe_encode_does_not_commit_gate_reference() {
        let sink = MockSink::new();
        let prompt = Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from(""))));
        let metrics = crate::metrics::PipelineMetrics::new();
        let stdb = sink as Arc<dyn EventSink>;
        let run_id: Arc<str> = "run-test".into();
        let session_id: Arc<str> = "sess-test".into();
        let mut gate = GateStreamState::new(crate::gate::GateConfig::default(), 6, 3);
        let ctx = KeyframeContext {
            run_id: &run_id,
            session_id: &session_id,
            prompt: &prompt,
            stdb: &stdb,
            metrics: &metrics,
        };

        assert!(gate
            .on_frame(
                make_stream_frame(0).signal,
                0,
                FrameCoordinates::full_frame(640, 480),
                &ctx,
                || None,
            )
            .is_none());

        let second = gate
            .on_frame(
                make_stream_frame(1).signal,
                33,
                FrameCoordinates::full_frame(640, 480),
                &ctx,
                || Some([0xff_u8, 0xd8, 0xff, 0xd9].into()),
            )
            .expect("uncommitted failed keep should not suppress the next keep");
        assert_eq!(second.frame_index, 1);
    }

    #[test]
    fn over_rate_clip_frames_are_not_encoded() {
        let yuv = tiny_yuv_frame(64, 64, 128);
        let mut prev_signal = None;
        let mut gate = ClipRateGate::new(1);
        let mut encode_calls = 0usize;

        let first = build_clip_stream_frame_from_yuv(
            &yuv,
            0,
            0,
            FrameCoordinates::full_frame(64, 64),
            &mut prev_signal,
            &mut gate,
            || {
                encode_calls += 1;
                Some([0xff_u8, 0xd8, 0xff, 0xd9].into())
            },
        );
        assert!(first.is_some());

        let second = build_clip_stream_frame_from_yuv(
            &yuv,
            1,
            500,
            FrameCoordinates::full_frame(64, 64),
            &mut prev_signal,
            &mut gate,
            || {
                encode_calls += 1;
                Some([0xff_u8, 0xd8, 0xff, 0xd9].into())
            },
        );

        let second = second.expect("over-rate frame should still forward its signal");
        assert!(second.jpeg.is_none());
        assert_eq!(encode_calls, 1);
    }

    #[test]
    fn failed_vlm_result_does_not_advance_temporal_context() {
        let mut last_description: Arc<str> = Arc::from("stable scene");
        let mut last_pts_ms = 120;

        assert!(
            advance_temporal_context(None, 200, &mut last_description, &mut last_pts_ms,).is_none()
        );
        assert_eq!(&*last_description, "stable scene");
        assert_eq!(last_pts_ms, 120);

        let description: Arc<str> = Arc::from("new scene");
        let previous = advance_temporal_context(
            Some(&description),
            240,
            &mut last_description,
            &mut last_pts_ms,
        )
        .expect("successful description should advance context");
        assert_eq!(&*previous, "stable scene");
        assert_eq!(&*last_description, "new scene");
        assert_eq!(last_pts_ms, 240);
    }

    /// Reference Jaccard over word-hash sets, exactly the `HashSet` version the
    /// stack implementation replaced. The two must agree bit-for-bit.
    fn jaccard_reference(prev: &str, curr: &str) -> f32 {
        use std::collections::HashSet;
        let a: HashSet<u64> = prev.split_whitespace().map(hash_word).collect();
        let b: HashSet<u64> = curr.split_whitespace().map(hash_word).collect();
        let inter = a.intersection(&b).count();
        let union = a.union(&b).count();
        if union == 0 {
            1.0
        } else {
            inter as f32 / union as f32
        }
    }

    #[test]
    fn stack_jaccard_matches_hashset_reference() {
        let cases = [
            ("", ""),
            ("a", ""),
            ("the cat sat on the mat", "the cat sat on the mat"),
            ("the cat sat on the mat", "a dog ran across the yard"),
            (
                "user opens the settings panel",
                "user opens the settings menu",
            ),
            // Repeated words must fold to a set, not inflate the counts.
            ("code code code review", "code review review review"),
            ("reordered words here now", "now here words reordered"),
        ];
        for (prev, curr) in cases {
            let got = jaccard_word_overlap(prev, curr);
            let want = jaccard_reference(prev, curr);
            assert!(
                (got - want).abs() < 1e-6,
                "jaccard({prev:?}, {curr:?}) = {got}, reference = {want}",
            );
        }
    }

    #[test]
    fn stack_jaccard_is_symmetric_and_bounded() {
        let got = jaccard_word_overlap("alpha beta gamma", "beta gamma delta");
        assert!((0.0..=1.0).contains(&got));
        assert!((got - jaccard_word_overlap("beta gamma delta", "alpha beta gamma")).abs() < 1e-6);
        // 2 shared of 4 distinct → exactly 0.5.
        assert!((got - 0.5).abs() < 1e-6, "expected 0.5, got {got}");
    }
}

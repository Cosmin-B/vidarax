//! WebRTC decode, analysis, and VLM worker pools.
//!
//! The pipeline is ordered through bounded `kanal` queues. Closing an upstream
//! sender propagates shutdown to downstream worker threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arc_swap::{ArcSwap, ArcSwapOption};
use base64::Engine as _;

use crate::dedup::DedupFilter;
use crate::gate::{GateConfig, GateEventType};
use crate::loop_detector::LoopDetector;
use crate::metrics::PipelineMetrics;
use crate::pipeline::{TwoPassConfig, TwoPassPipeline};
use crate::provider::{InferenceImage, InferenceProvider, InferenceRequest};
use crate::tiered_vlm::{run_tiered, DistillationConfig, TieredVlmConfig};
#[cfg(feature = "training")]
use crate::training_data::TrainingStore;
#[cfg(not(feature = "training"))]
#[allow(dead_code)]
pub struct TrainingStore;
use crate::webrtc::decode::{
    Decoder, DecoderBackend, DecoderConfig, VideoCodec, YuvFrame,
    FFMPEG_YUV_PENDING_POOL_ALLOWANCE, FFMPEG_YUV_READER_QUEUE_CAPACITY,
};
use crate::webrtc::recycle::{RecycledBytes, VecPool};
use crate::webrtc::session::RtpFrame;
use crate::webrtc::signals::{yuv_to_frame_signal, yuv_to_jpeg};

const DEFAULT_DECODE_WIDTH: u32 = 1920;
const DEFAULT_DECODE_HEIGHT: u32 = 1080;
const DEFAULT_LOOP_WINDOW: u32 = 6;
const DEFAULT_LOOP_REPEAT_THRESHOLD: usize = 3;
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
        DecoderBackend::Software => backend.min_yuv_pool_slots(),
    }
}

pub fn jpeg_pool_slots(analysis_workers: usize, vlm_workers: usize) -> usize {
    let analysis_workers = per_stream_analysis_workers(analysis_workers);
    let vlm_workers = per_stream_vlm_workers(vlm_workers);
    let decode_to_analysis = STREAM_FRAME_QUEUE_CAPACITY + analysis_workers + 1;
    let normal_path =
        VLM_WORK_QUEUE_CAPACITY + vlm_workers + JPEG_SINK_EVENT_POOL_ALLOWANCE + 1;
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
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String>;

    /// Persist a keyframe with its JPEG thumbnail (blocking).
    fn store_keyframe_sync(
        &self,
        run_id: &str,
        frame_index: u64,
        pts_ms: u64,
        event_type: &str,
        description: &str,
        jpeg_data: &[u8],
    ) -> Result<(), String>;
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
}

/// Work item forwarded to VLM workers when a keyframe is decided upon.
#[derive(Debug, Clone)]
pub struct KeyframeWork {
    /// Session run identifier — shared via `Arc<str>` so cloning is pointer-width.
    pub run_id: Arc<str>,
    /// Session identifier — shared via `Arc<str>` so cloning is pointer-width.
    pub session_id: Arc<str>,
    pub frame_index: u64,
    pub pts_ms: u64,
    /// Gate reason code: `"scene_cut"` | `"periodic_keepalive"` | `"initial_frame"`.
    pub event_type: &'static str,
    /// Gate confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Gate-derived novelty score (0=familiar, 1=novel).
    pub novelty_score: f32,
    /// Gate-derived motion score (0=static, 1=high motion).
    pub motion_score: f32,
    /// Raw JPEG bytes — base64-encoded on-demand for VLM, stored raw in SpacetimeDB.
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

fn build_stream_frame_from_yuv(
    yuv: &YuvFrame,
    seq: u64,
    pts_ms: u64,
    prev_signal: &mut Option<crate::gate::FrameSignal>,
    ycbcr_scratch: &mut Vec<u8>,
    jpeg_pool: &VecPool,
) -> StreamFrame {
    let signal = yuv_to_frame_signal(yuv, seq, pts_ms, prev_signal.as_ref());
    let jpeg = yuv_to_jpeg(yuv, 75, ycbcr_scratch, jpeg_pool);
    *prev_signal = Some(signal);

    StreamFrame {
        signal,
        jpeg: Some(jpeg),
        pts_ms,
        seq,
    }
}

// ─── Decode workers ───────────────────────────────────────────────────────────

/// Spawn decode workers for one ordered media stream.
///
/// One stream uses one stateful decoder. Codec detection is lazy on the first
/// frame, and the decoder is rebuilt if a later session uses another codec on
/// the same worker.
/// `cores` is API-compatible only; one ordered stream gets one stateful decoder, so parallelism is across sessions, not within it.
// Spawns decode workers; each channel, pool size, metrics handle, and span is configured separately.
#[allow(clippy::too_many_arguments)]
pub fn spawn_decode_workers(
    cores: usize,
    rtp_rx: kanal::Receiver<RtpFrame>,
    frame_tx: kanal::Sender<StreamFrame>,
    gpu: bool,
    output_pool_slots: usize,
    jpeg_pool_slots: usize,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
) {
    for i in 0..per_stream_decode_workers(cores) {
        let rtp_rx = rtp_rx.clone();
        let frame_tx = frame_tx.clone();
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();

        std::thread::Builder::new()
            .name(format!("vx-decode-{i}"))
            .spawn(move || {
                use crate::webrtc::decode::VideoCodec;

                // Lazy decoder: created on the first frame so we know the codec.
                let mut decoder: Option<Decoder> = None;
                let mut active_codec: Option<VideoCodec> = None;
                let mut prev_signal: Option<crate::gate::FrameSignal> = None;
                // Reused across frames to avoid per-frame 6MB YCbCr allocation.
                let mut ycbcr_scratch: Vec<u8> = Vec::with_capacity(
                    DEFAULT_DECODE_WIDTH as usize * DEFAULT_DECODE_HEIGHT as usize * 3,
                );
                let jpeg_pool = VecPool::with_slots(jpeg_pool_slots);

                'rtp_frames: while let Ok(frame) = rtp_rx.recv() {
                    let _guard = session_span.enter();
                    metrics.inc_rtp_received();

                    // Re-initialise the decoder when the codec changes (e.g. a
                    // new session with a different codec arrives on the same worker).
                    if active_codec != Some(frame.codec) {
                        let config = DecoderConfig {
                            gpu_available: gpu,
                            codec: frame.codec,
                            width: DEFAULT_DECODE_WIDTH,
                            height: DEFAULT_DECODE_HEIGHT,
                            output_pool_slots,
                        };
                        decoder = Some(Decoder::new_with_metrics(&config, Arc::clone(&metrics)));
                        active_codec = Some(frame.codec);
                    }

                    let dec = decoder.as_mut().expect("decoder initialised above");

                    let decode_start = std::time::Instant::now();
                    let yuv = match dec.decode(&frame.nals) {
                        Ok(yuv) => yuv,
                        Err(_) => continue 'rtp_frames,
                    };

                    let sf = build_stream_frame_from_yuv(
                        &yuv,
                        frame.seq,
                        frame.pts_ms,
                        &mut prev_signal,
                        &mut ycbcr_scratch,
                        &jpeg_pool,
                    );

                    let decode_us = decode_start.elapsed().as_micros() as u64;
                    metrics.decode_latency_us.record(decode_us);

                    metrics.inc_frames_decoded();
                    if frame_tx.send(sf).is_err() {
                        return; // downstream dropped — shut down
                    }
                }
            })
            .expect("decode thread spawn failed");
    }
}

// ─── Analysis workers ─────────────────────────────────────────────────────────

/// Spawn analysis workers.
///
/// Normal mode runs the gate engine and forwards kept keyframes to the VLM
/// queue with `try_send`. Clip mode bypasses the gate and forwards accepted
/// frames to the clip accumulator with `try_send`, dropping if its queue is full.
// Spawns analysis workers; the worker owns separate queue, sink, prompt, metrics, and span handles.
#[allow(clippy::too_many_arguments)]
pub fn spawn_analysis_workers(
    cores: usize,
    frame_rx: kanal::Receiver<StreamFrame>,
    vlm_tx: kanal::Sender<KeyframeWork>,
    clip_tx: Option<kanal::Sender<StreamFrame>>,
    stdb: Arc<dyn EventSink>,
    run_id: Arc<str>,
    session_id: Arc<str>,
    prompt: Arc<ArcSwap<Arc<str>>>,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
) {
    for i in 0..per_stream_analysis_workers(cores) {
        let frame_rx = frame_rx.clone();
        let vlm_tx = vlm_tx.clone();
        let clip_tx = clip_tx.clone();
        let stdb = Arc::clone(&stdb);
        let run_id = Arc::clone(&run_id);
        let session_id = Arc::clone(&session_id);
        let prompt = Arc::clone(&prompt);
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();

        std::thread::Builder::new()
            .name(format!("vx-analysis-{i}"))
            .spawn(move || {
                let mut pipeline =
                    TwoPassPipeline::new(TwoPassConfig::default(), GateConfig::default());
                let mut loop_det =
                    LoopDetector::new(DEFAULT_LOOP_WINDOW, DEFAULT_LOOP_REPEAT_THRESHOLD);
                // True while the loop detector considers the stream stuck.
                // Cleared when the detector stops firing for a full window.
                let mut loop_active = false;

                while let Ok(mut sf) = frame_rx.recv() {
                    let _guard = session_span.enter();

                    // ── Loop detection (always active) ───────────────────
                    let loop_fired = loop_det.check(sf.signal.perceptual_hash);
                    if loop_fired {
                        // Only emit the event the first time we enter a loop
                        // to avoid flooding the sink with repeated notices.
                        if !loop_active {
                            metrics.inc_loop_detected();
                            let _ = stdb.emit_event_sync(
                                &run_id,
                                &session_id,
                                sf.signal.frame_index,
                                sf.pts_ms,
                                "loop_detected",
                                0.9,
                                "loop detected via perceptual-hash ring buffer",
                            );
                        }
                        loop_active = true;
                    } else {
                        // LoopDetector returns false when the window no longer
                        // has enough repeated hashes — the scene has changed.
                        loop_active = false;
                    }

                    if let Some(ref clip_tx) = clip_tx {
                        let _ = clip_tx.try_send(sf);
                    } else {
                        let gate_start = std::time::Instant::now();
                        let metas = pipeline.analyze_batch(&[sf.signal]);
                        let gate_us = gate_start.elapsed().as_micros() as u64;
                        metrics.gate_latency_us.record(gate_us);
                        let meta = match metas.first() {
                            Some(m) => *m,
                            None => continue,
                        };

                        if meta.gate_event == GateEventType::KeepKeyframe {
                            let jpeg_bytes = sf.jpeg.take().unwrap_or_default();

                            let event_type: &'static str = if meta.scene_cut {
                                "scene_cut"
                            } else {
                                "periodic_keepalive"
                            };

                            let work = KeyframeWork {
                                run_id: Arc::clone(&run_id),
                                session_id: Arc::clone(&session_id),
                                frame_index: sf.signal.frame_index,
                                pts_ms: sf.pts_ms,
                                event_type,
                                confidence: meta.confidence,
                                novelty_score: meta.novelty_score,
                                motion_score: meta.motion_score,
                                jpeg_bytes,
                                prompt: Arc::clone(&*prompt.load_full()),
                                loop_active,
                            };

                            // Non-blocking: drop if VLM queue is full to avoid
                            // stalling the decode → analysis pipeline.
                            if vlm_tx.try_send(work).is_ok() {
                                metrics.inc_keyframes();
                            } else {
                                metrics.inc_keyframes_dropped();
                            }
                        }
                    }
                }
            })
            .expect("analysis thread spawn failed");
    }
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
        let mut current = self.in_flight.load(Ordering::Relaxed);
        loop {
            if current >= JPEG_SINK_EVENT_POOL_ALLOWANCE {
                return None;
            }
            match self.in_flight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(SinkJpegPermit {
                        in_flight: Arc::clone(&self.in_flight),
                    });
                }
                Err(next) => current = next,
            }
        }
    }
}

struct SinkJpegPermit {
    in_flight: Arc<AtomicUsize>,
}

impl Drop for SinkJpegPermit {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::Release);
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
        event_type: &'static str,
        confidence: f32,
        description: Arc<str>,
    },
    StoreKeyframe {
        run_id: Arc<str>,
        frame_index: u64,
        pts_ms: u64,
        event_type: &'static str,
        description: Arc<str>,
        jpeg_bytes: RecycledBytes,
        _jpeg_permit: SinkJpegPermit,
    },
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
    pub training_store: Option<Arc<Mutex<TrainingStore>>>,
    pub distillation: DistillationConfig,
}

/// Upper bound on sessions tracked by the per-session token budget map.
pub(super) const VLM_TOKEN_BUDGET_MAX_SESSIONS: usize = 4096;

/// Returns the token-budget window for `session`, inserting it if absent and
/// keeping the map within `VLM_TOKEN_BUDGET_MAX_SESSIONS` (stale windows are
/// dropped first, then an arbitrary entry) so a long-lived worker cannot grow
/// the map unbounded.
pub(super) fn token_budget_entry<'a>(
    budget: &'a mut std::collections::HashMap<Arc<str>, (std::time::Instant, u32)>,
    session: &Arc<str>,
    now: std::time::Instant,
) -> &'a mut (std::time::Instant, u32) {
    if !budget.contains_key(session.as_ref()) {
        if budget.len() >= VLM_TOKEN_BUDGET_MAX_SESSIONS {
            budget.retain(|_, (ts, _)| now.duration_since(*ts).as_secs() < 1);
            if budget.len() >= VLM_TOKEN_BUDGET_MAX_SESSIONS {
                if let Some(key) = budget.keys().next().cloned() {
                    budget.remove(&key);
                }
            }
        }
        budget.insert(Arc::clone(session), (now, 0));
    }
    budget.get_mut(session.as_ref()).expect("entry inserted above")
}

/// Spawn VLM inference worker threads with 3-tier routing + training pair collection.
///
/// **Tier 1 — KNN cache** (when `distillation.enabled` and embedding server is reachable):
/// Fetches a SigLIP2 embedding for the frame, then asks the `TrainingStore` for the
/// nearest-neighbour label.  If a confident match is found, the KNN result is used
/// directly and the VLM call is skipped.
///
/// **Tier 2 — specialist / fast VLM** (`config.first_pass_model`):
/// Called when KNN misses or is disabled.  Quick, low-cost inference.
///
/// **Tier 3 — teacher / accurate VLM** (`config.second_pass_model`):
/// Called when the specialist confidence is below `config.second_pass_threshold`.
///
/// **Training pair collection**: After any inference, if `distillation.enabled`,
/// the frame embedding and label are stored in the `TrainingStore` according to
/// `collection_rate` (deterministic per-frame sampling).  Oldest pairs are
/// evicted automatically when `max_pairs_per_tenant` is exceeded.
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
pub fn spawn_vlm_workers<I>(params: VlmWorkerParams<I>)
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
        training_store,
        distillation,
    } = params;
    #[cfg(not(feature = "training"))]
    let _training_store = training_store;

    // FIFO channel between VLM workers and the SpacetimeDB writer.
    let (event_tx, event_rx) = kanal::bounded::<SinkEvent>(SINK_EVENT_QUEUE_CAPACITY);

    // The writer thread needs its own Arc<PipelineMetrics> clone so it can
    // record HTTP POST latency without contending with VLM worker threads.
    let writer_metrics = Arc::clone(&metrics);

    // Dedicated writer thread: drains the FIFO and calls blocking sink methods.
    std::thread::Builder::new()
        .name("vx-stdb-writer".to_string())
        .spawn(move || {
            while let Ok(event) = event_rx.recv() {
                let emit_start = std::time::Instant::now();
                match event {
                    SinkEvent::Emit {
                        run_id, session_id, frame_index, pts_ms,
                        event_type, confidence, description,
                    } => {
                        let _ = stdb.emit_event_sync(
                            &run_id, &session_id, frame_index, pts_ms,
                            event_type, confidence, &description,
                        );
                    }
                    SinkEvent::StoreKeyframe {
                        run_id, frame_index, pts_ms, event_type, description, jpeg_bytes, ..
                    } => {
                        let _ = stdb.store_keyframe_sync(
                            &run_id, frame_index, pts_ms, event_type, &description, &jpeg_bytes,
                        );
                    }
                }
                let emit_ms = emit_start.elapsed().as_millis() as u64;
                writer_metrics.stdb_emit_latency_ms.record(emit_ms);
            }
        })
        .expect("stdb writer thread spawn failed");

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
        #[cfg(feature = "training")]
        let training_store = training_store.clone();
        #[cfg(feature = "training")]
        let distillation = distillation.clone();
        #[cfg(not(feature = "training"))]
        let _distillation = distillation.clone();

        std::thread::Builder::new()
            .name(format!("vx-vlm-{i}"))
            .spawn(move || {
                // Per-session token budget: (window_start, tokens_emitted_in_window).
                // Key is Arc<str>: clone is pointer-width; Hash/Eq compare string content.
                let mut token_budget: std::collections::HashMap<
                    Arc<str>,
                    (std::time::Instant, u32),
                > = std::collections::HashMap::new();

                // One HTTP client per thread for embedding calls (training feature only).
                #[cfg(feature = "training")]
                let embed_url = distillation.embedding_server_url.clone();
                #[cfg(feature = "training")]
                let http_client: Option<reqwest::blocking::Client> =
                    if embed_url.is_some() {
                        reqwest::blocking::Client::builder()
                            .timeout(std::time::Duration::from_secs(2))
                            .build()
                            .ok()
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
                let mut prev_word_hashes = std::collections::HashSet::new();
                let mut curr_word_hashes = std::collections::HashSet::new();

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
                        let entry = token_budget_entry(&mut token_budget, &work.session_id, now);
                        if now.duration_since(entry.0).as_secs() >= 1 {
                            *entry = (now, 0); // reset 1-second window
                        }
                        if entry.1 >= max_output_tokens_per_second {
                            metrics.inc_keyframes_dropped();
                            continue; // backpressure: drop this inference
                        }
                    }

                    metrics.inc_vlm_inferences();

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

                    // Base64-encode the JPEG once for both embedding and VLM calls.
                    jpeg_b64.clear();
                    base64::engine::general_purpose::STANDARD
                        .encode_string(&work.jpeg_bytes, &mut jpeg_b64);

                    // ── Step 1 (training only): fetch embedding for KNN + pair collection ──
                    #[cfg(feature = "training")]
                    let embedding = fetch_frame_embedding(
                        http_client.as_ref(),
                        embed_url.as_deref(),
                        &jpeg_b64,
                    );

                    // ── Tier 1 (training only): KNN classification ──────────────────
                    #[cfg(feature = "training")]
                    let knn_hit: Option<(String, bool)> = if distillation.enabled {
                        embedding.as_ref().and_then(|emb| {
                            let result = training_store.as_ref()?.lock().ok().and_then(|store| {
                                store
                                    .knn_classify(
                                        &work.run_id,
                                        emb,
                                        distillation.knn_k,
                                        distillation.distance_threshold,
                                    )
                                    .unwrap_or(None)
                            })?;
                            tracing::info!(
                                run_id = %work.run_id,
                                label = %result.label,
                                avg_distance = result.avg_distance,
                                votes = result.votes,
                                total = result.total,
                                "tier1_knn_hit: skipping vlm inference"
                            );
                            Some((result.label, false))
                        })
                    } else {
                        None
                    };
                    #[cfg(not(feature = "training"))]
                    let knn_hit: Option<(String, bool)> = None;

                    // ── Tiers 2+3: VLM inference (when KNN misses or disabled) ──────
                    let vlm_start = std::time::Instant::now();
                    let (description_str, used_second_pass) = if let Some(hit) = knn_hit {
                        hit
                    } else {
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

                        match run_tiered(provider.as_ref(), &config, request, 1024, 10_000) {
                            Ok(output) => {
                                input_images = output.request.input_images;
                                if let Some(image) = input_images.get_mut(0) {
                                    std::mem::swap(&mut image.data_base64, &mut jpeg_b64);
                                }
                                (output.result.output_text, output.used_second_pass)
                            }
                            Err(err) => {
                                input_images = err.request.input_images;
                                if let Some(image) = input_images.get_mut(0) {
                                    std::mem::swap(&mut image.data_base64, &mut jpeg_b64);
                                }
                                (format!("vlm_error: {:?}", err.error), false)
                            }
                        }
                    };
                    let vlm_elapsed_ms = vlm_start.elapsed().as_millis() as u64;
                    metrics.vlm_latency_ms.record(vlm_elapsed_ms);

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

                    // Compare current and previous descriptions to detect
                    // coarse state transitions without sending a second image.
                    let prev_desc = Arc::clone(&last_description);
                    let state_changed = if prev_desc.is_empty() {
                        // First frame — always emit as initial state.
                        true
                    } else {
                        // Jaccard word-overlap: compare the *set* of words so
                        // that paraphrases or reordering don't fool the check,
                        // unlike the previous positional char comparison.
                        prev_word_hashes.clear();
                        curr_word_hashes.clear();
                        prev_word_hashes.extend(prev_desc.split_whitespace().map(hash_word));
                        curr_word_hashes.extend(description.split_whitespace().map(hash_word));
                        let intersection = prev_word_hashes.intersection(&curr_word_hashes).count();
                        let union = prev_word_hashes.union(&curr_word_hashes).count();
                        let jaccard = if union == 0 {
                            1.0
                        } else {
                            intersection as f32 / union as f32
                        };
                        prev_word_hashes.clear();
                        curr_word_hashes.clear();
                        jaccard < 0.5
                    };

                    // Update temporal context for the next keyframe.
                    last_description = truncate_arc_str(&description, 200);
                    last_pts_ms = work.pts_ms;

                    // Always emit the VLM description event.
                    if dedup.should_emit(&description) {
                        let _ = event_tx.send(SinkEvent::Emit {
                            run_id: Arc::clone(&work.run_id),
                            session_id: Arc::clone(&work.session_id),
                            frame_index: work.frame_index,
                            pts_ms: work.pts_ms,
                            event_type,
                            confidence: work.confidence,
                            description: Arc::clone(&description),
                        });
                    }

                    // Emit state_transition when the scene changed.
                    if state_changed && !prev_desc.is_empty() {
                        let transition_desc = format!(
                            "{{\"from_state\":{},\"to_state\":{},\"trigger\":\"{}\",\"confidence\":{:.2}}}",
                            serde_json::Value::String(prev_desc.to_string()),
                            serde_json::Value::String(description.to_string()),
                            work.event_type,
                            work.confidence,
                        );
                        let _ = event_tx.send(SinkEvent::Emit {
                            run_id: Arc::clone(&work.run_id),
                            session_id: Arc::clone(&work.session_id),
                            frame_index: work.frame_index,
                            pts_ms: work.pts_ms,
                            event_type: "state_transition",
                            confidence: work.confidence,
                            description: Arc::from(transition_desc),
                        });
                    }
                    // ── Training pair collection ─────────────────────────────────────
                    #[cfg(feature = "training")]
                    if distillation.enabled {
                        if let Some(emb) = &embedding {
                            if sample_frame(work.frame_index, distillation.collection_rate) {
                                collect_training_pair(
                                    &work,
                                    emb,
                                    &description,
                                    event_type,
                                    &distillation,
                                    training_store.as_ref(),
                                );
                            }
                        }
                    }

                    if let Some(event) = store_keyframe_event_with_backlog(
                        &jpeg_sink_backlog,
                        &metrics,
                        Arc::clone(&work.run_id),
                        work.frame_index,
                        work.pts_ms,
                        work.event_type,
                        description,
                        work.jpeg_bytes,
                    ) {
                        let _ = event_tx.send(event);
                    }
                }
                // event_tx clone is dropped here; once all worker clones drop,
                // the outer event_tx also drops, closing the writer channel.
            })
            .expect("vlm thread spawn failed");
    }
}

// ─── VLM worker helpers ────────────────────────────────────────────────────────

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

/// POST `jpeg_b64` to the SigLIP2 embedding server and return a 768-dim vector.
///
/// Returns `None` on any error (unreachable server, timeout, malformed response).
/// Callers should treat `None` as "embedding unavailable" and skip KNN / training
/// collection gracefully.
#[cfg(feature = "training")]
fn fetch_frame_embedding(
    client: Option<&reqwest::blocking::Client>,
    url: Option<&str>,
    jpeg_b64: &str,
) -> Option<[f32; 768]> {
    let (client, url) = (client?, url?);
    let endpoint = format!("{url}/embed");
    let resp: serde_json::Value = client
        .post(&endpoint)
        .json(&serde_json::json!({"image_b64": jpeg_b64}))
        .send()
        .ok()?
        .json()
        .ok()?;
    let arr = resp.get("embedding")?.as_array()?;
    if arr.len() != 768 {
        return None;
    }
    let mut emb = [0f32; 768];
    for (i, v) in arr.iter().enumerate() {
        emb[i] = v.as_f64()? as f32;
    }
    Some(emb)
}

/// Deterministic per-frame sampling: returns `true` for approximately
/// `rate * 100%` of frames, determined by `frame_index % 1000`.
#[cfg(feature = "training")]
fn sample_frame(frame_index: u64, rate: f32) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    (frame_index % 1000) < (rate * 1000.0) as u64
}

/// Write one `(frame, label, embedding)` training triple to the store.
///
/// Evicts the oldest pairs when the tenant's count exceeds `max_pairs_per_tenant`.
/// All errors are logged as warnings rather than propagated — training collection
/// must never interrupt the real-time pipeline.
#[cfg(feature = "training")]
fn collect_training_pair(
    work: &KeyframeWork,
    embedding: &[f32; 768],
    description: &str,
    event_type: &str,
    distillation: &DistillationConfig,
    training_store: Option<&Arc<Mutex<TrainingStore>>>,
) {
    let store = match training_store {
        Some(s) => s,
        None => return,
    };
    let jpeg_bytes = work.jpeg_bytes.as_ref();
    let label_json = serde_json::json!({
        "event_type": event_type,
        "description": description,
    })
    .to_string();

    match store.lock() {
        Ok(guard) => {
            match guard.store_pair(
                &work.run_id,
                jpeg_bytes,
                &label_json,
                &distillation.teacher_model,
                work.confidence,
                embedding,
            ) {
                Ok(_row_id) => {
                    let _ = guard.evict_oldest(&work.run_id, distillation.max_pairs_per_tenant);
                    let pairs_count = guard.pair_count(&work.run_id).unwrap_or(0);
                    tracing::info!(
                        tenant_id = %work.run_id,
                        pairs_count,
                        "training pair stored"
                    );
                }
                Err(e) => {
                    tracing::warn!("failed to store training pair: {e}");
                }
            }
        }
        Err(_) => {
            tracing::warn!("training store mutex poisoned; skipping pair collection");
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        build_stream_frame_from_yuv, decode_output_pool_slots, jpeg_pool_slots,
        token_budget_entry, EventSink, KeyframeWork, StreamFrame, CLIP_FRAME_QUEUE_CAPACITY,
        CLIP_WORK_QUEUE_CAPACITY,
        FFMPEG_YUV_READER_QUEUE_CAPACITY, JPEG_POOL_SLOT_CEILING, JPEG_SINK_EVENT_POOL_ALLOWANCE,
        SINK_EVENT_QUEUE_CAPACITY, STREAM_FRAME_QUEUE_CAPACITY, VLM_TOKEN_BUDGET_MAX_SESSIONS,
        VLM_WORK_QUEUE_CAPACITY,
    };
    use crate::webrtc::clip::MAX_CLIP_FRAMES_PER_REQUEST;
    use crate::webrtc::decode::{
        VideoCodec, YuvFrame, FFMPEG_YUV_PENDING_POOL_ALLOWANCE,
        FFMPEG_YUV_READER_POOL_MIN_SLOTS, SOFTWARE_YUV_POOL_MIN_SLOTS,
    };
    use crate::webrtc::recycle::VecPool;
    use crate::gate::FrameSignal;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

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
            event_type: &str,
            _confidence: f32,
            _description: &str,
        ) -> Result<(), String> {
            self.events.lock().unwrap().push(event_type.to_string());
            Ok(())
        }

        fn store_keyframe_sync(
            &self,
            _run_id: &str,
            _frame_index: u64,
            _pts_ms: u64,
            event_type: &str,
            _description: &str,
            _jpeg_data: &[u8],
        ) -> Result<(), String> {
            self.keyframes.lock().unwrap().push(event_type.to_string());
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
        sink.emit_event_sync("r", "s", 0, 0, "vlm", 0.9, "hello").unwrap();
        sink.store_keyframe_sync("r", 0, 0, "scene_cut", "hello", b"").unwrap();
        assert_eq!(sink.events.lock().unwrap().as_slice(), ["vlm"]);
        assert_eq!(sink.keyframes.lock().unwrap().as_slice(), ["scene_cut"]);
    }

    #[test]
    fn token_budget_entry_evicts_when_session_bound_is_reached() {
        let mut budget: HashMap<Arc<str>, (Instant, u32)> = HashMap::new();
        let now = Instant::now();

        for i in 0..(VLM_TOKEN_BUDGET_MAX_SESSIONS + 4) {
            let session: Arc<str> = Arc::from(format!("session-{i}"));
            token_budget_entry(&mut budget, &session, now);
        }

        assert_eq!(budget.len(), VLM_TOKEN_BUDGET_MAX_SESSIONS);
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
        assert_eq!(decode_output_pool_slots(true, VideoCodec::Vp8), expected);
        assert_eq!(decode_output_pool_slots(false, VideoCodec::Vp8), expected);
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
        assert_eq!(super::jpeg_sink_event_backlog_capacity(), JPEG_SINK_EVENT_POOL_ALLOWANCE);
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
            "scene_cut",
            Arc::from("description"),
            [0xff_u8, 0xd8, 0xff, 0xd9].into(),
        )
        .is_some());
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
    fn analysis_workers_emit_loop_event() {
        use super::spawn_analysis_workers;

        let sink = MockSink::new();
        let prompt = Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from(""))));
        let (frame_tx, frame_rx) = kanal::bounded::<StreamFrame>(64);
        let (vlm_tx, _vlm_rx) = kanal::bounded::<KeyframeWork>(64);

        spawn_analysis_workers(
            1,
            frame_rx,
            vlm_tx,
            None, // no clip mode
            Arc::clone(&sink) as Arc<dyn EventSink>,
            "run-test".into(),
            "sess-test".into(),
            prompt,
            Arc::new(crate::metrics::PipelineMetrics::new()),
            tracing::Span::none(),
        );

        // Send 8 frames with the same hash to trigger loop detection.
        let same_hash_frame = StreamFrame {
            signal: FrameSignal {
                frame_index: 0,
                pts_ms: 0,
                perceptual_hash: 0xAAAA_AAAA_AAAA_AAAA,
                luma_mean: 0.4,
                flicker_score: 0.0,
                ghosting_score: 0.0,
                noise_variance_score: 0.0,
            },
            jpeg: Some([0xff_u8, 0xd8, 0xff, 0xd9].into()),
            pts_ms: 0,
            seq: 0,
        };

        for i in 0..8u64 {
            let mut sf = same_hash_frame.clone();
            sf.seq = i;
            sf.signal.frame_index = i;
            sf.pts_ms = i * 33;
            frame_tx.send(sf).unwrap();
        }

        drop(frame_tx); // signal EOF
        std::thread::sleep(std::time::Duration::from_millis(200));

        let events = sink.events.lock().unwrap();
        assert!(
            events.iter().any(|e| e == "loop_detected"),
            "expected at least one loop_detected event, got: {events:?}"
        );
    }

    #[test]
    fn analysis_workers_load_latest_prompt_for_keyframes() {
        use super::spawn_analysis_workers;

        let sink = MockSink::new();
        let prompt = Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from(""))));
        let (frame_tx, frame_rx) = kanal::bounded::<StreamFrame>(64);
        let (vlm_tx, vlm_rx) = kanal::bounded::<KeyframeWork>(64);

        spawn_analysis_workers(
            1,
            frame_rx,
            vlm_tx,
            None,
            Arc::clone(&sink) as Arc<dyn EventSink>,
            "run-test".into(),
            "sess-test".into(),
            Arc::clone(&prompt),
            Arc::new(crate::metrics::PipelineMetrics::new()),
            tracing::Span::none(),
        );

        frame_tx.send(make_stream_frame(0)).unwrap();
        let first = vlm_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("initial keyframe work");
        assert_eq!(&*first.prompt, "");

        prompt.store(Arc::new(Arc::from("describe updated prompt")));

        let mut scene_cut = make_stream_frame(1);
        scene_cut.signal.perceptual_hash = !scene_cut.signal.perceptual_hash;
        frame_tx.send(scene_cut).unwrap();

        let second = vlm_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("scene-cut keyframe work");
        assert_eq!(&*second.prompt, "describe updated prompt");

        drop(frame_tx);
    }
}

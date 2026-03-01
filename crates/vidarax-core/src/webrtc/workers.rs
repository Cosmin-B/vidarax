//! Core streaming analysis worker pools.
//!
//! Three pools form the real-time pipeline:
//!
//! ```text
//! WebRTC peer (kanal::Receiver<RtpFrame>)
//!   ↓
//! decode_workers   — H.264 → YUV → FrameSignal + JPEG
//!   ↓ kanal::Sender<StreamFrame>
//! analysis_workers — TwoPassPipeline + LoopDetector
//!   ↓ kanal::Sender<KeyframeWork>  (non-blocking; drops if full)
//! vlm_workers      — N threads; VLM inference → SpacetimeDB events + keyframes
//! ```
//!
//! All worker threads are `std::thread::spawn`ed (long-running, no async).
//! They each terminate when their input `kanal` channel is closed (sender
//! dropped), so shutdown is cooperative and propagates automatically from the
//! decode stage forward.
//!
//! # SpacetimeDB integration
//!
//! Workers accept `Arc<dyn EventSink>`.  Implement this trait on your
//! SpacetimeDB client and pass it in.  The sync methods must not block
//! indefinitely; they are called from worker threads that must keep up with
//! the frame rate.

use std::sync::Arc;

use base64::Engine as _;

use crate::gate::{GateConfig, GateEventType};
use crate::loop_detector::LoopDetector;
use crate::metrics::PipelineMetrics;
use crate::pipeline::{TwoPassConfig, TwoPassPipeline};
use crate::provider::{InferenceImage, InferenceProvider, InferenceRequest};
use crate::tiered_vlm::TieredVlmConfig;
use crate::webrtc::decode::{Decoder, DecoderConfig};
use crate::webrtc::session::RtpFrame;
use crate::webrtc::signals::{yuv_to_frame_signal, yuv_to_jpeg};

// ─── EventSink trait ──────────────────────────────────────────────────────────

/// Abstraction over SpacetimeDB event writes used by the worker pools.
///
/// Implement this on your `SpacetimeClient` (or a test mock) and pass
/// `Arc<dyn EventSink>` to the spawn functions.
///
/// # Thread-safety
///
/// The `Send + Sync` bounds are required because worker threads hold
/// `Arc<dyn EventSink>` and call these methods concurrently.  The
/// implementation must be safe to call from multiple threads simultaneously.
pub trait EventSink: Send + Sync {
    /// Emit a real-time agent event (blocking; must not hold locks indefinitely).
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
    /// Stored as a reference-counted slice so downstream consumers can clone
    /// the pointer (16 bytes) rather than copying the full JPEG payload.
    pub jpeg: Option<Arc<[u8]>>,
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
    pub event_type: String,
    /// Gate confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Raw JPEG bytes — base64-encoded on-demand for VLM, stored raw in SpacetimeDB.
    ///
    /// Shared via `Arc<[u8]>` so cloning this work item copies only a pointer.
    pub jpeg_bytes: Arc<[u8]>,
    /// Semantic prompt to pass to the VLM.
    pub prompt: String,
}

// ─── Decode workers ───────────────────────────────────────────────────────────

/// Spawn `cores` H.264 decode worker threads.
///
/// Each thread:
/// 1. Constructs a [`Decoder`] using the selected backend (`gpu` flag).
/// 2. Decodes every [`RtpFrame`] to planar YUV 4:2:0.
/// 3. Computes a [`crate::gate::FrameSignal`] from the luma plane.
/// 4. Encodes a JPEG thumbnail at quality 75.
/// 5. Sends the [`StreamFrame`] to `frame_tx`.
///
/// Threads exit when `rtp_rx` is closed (all senders dropped).
/// `rtp_rx` is cloned so all `cores` workers share the same channel (MPMC).
///
/// `metrics` counters are incremented for each received / decoded frame.
/// `session_span` is entered inside each thread so all log events are
/// attributed to the owning session.
pub fn spawn_decode_workers(
    cores: usize,
    rtp_rx: kanal::Receiver<RtpFrame>,
    frame_tx: kanal::Sender<StreamFrame>,
    gpu: bool,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
) {
    for i in 0..cores.max(1) {
        let rtp_rx = rtp_rx.clone();
        let frame_tx = frame_tx.clone();
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();

        std::thread::Builder::new()
            .name(format!("vx-decode-{i}"))
            .spawn(move || {

                let config = DecoderConfig { gpu_available: gpu };
                let mut decoder = Decoder::new(&config);
                let mut prev_signal: Option<crate::gate::FrameSignal> = None;

                while let Ok(frame) = rtp_rx.recv() {
                    let _guard = session_span.enter();
                    metrics.inc_rtp_received();

                    let yuv = match decoder.decode(&frame.nals) {
                        Ok(y) => y,
                        Err(_) => continue, // SPS/PPS or incomplete NAL — skip
                    };

                    let signal = yuv_to_frame_signal(
                        &yuv,
                        frame.seq,
                        frame.pts_ms,
                        prev_signal.as_ref(),
                    );
                    // Allocate once; all downstream consumers share the Arc pointer.
                    let jpeg: Arc<[u8]> = Arc::from(yuv_to_jpeg(&yuv, 75));
                    prev_signal = Some(signal);

                    metrics.inc_frames_decoded();
                    let sf = StreamFrame {
                        signal,
                        jpeg: Some(jpeg),
                        pts_ms: frame.pts_ms,
                        seq: frame.seq,
                    };

                    if frame_tx.send(sf).is_err() {
                        break; // downstream dropped — shut down
                    }
                }
            })
            .expect("decode thread spawn failed");
    }
}

// ─── Analysis workers ─────────────────────────────────────────────────────────

/// Spawn `cores` analysis worker threads.
///
/// Each thread maintains its own stateful [`TwoPassPipeline`] and
/// [`LoopDetector`].  For every [`StreamFrame`] received:
///
/// - **Loop detection** (perceptual-hash ring buffer): if a loop is detected,
///   `stdb.emit_event_sync("loop_detected", …)` is called immediately.
/// - **Normal mode** (`clip_tx` is `None`): the gate engine ([`TwoPassPipeline`])
///   decides which frames are keyframes; those are base64-encoded and forwarded
///   to `vlm_tx` via a non-blocking try-send (dropped when VLM queue is full).
/// - **Clip mode** (`clip_tx` is `Some`): the gate engine is bypassed; every
///   accepted [`StreamFrame`] is forwarded (non-blocking) to the
///   [`crate::webrtc::clip::ClipAccumulator`] channel.
///
/// Threads exit when `frame_rx` is closed.
pub fn spawn_analysis_workers(
    cores: usize,
    frame_rx: kanal::Receiver<StreamFrame>,
    vlm_tx: kanal::Sender<KeyframeWork>,
    clip_tx: Option<kanal::Sender<StreamFrame>>,
    stdb: Arc<dyn EventSink>,
    run_id: Arc<str>,
    session_id: Arc<str>,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
) {
    for i in 0..cores.max(1) {
        let frame_rx = frame_rx.clone();
        let vlm_tx = vlm_tx.clone();
        let clip_tx = clip_tx.clone();
        let stdb = Arc::clone(&stdb);
        let run_id = Arc::clone(&run_id);
        let session_id = Arc::clone(&session_id);
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();

        std::thread::Builder::new()
            .name(format!("vx-analysis-{i}"))
            .spawn(move || {

                let mut pipeline =
                    TwoPassPipeline::new(TwoPassConfig::default(), GateConfig::default());
                let mut loop_det = LoopDetector::new(6, 3);

                while let Ok(sf) = frame_rx.recv() {
                    let _guard = session_span.enter();

                    // ── Loop detection (always active) ───────────────────
                    if loop_det.check(sf.signal.perceptual_hash) {
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

                    if let Some(ref clip_tx) = clip_tx {
                        // ── Clip mode: forward every frame to accumulator ─
                        // Non-blocking: drop if accumulator queue is full.
                        let _ = clip_tx.try_send(sf);
                    } else {
                        // ── Normal mode: gate engine → VLM ───────────────
                        let metas = pipeline.analyze_batch(&[sf.signal]);
                        let meta = match metas.first() {
                            Some(m) => *m,
                            None => continue,
                        };

                        if meta.gate_event == GateEventType::KeepKeyframe {
                            // Clone the Arc pointer (16 bytes), not the JPEG payload.
                            let jpeg_bytes: Arc<[u8]> = sf
                                .jpeg
                                .clone()
                                .unwrap_or_else(|| Arc::from([] as [u8; 0]));

                            let event_type = if meta.scene_cut {
                                "scene_cut"
                            } else {
                                "periodic_keepalive"
                            };

                            let work = KeyframeWork {
                                run_id: Arc::clone(&run_id),
                                session_id: Arc::clone(&session_id),
                                frame_index: sf.signal.frame_index,
                                pts_ms: sf.pts_ms,
                                event_type: event_type.to_string(),
                                confidence: meta.confidence,
                                jpeg_bytes,
                                prompt: String::new(),
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

/// Spawn `n` VLM inference worker threads with optional tiered routing.
///
/// When `config.is_tiered()`, workers call the first-pass model first, then
/// re-infer with the second-pass model if confidence is below threshold.
///
/// Each thread:
/// 1. Pulls [`KeyframeWork`] from the shared `vlm_rx` channel.
/// 2. Enforces `max_output_tokens_per_second` backpressure per session.
/// 3. Calls `provider.infer()` with the first-pass model.
/// 4. If tiered and confidence is low, re-infers with the second-pass model.
/// 5. Emits a `vlm` or `vlm_tiered` agent event to SpacetimeDB.
/// 6. Stores the keyframe (JPEG + description) in SpacetimeDB.
///
/// VLM inference errors are logged as the description text rather than
/// crashing the worker, so a transient provider failure does not interrupt
/// the pipeline.  Threads exit when `vlm_rx` is closed.
pub fn spawn_vlm_workers<I>(
    n: usize,
    vlm_rx: kanal::Receiver<KeyframeWork>,
    provider: Arc<I>,
    stdb: Arc<dyn EventSink>,
    config: TieredVlmConfig,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
    max_output_tokens_per_second: u32,
) where
    I: InferenceProvider + 'static,
{
    for i in 0..n.max(1) {
        let vlm_rx = vlm_rx.clone();
        let provider = Arc::clone(&provider);
        let stdb = Arc::clone(&stdb);
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();

        std::thread::Builder::new()
            .name(format!("vx-vlm-{i}"))
            .spawn(move || {
                // Per-session token budget: (window_start, tokens_emitted_in_window).
                // Key is Arc<str>: clone is pointer-width; Hash/Eq compare string content.
                let mut token_budget: std::collections::HashMap<
                    Arc<str>,
                    (std::time::Instant, u32),
                > = std::collections::HashMap::new();

                while let Ok(work) = vlm_rx.recv() {
                    let _guard = session_span.enter();

                    // Token rate limit: skip this frame if the session has
                    // already exceeded the per-second output token budget.
                    if max_output_tokens_per_second > 0 {
                        let now = std::time::Instant::now();
                        let entry = token_budget
                            .entry(work.session_id.clone())
                            .or_insert((now, 0));
                        if now.duration_since(entry.0).as_secs() >= 1 {
                            *entry = (now, 0); // reset 1-second window
                        }
                        if entry.1 >= max_output_tokens_per_second {
                            metrics.inc_keyframes_dropped();
                            continue; // backpressure: drop this inference
                        }
                    }

                    metrics.inc_vlm_inferences();
                    let prompt = if work.prompt.is_empty() {
                        "Briefly describe what is happening in this video frame.".to_string()
                    } else {
                        work.prompt.clone()
                    };

                    // First pass (fast model)
                    let first_request = InferenceRequest {
                        model: config.first_pass_model.clone(),
                        prompt: prompt.clone(),
                        input_images: vec![InferenceImage {
                            media_type: "image/jpeg",
                            data_base64: base64::engine::general_purpose::STANDARD.encode(&work.jpeg_bytes),
                        }],
                        max_tokens: 128,
                        temperature: 0.0,
                        timeout_ms: 5_000,
                        allow_fallback: true,
                        output_schema: None,
                    };

                    let (description, used_second_pass) = match provider.infer(&first_request) {
                        Ok(result) => {
                            let first_conf = parse_confidence_from_output(&result.output_text);
                            if config.needs_second_pass(first_conf) {
                                // Second pass (accurate model)
                                let second_request = InferenceRequest {
                                    model: config.second_pass_model.clone(),
                                    prompt,
                                    input_images: first_request.input_images,
                                    max_tokens: config.second_pass_max_tokens,
                                    temperature: 0.0,
                                    timeout_ms: 10_000,
                                    allow_fallback: true,
                                    output_schema: None,
                                };
                                match provider.infer(&second_request) {
                                    Ok(second) => (second.output_text, true),
                                    Err(_) => (result.output_text, false), // fallback to first
                                }
                            } else {
                                (result.output_text, false)
                            }
                        }
                        Err(err) => (format!("vlm_error: {err:?}"), false),
                    };

                    // Charge output tokens against the session budget.
                    // Approximate: 4 bytes per token (UTF-8 average).
                    if max_output_tokens_per_second > 0 {
                        let token_count = (description.len() / 4).max(1) as u32;
                        if let Some(entry) = token_budget.get_mut(work.session_id.as_ref()) {
                            entry.1 = entry.1.saturating_add(token_count);
                        }
                    }

                    let event_type = if used_second_pass { "vlm_tiered" } else { "vlm" };

                    let _ = stdb.emit_event_sync(
                        &work.run_id,
                        &work.session_id,
                        work.frame_index,
                        work.pts_ms,
                        event_type,
                        work.confidence,
                        &description,
                    );
                    let _ = stdb.store_keyframe_sync(
                        &work.run_id,
                        work.frame_index,
                        work.pts_ms,
                        &work.event_type,
                        &description,
                        &work.jpeg_bytes,
                    );
                }
            })
            .expect("vlm thread spawn failed");
    }
}

/// Try to extract a confidence float from VLM JSON output.
///
/// Looks for `"confidence": 0.XX` in JSON output, or falls back to `0.5`
/// (which triggers a second pass at the default threshold of 0.7).
fn parse_confidence_from_output(text: &str) -> f32 {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(conf) = val.get("confidence").and_then(|v| v.as_f64()) {
            return conf as f32;
        }
    }
    // Default: assume low confidence to trigger second pass when tiered.
    0.5
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{EventSink, KeyframeWork, StreamFrame};
    use crate::gate::FrameSignal;
    use std::sync::{Arc, Mutex};

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
            jpeg: Some(Arc::from([0xff_u8, 0xd8, 0xff, 0xd9] as [u8; 4])), // minimal JPEG markers
            pts_ms: seq * 33,
            seq,
        }
    }

    #[test]
    fn stream_frame_and_keyframe_work_are_clone_debug() {
        let sf = make_stream_frame(0);
        let _ = sf.clone();
        let _ = format!("{:?}", sf);

        let kw = KeyframeWork {
            run_id: "r1".into(),
            session_id: "s1".into(),
            frame_index: 0,
            pts_ms: 0,
            event_type: "scene_cut".into(),
            confidence: 0.9,
            jpeg_bytes: Arc::from([0xFF_u8, 0xD8, 0xFF, 0xD9] as [u8; 4]),
            prompt: String::new(),
        };
        let _ = kw.clone();
        let _ = format!("{:?}", kw);
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
    fn analysis_workers_emit_loop_event() {
        use super::spawn_analysis_workers;

        let sink = MockSink::new();
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
            jpeg: Some(Arc::from([0xff_u8, 0xd8, 0xff, 0xd9] as [u8; 4])),
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
}

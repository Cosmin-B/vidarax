//! Core streaming analysis worker pools.
//!
//! Three pools form the real-time pipeline:
//!
//! ```text
//! WebRTC peer (kanal::Receiver<RtpFrame>)
//!   ↓
//! decode_workers   — core-pinned; H.264 → YUV → FrameSignal + JPEG
//!   ↓ kanal::Sender<StreamFrame>
//! analysis_workers — core-pinned; TwoPassPipeline + LoopDetector
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
use crate::pipeline::{TwoPassConfig, TwoPassPipeline};
use crate::provider::{InferenceImage, InferenceProvider, InferenceRequest};
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
        jpeg_b64: &str,
    ) -> Result<(), String>;
}

// ─── Pipeline types ───────────────────────────────────────────────────────────

/// Decoded video frame ready for the analysis stage.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Gate-engine signal computed from the luma plane.
    pub signal: crate::gate::FrameSignal,
    /// JPEG thumbnail of the decoded frame (`Some` after successful decode).
    pub jpeg: Option<Vec<u8>>,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Per-session monotonically increasing frame index (== `signal.frame_index`).
    pub seq: u64,
}

/// Work item forwarded to VLM workers when a keyframe is decided upon.
#[derive(Debug, Clone)]
pub struct KeyframeWork {
    pub run_id: String,
    pub session_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    /// Gate reason code: `"scene_cut"` | `"periodic_keepalive"` | `"initial_frame"`.
    pub event_type: String,
    /// Gate confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Base64-encoded JPEG thumbnail for VLM input.
    pub jpeg_b64: String,
}

// ─── Decode workers ───────────────────────────────────────────────────────────

/// Spawn `cores` H.264 decode worker threads.
///
/// Each thread:
/// 1. Is pinned to a physical CPU core (best-effort via `core_affinity`).
/// 2. Constructs a [`Decoder`] using the selected backend (`gpu` flag).
/// 3. Decodes every [`RtpFrame`] to planar YUV 4:2:0.
/// 4. Computes a [`crate::gate::FrameSignal`] from the luma plane.
/// 5. Encodes a JPEG thumbnail at quality 75.
/// 6. Sends the [`StreamFrame`] to `frame_tx`.
///
/// Threads exit when `rtp_rx` is closed (all senders dropped).
/// `rtp_rx` is cloned so all `cores` workers share the same channel (MPMC).
pub fn spawn_decode_workers(
    cores: usize,
    rtp_rx: kanal::Receiver<RtpFrame>,
    frame_tx: kanal::Sender<StreamFrame>,
    gpu: bool,
) {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();

    for i in 0..cores.max(1) {
        let rtp_rx = rtp_rx.clone();
        let frame_tx = frame_tx.clone();
        let core_id = core_ids.get(i).copied();

        std::thread::Builder::new()
            .name(format!("vx-decode-{i}"))
            .spawn(move || {
                if let Some(cid) = core_id {
                    core_affinity::set_for_current(cid);
                }

                let config = DecoderConfig { gpu_available: gpu };
                let mut decoder = Decoder::new(&config);
                let mut prev_signal: Option<crate::gate::FrameSignal> = None;

                while let Ok(frame) = rtp_rx.recv() {
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
                    let jpeg = yuv_to_jpeg(&yuv, 75);
                    prev_signal = Some(signal);

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
/// - **Gate engine** ([`TwoPassPipeline`]): if the gate decides
///   [`GateEventType::KeepKeyframe`], the JPEG is base64-encoded and a
///   [`KeyframeWork`] is pushed to `vlm_tx` via a *non-blocking* try-send
///   (dropped when the VLM queue is full to avoid stalling decode).
///
/// Threads exit when `frame_rx` is closed.
pub fn spawn_analysis_workers(
    cores: usize,
    frame_rx: kanal::Receiver<StreamFrame>,
    vlm_tx: kanal::Sender<KeyframeWork>,
    stdb: Arc<dyn EventSink>,
    run_id: String,
    session_id: String,
) {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();

    for i in 0..cores.max(1) {
        let frame_rx = frame_rx.clone();
        let vlm_tx = vlm_tx.clone();
        let stdb = Arc::clone(&stdb);
        let run_id = run_id.clone();
        let session_id = session_id.clone();
        let core_id = core_ids.get(i).copied();

        std::thread::Builder::new()
            .name(format!("vx-analysis-{i}"))
            .spawn(move || {
                if let Some(cid) = core_id {
                    core_affinity::set_for_current(cid);
                }

                let mut pipeline =
                    TwoPassPipeline::new(TwoPassConfig::default(), GateConfig::default());
                let mut loop_det = LoopDetector::new(6, 3);

                while let Ok(sf) = frame_rx.recv() {
                    // ── Loop detection ───────────────────────────────────
                    if loop_det.check(sf.signal.perceptual_hash) {
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

                    // ── Gate engine ──────────────────────────────────────
                    let metas = pipeline.analyze_batch(&[sf.signal]);
                    let meta = match metas.first() {
                        Some(m) => *m,
                        None => continue,
                    };

                    if meta.gate_event == GateEventType::KeepKeyframe {
                        let jpeg_b64 = sf
                            .jpeg
                            .as_deref()
                            .map(|j| {
                                base64::engine::general_purpose::STANDARD.encode(j)
                            })
                            .unwrap_or_default();

                        let event_type = if meta.scene_cut {
                            "scene_cut"
                        } else {
                            "periodic_keepalive"
                        };

                        let work = KeyframeWork {
                            run_id: run_id.clone(),
                            session_id: session_id.clone(),
                            frame_index: sf.signal.frame_index,
                            pts_ms: sf.pts_ms,
                            event_type: event_type.to_string(),
                            confidence: meta.confidence,
                            jpeg_b64,
                        };

                        // Non-blocking: drop if VLM queue is full to avoid
                        // stalling the decode → analysis pipeline.
                        let _ = vlm_tx.try_send(work);
                    }
                }
            })
            .expect("analysis thread spawn failed");
    }
}

// ─── VLM workers ──────────────────────────────────────────────────────────────

/// Spawn `n` VLM inference worker threads.
///
/// Each thread:
/// 1. Pulls [`KeyframeWork`] from the shared `vlm_rx` channel.
/// 2. Calls `provider.infer()` with the base64 JPEG as input.
/// 3. Emits a `vlm` agent event to SpacetimeDB.
/// 4. Stores the keyframe (JPEG + description) in SpacetimeDB.
///
/// VLM inference errors are logged as the description text rather than
/// crashing the worker, so a transient provider failure does not interrupt
/// the pipeline.  Threads exit when `vlm_rx` is closed.
pub fn spawn_vlm_workers<I>(
    n: usize,
    vlm_rx: kanal::Receiver<KeyframeWork>,
    provider: Arc<I>,
    stdb: Arc<dyn EventSink>,
) where
    I: InferenceProvider + 'static,
{
    for i in 0..n.max(1) {
        let vlm_rx = vlm_rx.clone();
        let provider = Arc::clone(&provider);
        let stdb = Arc::clone(&stdb);

        std::thread::Builder::new()
            .name(format!("vx-vlm-{i}"))
            .spawn(move || {
                while let Ok(work) = vlm_rx.recv() {
                    let request = InferenceRequest {
                        model: "openbmb/MiniCPM-V-4.5".to_string(),
                        prompt: "Briefly describe what is happening in this video frame."
                            .to_string(),
                        input_images: vec![InferenceImage {
                            media_type: "image/jpeg".to_string(),
                            data_base64: work.jpeg_b64.clone(),
                        }],
                        max_tokens: 128,
                        temperature: 0.0,
                        timeout_ms: 5_000,
                        allow_fallback: true,
                    };

                    let description = match provider.infer(&request) {
                        Ok(result) => result.output_text,
                        Err(err) => format!("vlm_error: {err:?}"),
                    };

                    // Emit the VLM-annotated agent event.
                    let _ = stdb.emit_event_sync(
                        &work.run_id,
                        &work.session_id,
                        work.frame_index,
                        work.pts_ms,
                        "vlm",
                        work.confidence,
                        &description,
                    );

                    // Persist the keyframe with its thumbnail and description.
                    let _ = stdb.store_keyframe_sync(
                        &work.run_id,
                        work.frame_index,
                        work.pts_ms,
                        &work.event_type,
                        &description,
                        &work.jpeg_b64,
                    );
                }
            })
            .expect("vlm thread spawn failed");
    }
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
            _jpeg_b64: &str,
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
            jpeg: Some(vec![0xff, 0xd8, 0xff, 0xd9]), // minimal JPEG markers
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
            jpeg_b64: "YWJj".into(),
        };
        let _ = kw.clone();
        let _ = format!("{:?}", kw);
    }

    #[test]
    fn mock_sink_records_calls() {
        let sink = MockSink::new();
        sink.emit_event_sync("r", "s", 0, 0, "vlm", 0.9, "hello").unwrap();
        sink.store_keyframe_sync("r", 0, 0, "scene_cut", "hello", "").unwrap();
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
            Arc::clone(&sink) as Arc<dyn EventSink>,
            "run-test".into(),
            "sess-test".into(),
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
            jpeg: Some(vec![0xff, 0xd8, 0xff, 0xd9]),
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

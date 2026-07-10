//! Clip-mode: temporal multi-frame VLM inference.
//!
//! Instead of processing one keyframe at a time, clip mode collects a sliding
//! window of frames and submits them as a **multi-image** VLM call.  This lets
//! the model reason across temporal context (e.g. "did the person sit down
//! between frame 1 and frame 6?") rather than evaluating isolated snapshots.
//!
//! # Pipeline (clip mode)
//!
//! ```text
//! analysis_workers  ──┐
//!   (StreamFrame)     │  kanal::Sender<StreamFrame>
//!                     ↓
//!               ClipAccumulator thread
//!                     │  kanal::Sender<ClipWork>
//!                     ↓
//!               clip_vlm_workers  ──→  VLM (multi-image) ──→ EventSink
//! ```
//!
//! # Rate control
//!
//! The accumulator down-samples incoming `StreamFrame`s to `target_fps` by
//! comparing presentation timestamps.  It collects frames until
//! `clip_length_seconds` of video time has elapsed, then emits one `ClipWork`
//! and starts a fresh window.  A minimum inter-emission delay of
//! `delay_seconds` (wall-clock) prevents bursting.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwapOption;
use base64::Engine as _;

use crate::gate::FrameSignal;
use crate::metrics::PipelineMetrics;
use crate::provider::{InferenceImage, InferenceObserver, InferenceProvider, InferenceRequest};
use crate::tiered_vlm::{run_tiered, TieredVlmConfig};
use crate::webrtc::recycle::RecycledBytes;
use crate::webrtc::workers::{
    per_stream_vlm_workers, prune_stale_token_budget_entries, token_budget_entry, EventSink,
    StreamFrame,
};

// ─── ClipConfig ───────────────────────────────────────────────────────────────

pub const MAX_CLIP_TARGET_FPS: u32 = 30;
pub const MAX_CLIP_LENGTH_SECONDS: u32 = 60;
/// Maximum JPEG frames retained and sent in one multi-image VLM clip request.
///
/// At typical JPEG sizes this keeps a clip payload bounded while preserving
/// enough temporal coverage for short interaction analysis. Longer windows keep
/// a uniformly downsampled set across the configured clip duration.
pub const MAX_CLIP_FRAMES_PER_REQUEST: usize = 64;

/// Parameters controlling clip-mode temporal batching.
///
/// # Constraints
/// - `target_fps` must be in 1–30.
/// - `clip_length_seconds` must be in 0.1–60.
/// - `delay_seconds` must be in 0.0–60.
/// - `target_fps * clip_length_seconds >= 3` (ensures at least 3 frames per clip).
///
/// # Examples
///
/// ```
/// use vidarax_core::webrtc::clip::ClipConfig;
///
/// let cfg = ClipConfig::default();
/// assert!(cfg.validate().is_ok());
///
/// let bad = ClipConfig { target_fps: 1, clip_length_seconds: 0.1, delay_seconds: 0.0 };
/// assert!(bad.validate().is_err());   // 1 * 0.1 < 3
/// ```
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ClipConfig {
    /// Frames per second to sample from the decoded stream (1–30, default 6).
    pub target_fps: u32,
    /// Duration of each clip window in seconds (0.1–60, default 0.5).
    pub clip_length_seconds: f32,
    /// Minimum delay between clip emissions in seconds (0–60, default 0.5).
    pub delay_seconds: f32,
}

impl ClipConfig {
    /// Validate all fields and the composite frame-count constraint.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_fps < 1 || self.target_fps > MAX_CLIP_TARGET_FPS {
            return Err(format!(
                "target_fps must be between 1 and 30, got {}",
                self.target_fps
            ));
        }
        if !self.clip_length_seconds.is_finite()
            || self.clip_length_seconds < 0.1
            || self.clip_length_seconds > MAX_CLIP_LENGTH_SECONDS as f32
        {
            return Err(format!(
                "clip_length_seconds must be between 0.1 and 60, got {}",
                self.clip_length_seconds
            ));
        }
        if !self.delay_seconds.is_finite() || self.delay_seconds < 0.0 || self.delay_seconds > 60.0
        {
            return Err(format!(
                "delay_seconds must be between 0 and 60, got {}",
                self.delay_seconds
            ));
        }
        let min_frames = (self.target_fps as f32 * self.clip_length_seconds).floor() as u32;
        if min_frames < 3 {
            return Err(format!(
                "target_fps ({}) * clip_length_seconds ({}) must yield >= 3 frames, got {}",
                self.target_fps, self.clip_length_seconds, min_frames
            ));
        }
        Ok(())
    }
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            target_fps: 6,
            clip_length_seconds: 0.5,
            delay_seconds: 0.5,
        }
    }
}

// ─── ClipWork ─────────────────────────────────────────────────────────────────

/// A batch of frames assembled by [`ClipAccumulator`] for multi-image VLM inference.
#[derive(Debug, Clone)]
pub struct ClipWork {
    pub run_id: Arc<str>,
    pub session_id: Arc<str>,
    /// Frames in this clip: `(signal, JPEG bytes)`.
    ///
    /// JPEG buffers are recycled byte handles moved into clip work without
    /// copying the payload on the runtime path.
    pub frames: VecDeque<(FrameSignal, RecycledBytes)>,
    /// PTS of the first frame in the batch (milliseconds).
    pub pts_start: u64,
    /// PTS of the last frame in the batch (milliseconds).
    pub pts_end: u64,
    /// Semantic prompt forwarded to the VLM.
    pub prompt: Arc<str>,
}

#[derive(Debug, Clone)]
pub struct ClipRateGate {
    /// Minimum inter-sample distance in ms (1000 / target_fps).
    sample_interval_ms: u64,
    /// PTS of the last frame accepted into the buffer.
    last_accepted_pts: Option<u64>,
}

impl ClipRateGate {
    pub fn new(target_fps: u32) -> Self {
        Self {
            sample_interval_ms: 1000u64 / (target_fps as u64).max(1),
            last_accepted_pts: None,
        }
    }

    pub fn should_keep(&self, pts_ms: u64) -> bool {
        let Some(last_pts) = self.last_accepted_pts else {
            return true;
        };
        pts_ms.saturating_sub(last_pts) >= self.sample_interval_ms
    }

    pub fn commit(&mut self, pts_ms: u64) {
        self.last_accepted_pts = Some(pts_ms);
    }
}

// ─── ClipAccumulator ──────────────────────────────────────────────────────────

/// Stateful accumulator that down-samples and batches [`StreamFrame`]s.
///
/// Feed frames via [`ClipAccumulator::push`]; it returns a [`ClipWork`] whenever
/// the window is full and the delay constraint is satisfied.
///
/// Each accumulator is **single-threaded** — wrap in a dedicated OS thread when
/// integrating into the pipeline.
pub struct ClipAccumulator {
    config: ClipConfig,
    run_id: Arc<str>,
    session_id: Arc<str>,
    prompt: Arc<str>,
    /// Buffered frames for the current window.
    buffer: VecDeque<(FrameSignal, RecycledBytes)>,
    /// PTS-based sampling state shared with the decode-side pre-encode gate.
    rate_gate: ClipRateGate,
    /// PTS of the first accepted frame in the current logical clip window.
    window_start_pts: Option<u64>,
    /// Wall-clock instant of the last emission (for delay enforcement).
    last_emit: Option<Instant>,
}

impl ClipAccumulator {
    /// Create a new accumulator.  `prompt` is forwarded verbatim to each
    /// emitted [`ClipWork`]; pass an empty string to use the worker default.
    ///
    /// # Panics
    ///
    /// Does **not** panic; validation must be done by the caller via
    /// [`ClipConfig::validate`] before constructing.
    pub fn new(
        config: ClipConfig,
        run_id: Arc<str>,
        session_id: Arc<str>,
        prompt: Arc<str>,
    ) -> Self {
        let rate_gate = ClipRateGate::new(config.target_fps);
        Self {
            config,
            run_id,
            session_id,
            prompt,
            buffer: VecDeque::new(),
            rate_gate,
            window_start_pts: None,
            last_emit: None,
        }
    }

    /// Accept a frame.  Returns `Some(ClipWork)` when a clip is ready.
    ///
    /// Frames are silently dropped when:
    /// - They arrive too soon (below `target_fps` interval based on PTS).
    /// - The frame has no JPEG data.
    /// - The window is not yet full.
    /// - The inter-emission delay has not elapsed.
    pub fn push(&mut self, mut sf: StreamFrame) -> Option<ClipWork> {
        // ── Rate-limit to target_fps ───────────────────────────────────────
        if !self.rate_gate.should_keep(sf.pts_ms) {
            return None; // too soon
        }

        // ── Accept JPEG ────────────────────────────────────────────────────
        let jpeg_bytes = match sf.jpeg.take() {
            Some(arc) if !arc.is_empty() => arc,
            _ => return None, // no image data
        };

        self.rate_gate.commit(sf.pts_ms);
        if self.window_start_pts.is_none() {
            self.window_start_pts = Some(sf.pts_ms);
        }
        self.buffer.push_back((sf.signal, jpeg_bytes));

        // ── Check window duration ──────────────────────────────────────────
        let window_start_pts = self.window_start_pts.unwrap_or(sf.pts_ms);
        let window_ms = (self.config.clip_length_seconds * 1000.0) as u64;
        if self.buffer.len() > MAX_CLIP_FRAMES_PER_REQUEST {
            downsample_clip_buffer(&mut self.buffer, window_start_pts, window_ms);
        }
        let elapsed_pts_ms = sf.pts_ms.saturating_sub(window_start_pts);

        if elapsed_pts_ms < window_ms {
            return None; // window not yet full
        }

        // ── Enforce inter-emission delay ───────────────────────────────────
        let delay_ms = (self.config.delay_seconds * 1000.0) as u64;
        let now = Instant::now();
        if let Some(last) = self.last_emit {
            if delay_ms > 0 && last.elapsed().as_millis() < delay_ms as u128 {
                // Slide the window forward by dropping the oldest frame — O(1) with VecDeque.
                self.buffer.pop_front();
                self.window_start_pts = self.buffer.front().map(|(sig, _)| sig.pts_ms);
                return None;
            }
        }

        // ── Emit ───────────────────────────────────────────────────────────
        let deque = std::mem::take(&mut self.buffer);
        let pts_start = deque.front().map(|(s, _)| s.pts_ms).unwrap_or(0);
        let pts_end = deque.back().map(|(s, _)| s.pts_ms).unwrap_or(pts_start);
        self.last_emit = Some(now);
        self.window_start_pts = None;
        Some(ClipWork {
            run_id: Arc::clone(&self.run_id),
            session_id: Arc::clone(&self.session_id),
            frames: deque,
            pts_start,
            pts_end,
            prompt: Arc::clone(&self.prompt),
        })
    }
}

// ─── spawn_clip_accumulator ───────────────────────────────────────────────────

/// Spawn a single thread that runs a [`ClipAccumulator`].
///
/// The thread reads [`StreamFrame`]s from `frame_rx`, passes each to the
/// accumulator, and forwards any emitted [`ClipWork`] to `clip_tx`.
///
/// The thread exits when `frame_rx` is closed (all senders dropped).
/// `session_span` is entered per-loop-iteration so tracing events are
/// attributed to the correct session.
pub fn spawn_clip_accumulator(
    frame_rx: kanal::Receiver<StreamFrame>,
    clip_tx: kanal::Sender<ClipWork>,
    config: ClipConfig,
    run_id: Arc<str>,
    session_id: Arc<str>,
    prompt: Arc<str>,
    session_span: tracing::Span,
) {
    std::thread::Builder::new()
        .name("vx-clip-acc".to_string())
        .spawn(move || {
            let mut acc = ClipAccumulator::new(config, run_id, session_id, prompt);
            while let Ok(sf) = frame_rx.recv() {
                let _guard = session_span.enter();
                if let Some(clip) = acc.push(sf) {
                    if clip_tx.send(clip).is_err() {
                        break; // downstream dropped — shut down
                    }
                }
            }
        })
        .expect("clip accumulator thread spawn failed");
}

// ─── spawn_clip_vlm_workers ───────────────────────────────────────────────────

/// Spawn `n` VLM inference worker threads that consume [`ClipWork`].
///
/// Each worker builds a **multi-image** [`InferenceRequest`] from all frames
/// in the clip and calls the VLM provider.  Tiered routing is applied
/// identically to [`crate::webrtc::workers::spawn_vlm_workers`].
///
/// Workers exit when `clip_rx` is closed.
// Spawns clip VLM workers; each borrowed handle is distinct, a params struct would only add indirection.
#[allow(clippy::too_many_arguments)]
pub fn spawn_clip_vlm_workers<I>(
    n: usize,
    clip_rx: kanal::Receiver<ClipWork>,
    provider: Arc<I>,
    stdb: Arc<dyn EventSink>,
    config: TieredVlmConfig,
    metrics: Arc<PipelineMetrics>,
    session_span: tracing::Span,
    max_output_tokens_per_second: u32,
    // Shared guided-JSON schema handle.  When the inner `Option` is `Some`,
    // the schema is passed to the first-pass VLM request and `max_tokens`
    // is raised to 1024 to accommodate structured output.
    guided_json: Arc<ArcSwapOption<Arc<str>>>,
    // Where tiered VLM inference outcomes are recorded for `/metrics`. `None`
    // when the caller has no metrics sink wired up (e.g. tests).
    observer: Option<Arc<dyn InferenceObserver>>,
) where
    I: InferenceProvider + 'static,
{
    for i in 0..clip_vlm_worker_count(n) {
        let clip_rx = clip_rx.clone();
        let provider = Arc::clone(&provider);
        let stdb = Arc::clone(&stdb);
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let session_span = session_span.clone();
        let guided_json = Arc::clone(&guided_json);
        let observer = observer.clone();
        let mut token_budget = std::collections::HashMap::new();

        std::thread::Builder::new()
            .name(format!("vx-clip-vlm-{i}"))
            .spawn(move || {
                while let Ok(work) = clip_rx.recv() {
                    let _guard = session_span.enter();
                    if max_output_tokens_per_second > 0 {
                        let now = std::time::Instant::now();
                        prune_stale_token_budget_entries(&mut token_budget, now);
                        let entry = token_budget_entry(&mut token_budget, &work.session_id, now);
                        if now.duration_since(entry.0).as_secs() >= 1 {
                            *entry = (now, 0);
                        }
                        if entry.1 >= max_output_tokens_per_second {
                            metrics.inc_keyframes_dropped();
                            continue;
                        }
                    }
                    metrics.inc_vlm_inferences();

                    let prompt: Arc<str> = if work.prompt.is_empty() {
                        Arc::from("Briefly describe what is happening across these video frames.")
                    } else {
                        Arc::clone(&work.prompt)
                    };

                    // Build multi-image request — all frames sent together.
                    let input_images: Vec<InferenceImage> = work
                        .frames
                        .iter()
                        .map(|(_, jpeg_bytes)| InferenceImage {
                            media_type: "image/jpeg",
                            data_base64: base64::engine::general_purpose::STANDARD
                                .encode(jpeg_bytes),
                        })
                        .collect();

                    // Snapshot the current guided_json schema once per inference.
                    let current_guided_json: Option<Arc<str>> =
                        guided_json.load_full().map(|schema| Arc::clone(&*schema));

                    let request = InferenceRequest {
                        model: Arc::clone(&config.first_pass_model),
                        prompt: Arc::clone(&prompt),
                        input_images,
                        input_videos: vec![],
                        // Allow more tokens and time for multi-frame analysis.
                        max_tokens: 256,
                        temperature: 0.0,
                        timeout_ms: 15_000,
                        allow_fallback: true,
                        guided_json: current_guided_json,
                    };

                    let clip_call_start = std::time::Instant::now();
                    let (description, used_second_pass) = match run_tiered(
                        provider.as_ref(),
                        &config,
                        request,
                        1024,
                        20_000,
                        observer.as_deref(),
                    ) {
                        Ok(output) => (output.result.output_text, output.used_second_pass),
                        Err(err) => {
                            if let Some(o) = observer.as_deref() {
                                o.record_error(
                                    provider.kind(),
                                    clip_call_start.elapsed().as_millis() as u64,
                                );
                            }
                            (format!("clip_vlm_error: {:?}", err.error), false)
                        }
                    };

                    if max_output_tokens_per_second > 0 {
                        let token_count = (description.len() / 4).max(1) as u32;
                        if let Some(entry) = token_budget.get_mut(work.session_id.as_ref()) {
                            entry.1 = entry.1.saturating_add(token_count);
                        }
                    }

                    let event_type = if used_second_pass {
                        "clip_vlm_tiered"
                    } else {
                        "clip_vlm"
                    };

                    // Use the last frame's signal for metadata.
                    let (last_signal, last_jpeg) =
                        work.frames.back().cloned().unwrap_or_else(|| {
                            (
                                FrameSignal {
                                    frame_index: 0,
                                    pts_ms: work.pts_end,
                                    perceptual_hash: 0,
                                    luma_mean: 0.0,
                                    flicker_score: 0.0,
                                    ghosting_score: 0.0,
                                    noise_variance_score: 0.0,
                                },
                                RecycledBytes::default(),
                            )
                        });

                    let _ = stdb.emit_event_sync(
                        &work.run_id,
                        &work.session_id,
                        last_signal.frame_index,
                        work.pts_end,
                        event_type,
                        0.9,
                        &description,
                    );
                    let _ = stdb.store_keyframe_sync(
                        &work.run_id,
                        last_signal.frame_index,
                        work.pts_end,
                        event_type,
                        &description,
                        &last_jpeg,
                    );
                }
            })
            .expect("clip vlm thread spawn failed");
    }
}

fn clip_vlm_worker_count(configured: usize) -> usize {
    per_stream_vlm_workers(configured)
}

fn downsample_clip_buffer(
    buffer: &mut VecDeque<(FrameSignal, RecycledBytes)>,
    window_start_pts: u64,
    window_ms: u64,
) {
    if buffer.len() <= MAX_CLIP_FRAMES_PER_REQUEST {
        return;
    }

    let last_slot = (MAX_CLIP_FRAMES_PER_REQUEST - 1) as u64;
    let mut slots: [Option<(FrameSignal, RecycledBytes)>; MAX_CLIP_FRAMES_PER_REQUEST] =
        std::array::from_fn(|_| None);

    let original_len = buffer.len();
    for (idx, frame) in buffer.drain(..).enumerate() {
        if idx == 0 {
            slots[0] = Some(frame);
            continue;
        }
        if idx + 1 == original_len {
            slots[MAX_CLIP_FRAMES_PER_REQUEST - 1] = Some(frame);
            continue;
        }

        let elapsed = frame
            .0
            .pts_ms
            .saturating_sub(window_start_pts)
            .min(window_ms);
        let mut slot = if window_ms == 0 {
            0
        } else {
            ((elapsed * last_slot + window_ms / 2) / window_ms) as usize
        };
        slot = slot.clamp(1, MAX_CLIP_FRAMES_PER_REQUEST - 2);
        let target_pts = if last_slot == 0 {
            window_start_pts
        } else {
            window_start_pts + ((slot as u64 * window_ms + last_slot / 2) / last_slot)
        };
        let new_distance = frame.0.pts_ms.abs_diff(target_pts);

        match &slots[slot] {
            Some((existing, _)) if existing.pts_ms.abs_diff(target_pts) <= new_distance => {}
            _ => slots[slot] = Some(frame),
        }
    }

    buffer.extend(slots.into_iter().flatten());
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::FrameSignal;
    use crate::webrtc::workers::StreamFrame;

    fn make_frame(seq: u64, pts_ms: u64) -> StreamFrame {
        StreamFrame {
            signal: FrameSignal {
                frame_index: seq,
                pts_ms,
                perceptual_hash: seq.wrapping_mul(0xDEAD_BEEF),
                luma_mean: 0.5,
                flicker_score: 0.0,
                ghosting_score: 0.0,
                noise_variance_score: 0.0,
            },
            jpeg: Some([0xff_u8, 0xd8, 0xaa, 0xbb, 0xff, 0xd9].into()),
            pts_ms,
            seq,
        }
    }

    #[test]
    fn clip_accumulator_caps_retained_frames_across_full_window() {
        let cfg = ClipConfig {
            target_fps: 30,
            clip_length_seconds: MAX_CLIP_LENGTH_SECONDS as f32,
            delay_seconds: 0.0,
        };
        let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());
        let mut clip = None;

        for i in 0..=1800_u64 {
            clip = acc.push(make_frame(i, i * 34)).or(clip);
            if clip.is_some() {
                break;
            }
        }

        let clip = clip.expect("long clip should emit");
        assert_eq!(clip.frames.len(), MAX_CLIP_FRAMES_PER_REQUEST);
        assert!(
            clip.frames.front().unwrap().0.pts_ms <= 100,
            "first retained frame should stay near the window start, got {}ms",
            clip.frames.front().unwrap().0.pts_ms
        );
        assert_eq!(clip.frames.back().unwrap().0.pts_ms, clip.pts_end);
    }

    #[test]
    fn clip_timestamps_match_retained_first_and_last_frames_after_downsample() {
        let mut buffer = VecDeque::new();
        for i in 0..63_u64 {
            let mut frame = make_frame(i, i * 900);
            buffer.push_back((frame.signal, frame.jpeg.take().unwrap()));
        }
        let mut near_window_end = make_frame(63, 60_000);
        buffer.push_back((near_window_end.signal, near_window_end.jpeg.take().unwrap()));
        let mut triggering = make_frame(64, 60_900);
        buffer.push_back((triggering.signal, triggering.jpeg.take().unwrap()));

        downsample_clip_buffer(&mut buffer, 0, 60_000);

        let pts_start = buffer.front().map(|(s, _)| s.pts_ms).unwrap();
        let pts_end = buffer.back().map(|(s, _)| s.pts_ms).unwrap();

        assert_eq!(pts_start, 0);
        assert_eq!(pts_end, 60_900);
        assert_eq!(buffer.front().unwrap().0.pts_ms, pts_start);
        assert_eq!(buffer.back().unwrap().0.pts_ms, pts_end);
    }

    #[test]
    fn clip_vlm_workers_are_clamped_to_one_per_stream() {
        assert_eq!(clip_vlm_worker_count(0), 1);
        assert_eq!(clip_vlm_worker_count(1), 1);
        assert_eq!(clip_vlm_worker_count(8), 1);
    }

    // ── ClipConfig validation ──────────────────────────────────────────────

    #[test]
    fn default_config_is_valid() {
        assert!(ClipConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_target_fps_out_of_range() {
        let zero_fps = ClipConfig {
            target_fps: 0,
            ..ClipConfig::default()
        };
        assert!(zero_fps.validate().is_err());

        let high_fps = ClipConfig {
            target_fps: 31,
            ..ClipConfig::default()
        };
        assert!(high_fps.validate().is_err());
    }

    #[test]
    fn rejects_clip_length_too_short() {
        let cfg = ClipConfig {
            clip_length_seconds: 0.01,
            ..ClipConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_min_frame_constraint() {
        // target_fps=1, clip_length=0.1 → 0.1 < 3 frames
        let cfg = ClipConfig {
            target_fps: 1,
            clip_length_seconds: 0.1,
            delay_seconds: 0.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_boundary_frame_count() {
        // target_fps=3, clip_length=1.0 → 3 frames — exactly at minimum
        let cfg = ClipConfig {
            target_fps: 3,
            clip_length_seconds: 1.0,
            delay_seconds: 0.0,
        };
        assert!(cfg.validate().is_ok());
    }

    // ── ClipAccumulator batching ───────────────────────────────────────────

    #[test]
    fn emits_clip_after_window_fills() {
        // target_fps=10 → sample_interval=100ms, clip_window=500ms
        let cfg = ClipConfig {
            target_fps: 10,
            clip_length_seconds: 0.5,
            delay_seconds: 0.0, // no delay so first window triggers immediately
        };
        let mut acc = ClipAccumulator::new(cfg, "run1".into(), "sess1".into(), "describe".into());

        // Send frames at 10 fps (100ms apart), for 600ms → 7 frames.
        // Window requires 500ms elapsed since first frame.
        let mut result = None;
        for i in 0..7u64 {
            result = acc.push(make_frame(i, i * 100));
            if result.is_some() {
                break;
            }
        }

        let clip = result.expect("should have emitted a clip");
        assert!(!clip.frames.is_empty(), "clip must contain frames");
        assert!(clip.pts_end >= clip.pts_start);
        assert_eq!(&*clip.run_id, "run1");
        assert_eq!(&*clip.session_id, "sess1");
    }

    #[test]
    fn rate_limits_to_target_fps() {
        // target_fps=1 → sample_interval=1000ms
        let cfg = ClipConfig {
            target_fps: 1,
            clip_length_seconds: 5.0, // 1 fps * 5s = 5 frames
            delay_seconds: 0.0,
        };
        let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

        // Send 10 frames at 500ms apart — only every other one should be accepted.
        for i in 0..10u64 {
            acc.push(make_frame(i, i * 500));
        }
        // At 500ms spacing with 1000ms interval, only frames at 0, 1000, 2000, 3000, 4000ms
        // get accepted (5 frames). Window = 5s, but pts_end at 4000ms < 5000ms, so no emit.
        // Final frame at i=9 → pts=4500ms still < 5000ms.
        // Actually window needs 5000ms elapsed. At i=9 pts=4500, first accepted pts=0.
        // 4500 - 0 = 4500 < 5000. No emit yet.

        // Send frame at 5100ms: accepted (5100-4000=1100 >= 1000), window elapsed = 5100-0 = 5100 >= 5000
        let result = acc.push(make_frame(10, 5100));
        assert!(result.is_some(), "should emit after 5s of window at 1fps");
        let clip = result.unwrap();
        // Should have exactly the accepted frames: 0, 1000, 2000, 3000, 4000, 5100 → 6 frames
        assert!(
            clip.frames.len() >= 5,
            "expected at least 5 frames, got {}",
            clip.frames.len()
        );
    }

    #[test]
    fn drops_frames_without_jpeg() {
        let cfg = ClipConfig {
            target_fps: 30,
            clip_length_seconds: 0.1,
            delay_seconds: 0.0,
        };
        let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

        let mut no_jpeg = make_frame(0, 0);
        no_jpeg.jpeg = None;
        let result = acc.push(no_jpeg);
        assert!(result.is_none());
    }

    #[test]
    fn no_emit_before_window_elapsed() {
        let cfg = ClipConfig {
            target_fps: 10,
            clip_length_seconds: 2.0, // 2 seconds window
            delay_seconds: 0.0,
        };
        let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

        // Send 10 frames at 100ms each → 1 second elapsed, below 2s window
        for i in 0..10u64 {
            let r = acc.push(make_frame(i, i * 100));
            assert!(r.is_none(), "should not emit before window fills");
        }
    }

    #[test]
    fn clip_work_contains_run_and_session_ids() {
        let cfg = ClipConfig {
            target_fps: 10,
            clip_length_seconds: 0.5,
            delay_seconds: 0.0,
        };
        let mut acc = ClipAccumulator::new(
            cfg,
            "my-run".into(),
            "my-session".into(),
            "test prompt".into(),
        );

        let mut clip = None;
        for i in 0..10u64 {
            clip = acc.push(make_frame(i, i * 100));
            if clip.is_some() {
                break;
            }
        }

        let c = clip.expect("clip should be emitted");
        assert_eq!(&*c.run_id, "my-run");
        assert_eq!(&*c.session_id, "my-session");
        assert_eq!(&*c.prompt, "test prompt");
    }
}

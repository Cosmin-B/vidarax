pub const GATE_KEEPALIVE_EVERY_FRAMES: u64 = 30;
pub const GATE_SCENE_CUT_HAMMING_THRESHOLD: u32 = 18;
pub const GATE_LUMA_SHIFT_THRESHOLD: f32 = 0.15;
pub const GATE_FLICKER_THRESHOLD: f32 = 0.55;
pub const GATE_GHOSTING_THRESHOLD: f32 = 0.55;
pub const GATE_NOISE_VARIANCE_THRESHOLD: f32 = 0.55;

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub keepalive_every_frames: u64,
    pub scene_cut_hamming_threshold: u32,
    pub luma_shift_threshold: f32,
    pub flicker_threshold: f32,
    pub ghosting_threshold: f32,
    pub noise_variance_threshold: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            keepalive_every_frames: GATE_KEEPALIVE_EVERY_FRAMES,
            scene_cut_hamming_threshold: GATE_SCENE_CUT_HAMMING_THRESHOLD,
            luma_shift_threshold: GATE_LUMA_SHIFT_THRESHOLD,
            flicker_threshold: GATE_FLICKER_THRESHOLD,
            ghosting_threshold: GATE_GHOSTING_THRESHOLD,
            noise_variance_threshold: GATE_NOISE_VARIANCE_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameSignal {
    pub frame_index: u64,
    pub pts_ms: u64,
    pub perceptual_hash: u64,
    pub luma_mean: f32,
    pub flicker_score: f32,
    pub ghosting_score: f32,
    pub noise_variance_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateEventType {
    KeepKeyframe,
    SuspectArtifact,
    Skip,
}

/// Reason a gate decision was made.
///
/// Replaces `&'static str` on [`GateEvent`], shrinking the struct from
/// 48 bytes to 32 bytes and enabling exhaustive `match` without string
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReasonCode {
    InitialFrame,
    SceneCut,
    PeriodicKeepalive,
    ExposureShift,
    FlickerSuspected,
    GhostingSuspected,
    NoiseVarianceSpike,
    NoTrigger,
}

impl GateReasonCode {
    /// Return the canonical string label for this reason code.
    ///
    /// Preserves the same byte sequence as the old `&'static str` field so
    /// that external consumers (schemas, FNV hashes, etc.) remain compatible.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InitialFrame => "initial_frame",
            Self::SceneCut => "scene_cut",
            Self::PeriodicKeepalive => "periodic_keepalive",
            Self::ExposureShift => "exposure_shift",
            Self::FlickerSuspected => "flicker_suspected",
            Self::GhostingSuspected => "ghosting_suspected",
            Self::NoiseVarianceSpike => "noise_variance_spike",
            Self::NoTrigger => "no_trigger",
        }
    }
}

/// Gate decision for a single frame.
///
/// Size: 32 bytes (was 48 bytes with `&'static str` reason_code).
#[derive(Debug, Clone, PartialEq)]
pub struct GateEvent {
    pub event_type: GateEventType,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub confidence: f32,
    pub reason_code: GateReasonCode,
}

pub struct GateEngine {
    config: GateConfig,
    initialized: bool,
    last_kept_frame_index: u64,
    last_kept_hash: u64,
    last_kept_luma: f32,
}

impl GateEngine {
    pub fn new(config: GateConfig) -> Self {
        Self {
            config,
            initialized: false,
            last_kept_frame_index: 0,
            last_kept_hash: 0,
            last_kept_luma: 0.0,
        }
    }

    /// Evaluate `s` against all gate conditions and return the highest-priority
    /// triggered event.
    ///
    /// Uses a branchless bitmask + `trailing_zeros()` dispatch: each condition
    /// is evaluated as a `u8` bool, packed into a bitmask (bit 0 = highest
    /// priority), and `trailing_zeros()` selects the winning condition index.
    /// Bit 7 is always set as the "no trigger" sentinel so the result is never
    /// out-of-bounds.
    pub fn process(&mut self, s: FrameSignal) -> GateEvent {
        // Pre-read state before any mutation.
        let hash_distance = (self.last_kept_hash ^ s.perceptual_hash).count_ones();
        let frames_since_keep = s.frame_index.saturating_sub(self.last_kept_frame_index);
        let luma_shift = (self.last_kept_luma - s.luma_mean).abs();

        // Each bit encodes one trigger condition; lower bit = higher priority.
        // Bit 7 is always set so trailing_zeros() always yields a valid index.
        let mask: u8 = (!self.initialized as u8)
            | ((hash_distance >= self.config.scene_cut_hamming_threshold) as u8) << 1
            | ((frames_since_keep >= self.config.keepalive_every_frames) as u8) << 2
            | ((luma_shift >= self.config.luma_shift_threshold) as u8) << 3
            | ((s.flicker_score >= self.config.flicker_threshold) as u8) << 4
            | ((s.ghosting_score >= self.config.ghosting_threshold) as u8) << 5
            | ((s.noise_variance_score >= self.config.noise_variance_threshold) as u8) << 6
            | 0x80; // sentinel: NoTrigger at index 7

        let idx = mask.trailing_zeros() as usize;

        static EVENT_TYPES: [GateEventType; 8] = [
            GateEventType::KeepKeyframe,    // 0: initial_frame
            GateEventType::KeepKeyframe,    // 1: scene_cut
            GateEventType::KeepKeyframe,    // 2: periodic_keepalive
            GateEventType::SuspectArtifact, // 3: exposure_shift
            GateEventType::SuspectArtifact, // 4: flicker_suspected
            GateEventType::SuspectArtifact, // 5: ghosting_suspected
            GateEventType::SuspectArtifact, // 6: noise_variance_spike
            GateEventType::Skip,            // 7: no_trigger
        ];

        static REASONS: [GateReasonCode; 8] = [
            GateReasonCode::InitialFrame,
            GateReasonCode::SceneCut,
            GateReasonCode::PeriodicKeepalive,
            GateReasonCode::ExposureShift,
            GateReasonCode::FlickerSuspected,
            GateReasonCode::GhostingSuspected,
            GateReasonCode::NoiseVarianceSpike,
            GateReasonCode::NoTrigger,
        ];

        // Confidence is data-dependent; compute per-slot and index in.
        let confidences: [f32; 8] = [
            1.0,                                              // 0: initial_frame
            (hash_distance as f32 / 64.0).clamp(0.0, 1.0),  // 1: scene_cut
            1.0,                                              // 2: periodic_keepalive
            luma_shift.clamp(0.0, 1.0),                      // 3: exposure_shift
            s.flicker_score.clamp(0.0, 1.0),                 // 4: flicker_suspected
            s.ghosting_score.clamp(0.0, 1.0),                // 5: ghosting_suspected
            s.noise_variance_score.clamp(0.0, 1.0),          // 6: noise_variance_spike
            0.0,                                              // 7: no_trigger
        ];

        // Capture keyframe for the three KeepKeyframe conditions (indices 0–2).
        if idx < 3 {
            self.capture_keyframe(s);
        }

        GateEvent {
            event_type: EVENT_TYPES[idx],
            frame_index: s.frame_index,
            pts_ms: s.pts_ms,
            confidence: confidences[idx],
            reason_code: REASONS[idx],
        }
    }

    fn capture_keyframe(&mut self, s: FrameSignal) {
        self.initialized = true;
        self.last_kept_frame_index = s.frame_index;
        self.last_kept_hash = s.perceptual_hash;
        self.last_kept_luma = s.luma_mean;
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameSignal, GateConfig, GateEngine, GateEventType, GateReasonCode};

    fn s(frame: u64, hash: u64) -> FrameSignal {
        FrameSignal {
            frame_index: frame,
            pts_ms: frame * 33,
            perceptual_hash: hash,
            luma_mean: 0.4,
            flicker_score: 0.0,
            ghosting_score: 0.0,
            noise_variance_score: 0.0,
        }
    }

    // Reference (if-return) implementation used only for parity testing.
    // Accesses private fields through the cfg(test) impl block.
    impl GateEngine {
        fn process_reference(&mut self, s: FrameSignal) -> super::GateEvent {
            use super::{GateEvent, GateEventType, GateReasonCode};
            if !self.initialized {
                self.capture_keyframe(s);
                return GateEvent {
                    event_type: GateEventType::KeepKeyframe,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: 1.0,
                    reason_code: GateReasonCode::InitialFrame,
                };
            }
            let hash_distance = (self.last_kept_hash ^ s.perceptual_hash).count_ones();
            if hash_distance >= self.config.scene_cut_hamming_threshold {
                self.capture_keyframe(s);
                return GateEvent {
                    event_type: GateEventType::KeepKeyframe,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: (hash_distance as f32 / 64.0).clamp(0.0, 1.0),
                    reason_code: GateReasonCode::SceneCut,
                };
            }
            let frames_since_keep = s.frame_index.saturating_sub(self.last_kept_frame_index);
            if frames_since_keep >= self.config.keepalive_every_frames {
                self.capture_keyframe(s);
                return GateEvent {
                    event_type: GateEventType::KeepKeyframe,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: 1.0,
                    reason_code: GateReasonCode::PeriodicKeepalive,
                };
            }
            let luma_shift = (self.last_kept_luma - s.luma_mean).abs();
            if luma_shift >= self.config.luma_shift_threshold {
                return GateEvent {
                    event_type: GateEventType::SuspectArtifact,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: luma_shift.clamp(0.0, 1.0),
                    reason_code: GateReasonCode::ExposureShift,
                };
            }
            if s.flicker_score >= self.config.flicker_threshold {
                return GateEvent {
                    event_type: GateEventType::SuspectArtifact,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: s.flicker_score.clamp(0.0, 1.0),
                    reason_code: GateReasonCode::FlickerSuspected,
                };
            }
            if s.ghosting_score >= self.config.ghosting_threshold {
                return GateEvent {
                    event_type: GateEventType::SuspectArtifact,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: s.ghosting_score.clamp(0.0, 1.0),
                    reason_code: GateReasonCode::GhostingSuspected,
                };
            }
            if s.noise_variance_score >= self.config.noise_variance_threshold {
                return GateEvent {
                    event_type: GateEventType::SuspectArtifact,
                    frame_index: s.frame_index,
                    pts_ms: s.pts_ms,
                    confidence: s.noise_variance_score.clamp(0.0, 1.0),
                    reason_code: GateReasonCode::NoiseVarianceSpike,
                };
            }
            GateEvent {
                event_type: GateEventType::Skip,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: 0.0,
                reason_code: GateReasonCode::NoTrigger,
            }
        }
    }

    #[test]
    fn keeps_initial_frame() {
        let mut gate = GateEngine::new(GateConfig::default());
        let ev = gate.process(s(0, 0xAAAA));
        assert_eq!(ev.event_type, GateEventType::KeepKeyframe);
        assert_eq!(ev.reason_code, GateReasonCode::InitialFrame);
    }

    #[test]
    fn triggers_scene_cut_on_hash_jump() {
        let cfg = GateConfig {
            scene_cut_hamming_threshold: 4,
            ..GateConfig::default()
        };
        let mut gate = GateEngine::new(cfg);
        gate.process(s(0, 0));
        let ev = gate.process(s(1, u64::MAX));
        assert_eq!(ev.event_type, GateEventType::KeepKeyframe);
        assert_eq!(ev.reason_code, GateReasonCode::SceneCut);
    }

    #[test]
    fn triggers_keepalive() {
        let cfg = GateConfig {
            keepalive_every_frames: 2,
            ..GateConfig::default()
        };
        let mut gate = GateEngine::new(cfg);
        gate.process(s(0, 123));
        assert_eq!(gate.process(s(1, 123)).event_type, GateEventType::Skip);
        let ev = gate.process(s(2, 123));
        assert_eq!(ev.event_type, GateEventType::KeepKeyframe);
        assert_eq!(ev.reason_code, GateReasonCode::PeriodicKeepalive);
    }

    #[test]
    fn flags_temporal_artifact_signal() {
        let mut gate = GateEngine::new(GateConfig::default());
        gate.process(s(0, 1));
        let mut sample = s(1, 1);
        sample.flicker_score = 0.8;
        let ev = gate.process(sample);
        assert_eq!(ev.event_type, GateEventType::SuspectArtifact);
        assert_eq!(ev.reason_code, GateReasonCode::FlickerSuspected);
    }

    /// Parity test: branchless `process()` must produce identical output to
    /// the reference if-return implementation for every trigger path.
    #[test]
    fn branchless_matches_reference_for_all_trigger_paths() {
        let cfg = GateConfig {
            keepalive_every_frames: 3,
            scene_cut_hamming_threshold: 8,
            luma_shift_threshold: 0.2,
            flicker_threshold: 0.6,
            ghosting_threshold: 0.6,
            noise_variance_threshold: 0.6,
        };

        // Each entry is a sequence of (frame_index, hash, luma, flicker, ghosting, noise)
        // tuples fed to both engines.  The last signal in each sequence exercises the
        // named trigger path.
        let sequences: &[(&str, &[(u64, u64, f32, f32, f32, f32)])] = &[
            ("InitialFrame",       &[(0, 0, 0.5, 0.0, 0.0, 0.0)]),
            ("NoTrigger",          &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, 0, 0.5, 0.0, 0.0, 0.0)]),
            ("SceneCut",           &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, u64::MAX, 0.5, 0.0, 0.0, 0.0)]),
            ("PeriodicKeepalive",  &[(0, 0, 0.5, 0.0, 0.0, 0.0), (3, 0, 0.5, 0.0, 0.0, 0.0)]),
            ("ExposureShift",      &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, 0, 0.8, 0.0, 0.0, 0.0)]),
            ("FlickerSuspected",   &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, 0, 0.5, 0.9, 0.0, 0.0)]),
            ("GhostingSuspected",  &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, 0, 0.5, 0.0, 0.9, 0.0)]),
            ("NoiseVarianceSpike", &[(0, 0, 0.5, 0.0, 0.0, 0.0), (1, 0, 0.5, 0.0, 0.0, 0.7)]),
        ];

        for (label, seq) in sequences {
            let mut branchless = GateEngine::new(cfg.clone());
            let mut reference = GateEngine::new(cfg.clone());

            for &(frame_index, hash, luma_mean, flicker_score, ghosting_score, noise_variance_score) in *seq {
                let sig = FrameSignal {
                    frame_index,
                    pts_ms: frame_index * 33,
                    perceptual_hash: hash,
                    luma_mean,
                    flicker_score,
                    ghosting_score,
                    noise_variance_score,
                };
                let br_ev = branchless.process(sig);
                let ref_ev = reference.process_reference(sig);
                assert_eq!(
                    br_ev, ref_ev,
                    "path '{label}' frame={frame_index}: branchless={br_ev:?} vs reference={ref_ev:?}"
                );
            }
        }
    }
}

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
            keepalive_every_frames: 30,
            scene_cut_hamming_threshold: 18,
            luma_shift_threshold: 0.15,
            flicker_threshold: 0.55,
            ghosting_threshold: 0.55,
            noise_variance_threshold: 0.55,
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

#[derive(Debug, Clone, PartialEq)]
pub struct GateEvent {
    pub event_type: GateEventType,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub confidence: f32,
    pub reason_code: &'static str,
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

    pub fn process(&mut self, s: FrameSignal) -> GateEvent {
        if !self.initialized {
            self.capture_keyframe(s);
            return GateEvent {
                event_type: GateEventType::KeepKeyframe,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: 1.0,
                reason_code: "initial_frame",
            };
        }

        let hash_distance = (self.last_kept_hash ^ s.perceptual_hash).count_ones();
        if hash_distance >= self.config.scene_cut_hamming_threshold {
            self.capture_keyframe(s);
            return GateEvent {
                event_type: GateEventType::KeepKeyframe,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: normalize(hash_distance as f32 / 64.0),
                reason_code: "scene_cut",
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
                reason_code: "periodic_keepalive",
            };
        }

        let luma_shift = (self.last_kept_luma - s.luma_mean).abs();
        if luma_shift >= self.config.luma_shift_threshold {
            return GateEvent {
                event_type: GateEventType::SuspectArtifact,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: normalize(luma_shift),
                reason_code: "exposure_shift",
            };
        }

        if s.flicker_score >= self.config.flicker_threshold {
            return GateEvent {
                event_type: GateEventType::SuspectArtifact,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: normalize(s.flicker_score),
                reason_code: "flicker_suspected",
            };
        }

        if s.ghosting_score >= self.config.ghosting_threshold {
            return GateEvent {
                event_type: GateEventType::SuspectArtifact,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: normalize(s.ghosting_score),
                reason_code: "ghosting_suspected",
            };
        }

        if s.noise_variance_score >= self.config.noise_variance_threshold {
            return GateEvent {
                event_type: GateEventType::SuspectArtifact,
                frame_index: s.frame_index,
                pts_ms: s.pts_ms,
                confidence: normalize(s.noise_variance_score),
                reason_code: "noise_variance_spike",
            };
        }

        GateEvent {
            event_type: GateEventType::Skip,
            frame_index: s.frame_index,
            pts_ms: s.pts_ms,
            confidence: 0.0,
            reason_code: "no_trigger",
        }
    }

    fn capture_keyframe(&mut self, s: FrameSignal) {
        self.initialized = true;
        self.last_kept_frame_index = s.frame_index;
        self.last_kept_hash = s.perceptual_hash;
        self.last_kept_luma = s.luma_mean;
    }
}

#[inline]
fn normalize(v: f32) -> f32 {
    if v.is_nan() || v.is_sign_negative() {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameSignal, GateConfig, GateEngine, GateEventType};

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

    #[test]
    fn keeps_initial_frame() {
        let mut gate = GateEngine::new(GateConfig::default());
        let ev = gate.process(s(0, 0xAAAA));
        assert_eq!(ev.event_type, GateEventType::KeepKeyframe);
        assert_eq!(ev.reason_code, "initial_frame");
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
        assert_eq!(ev.reason_code, "scene_cut");
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
        assert_eq!(ev.reason_code, "periodic_keepalive");
    }

    #[test]
    fn flags_temporal_artifact_signal() {
        let mut gate = GateEngine::new(GateConfig::default());
        gate.process(s(0, 1));
        let mut sample = s(1, 1);
        sample.flicker_score = 0.8;
        let ev = gate.process(sample);
        assert_eq!(ev.event_type, GateEventType::SuspectArtifact);
        assert_eq!(ev.reason_code, "flicker_suspected");
    }
}

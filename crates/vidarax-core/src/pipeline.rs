use crate::gate::{FrameSignal, GateConfig, GateEngine, GateEventType};

#[derive(Debug, Clone, Copy)]
pub struct TwoPassConfig {
    pub window_size: usize,
    pub segment_ms: u64,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            window_size: 16,
            segment_ms: 250,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMetadata {
    pub frame_index: u64,
    pub pts_ms: u64,
    pub gate_event: GateEventType,
    pub scene_cut: bool,
    pub suspect_artifact: bool,
    pub novelty_score: f32,
    pub temporal_stability: f32,
    pub motion_score: f32,
    pub confidence: f32,
    pub segment_start_ms: u64,
    pub segment_end_ms: u64,
}

pub struct TwoPassPipeline {
    gate: GateEngine,
    config: TwoPassConfig,
    window: Vec<WindowSample>,
    window_len: usize,
    cursor: usize,
    previous_hash: Option<u64>,
}

impl TwoPassPipeline {
    pub fn new(config: TwoPassConfig, gate_config: GateConfig) -> Self {
        let window_size = config.window_size.max(2);
        Self {
            gate: GateEngine::new(gate_config),
            config: TwoPassConfig {
                window_size,
                segment_ms: config.segment_ms.max(1),
            },
            window: vec![WindowSample::default(); window_size],
            window_len: 0,
            cursor: 0,
            previous_hash: None,
        }
    }

    pub fn analyze_batch(&mut self, frames: &[FrameSignal]) -> Vec<FrameMetadata> {
        // Pass 1: compute deterministic gate events first.
        let mut pass1 = Vec::with_capacity(frames.len());
        for frame in frames {
            pass1.push(self.gate.process(*frame));
        }

        // Pass 2: derive contextual metadata from a bounded sliding window.
        let mut out = Vec::with_capacity(frames.len());
        for (frame, gate_event) in frames.iter().zip(pass1.iter()) {
            let novelty = self.novelty_against_window(frame.perceptual_hash);
            let stability = self.temporal_stability(frame.luma_mean);
            let motion = self.motion_score(frame.perceptual_hash);
            let confidence = normalize(0.45 * novelty + 0.35 * (1.0 - stability) + 0.20 * motion);

            let segment_start_ms = (frame.pts_ms / self.config.segment_ms) * self.config.segment_ms;
            let segment_end_ms = segment_start_ms + self.config.segment_ms;
            out.push(FrameMetadata {
                frame_index: frame.frame_index,
                pts_ms: frame.pts_ms,
                gate_event: gate_event.event_type,
                scene_cut: gate_event.reason_code == "scene_cut",
                suspect_artifact: gate_event.event_type == GateEventType::SuspectArtifact,
                novelty_score: novelty,
                temporal_stability: stability,
                motion_score: motion,
                confidence,
                segment_start_ms,
                segment_end_ms,
            });

            self.push_window(frame.perceptual_hash, frame.luma_mean);
            self.previous_hash = Some(frame.perceptual_hash);
        }

        out
    }

    fn novelty_against_window(&self, hash: u64) -> f32 {
        if self.window_len == 0 {
            return 1.0;
        }
        let mut total = 0.0f32;
        for sample in self.window.iter().take(self.window_len) {
            total += hamming_similarity(hash, sample.perceptual_hash);
        }
        normalize(total / self.window_len as f32)
    }

    fn temporal_stability(&self, luma_mean: f32) -> f32 {
        if self.window_len == 0 {
            return 0.0;
        }
        let mut drift = 0.0f32;
        for sample in self.window.iter().take(self.window_len) {
            drift += (sample.luma_mean - luma_mean).abs();
        }
        normalize(drift / self.window_len as f32)
    }

    fn motion_score(&self, hash: u64) -> f32 {
        self.previous_hash
            .map(|prev| hamming_similarity(prev, hash))
            .unwrap_or(0.0)
    }

    fn push_window(&mut self, perceptual_hash: u64, luma_mean: f32) {
        self.window[self.cursor] = WindowSample {
            perceptual_hash,
            luma_mean,
        };
        if self.window_len < self.window.len() {
            self.window_len += 1;
        }
        self.cursor = (self.cursor + 1) % self.window.len();
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WindowSample {
    perceptual_hash: u64,
    luma_mean: f32,
}

#[inline]
fn hamming_similarity(a: u64, b: u64) -> f32 {
    ((a ^ b).count_ones() as f32) / 64.0
}

#[inline]
fn normalize(value: f32) -> f32 {
    if !value.is_finite() || value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameMetadata, TwoPassConfig, TwoPassPipeline};
    use crate::gate::{FrameSignal, GateConfig, GateEventType};

    fn frame(frame_index: u64, hash: u64, luma: f32) -> FrameSignal {
        FrameSignal {
            frame_index,
            pts_ms: frame_index * 33,
            perceptual_hash: hash,
            luma_mean: luma,
            flicker_score: 0.0,
            ghosting_score: 0.0,
            noise_variance_score: 0.0,
        }
    }

    #[test]
    fn produces_deterministic_metadata_shape() {
        let mut p = TwoPassPipeline::new(
            TwoPassConfig {
                window_size: 4,
                segment_ms: 250,
            },
            GateConfig::default(),
        );
        let out = p.analyze_batch(&[frame(0, 0xAAAA, 0.2), frame(1, 0xBBBB, 0.4)]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].gate_event, GateEventType::KeepKeyframe);
        assert!(out[0].segment_end_ms >= out[0].segment_start_ms);
    }

    #[test]
    fn detects_scene_cut_and_scores_motion() {
        let mut p = TwoPassPipeline::new(
            TwoPassConfig::default(),
            GateConfig {
                scene_cut_hamming_threshold: 4,
                ..GateConfig::default()
            },
        );
        let out = p.analyze_batch(&[frame(0, 0, 0.2), frame(1, u64::MAX, 0.2)]);
        assert_eq!(out[1].gate_event, GateEventType::KeepKeyframe);
        assert!(out[1].scene_cut);
        assert!(out[1].motion_score > 0.5);
    }

    #[test]
    fn metadata_values_are_normalized() {
        let mut p = TwoPassPipeline::new(TwoPassConfig::default(), GateConfig::default());
        let out: Vec<FrameMetadata> = p.analyze_batch(&[frame(0, 1, 0.0), frame(1, 2, 1.0)]);
        for m in out {
            assert!((0.0..=1.0).contains(&m.novelty_score));
            assert!((0.0..=1.0).contains(&m.temporal_stability));
            assert!((0.0..=1.0).contains(&m.motion_score));
            assert!((0.0..=1.0).contains(&m.confidence));
        }
    }
}

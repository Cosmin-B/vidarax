use crate::gate::{FrameSignal, GateConfig, GateEngine, GateEvent, GateEventType, GateReasonCode};

pub const TWO_PASS_CONFIDENCE_NOVELTY_WEIGHT: f32 = 0.45;
pub const TWO_PASS_CONFIDENCE_INSTABILITY_WEIGHT: f32 = 0.35;
pub const TWO_PASS_CONFIDENCE_MOTION_WEIGHT: f32 = 0.20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPassWeights {
    pub novelty: f32,
    pub instability: f32,
    pub motion: f32,
}

impl Default for TwoPassWeights {
    fn default() -> Self {
        Self {
            novelty: TWO_PASS_CONFIDENCE_NOVELTY_WEIGHT,
            instability: TWO_PASS_CONFIDENCE_INSTABILITY_WEIGHT,
            motion: TWO_PASS_CONFIDENCE_MOTION_WEIGHT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TwoPassConfig {
    pub window_size: usize,
    pub segment_ms: u64,
    pub confidence_weights: TwoPassWeights,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            window_size: 16,
            segment_ms: 250,
            confidence_weights: TwoPassWeights::default(),
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
    pass1_buf: Vec<GateEvent>,
    out_buf: Vec<FrameMetadata>,
}

impl TwoPassPipeline {
    pub fn new(config: TwoPassConfig, gate_config: GateConfig) -> Self {
        let window_size = config.window_size.max(2);
        Self {
            gate: GateEngine::new(gate_config),
            config: TwoPassConfig {
                window_size,
                segment_ms: config.segment_ms.max(1),
                confidence_weights: config.confidence_weights,
            },
            window: vec![WindowSample::default(); window_size],
            window_len: 0,
            cursor: 0,
            previous_hash: None,
            pass1_buf: Vec::new(),
            out_buf: Vec::new(),
        }
    }

    pub fn analyze_batch(&mut self, frames: &[FrameSignal]) -> &[FrameMetadata] {
        // Pass 1: compute deterministic gate events; reuse allocation.
        self.pass1_buf.clear();
        for frame in frames {
            self.pass1_buf.push(self.gate.process(*frame));
        }

        // Pass 2: derive contextual metadata from a bounded sliding window.
        // Index-based loop avoids holding a borrow on pass1_buf across mutable
        // calls to window_metrics / push_window.
        self.out_buf.clear();
        for i in 0..frames.len() {
            let frame = &frames[i];
            // Both fields are Copy enums — no reference retained.
            let event_type = self.pass1_buf[i].event_type;
            let reason_code = self.pass1_buf[i].reason_code;

            let (novelty, stability) = self.window_metrics(frame.perceptual_hash, frame.luma_mean);
            let motion = self.motion_score(frame.perceptual_hash);
            let weights = self.config.confidence_weights;
            let confidence = (weights.novelty * novelty
                + weights.instability * (1.0 - stability)
                + weights.motion * motion)
                .clamp(0.0, 1.0);

            let segment_start_ms = (frame.pts_ms / self.config.segment_ms) * self.config.segment_ms;
            let segment_end_ms = segment_start_ms + self.config.segment_ms;
            self.out_buf.push(FrameMetadata {
                frame_index: frame.frame_index,
                pts_ms: frame.pts_ms,
                gate_event: event_type,
                scene_cut: reason_code == GateReasonCode::SceneCut,
                suspect_artifact: event_type == GateEventType::SuspectArtifact,
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

        &self.out_buf
    }

    /// Fused single-pass over the sliding window: computes both novelty
    /// (hamming similarity) and temporal stability (luma drift) in one loop.
    fn window_metrics(&self, hash: u64, luma_mean: f32) -> (f32, f32) {
        if self.window_len == 0 {
            return (1.0, 0.0);
        }
        let mut similarity = 0.0f32;
        let mut drift = 0.0f32;
        for sample in self.window.iter().take(self.window_len) {
            similarity += hamming_similarity(hash, sample.perceptual_hash);
            drift += (sample.luma_mean - luma_mean).abs();
        }
        let inv = 1.0 / self.window_len as f32;
        ((similarity * inv).clamp(0.0, 1.0), (drift * inv).clamp(0.0, 1.0))
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

#[cfg(test)]
mod tests {
    use super::{
        TwoPassConfig, TwoPassPipeline, TwoPassWeights,
        TWO_PASS_CONFIDENCE_INSTABILITY_WEIGHT, TWO_PASS_CONFIDENCE_MOTION_WEIGHT,
        TWO_PASS_CONFIDENCE_NOVELTY_WEIGHT,
    };
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
                confidence_weights: Default::default(),
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
        let out = p.analyze_batch(&[frame(0, 1, 0.0), frame(1, 2, 1.0)]);
        for m in out {
            assert!((0.0..=1.0).contains(&m.novelty_score));
            assert!((0.0..=1.0).contains(&m.temporal_stability));
            assert!((0.0..=1.0).contains(&m.motion_score));
            assert!((0.0..=1.0).contains(&m.confidence));
        }
    }

    #[test]
    fn default_confidence_weights_are_named_and_preserved() {
        let weights = TwoPassWeights::default();

        assert_eq!(weights.novelty, TWO_PASS_CONFIDENCE_NOVELTY_WEIGHT);
        assert_eq!(weights.instability, TWO_PASS_CONFIDENCE_INSTABILITY_WEIGHT);
        assert_eq!(weights.motion, TWO_PASS_CONFIDENCE_MOTION_WEIGHT);
        assert_eq!(weights.novelty, 0.45);
        assert_eq!(weights.instability, 0.35);
        assert_eq!(weights.motion, 0.20);
    }

    #[test]
    fn confidence_weights_are_configurable() {
        let mut p = TwoPassPipeline::new(
            TwoPassConfig {
                window_size: 4,
                segment_ms: 250,
                confidence_weights: TwoPassWeights {
                    novelty: 0.0,
                    instability: 0.0,
                    motion: 1.0,
                },
            },
            GateConfig::default(),
        );

        let out = p.analyze_batch(&[frame(0, 0, 0.2), frame(1, u64::MAX, 0.2)]);

        assert_eq!(out[1].confidence, out[1].motion_score);
    }
}

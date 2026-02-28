use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct MarkerInput {
    pub frame_index: u64,
    pub pts_ms: u64,
    pub event_type: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMarker {
    pub marker_id: String,
    pub run_id: String,
    pub stream_id: String,
    pub event_type: String,
    pub status: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_pts_ms: u64,
    pub end_pts_ms: u64,
    pub confidence: f32,
    pub supersedes_marker_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MarkerConfig {
    pub correction_window_frames: u64,
    pub provisional_confidence_threshold: f32,
}

impl Default for MarkerConfig {
    fn default() -> Self {
        Self {
            correction_window_frames: 3,
            provisional_confidence_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
struct MarkerSegment {
    event_type: String,
    start_frame: u64,
    end_frame: u64,
    start_pts_ms: u64,
    end_pts_ms: u64,
    confidence_sum: f32,
    count: u64,
}

impl MarkerSegment {
    fn new(input: &MarkerInput) -> Self {
        Self {
            event_type: input.event_type.clone(),
            start_frame: input.frame_index,
            end_frame: input.frame_index,
            start_pts_ms: input.pts_ms,
            end_pts_ms: input.pts_ms,
            confidence_sum: input.confidence,
            count: 1,
        }
    }

    fn extend(&mut self, input: &MarkerInput) {
        self.end_frame = input.frame_index;
        self.end_pts_ms = input.pts_ms;
        self.confidence_sum += input.confidence;
        self.count += 1;
    }

    fn confidence(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.confidence_sum / self.count as f32).clamp(0.0, 1.0)
        }
    }
}

pub fn build_marker_lifecycle(
    run_id: &str,
    stream_id: &str,
    inputs: &[MarkerInput],
    config: &MarkerConfig,
) -> Vec<SemanticMarker> {
    let segments = segment_inputs(inputs);
    let mut out = Vec::new();

    for (idx, segment) in segments.iter().enumerate() {
        let confidence = segment.confidence();
        let status = if confidence >= config.provisional_confidence_threshold
            || segment.event_type == "scene_cut"
        {
            "exact"
        } else {
            "provisional"
        };

        let marker_id = format!("mk-{idx:08x}");
        out.push(SemanticMarker {
            marker_id: marker_id.clone(),
            run_id: run_id.to_string(),
            stream_id: stream_id.to_string(),
            event_type: segment.event_type.clone(),
            status: status.to_string(),
            start_frame: segment.start_frame,
            end_frame: segment.end_frame,
            start_pts_ms: segment.start_pts_ms,
            end_pts_ms: segment.end_pts_ms,
            confidence,
            supersedes_marker_id: None,
        });

        if status == "provisional" {
            if let Some(next) = segments.get(idx + 1) {
                if next.event_type == segment.event_type
                    && next.start_frame.saturating_sub(segment.end_frame)
                        <= config.correction_window_frames
                {
                    let finalized_id = format!("mkf-{idx:08x}");
                    out.push(SemanticMarker {
                        marker_id: finalized_id,
                        run_id: run_id.to_string(),
                        stream_id: stream_id.to_string(),
                        event_type: segment.event_type.clone(),
                        status: "finalized".to_string(),
                        start_frame: segment.start_frame,
                        end_frame: next.end_frame,
                        start_pts_ms: segment.start_pts_ms,
                        end_pts_ms: next.end_pts_ms,
                        confidence: ((confidence + next.confidence()) / 2.0).clamp(0.0, 1.0),
                        supersedes_marker_id: Some(marker_id),
                    });
                }
            }
        }
    }

    out
}

fn segment_inputs(inputs: &[MarkerInput]) -> Vec<MarkerSegment> {
    let mut segments = Vec::new();
    let mut current: Option<MarkerSegment> = None;

    for input in inputs {
        match current.as_mut() {
            Some(segment)
                if segment.event_type == input.event_type
                    && input.frame_index <= segment.end_frame.saturating_add(1) =>
            {
                segment.extend(input);
            }
            Some(segment) => {
                segments.push(segment.clone());
                current = Some(MarkerSegment::new(input));
            }
            None => {
                current = Some(MarkerSegment::new(input));
            }
        }
    }

    if let Some(segment) = current {
        segments.push(segment);
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::{build_marker_lifecycle, MarkerConfig, MarkerInput};

    #[test]
    fn emits_provisional_and_finalized_pairs() {
        let markers = build_marker_lifecycle(
            "run-1",
            "stream-0",
            &[
                MarkerInput {
                    frame_index: 1,
                    pts_ms: 33,
                    event_type: "context_observation".to_string(),
                    confidence: 0.3,
                },
                MarkerInput {
                    frame_index: 2,
                    pts_ms: 66,
                    event_type: "context_observation".to_string(),
                    confidence: 0.4,
                },
                MarkerInput {
                    frame_index: 4,
                    pts_ms: 132,
                    event_type: "context_observation".to_string(),
                    confidence: 0.5,
                },
            ],
            &MarkerConfig::default(),
        );

        assert!(markers.iter().any(|m| m.status == "provisional"));
        assert!(markers.iter().any(|m| m.status == "finalized"));
    }
}

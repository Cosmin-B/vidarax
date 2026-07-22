//! Deterministic restricted-zone activity assertions.
//!
//! A live pipeline crops to [`RestrictedZonePolicy::region`] before computing
//! its perceptual-hash motion score. This module applies hysteresis to that
//! score and reports state transitions. It deliberately claims only visual
//! activity: subject identity requires an explicit detector or semantic
//! confirmation represented by [`ConfirmedZoneSubject`].

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::coordinates::NormalizedRect;
use crate::crop::CropRegion;

pub const RESTRICTED_ZONE_ACTIVITY_EVENT: &str = "restricted_zone_activity_entered";
pub const RESTRICTED_ZONE_MOTION_MODEL: &str = "vidarax.motion-phash.v1";
pub const RESTRICTED_ZONE_COORDINATE_SPACE: &str = "normalized_image";
const MAX_ID_BYTES: usize = 128;
const MAX_CONFIRMATION_LABEL_BYTES: usize = 128;
const MAX_TRANSITION_FRAMES: u16 = 300;

/// Generation-static policy for one normalized rectangular image zone.
///
/// The live pipeline uses the region as its decode crop, so the motion score
/// covers exactly this source-image rectangle. Updating the value requires a
/// new pipeline generation; stale work cannot silently inherit a new policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestrictedZonePolicy {
    pub policy_id: String,
    pub policy_version: u64,
    pub device_id: String,
    pub region: NormalizedRect,
    pub enter_motion_score: f32,
    pub exit_motion_score: f32,
    pub enter_after_frames: u16,
    pub exit_after_frames: u16,
}

impl RestrictedZonePolicy {
    pub fn validate(&self) -> Result<(), &'static str> {
        validate_identifier(&self.policy_id, "policy_id")?;
        validate_identifier(&self.device_id, "device_id")?;
        if self.policy_version == 0 {
            return Err("policy_version must be greater than zero");
        }
        validate_region(self.region)?;
        if !self.enter_motion_score.is_finite() || !(0.0..=1.0).contains(&self.enter_motion_score) {
            return Err("enter_motion_score must be finite and in [0, 1]");
        }
        if !self.exit_motion_score.is_finite() || !(0.0..=1.0).contains(&self.exit_motion_score) {
            return Err("exit_motion_score must be finite and in [0, 1]");
        }
        if self.exit_motion_score >= self.enter_motion_score {
            return Err("exit_motion_score must be lower than enter_motion_score");
        }
        if !(1..=MAX_TRANSITION_FRAMES).contains(&self.enter_after_frames) {
            return Err("enter_after_frames must be in [1, 300]");
        }
        if !(1..=MAX_TRANSITION_FRAMES).contains(&self.exit_after_frames) {
            return Err("exit_after_frames must be in [1, 300]");
        }
        Ok(())
    }

    pub fn crop(&self) -> CropRegion {
        CropRegion {
            x: self.region.x,
            y: self.region.y,
            width: self.region.width,
            height: self.region.height,
        }
    }
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), &'static str> {
    if value.is_empty() || value.len() > MAX_ID_BYTES || value.chars().any(char::is_control) {
        return Err(match field {
            "policy_id" => "policy_id must be 1..=128 bytes and contain no control characters",
            _ => "device_id must be 1..=128 bytes and contain no control characters",
        });
    }
    Ok(())
}

fn validate_region(region: NormalizedRect) -> Result<(), &'static str> {
    let values = [region.x, region.y, region.width, region.height];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("region values must be finite");
    }
    if region.x < 0.0
        || region.y < 0.0
        || region.width <= 0.0
        || region.height <= 0.0
        || region.x + region.width > 1.0
        || region.y + region.height > 1.0
    {
        return Err("region must be a non-empty rectangle within normalized image bounds");
    }
    Ok(())
}

/// Optional subject-specific confirmation attached only by a detector or a
/// structured semantic result. The local motion gate always leaves this unset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedZoneSubject {
    pub label: String,
    pub confidence: f32,
    pub bounds: Option<NormalizedRect>,
}

impl ConfirmedZoneSubject {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.label.is_empty()
            || self.label.len() > MAX_CONFIRMATION_LABEL_BYTES
            || self.label.chars().any(char::is_control)
        {
            return Err("subject label must be 1..=128 bytes and contain no control characters");
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("subject confidence must be finite and in [0, 1]");
        }
        if let Some(bounds) = self.bounds {
            validate_region(bounds)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneActivityTransition {
    Entered,
    Exited,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoneActivityObservation {
    pub transition: ZoneActivityTransition,
    pub motion_score: f32,
    pub threshold: f32,
    pub consecutive_frames: u16,
}

/// Ordered per-stream state for one restricted-zone policy.
///
/// `observe` performs no allocation and touches no shared state. Hysteresis
/// prevents a score near the boundary from producing alternating enter/exit
/// assertions on successive frames.
#[derive(Debug)]
pub struct RestrictedZoneState {
    policy: Arc<RestrictedZonePolicy>,
    active: bool,
    enter_streak: u16,
    exit_streak: u16,
}

impl RestrictedZoneState {
    pub fn try_new(policy: Arc<RestrictedZonePolicy>) -> Result<Self, &'static str> {
        policy.validate()?;
        Ok(Self {
            policy,
            active: false,
            enter_streak: 0,
            exit_streak: 0,
        })
    }

    pub fn policy(&self) -> &RestrictedZonePolicy {
        &self.policy
    }

    pub fn policy_handle(&self) -> Arc<RestrictedZonePolicy> {
        Arc::clone(&self.policy)
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn observe(&mut self, motion_score: f32) -> Option<ZoneActivityObservation> {
        if self.active {
            self.enter_streak = 0;
            if motion_score <= self.policy.exit_motion_score {
                self.exit_streak = self.exit_streak.saturating_add(1);
                if self.exit_streak >= self.policy.exit_after_frames {
                    let consecutive_frames = self.exit_streak;
                    self.exit_streak = 0;
                    self.active = false;
                    return Some(ZoneActivityObservation {
                        transition: ZoneActivityTransition::Exited,
                        motion_score,
                        threshold: self.policy.exit_motion_score,
                        consecutive_frames,
                    });
                }
            } else {
                self.exit_streak = 0;
            }
            return None;
        }

        self.exit_streak = 0;
        if motion_score >= self.policy.enter_motion_score {
            self.enter_streak = self.enter_streak.saturating_add(1);
            if self.enter_streak >= self.policy.enter_after_frames {
                let consecutive_frames = self.enter_streak;
                self.enter_streak = 0;
                self.active = true;
                return Some(ZoneActivityObservation {
                    transition: ZoneActivityTransition::Entered,
                    motion_score,
                    threshold: self.policy.enter_motion_score,
                    consecutive_frames,
                });
            }
        } else {
            self.enter_streak = 0;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RestrictedZonePolicy {
        RestrictedZonePolicy {
            policy_id: "loading-bay-east".to_string(),
            policy_version: 3,
            device_id: "camera-17".to_string(),
            region: NormalizedRect {
                x: 0.25,
                y: 0.20,
                width: 0.50,
                height: 0.60,
            },
            enter_motion_score: 0.40,
            exit_motion_score: 0.15,
            enter_after_frames: 2,
            exit_after_frames: 3,
        }
    }

    #[test]
    fn policy_round_trips_as_a_small_serializable_value() {
        let original = policy();
        let json = serde_json::to_string(&original).unwrap();
        let decoded: RestrictedZonePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, original);
        assert!(!json.contains("base64"));
    }

    #[test]
    fn validates_coordinate_and_hysteresis_contract() {
        let mut invalid = policy();
        invalid.region.width = 0.80;
        assert!(invalid.validate().is_err());

        let mut invalid = policy();
        invalid.exit_motion_score = invalid.enter_motion_score;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn emits_one_enter_and_one_exit_after_configured_streaks() {
        let mut state = RestrictedZoneState::try_new(Arc::new(policy())).unwrap();

        assert_eq!(state.observe(0.50), None);
        let entered = state.observe(0.45).unwrap();
        assert_eq!(entered.transition, ZoneActivityTransition::Entered);
        assert_eq!(entered.consecutive_frames, 2);
        assert!(state.is_active());

        assert_eq!(state.observe(0.80), None);
        assert_eq!(state.observe(0.10), None);
        assert_eq!(state.observe(0.10), None);
        let exited = state.observe(0.10).unwrap();
        assert_eq!(exited.transition, ZoneActivityTransition::Exited);
        assert!(!state.is_active());
    }

    #[test]
    fn interrupted_streaks_reset_deterministically() {
        let mut state = RestrictedZoneState::try_new(Arc::new(policy())).unwrap();
        assert_eq!(state.observe(0.50), None);
        assert_eq!(state.observe(0.20), None);
        assert_eq!(state.observe(0.50), None);
        assert_eq!(state.observe(0.50).unwrap().consecutive_frames, 2);
    }

    #[test]
    fn confirmation_hook_rejects_unbounded_or_invalid_subjects() {
        let subject = ConfirmedZoneSubject {
            label: "person".to_string(),
            confidence: 0.92,
            bounds: Some(NormalizedRect {
                x: 0.3,
                y: 0.2,
                width: 0.1,
                height: 0.4,
            }),
        };
        assert!(subject.validate().is_ok());
    }
}

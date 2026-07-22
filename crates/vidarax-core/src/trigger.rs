//! Allocation-free execution for validated trigger programs.

use std::sync::Arc;

use vidarax_contracts::triggers::{
    TriggerInstruction, TriggerProgram, TriggerSample, TriggerSignal, MAX_TRIGGER_ACTIONS,
    MAX_TRIGGER_STACK, MAX_TRIGGER_STATE_SLOTS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriggerEvaluation {
    action_indices: [u8; MAX_TRIGGER_ACTIONS],
    action_count: u8,
    pub missing_signal: bool,
}

impl TriggerEvaluation {
    fn empty() -> Self {
        Self {
            action_indices: [0; MAX_TRIGGER_ACTIONS],
            action_count: 0,
            missing_signal: false,
        }
    }

    pub fn fired(self) -> bool {
        self.action_count > 0
    }

    pub fn actions<'a>(
        &'a self,
        program: &'a TriggerProgram,
    ) -> impl Iterator<Item = &'a TriggerInstruction> + 'a {
        self.action_indices[..usize::from(self.action_count)]
            .iter()
            .map(|index| &program.instructions[usize::from(*index)])
    }

    fn push_action(&mut self, instruction: usize) {
        if usize::from(self.action_count) >= MAX_TRIGGER_ACTIONS {
            return;
        }
        self.action_indices[usize::from(self.action_count)] = instruction as u8;
        self.action_count += 1;
    }
}

/// Per-stream state for a generation-static program.
///
/// All arrays are fixed at the ISA maximum. A frame evaluation performs no
/// allocation and no locking. Programs cannot jump backward, so they execute
/// at most `MAX_TRIGGER_INSTRUCTIONS` operations per sample.
#[derive(Debug)]
pub struct TriggerVm {
    program: Arc<TriggerProgram>,
    sustain_streaks: [u16; MAX_TRIGGER_STATE_SLOTS],
    edge_values: [bool; MAX_TRIGGER_STATE_SLOTS],
    cooldown_last_ms: [u64; MAX_TRIGGER_STATE_SLOTS],
    cooldown_initialized: u16,
}

impl TriggerVm {
    pub fn try_new(program: Arc<TriggerProgram>) -> Result<Self, String> {
        program.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            program,
            sustain_streaks: [0; MAX_TRIGGER_STATE_SLOTS],
            edge_values: [false; MAX_TRIGGER_STATE_SLOTS],
            cooldown_last_ms: [0; MAX_TRIGGER_STATE_SLOTS],
            cooldown_initialized: 0,
        })
    }

    pub fn program(&self) -> &TriggerProgram {
        &self.program
    }

    pub fn program_handle(&self) -> Arc<TriggerProgram> {
        Arc::clone(&self.program)
    }

    pub fn evaluate(&mut self, sample: &TriggerSample) -> TriggerEvaluation {
        let mut stack = [0.0f32; MAX_TRIGGER_STACK];
        let mut stack_len = 0usize;
        let mut evaluation = TriggerEvaluation::empty();
        let mut pc = 0usize;

        while pc < self.program.instructions.len() {
            match &self.program.instructions[pc] {
                TriggerInstruction::Load { signal } => {
                    if stack_len >= MAX_TRIGGER_STACK {
                        break;
                    }
                    let value = read_signal(sample, signal);
                    evaluation.missing_signal |= value.is_none();
                    stack[stack_len] = value.unwrap_or(f32::NAN);
                    stack_len += 1;
                }
                TriggerInstruction::Constant { value } => {
                    if stack_len >= MAX_TRIGGER_STACK {
                        break;
                    }
                    stack[stack_len] = *value;
                    stack_len += 1;
                }
                TriggerInstruction::GreaterThan => {
                    if !binary(&mut stack, &mut stack_len, |a, b| a > b) {
                        break;
                    }
                }
                TriggerInstruction::GreaterOrEqual => {
                    if !binary(&mut stack, &mut stack_len, |a, b| a >= b) {
                        break;
                    }
                }
                TriggerInstruction::LessThan => {
                    if !binary(&mut stack, &mut stack_len, |a, b| a < b) {
                        break;
                    }
                }
                TriggerInstruction::LessOrEqual => {
                    if !binary(&mut stack, &mut stack_len, |a, b| a <= b) {
                        break;
                    }
                }
                TriggerInstruction::Equal => {
                    if !binary(&mut stack, &mut stack_len, |a, b| a == b) {
                        break;
                    }
                }
                TriggerInstruction::And => {
                    if !binary(&mut stack, &mut stack_len, |a, b| truthy(a) && truthy(b)) {
                        break;
                    }
                }
                TriggerInstruction::Or => {
                    if !binary(&mut stack, &mut stack_len, |a, b| truthy(a) || truthy(b)) {
                        break;
                    }
                }
                TriggerInstruction::Not => {
                    if stack_len == 0 {
                        break;
                    }
                    stack[stack_len - 1] = bool_value(!truthy(stack[stack_len - 1]));
                }
                TriggerInstruction::SustainFrames { slot, frames } => {
                    if stack_len == 0 {
                        break;
                    }
                    let slot = usize::from(*slot);
                    if truthy(stack[stack_len - 1]) {
                        self.sustain_streaks[slot] = self.sustain_streaks[slot].saturating_add(1);
                    } else {
                        self.sustain_streaks[slot] = 0;
                    }
                    stack[stack_len - 1] = bool_value(self.sustain_streaks[slot] >= *frames);
                }
                TriggerInstruction::RisingEdge { slot } => {
                    if stack_len == 0 {
                        break;
                    }
                    let slot = usize::from(*slot);
                    let current = truthy(stack[stack_len - 1]);
                    let rising = current && !self.edge_values[slot];
                    self.edge_values[slot] = current;
                    stack[stack_len - 1] = bool_value(rising);
                }
                TriggerInstruction::CooldownMs { slot, duration_ms } => {
                    if stack_len == 0 {
                        break;
                    }
                    let slot = usize::from(*slot);
                    let condition = truthy(stack[stack_len - 1]);
                    let bit = 1u16 << slot;
                    let initialized = self.cooldown_initialized & bit != 0;
                    let elapsed = sample.pts_ms.saturating_sub(self.cooldown_last_ms[slot]);
                    let allowed = condition && (!initialized || elapsed >= *duration_ms);
                    if allowed {
                        self.cooldown_initialized |= bit;
                        self.cooldown_last_ms[slot] = sample.pts_ms;
                    }
                    stack[stack_len - 1] = bool_value(allowed);
                }
                TriggerInstruction::JumpIfFalse { target } => {
                    if stack_len == 0 {
                        break;
                    }
                    stack_len -= 1;
                    if !truthy(stack[stack_len]) {
                        pc = usize::from(*target);
                        continue;
                    }
                }
                TriggerInstruction::Emit { .. }
                | TriggerInstruction::Capture { .. }
                | TriggerInstruction::Notify { .. } => evaluation.push_action(pc),
                TriggerInstruction::Halt => break,
            }
            pc += 1;
        }
        evaluation
    }
}

#[inline]
fn binary(
    stack: &mut [f32; MAX_TRIGGER_STACK],
    stack_len: &mut usize,
    operation: impl FnOnce(f32, f32) -> bool,
) -> bool {
    if *stack_len < 2 {
        return false;
    }
    let right = stack[*stack_len - 1];
    let left = stack[*stack_len - 2];
    *stack_len -= 1;
    stack[*stack_len - 1] = bool_value(operation(left, right));
    true
}

#[inline]
fn truthy(value: f32) -> bool {
    value.is_finite() && value != 0.0
}

#[inline]
fn bool_value(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn read_signal(sample: &TriggerSample, signal: &TriggerSignal) -> Option<f32> {
    let scalar = match signal {
        TriggerSignal::MotionScore => sample.motion_score,
        TriggerSignal::NoveltyScore => sample.novelty_score,
        TriggerSignal::Confidence => sample.confidence,
        TriggerSignal::ModelUncertainty => sample.model_uncertainty,
        TriggerSignal::TeacherDisagreement => sample.teacher_disagreement,
        TriggerSignal::ObjectCount { .. }
        | TriggerSignal::MinimumDistanceMm { .. }
        | TriggerSignal::TimeToCollisionMs { .. }
        | TriggerSignal::DwellMs { .. } => None,
    };
    scalar.or_else(|| {
        sample
            .observations
            .iter()
            .find(|observation| &observation.signal == signal)
            .map(|observation| observation.value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vidarax_contracts::triggers::{compile_trigger, CaptureKind, TriggerObservation};

    fn sample(pts_ms: u64, motion_score: f32) -> TriggerSample {
        TriggerSample {
            pts_ms,
            motion_score: Some(motion_score),
            novelty_score: Some(motion_score),
            confidence: Some(0.9),
            model_uncertainty: None,
            teacher_disagreement: None,
            observations: Vec::new(),
        }
    }

    #[test]
    fn sustained_rising_edge_fires_once_and_cooldown_rearms() {
        let program = Arc::new(
            compile_trigger(
                r#"trigger zone version 1
when motion_score >= 0.4 for 2 frames
edge rising
cooldown 1000ms
emit zone_entered
capture keyframe
notify webhook
end"#,
            )
            .unwrap(),
        );
        let mut vm = TriggerVm::try_new(Arc::clone(&program)).unwrap();
        assert!(!vm.evaluate(&sample(0, 0.6)).fired());
        let fired = vm.evaluate(&sample(100, 0.6));
        assert!(fired.fired());
        assert_eq!(fired.actions(&program).count(), 3);
        assert!(!vm.evaluate(&sample(200, 0.6)).fired());
        assert!(!vm.evaluate(&sample(300, 0.0)).fired());
        assert!(!vm.evaluate(&sample(1_200, 0.6)).fired());
        assert!(vm.evaluate(&sample(1_300, 0.6)).fired());
    }

    #[test]
    fn evaluates_geometry_observations_without_special_runtime_code() {
        let program = Arc::new(
            compile_trigger(
                r#"trigger near-miss version 1
when minimum_distance_mm:forklift,pedestrian <= 2500
and time_to_collision_ms:forklift,pedestrian <= 1800
emit near_miss
capture clip 3000ms 5000ms
end"#,
            )
            .unwrap(),
        );
        let mut vm = TriggerVm::try_new(Arc::clone(&program)).unwrap();
        let sample = TriggerSample {
            pts_ms: 5_000,
            motion_score: None,
            novelty_score: None,
            confidence: None,
            model_uncertainty: None,
            teacher_disagreement: None,
            observations: vec![
                TriggerObservation {
                    signal: TriggerSignal::MinimumDistanceMm {
                        left: "forklift".into(),
                        right: "pedestrian".into(),
                    },
                    value: 1_900.0,
                },
                TriggerObservation {
                    signal: TriggerSignal::TimeToCollisionMs {
                        left: "forklift".into(),
                        right: "pedestrian".into(),
                    },
                    value: 900.0,
                },
            ],
        };
        let result = vm.evaluate(&sample);
        assert!(result.fired());
        assert!(result.actions(&program).any(|action| matches!(
            action,
            TriggerInstruction::Capture {
                kind: CaptureKind::Clip,
                ..
            }
        )));
    }

    #[test]
    fn missing_detector_signal_fails_closed() {
        let program = Arc::new(
            compile_trigger(
                r#"trigger person version 1
when object_count:person@zone >= 1
emit person_seen
end"#,
            )
            .unwrap(),
        );
        let mut vm = TriggerVm::try_new(program).unwrap();
        let result = vm.evaluate(&sample(1, 1.0));
        assert!(!result.fired());
        assert!(result.missing_signal);
    }
}

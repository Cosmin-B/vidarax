//! Versioned trigger bytecode and its compact line-oriented source format.
//!
//! The wire format is typed JSON. The source format exists for operators and
//! source control; both compile to the same bounded, forward-only program.

use core::fmt;
use serde::{Deserialize, Serialize};

pub const TRIGGER_ISA_VERSION: u16 = 1;
pub const MAX_TRIGGER_INSTRUCTIONS: usize = 64;
pub const MAX_TRIGGER_STACK: usize = 16;
pub const MAX_TRIGGER_STATE_SLOTS: usize = 16;
pub const MAX_TRIGGER_ACTIONS: usize = 8;
pub const MAX_TRIGGER_ID_BYTES: usize = 128;
pub const MAX_CAPTURE_WINDOW_MS: u64 = 60_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerProgram {
    pub isa_version: u16,
    pub program_id: String,
    pub version: u64,
    pub instructions: Vec<TriggerInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TriggerSignal {
    MotionScore,
    NoveltyScore,
    Confidence,
    ObjectCount { label: String, zone: Option<String> },
    MinimumDistanceMm { left: String, right: String },
    TimeToCollisionMs { left: String, right: String },
    DwellMs { label: String, zone: String },
    ModelUncertainty,
    TeacherDisagreement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureKind {
    Keyframe,
    Clip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionChannel {
    Webhook,
    LocalOutput,
}

/// Forward-only stack instructions. Jumps may only target a later instruction,
/// which makes execution time statically bounded by the program length.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum TriggerInstruction {
    Load {
        signal: TriggerSignal,
    },
    Constant {
        value: f32,
    },
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
    Equal,
    And,
    Or,
    Not,
    SustainFrames {
        slot: u8,
        frames: u16,
    },
    RisingEdge {
        slot: u8,
    },
    CooldownMs {
        slot: u8,
        duration_ms: u64,
    },
    JumpIfFalse {
        target: u8,
    },
    Emit {
        event_type: String,
    },
    Capture {
        kind: CaptureKind,
        before_ms: u64,
        after_ms: u64,
    },
    Notify {
        channel: ActionChannel,
    },
    Halt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerCompileRequest {
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerEvaluateRequest {
    pub program: TriggerProgram,
    pub samples: Vec<TriggerSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerSample {
    pub pts_ms: u64,
    #[serde(default)]
    pub motion_score: Option<f32>,
    #[serde(default)]
    pub novelty_score: Option<f32>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub model_uncertainty: Option<f32>,
    #[serde(default)]
    pub teacher_disagreement: Option<f32>,
    #[serde(default)]
    pub observations: Vec<TriggerObservation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerObservation {
    pub signal: TriggerSignal,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerCompileResponse {
    pub program: TriggerProgram,
    pub instruction_count: usize,
    pub state_slots: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriggerError {
    pub line: Option<usize>,
    pub message: String,
}

impl TriggerError {
    fn at(line: usize, message: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            message: message.into(),
        }
    }

    fn program(message: impl Into<String>) -> Self {
        Self {
            line: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for TriggerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {line}: {}", self.message)
        } else {
            f.write_str(&self.message)
        }
    }
}

impl std::error::Error for TriggerError {}

impl TriggerProgram {
    pub fn validate(&self) -> Result<(), TriggerError> {
        if self.isa_version != TRIGGER_ISA_VERSION {
            return Err(TriggerError::program(format!(
                "unsupported trigger ISA version {}; expected {TRIGGER_ISA_VERSION}",
                self.isa_version
            )));
        }
        validate_identifier(&self.program_id, "program_id")?;
        if self.version == 0 {
            return Err(TriggerError::program(
                "program version must be greater than zero",
            ));
        }
        if self.instructions.is_empty() || self.instructions.len() > MAX_TRIGGER_INSTRUCTIONS {
            return Err(TriggerError::program(format!(
                "instruction count must be in [1, {MAX_TRIGGER_INSTRUCTIONS}]"
            )));
        }

        let mut stack_depth = 0usize;
        let mut max_slot = 0usize;
        let mut emit_count = 0usize;
        let mut action_count = 0usize;
        let mut halted = false;
        for (index, instruction) in self.instructions.iter().enumerate() {
            if halted {
                return Err(TriggerError::program(format!(
                    "instruction {index} is unreachable after halt"
                )));
            }
            match instruction {
                TriggerInstruction::Load { signal } => {
                    validate_signal(signal)?;
                    stack_depth += 1;
                }
                TriggerInstruction::Constant { value } => {
                    if !value.is_finite() {
                        return Err(TriggerError::program(format!(
                            "instruction {index} has a non-finite constant"
                        )));
                    }
                    stack_depth += 1;
                }
                TriggerInstruction::GreaterThan
                | TriggerInstruction::GreaterOrEqual
                | TriggerInstruction::LessThan
                | TriggerInstruction::LessOrEqual
                | TriggerInstruction::Equal
                | TriggerInstruction::And
                | TriggerInstruction::Or => {
                    require_stack(index, stack_depth, 2)?;
                    stack_depth -= 1;
                }
                TriggerInstruction::Not => require_stack(index, stack_depth, 1)?,
                TriggerInstruction::SustainFrames { slot, frames } => {
                    require_stack(index, stack_depth, 1)?;
                    validate_slot(index, *slot)?;
                    max_slot = max_slot.max(usize::from(*slot) + 1);
                    if *frames == 0 || *frames > 10_000 {
                        return Err(TriggerError::program(format!(
                            "instruction {index} sustain frames must be in [1, 10000]"
                        )));
                    }
                }
                TriggerInstruction::RisingEdge { slot } => {
                    require_stack(index, stack_depth, 1)?;
                    validate_slot(index, *slot)?;
                    max_slot = max_slot.max(usize::from(*slot) + 1);
                }
                TriggerInstruction::CooldownMs { slot, duration_ms } => {
                    require_stack(index, stack_depth, 1)?;
                    validate_slot(index, *slot)?;
                    max_slot = max_slot.max(usize::from(*slot) + 1);
                    if *duration_ms > 86_400_000 {
                        return Err(TriggerError::program(format!(
                            "instruction {index} cooldown exceeds 24 hours"
                        )));
                    }
                }
                TriggerInstruction::JumpIfFalse { target } => {
                    require_stack(index, stack_depth, 1)?;
                    stack_depth -= 1;
                    let target = usize::from(*target);
                    if target <= index || target >= self.instructions.len() {
                        return Err(TriggerError::program(format!(
                            "instruction {index} jump target must be forward and in bounds"
                        )));
                    }
                }
                TriggerInstruction::Emit { event_type } => {
                    validate_identifier(event_type, "event_type")?;
                    emit_count += 1;
                    action_count += 1;
                }
                TriggerInstruction::Capture {
                    before_ms,
                    after_ms,
                    ..
                } => {
                    if before_ms.saturating_add(*after_ms) > MAX_CAPTURE_WINDOW_MS {
                        return Err(TriggerError::program(format!(
                            "instruction {index} capture window exceeds {MAX_CAPTURE_WINDOW_MS}ms"
                        )));
                    }
                    action_count += 1;
                }
                TriggerInstruction::Notify { .. } => action_count += 1,
                TriggerInstruction::Halt => halted = true,
            }
            if stack_depth > MAX_TRIGGER_STACK {
                return Err(TriggerError::program(format!(
                    "instruction {index} exceeds stack depth {MAX_TRIGGER_STACK}"
                )));
            }
            if action_count > MAX_TRIGGER_ACTIONS {
                return Err(TriggerError::program(format!(
                    "program exceeds {MAX_TRIGGER_ACTIONS} actions"
                )));
            }
        }
        if !halted {
            return Err(TriggerError::program("program must end with halt"));
        }
        if emit_count != 1 {
            return Err(TriggerError::program(
                "program must contain exactly one emit action",
            ));
        }
        validate_stack_control_flow(&self.instructions)?;
        validate_atomic_action_block(&self.instructions)?;
        let _ = max_slot;
        Ok(())
    }

    pub fn state_slots(&self) -> usize {
        self.instructions
            .iter()
            .filter_map(|instruction| match instruction {
                TriggerInstruction::SustainFrames { slot, .. }
                | TriggerInstruction::RisingEdge { slot }
                | TriggerInstruction::CooldownMs { slot, .. } => Some(usize::from(*slot) + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

fn validate_atomic_action_block(instructions: &[TriggerInstruction]) -> Result<(), TriggerError> {
    let halt = instructions.len() - 1;
    let first_action = instructions
        .iter()
        .position(is_action)
        .expect("validated programs contain one emit action");

    if !instructions[first_action..halt].iter().all(is_action) {
        return Err(TriggerError::program(
            "emit, capture, and notify instructions must form one contiguous block before halt",
        ));
    }

    for (index, instruction) in instructions[..first_action].iter().enumerate() {
        if let TriggerInstruction::JumpIfFalse { target } = instruction {
            let target = usize::from(*target);
            if target > first_action && target != halt {
                return Err(TriggerError::program(format!(
                    "instruction {index} jumps into the action block; actions must execute atomically"
                )));
            }
        }
    }
    Ok(())
}

fn is_action(instruction: &TriggerInstruction) -> bool {
    matches!(
        instruction,
        TriggerInstruction::Emit { .. }
            | TriggerInstruction::Capture { .. }
            | TriggerInstruction::Notify { .. }
    )
}

fn validate_stack_control_flow(instructions: &[TriggerInstruction]) -> Result<(), TriggerError> {
    let mut depths = vec![None; instructions.len()];
    let mut pending = vec![0usize];
    depths[0] = Some(0usize);
    while let Some(index) = pending.pop() {
        let depth = depths[index].expect("queued instructions have a stack depth");
        let (required, produced) = match instructions[index] {
            TriggerInstruction::Load { .. } | TriggerInstruction::Constant { .. } => (0, 1),
            TriggerInstruction::GreaterThan
            | TriggerInstruction::GreaterOrEqual
            | TriggerInstruction::LessThan
            | TriggerInstruction::LessOrEqual
            | TriggerInstruction::Equal
            | TriggerInstruction::And
            | TriggerInstruction::Or => (2, 1),
            TriggerInstruction::Not
            | TriggerInstruction::SustainFrames { .. }
            | TriggerInstruction::RisingEdge { .. }
            | TriggerInstruction::CooldownMs { .. } => (1, 1),
            TriggerInstruction::JumpIfFalse { .. } => (1, 0),
            TriggerInstruction::Emit { .. }
            | TriggerInstruction::Capture { .. }
            | TriggerInstruction::Notify { .. }
            | TriggerInstruction::Halt => (0, 0),
        };
        if depth < required {
            return Err(TriggerError::program(format!(
                "instruction {index} can be reached with stack depth {depth}, but needs {required}"
            )));
        }
        let next_depth = depth - required + produced;
        if next_depth > MAX_TRIGGER_STACK {
            return Err(TriggerError::program(format!(
                "instruction {index} exceeds stack depth {MAX_TRIGGER_STACK}"
            )));
        }

        let mut successors = [None, None];
        match instructions[index] {
            TriggerInstruction::Halt => {}
            TriggerInstruction::JumpIfFalse { target } => {
                successors[0] = (index + 1 < instructions.len()).then_some(index + 1);
                successors[1] = Some(usize::from(target));
            }
            _ => successors[0] = (index + 1 < instructions.len()).then_some(index + 1),
        }
        for successor in successors.into_iter().flatten() {
            match depths[successor] {
                Some(existing) if existing != next_depth => {
                    return Err(TriggerError::program(format!(
                        "instruction {successor} has inconsistent incoming stack depths {existing} and {next_depth}"
                    )));
                }
                Some(_) => {}
                None => {
                    depths[successor] = Some(next_depth);
                    pending.push(successor);
                }
            }
        }
    }
    Ok(())
}

fn require_stack(index: usize, available: usize, needed: usize) -> Result<(), TriggerError> {
    if available < needed {
        return Err(TriggerError::program(format!(
            "instruction {index} needs {needed} stack values, found {available}"
        )));
    }
    Ok(())
}

fn validate_slot(index: usize, slot: u8) -> Result<(), TriggerError> {
    if usize::from(slot) >= MAX_TRIGGER_STATE_SLOTS {
        return Err(TriggerError::program(format!(
            "instruction {index} state slot must be below {MAX_TRIGGER_STATE_SLOTS}"
        )));
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &str) -> Result<(), TriggerError> {
    if value.is_empty()
        || value.len() > MAX_TRIGGER_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(TriggerError::program(format!(
            "{field} must be 1..={MAX_TRIGGER_ID_BYTES} bytes using letters, digits, '-', '_', '.', or ':'"
        )));
    }
    Ok(())
}

fn validate_signal(signal: &TriggerSignal) -> Result<(), TriggerError> {
    match signal {
        TriggerSignal::ObjectCount { label, zone } => {
            validate_identifier(label, "object label")?;
            if let Some(zone) = zone {
                validate_identifier(zone, "zone")?;
            }
        }
        TriggerSignal::MinimumDistanceMm { left, right }
        | TriggerSignal::TimeToCollisionMs { left, right } => {
            validate_identifier(left, "left object label")?;
            validate_identifier(right, "right object label")?;
        }
        TriggerSignal::DwellMs { label, zone } => {
            validate_identifier(label, "object label")?;
            validate_identifier(zone, "zone")?;
        }
        TriggerSignal::MotionScore
        | TriggerSignal::NoveltyScore
        | TriggerSignal::Confidence
        | TriggerSignal::ModelUncertainty
        | TriggerSignal::TeacherDisagreement => {}
    }
    Ok(())
}

#[derive(Debug)]
struct Condition {
    signal: TriggerSignal,
    comparison: TriggerInstruction,
    value: f32,
    sustain_frames: Option<u16>,
}

/// Compile a compact trigger source file into validated v1 bytecode.
///
/// ```text
/// trigger restricted-area-entry version 1
/// when motion_score >= 0.40 for 2 frames
/// and object_count:person@loading-bay >= 1
/// edge rising
/// cooldown 5000ms
/// emit restricted_area_entry
/// capture keyframe
/// notify webhook
/// end
/// ```
pub fn compile_trigger(source: &str) -> Result<TriggerProgram, TriggerError> {
    let mut header: Option<(String, u64)> = None;
    let mut conditions = Vec::new();
    let mut edge = false;
    let mut cooldown_ms = None;
    let mut actions = Vec::new();
    let mut saw_end = false;

    for (zero_line, raw) in source.lines().enumerate() {
        let line_number = zero_line + 1;
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if saw_end {
            return Err(TriggerError::at(line_number, "content after end"));
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first().copied() {
            Some("trigger") => {
                if header.is_some() || fields.len() != 4 || fields[2] != "version" {
                    return Err(TriggerError::at(
                        line_number,
                        "expected: trigger <program-id> version <positive-integer>",
                    ));
                }
                let version = fields[3]
                    .parse::<u64>()
                    .map_err(|_| TriggerError::at(line_number, "invalid program version"))?;
                header = Some((fields[1].to_string(), version));
            }
            Some("when" | "and") => {
                if header.is_none() {
                    return Err(TriggerError::at(
                        line_number,
                        "condition precedes trigger header",
                    ));
                }
                conditions.push(parse_condition(&fields, line_number)?);
            }
            Some("edge") if fields.as_slice() == ["edge", "rising"] => edge = true,
            Some("cooldown") if fields.len() == 2 => {
                cooldown_ms = Some(parse_duration_ms(fields[1], line_number)?);
            }
            Some("emit") if fields.len() == 2 => actions.push(TriggerInstruction::Emit {
                event_type: fields[1].to_string(),
            }),
            Some("capture") if fields.len() >= 2 => {
                let kind = match fields[1] {
                    "keyframe" if fields.len() == 2 => CaptureKind::Keyframe,
                    "clip" if fields.len() == 4 => CaptureKind::Clip,
                    _ => {
                        return Err(TriggerError::at(
                            line_number,
                            "expected: capture keyframe | capture clip <before-ms> <after-ms>",
                        ))
                    }
                };
                let (before_ms, after_ms) = if kind == CaptureKind::Clip {
                    (
                        parse_duration_ms(fields[2], line_number)?,
                        parse_duration_ms(fields[3], line_number)?,
                    )
                } else {
                    (0, 0)
                };
                actions.push(TriggerInstruction::Capture {
                    kind,
                    before_ms,
                    after_ms,
                });
            }
            Some("notify") if fields.len() == 2 => {
                let channel = match fields[1] {
                    "webhook" => ActionChannel::Webhook,
                    "local_output" => ActionChannel::LocalOutput,
                    _ => {
                        return Err(TriggerError::at(
                            line_number,
                            "notify channel must be webhook or local_output; emit already creates the durable event",
                        ))
                    }
                };
                actions.push(TriggerInstruction::Notify { channel });
            }
            Some("end") if fields.len() == 1 => saw_end = true,
            _ => {
                return Err(TriggerError::at(
                    line_number,
                    "unrecognized trigger statement",
                ))
            }
        }
    }

    let (program_id, version) =
        header.ok_or_else(|| TriggerError::program("missing trigger header"))?;
    if conditions.is_empty() {
        return Err(TriggerError::program(
            "trigger needs at least one condition",
        ));
    }
    if !saw_end {
        return Err(TriggerError::program("trigger source must end with end"));
    }

    let mut instructions = Vec::with_capacity(conditions.len() * 5 + actions.len() + 4);
    let mut next_slot = 0u8;
    for (index, condition) in conditions.into_iter().enumerate() {
        instructions.push(TriggerInstruction::Load {
            signal: condition.signal,
        });
        instructions.push(TriggerInstruction::Constant {
            value: condition.value,
        });
        instructions.push(condition.comparison);
        if let Some(frames) = condition.sustain_frames {
            instructions.push(TriggerInstruction::SustainFrames {
                slot: next_slot,
                frames,
            });
            next_slot = next_slot.saturating_add(1);
        }
        if index > 0 {
            instructions.push(TriggerInstruction::And);
        }
    }
    if edge {
        instructions.push(TriggerInstruction::RisingEdge { slot: next_slot });
        next_slot = next_slot.saturating_add(1);
    }
    if let Some(duration_ms) = cooldown_ms {
        instructions.push(TriggerInstruction::CooldownMs {
            slot: next_slot,
            duration_ms,
        });
    }
    let jump_index = instructions.len();
    instructions.push(TriggerInstruction::JumpIfFalse { target: 0 });
    instructions.extend(actions);
    instructions.push(TriggerInstruction::Halt);
    let halt = instructions.len() - 1;
    let target = u8::try_from(halt)
        .map_err(|_| TriggerError::program("compiled program exceeds jump address space"))?;
    instructions[jump_index] = TriggerInstruction::JumpIfFalse { target };

    let program = TriggerProgram {
        isa_version: TRIGGER_ISA_VERSION,
        program_id,
        version,
        instructions,
    };
    program.validate()?;
    Ok(program)
}

fn parse_condition(fields: &[&str], line: usize) -> Result<Condition, TriggerError> {
    if fields.len() != 4 && fields.len() != 7 {
        return Err(TriggerError::at(
            line,
            "expected: when <signal> <comparison> <value> [for <n> frames]",
        ));
    }
    if fields.len() == 7 && (fields[4] != "for" || fields[6] != "frames") {
        return Err(TriggerError::at(line, "expected 'for <n> frames'"));
    }
    let signal = parse_signal(fields[1], line)?;
    let comparison = match fields[2] {
        ">" => TriggerInstruction::GreaterThan,
        ">=" => TriggerInstruction::GreaterOrEqual,
        "<" => TriggerInstruction::LessThan,
        "<=" => TriggerInstruction::LessOrEqual,
        "==" => TriggerInstruction::Equal,
        _ => {
            return Err(TriggerError::at(
                line,
                "comparison must be >, >=, <, <=, or ==",
            ))
        }
    };
    let value = fields[3]
        .parse::<f32>()
        .map_err(|_| TriggerError::at(line, "condition value must be a number"))?;
    let sustain_frames = if fields.len() == 7 {
        Some(
            fields[5]
                .parse::<u16>()
                .map_err(|_| TriggerError::at(line, "frame count must be an integer"))?,
        )
    } else {
        None
    };
    Ok(Condition {
        signal,
        comparison,
        value,
        sustain_frames,
    })
}

fn parse_signal(raw: &str, line: usize) -> Result<TriggerSignal, TriggerError> {
    match raw {
        "motion_score" => return Ok(TriggerSignal::MotionScore),
        "novelty_score" => return Ok(TriggerSignal::NoveltyScore),
        "confidence" => return Ok(TriggerSignal::Confidence),
        "model_uncertainty" => return Ok(TriggerSignal::ModelUncertainty),
        "teacher_disagreement" => return Ok(TriggerSignal::TeacherDisagreement),
        _ => {}
    }
    let (kind, args) = raw
        .split_once(':')
        .ok_or_else(|| TriggerError::at(line, "unknown trigger signal"))?;
    match kind {
        "object_count" => {
            let (label, zone) = args.split_once('@').map_or((args, None), |(label, zone)| {
                (label, Some(zone.to_string()))
            });
            Ok(TriggerSignal::ObjectCount {
                label: label.to_string(),
                zone,
            })
        }
        "minimum_distance_mm" | "time_to_collision_ms" => {
            let (left, right) = args.split_once(',').ok_or_else(|| {
                TriggerError::at(line, "geometry signal requires two comma-separated labels")
            })?;
            if kind == "minimum_distance_mm" {
                Ok(TriggerSignal::MinimumDistanceMm {
                    left: left.to_string(),
                    right: right.to_string(),
                })
            } else {
                Ok(TriggerSignal::TimeToCollisionMs {
                    left: left.to_string(),
                    right: right.to_string(),
                })
            }
        }
        "dwell_ms" => {
            let (label, zone) = args
                .split_once('@')
                .ok_or_else(|| TriggerError::at(line, "dwell_ms requires label@zone"))?;
            Ok(TriggerSignal::DwellMs {
                label: label.to_string(),
                zone: zone.to_string(),
            })
        }
        _ => Err(TriggerError::at(line, "unknown trigger signal")),
    }
}

fn parse_duration_ms(raw: &str, line: usize) -> Result<u64, TriggerError> {
    let digits = raw.strip_suffix("ms").unwrap_or(raw);
    digits
        .parse::<u64>()
        .map_err(|_| TriggerError::at(line, "duration must be milliseconds, for example 5000ms"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_restricted_area_program() {
        let source = r#"
trigger restricted-area-entry version 3
when motion_score >= 0.40 for 2 frames
and object_count:person@loading-bay >= 1
edge rising
cooldown 5000ms
emit restricted_area_entry
capture keyframe
notify webhook
end
"#;
        let program = compile_trigger(source).unwrap();
        assert_eq!(program.program_id, "restricted-area-entry");
        assert_eq!(program.version, 3);
        assert!(program.state_slots() >= 2);
        assert!(program.instructions.len() <= MAX_TRIGGER_INSTRUCTIONS);
        program.validate().unwrap();
    }

    #[test]
    fn compiles_geometry_near_miss_program() {
        let source = r#"
trigger forklift-near-miss version 1
when object_count:forklift@yard >= 1
and object_count:pedestrian@yard >= 1
and minimum_distance_mm:forklift,pedestrian <= 2500
and time_to_collision_ms:forklift,pedestrian <= 1800 for 2 frames
edge rising
cooldown 10000ms
emit forklift_pedestrian_near_miss
capture clip 3000ms 5000ms
notify local_output
notify webhook
end
"#;
        let program = compile_trigger(source).unwrap();
        program.validate().unwrap();
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            TriggerInstruction::Capture {
                kind: CaptureKind::Clip,
                ..
            }
        )));
    }

    #[test]
    fn rejects_loops_and_unbounded_programs() {
        let mut program = TriggerProgram {
            isa_version: TRIGGER_ISA_VERSION,
            program_id: "bad".to_string(),
            version: 1,
            instructions: vec![
                TriggerInstruction::Constant { value: 1.0 },
                TriggerInstruction::JumpIfFalse { target: 0 },
                TriggerInstruction::Emit {
                    event_type: "bad".to_string(),
                },
                TriggerInstruction::Halt,
            ],
        };
        assert!(program.validate().is_err());
        program.instructions = vec![TriggerInstruction::Halt; MAX_TRIGGER_INSTRUCTIONS + 1];
        assert!(program.validate().is_err());
    }

    #[test]
    fn rejects_stack_underflow_on_a_jump_target() {
        let program = TriggerProgram {
            isa_version: TRIGGER_ISA_VERSION,
            program_id: "bad-stack".to_string(),
            version: 1,
            instructions: vec![
                TriggerInstruction::Constant { value: 0.0 },
                TriggerInstruction::JumpIfFalse { target: 4 },
                TriggerInstruction::Constant { value: 1.0 },
                TriggerInstruction::Not,
                TriggerInstruction::Not,
                TriggerInstruction::Emit {
                    event_type: "still_safe".to_string(),
                },
                TriggerInstruction::Halt,
            ],
        };
        let error = program.validate().unwrap_err().to_string();
        assert!(error.contains("reached with stack depth 0"), "{error}");
    }

    #[test]
    fn rejects_partial_action_paths_in_hand_written_bytecode() {
        let program = TriggerProgram {
            isa_version: TRIGGER_ISA_VERSION,
            program_id: "partial-actions".to_string(),
            version: 1,
            instructions: vec![
                TriggerInstruction::Load {
                    signal: TriggerSignal::MotionScore,
                },
                TriggerInstruction::Capture {
                    kind: CaptureKind::Keyframe,
                    before_ms: 0,
                    after_ms: 0,
                },
                TriggerInstruction::JumpIfFalse { target: 4 },
                TriggerInstruction::Emit {
                    event_type: "motion".to_string(),
                },
                TriggerInstruction::Halt,
            ],
        };

        assert!(program
            .validate()
            .unwrap_err()
            .to_string()
            .contains("contiguous block"));
    }
}

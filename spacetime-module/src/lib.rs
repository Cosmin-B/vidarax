//! SpacetimeDB module for Vidarax.
//!
//! Two public tables and their corresponding reducers:
//!
//! - [`AgentEvent`]: ephemeral broadcast table for real-time agent activity.
//! - [`KeyframeStore`]: persistent table for durable visual-memory keyframes.

use spacetimedb::{reducer, table, Identity, ReducerContext, Table, Timestamp};

// ---------------------------------------------------------------------------
// Input limits
//
// Every reducer below is reachable by any client that can open the module's
// WebSocket, so all arguments are untrusted. These caps sit far above anything
// the engine emits; they exist only to refuse the oversized or out-of-range
// payloads a hostile client could send to exhaust storage or poison the tables.
// A rejected reducer returns Err, which aborts its transaction without writing.
// ---------------------------------------------------------------------------

/// Identifiers (run_id, session_id) are UUID-like; 256 bytes is generous.
const MAX_ID_LEN: usize = 256;
/// Enum-like tags (event_type, category).
const MAX_TAG_LEN: usize = 64;
/// Free text (VLM description, user feedback).
const MAX_TEXT_LEN: usize = 64 * 1024;
/// Feedback rating is documented as 0..=10.
const MAX_RATING: u32 = 10;
/// A single keyframe JPEG, with headroom well past any frame the engine encodes.
const MAX_JPEG_BYTES: usize = 8 * 1024 * 1024;

fn check_len(field: &str, value: &str, max: usize) -> Result<(), String> {
    let len = value.len();
    if len > max {
        return Err(format!("{field} exceeds {max} bytes ({len} given)"));
    }
    Ok(())
}

fn check_confidence(confidence: f32) -> Result<(), String> {
    if !(0.0..=1.0).contains(&confidence) {
        return Err(format!(
            "confidence must be in [0.0, 1.0], got {confidence}"
        ));
    }
    Ok(())
}

fn validate_emit_event(
    run_id: &str,
    session_id: &str,
    event_type: &str,
    confidence: f32,
    description: &str,
) -> Result<(), String> {
    check_len("run_id", run_id, MAX_ID_LEN)?;
    check_len("session_id", session_id, MAX_ID_LEN)?;
    check_len("event_type", event_type, MAX_TAG_LEN)?;
    check_len("description", description, MAX_TEXT_LEN)?;
    check_confidence(confidence)
}

fn validate_submit_feedback(
    run_id: &str,
    session_id: &str,
    rating: u32,
    category: &str,
    feedback: &str,
) -> Result<(), String> {
    check_len("run_id", run_id, MAX_ID_LEN)?;
    check_len("session_id", session_id, MAX_ID_LEN)?;
    check_len("category", category, MAX_TAG_LEN)?;
    check_len("feedback", feedback, MAX_TEXT_LEN)?;
    if rating > MAX_RATING {
        return Err(format!("rating must be 0..={MAX_RATING}, got {rating}"));
    }
    Ok(())
}

fn validate_store_keyframe(
    run_id: &str,
    event_type: &str,
    description: &str,
    jpeg_len: usize,
) -> Result<(), String> {
    check_len("run_id", run_id, MAX_ID_LEN)?;
    check_len("event_type", event_type, MAX_TAG_LEN)?;
    check_len("description", description, MAX_TEXT_LEN)?;
    if jpeg_len > MAX_JPEG_BYTES {
        return Err(format!(
            "jpeg_data exceeds {MAX_JPEG_BYTES} bytes ({jpeg_len} given)"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tables (SpacetimeDB 2.0: `accessor =` replaces `name =`)
// ---------------------------------------------------------------------------

/// Real-time event broadcast. All subscribers receive inserts instantly.
#[table(accessor = agent_event, public)]
pub struct AgentEvent {
    #[primary_key]
    #[auto_inc]
    pub id: u64,

    /// Identity of the agent that emitted the event.
    pub agent_id: Identity,

    /// Run ID for filtering events by analysis session.
    pub run_id: String,

    /// Session ID for the WebRTC stream.
    pub session_id: String,

    /// Frame index within the stream.
    pub frame_index: u64,

    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,

    /// Event type: "scene_cut" | "loop_detected" | "vlm" | "goal_reached" | "artifact"
    pub event_type: String,

    /// Confidence score [0.0, 1.0].
    pub confidence: f32,

    /// Description (VLM output or gate engine reason).
    pub description: String,

    /// Wall-clock time at which the reducer was invoked.
    pub timestamp: Timestamp,
}

/// Persistent keyframe storage for visual memory.
/// Replayed to new subscribers for consistent history.
#[table(accessor = keyframe_store, public)]
pub struct KeyframeStore {
    #[primary_key]
    #[auto_inc]
    pub id: u64,

    /// Identity of the agent that produced this keyframe.
    pub agent_id: Identity,

    /// Run ID for filtering.
    pub run_id: String,

    /// Frame index within the stream.
    pub frame_index: u64,

    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,

    /// Event type that triggered this keyframe capture.
    pub event_type: String,

    /// VLM description of the keyframe content.
    pub description: String,

    /// Raw JPEG bytes of the keyframe (no base64 encoding overhead).
    pub jpeg_data: Vec<u8>,

    /// Wall-clock time at which the keyframe was stored.
    pub timestamp: Timestamp,
}

// ---------------------------------------------------------------------------
// Reducers (SpacetimeDB 2.0: ctx.sender() method instead of ctx.sender field)
// ---------------------------------------------------------------------------

/// Emit an event and broadcast to all subscribers.
#[reducer]
pub fn emit_event(
    ctx: &ReducerContext,
    run_id: String,
    session_id: String,
    frame_index: u64,
    pts_ms: u64,
    event_type: String,
    confidence: f32,
    description: String,
) -> Result<(), String> {
    validate_emit_event(&run_id, &session_id, &event_type, confidence, &description)?;
    ctx.db.agent_event().insert(AgentEvent {
        id: 0,
        agent_id: ctx.sender(),
        run_id,
        session_id,
        frame_index,
        pts_ms,
        event_type,
        confidence,
        description,
        timestamp: ctx.timestamp,
    });
    Ok(())
}

/// User feedback on analysis quality.
#[table(accessor = feedback, public)]
pub struct Feedback {
    #[primary_key]
    #[auto_inc]
    pub id: u64,

    /// Identity of the agent that submitted feedback.
    pub agent_id: Identity,

    /// Run ID the feedback applies to.
    pub run_id: String,

    /// Session ID for the WebRTC stream.
    pub session_id: String,

    /// Rating from 0 to 10.
    pub rating: u32,

    /// Category: "accuracy" | "latency" | "quality"
    pub category: String,

    /// Free-text feedback.
    pub feedback: String,

    /// Wall-clock time at which the reducer was invoked.
    pub timestamp: Timestamp,
}

/// Submit feedback for a run.
#[reducer]
pub fn submit_feedback(
    ctx: &ReducerContext,
    run_id: String,
    session_id: String,
    rating: u32,
    category: String,
    feedback: String,
) -> Result<(), String> {
    validate_submit_feedback(&run_id, &session_id, rating, &category, &feedback)?;
    ctx.db.feedback().insert(Feedback {
        id: 0,
        agent_id: ctx.sender(),
        run_id,
        session_id,
        rating,
        category,
        feedback,
        timestamp: ctx.timestamp,
    });
    Ok(())
}

/// Store a keyframe for persistent visual memory.
/// Accepts raw JPEG bytes — no base64 encoding needed.
#[reducer]
pub fn store_keyframe(
    ctx: &ReducerContext,
    run_id: String,
    frame_index: u64,
    pts_ms: u64,
    event_type: String,
    description: String,
    jpeg_data: Vec<u8>,
) -> Result<(), String> {
    validate_store_keyframe(&run_id, &event_type, &description, jpeg_data.len())?;
    ctx.db.keyframe_store().insert(KeyframeStore {
        id: 0,
        agent_id: ctx.sender(),
        run_id,
        frame_index,
        pts_ms,
        event_type,
        description,
        jpeg_data,
        timestamp: ctx.timestamp,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_realistic_payloads() {
        assert!(validate_emit_event("run-1", "sess-1", "vlm", 0.87, "a person waves").is_ok());
        assert!(validate_submit_feedback("run-1", "sess-1", 7, "accuracy", "solid result").is_ok());
        assert!(validate_store_keyframe("run-1", "scene_cut", "a doorway", 250 * 1024).is_ok());
    }

    #[test]
    fn boundary_values_are_inclusive() {
        let id = "a".repeat(MAX_ID_LEN);
        let tag = "a".repeat(MAX_TAG_LEN);
        let text = "a".repeat(MAX_TEXT_LEN);
        assert!(validate_emit_event(&id, &id, &tag, 1.0, &text).is_ok());
        assert!(validate_emit_event("r", "s", "t", 0.0, "d").is_ok());
        assert!(validate_submit_feedback("r", "s", MAX_RATING, &tag, &text).is_ok());
        assert!(validate_store_keyframe(&id, &tag, &text, MAX_JPEG_BYTES).is_ok());
    }

    #[test]
    fn rejects_oversized_strings() {
        let big_id = "a".repeat(MAX_ID_LEN + 1);
        let big_tag = "a".repeat(MAX_TAG_LEN + 1);
        let big_text = "a".repeat(MAX_TEXT_LEN + 1);
        assert!(validate_emit_event(&big_id, "s", "vlm", 0.5, "d").is_err());
        assert!(validate_emit_event("r", "s", &big_tag, 0.5, "d").is_err());
        assert!(validate_emit_event("r", "s", "vlm", 0.5, &big_text).is_err());
        assert!(validate_submit_feedback("r", "s", 5, &big_tag, "f").is_err());
        assert!(validate_submit_feedback("r", "s", 5, "accuracy", &big_text).is_err());
        assert!(validate_store_keyframe("r", &big_tag, "d", 0).is_err());
    }

    #[test]
    fn rejects_out_of_range_scalars() {
        assert!(validate_emit_event("r", "s", "vlm", 1.01, "d").is_err());
        assert!(validate_emit_event("r", "s", "vlm", -0.01, "d").is_err());
        assert!(validate_emit_event("r", "s", "vlm", f32::NAN, "d").is_err());
        assert!(validate_submit_feedback("r", "s", MAX_RATING + 1, "accuracy", "f").is_err());
    }

    #[test]
    fn rejects_oversized_jpeg() {
        assert!(validate_store_keyframe("r", "scene_cut", "d", MAX_JPEG_BYTES + 1).is_err());
    }
}

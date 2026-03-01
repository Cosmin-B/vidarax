//! SpacetimeDB module for Vidarax.
//!
//! Two public tables and their corresponding reducers:
//!
//! - [`AgentEvent`]: ephemeral broadcast table for real-time agent activity.
//! - [`KeyframeStore`]: persistent table for durable visual-memory keyframes.

use spacetimedb::{reducer, table, Identity, ReducerContext, Table, Timestamp};

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

    /// Base64-encoded JPEG of the keyframe.
    pub jpeg_b64: String,

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
) {
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
) {
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
}

/// Store a keyframe for persistent visual memory.
#[reducer]
pub fn store_keyframe(
    ctx: &ReducerContext,
    run_id: String,
    frame_index: u64,
    pts_ms: u64,
    event_type: String,
    description: String,
    jpeg_b64: String,
) {
    ctx.db.keyframe_store().insert(KeyframeStore {
        id: 0,
        agent_id: ctx.sender(),
        run_id,
        frame_index,
        pts_ms,
        event_type,
        description,
        jpeg_b64,
        timestamp: ctx.timestamp,
    });
}

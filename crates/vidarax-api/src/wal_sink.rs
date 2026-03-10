//! WAL-backed [`EventSink`] implementation.
//!
//! [`WalEventSink`] bridges the worker-thread [`EventSink`] trait into the
//! REST API's WAL/timeline store so that streaming VLM results emitted by
//! the real-time pipeline appear in `GET /v1/runs/{id}/events` without any
//! SpacetimeDB dependency.
//!
//! # Thread safety
//!
//! [`WalEventSink`] is `Send + Sync`.  All mutable state lives inside
//! [`AppState`] which uses lock-free atomics for the sequence counter and an
//! `ArcSwap`-guarded registry — both are safe to access from multiple worker
//! threads simultaneously.
//!
//! # Event kinds written to the WAL
//!
//! | EventSink method     | WAL `kind`        | Payload fields                                    |
//! |----------------------|-------------------|---------------------------------------------------|
//! | `emit_event_sync`    | `<event_type>`    | `session_id`, `frame_index`, `pts_ms`, `confidence`, `description` |
//! | `store_keyframe_sync`| `keyframe_stored` | `frame_index`, `pts_ms`, `event_type`, `description` |
//!
//! JPEG bytes are intentionally not stored in the WAL (the WAL is
//! a plain-text append log).  A future improvement could write the JPEG to a
//! side-car object store and record the URI in the payload.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use vidarax_api::wal_sink::WalEventSink;
//! use vidarax_api::AppState;
//! use vidarax_core::webrtc::workers::EventSink;
//!
//! # fn example(state: AppState) {
//! let sink: Arc<dyn EventSink> = Arc::new(WalEventSink::new(state, "run-abc123".to_string()));
//! # }
//! ```

use std::sync::Arc;

use serde_json::json;
use vidarax_core::webrtc::workers::EventSink;

use crate::state::AppState;

// ── WalEventSink ──────────────────────────────────────────────────────────────

/// An [`EventSink`] that persists events to the run's WAL timeline.
///
/// Construct one per WebRTC session and pass `Arc<WalEventSink>` (upcast to
/// `Arc<dyn EventSink>`) to the worker spawn functions.
///
/// All methods are synchronous and call [`AppState::append_run_event`], which
/// is safe to call from `std::thread` workers (no async runtime required).
pub struct WalEventSink {
    /// Shared application state — holds the WAL path, event sequence counter,
    /// and the in-memory run registry.
    state: AppState,
    /// The run ID this sink is scoped to.  Stored here so the sink can be
    /// constructed once and reused across many worker threads without
    /// re-allocating the string on every call.
    run_id: String,
}

impl WalEventSink {
    /// Create a new sink bound to `run_id`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vidarax_api::wal_sink::WalEventSink;
    /// use vidarax_api::AppState;
    ///
    /// # fn example(state: AppState) {
    /// let sink = WalEventSink::new(state, "run-abc123".to_string());
    /// # }
    /// ```
    pub fn new(state: AppState, run_id: String) -> Self {
        Self { state, run_id }
    }

    /// Convenience constructor that wraps `self` in an `Arc<dyn EventSink>`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use vidarax_core::webrtc::workers::EventSink;
    /// use vidarax_api::wal_sink::WalEventSink;
    /// use vidarax_api::AppState;
    ///
    /// # fn example(state: AppState) {
    /// let sink: Arc<dyn EventSink> = WalEventSink::arc(state, "run-abc123".to_string());
    /// # }
    /// ```
    pub fn arc(state: AppState, run_id: String) -> Arc<dyn EventSink> {
        Arc::new(Self::new(state, run_id))
    }

    /// The run ID this sink is bound to.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

// ── EventSink impl ────────────────────────────────────────────────────────────

impl EventSink for WalEventSink {
    /// Append a real-time agent event to the WAL.
    ///
    /// The WAL event `kind` is set to `event_type` (e.g. `"vlm"`,
    /// `"vlm_tiered"`, `"loop_detected"`, `"scene_cut"`).  The `run_id`
    /// argument from the trait call is preferred over the stored `self.run_id`
    /// so that the method stays correct even when the caller provides a
    /// different run_id (though in practice they will always match for a
    /// single-session sink).
    ///
    /// Errors are returned as `String` per the trait contract; the worker
    /// pool logs them as warnings and continues.
    fn emit_event_sync(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "session_id": session_id,
            "frame_index": frame_index,
            "pts_ms": pts_ms,
            "confidence": confidence,
            "description": description,
        });

        self.state
            .append_run_event(run_id, event_type, payload)
            .map(|_| ())
    }

    /// Append a keyframe metadata record to the WAL.
    ///
    /// JPEG bytes are not stored in the WAL (the WAL is a plain-text log).
    /// The `jpeg_data` slice is intentionally ignored; only the metadata fields
    /// are persisted so the `/events` endpoint can surface timing information.
    ///
    /// A future improvement may write JPEG bytes to a side-car blob store and
    /// record the resulting URI in the payload under `"jpeg_uri"`.
    fn store_keyframe_sync(
        &self,
        run_id: &str,
        frame_index: u64,
        pts_ms: u64,
        event_type: &str,
        description: &str,
        _jpeg_data: &[u8],
    ) -> Result<(), String> {
        let payload = json!({
            "frame_index": frame_index,
            "pts_ms": pts_ms,
            "event_type": event_type,
            "description": description,
        });

        self.state
            .append_run_event(run_id, "keyframe_stored", payload)
            .map(|_| ())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::WalEventSink;
    use crate::state::AppState;
    use std::sync::Arc;
    use vidarax_core::webrtc::workers::EventSink;

    fn temp_wal() -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("wal-sink-test-{nanos}.wal"))
    }

    fn make_state() -> (AppState, std::path::PathBuf) {
        let path = temp_wal();
        let state = AppState::with_wal_for_tests(path.clone());
        (state, path)
    }

    #[test]
    fn emit_event_writes_to_wal() {
        let (state, path) = make_state();
        let run_id = "run-abcdef1234567890";

        let sink = WalEventSink::new(state.clone(), run_id.to_string());
        sink.emit_event_sync(run_id, "sess-01", 42, 1400, "vlm", 0.92, "a dog is running")
            .expect("emit_event_sync should succeed");

        let events = state.read_run_events(run_id).expect("read_run_events failed");
        assert_eq!(events.len(), 1, "expected exactly one WAL event");
        let ev = &events[0];
        assert_eq!(ev.kind, "vlm");
        assert!(
            ev.payload.contains("a dog is running"),
            "payload should contain the description"
        );
        assert!(
            ev.payload.contains("\"frame_index\":42"),
            "payload should contain frame_index"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn store_keyframe_writes_keyframe_stored_kind() {
        let (state, path) = make_state();
        let run_id = "run-deadbeef00000000";

        let sink = WalEventSink::new(state.clone(), run_id.to_string());
        sink.store_keyframe_sync(run_id, 7, 231, "scene_cut", "a park scene", b"\xff\xd8\xff\xd9")
            .expect("store_keyframe_sync should succeed");

        let events = state.read_run_events(run_id).expect("read_run_events failed");
        assert_eq!(events.len(), 1);
        let ev = &events[0];
        assert_eq!(ev.kind, "keyframe_stored");
        assert!(ev.payload.contains("scene_cut"));
        assert!(ev.payload.contains("a park scene"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn multiple_events_accumulate_in_wal() {
        let (state, path) = make_state();
        let run_id = "run-cafebabe12345678";

        let sink = WalEventSink::new(state.clone(), run_id.to_string());
        for i in 0..5u64 {
            sink.emit_event_sync(
                run_id,
                "sess-multi",
                i,
                i * 33,
                "vlm",
                0.8,
                "frame description",
            )
            .unwrap();
        }
        sink.store_keyframe_sync(run_id, 3, 99, "periodic_keepalive", "desc", b"")
            .unwrap();

        let events = state.read_run_events(run_id).expect("read_run_events failed");
        assert_eq!(events.len(), 6);
        assert!(events.iter().any(|e| e.kind == "keyframe_stored"));
        assert_eq!(events.iter().filter(|e| e.kind == "vlm").count(), 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wal_sink_is_send_sync_and_arc_compatible() {
        let (state, path) = make_state();
        let run_id = "run-aabbccdd11223344";

        // Verify Arc<dyn EventSink> construction compiles and can be cloned.
        let sink: Arc<dyn EventSink> = WalEventSink::arc(state, run_id.to_string());
        let _clone = Arc::clone(&sink);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn run_id_accessor_returns_correct_value() {
        let (state, path) = make_state();
        let run_id = "run-0011223344556677";
        let sink = WalEventSink::new(state, run_id.to_string());
        assert_eq!(sink.run_id(), run_id);
        let _ = std::fs::remove_file(path);
    }
}

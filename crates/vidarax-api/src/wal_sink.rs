//! WAL-backed [`EventSink`] implementation.
//!
//! Events go to the local WAL. Keyframe JPEGs are written first to a
//! content-addressed blob directory and referenced from event metadata.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::json;
use sha2::{Digest, Sha256};
use vidarax_core::coordinates::{FrameCoordinates, IMAGE_COORDINATE_SCHEMA};
use vidarax_core::webrtc::workers::{
    EventSink, KeyframeEvent, RestrictedZoneEvidenceEvent, TriggerAssertionEvent,
};
use vidarax_core::zone::{
    RESTRICTED_ZONE_ACTIVITY_EVENT, RESTRICTED_ZONE_COORDINATE_SPACE, RESTRICTED_ZONE_MOTION_MODEL,
};

use crate::spacetime_client::{EmitEventArgs, SpacetimeClient};
use crate::state::AppState;

/// Synchronous sink for worker threads.
pub struct WalEventSink {
    state: AppState,
    keyframe_blob_root: PathBuf,
    spacetime_event_mirror: Option<SpacetimeClient>,
}

impl WalEventSink {
    pub fn new(state: AppState) -> Self {
        let keyframe_blob_root = state.keyframe_blob_root();
        let spacetime_event_mirror = state.spacetime_client().cloned();
        Self {
            state,
            keyframe_blob_root,
            spacetime_event_mirror,
        }
    }

    pub fn arc(state: AppState) -> Arc<dyn EventSink> {
        Arc::new(Self::new(state))
    }

    fn persist_keyframe_blob(&self, jpeg_data: &[u8]) -> Result<KeyframeBlob, String> {
        let digest = Sha256::digest(jpeg_data);
        let mut sha256 = String::with_capacity(64);
        for byte in digest {
            let _ = write!(sha256, "{byte:02x}");
        }
        let shard = &sha256[..2];
        let directory = self.keyframe_blob_root.join(shard);
        std::fs::create_dir_all(&directory).map_err(|err| {
            format!(
                "create keyframe sidecar directory {}: {err}",
                directory.display()
            )
        })?;
        let final_path = directory.join(format!("{sha256}.jpg"));
        let created = if final_path.exists() {
            false
        } else {
            write_blob_atomically(&final_path, jpeg_data)?
        };
        Ok(KeyframeBlob {
            image_ref: format!("keyframes/blobs/{shard}/{sha256}.jpg"),
            sha256,
            bytes: jpeg_data.len(),
            created,
        })
    }
}

struct KeyframeBlob {
    image_ref: String,
    sha256: String,
    bytes: usize,
    created: bool,
}

static NEXT_BLOB_TEMP: AtomicU64 = AtomicU64::new(0);

fn write_blob_atomically(final_path: &Path, data: &[u8]) -> Result<bool, String> {
    let sequence = NEXT_BLOB_TEMP.fetch_add(1, Ordering::Relaxed);
    let file_name = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid keyframe sidecar path {}", final_path.display()))?;
    let temp_path = final_path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let result = (|| -> Result<bool, String> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        apply_blob_permissions(&mut options);
        let mut file = options
            .open(&temp_path)
            .map_err(|err| format!("create keyframe sidecar {}: {err}", temp_path.display()))?;
        file.write_all(data)
            .map_err(|err| format!("write keyframe sidecar {}: {err}", temp_path.display()))?;
        // Match the WAL policy: flush, but do not fsync each keyframe.
        file.flush()
            .map_err(|err| format!("flush keyframe sidecar {}: {err}", temp_path.display()))?;
        drop(file);
        if let Err(err) = std::fs::rename(&temp_path, final_path) {
            // Another writer may have committed the same hash first.
            if final_path.exists() {
                let _ = std::fs::remove_file(&temp_path);
                return Ok(false);
            }
            return Err(format!(
                "commit keyframe sidecar {} -> {}: {err}",
                temp_path.display(),
                final_path.display()
            ));
        }
        Ok(true)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

#[cfg(unix)]
fn apply_blob_permissions(options: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
}

#[cfg(not(unix))]
fn apply_blob_permissions(_options: &mut OpenOptions) {}

impl EventSink for WalEventSink {
    fn emit_event_sync(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "session_id": session_id,
            "frame_index": frame_index,
            "pts_ms": pts_ms,
            "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
            "coordinates": coordinates,
            "confidence": confidence,
            "description": description,
        });

        self.state.append_run_event(run_id, event_type, payload)?;
        if let Some(mirror) = &self.spacetime_event_mirror {
            if let Err(err) = mirror.emit_event_sync(&EmitEventArgs {
                run_id,
                session_id,
                frame_index,
                pts_ms,
                event_type,
                confidence,
                description,
            }) {
                tracing::warn!(%err, "SpacetimeDB event mirror failed after local WAL commit");
            }
        }
        Ok(())
    }

    fn emit_event_nonblocking(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        coordinates: FrameCoordinates,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "session_id": session_id,
            "frame_index": frame_index,
            "pts_ms": pts_ms,
            "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
            "coordinates": coordinates,
            "confidence": confidence,
            "description": description,
        });

        // Mirroring would violate this method's nonblocking contract.
        self.state
            .append_run_event_nonblocking(run_id, event_type, payload)
    }

    /// Write the JPEG before appending its metadata event.
    fn store_keyframe_sync(&self, event: KeyframeEvent<'_>) -> Result<(), String> {
        let started = std::time::Instant::now();
        let blob = match self.persist_keyframe_blob(event.jpeg_data) {
            Ok(blob) => blob,
            Err(err) => {
                self.state.pipeline_metrics().inc_keyframe_blob_failure();
                self.state
                    .pipeline_metrics()
                    .keyframe_blob_latency_ms
                    .record(started.elapsed().as_millis() as u64);
                return Err(err);
            }
        };
        self.state
            .pipeline_metrics()
            .keyframe_blob_latency_ms
            .record(started.elapsed().as_millis() as u64);
        if blob.created {
            self.state
                .pipeline_metrics()
                .record_keyframe_blob_written(blob.bytes as u64);
        } else {
            self.state.pipeline_metrics().inc_keyframe_blob_reused();
        }
        let payload = json!({
            "frame_index": event.frame_index,
            "pts_ms": event.pts_ms,
            "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
            "coordinates": event.coordinates,
            "event_type": event.event_type,
            "description": event.description,
            "image_ref": blob.image_ref,
            "image_media_type": "image/jpeg",
            "image_bytes": blob.bytes,
            "image_sha256": blob.sha256,
        });

        self.state
            .append_run_event(event.run_id, "keyframe_stored", payload)
            .map(|_| ())
    }

    /// Write binary evidence before publishing the assertion that references
    /// it. The WAL never contains image bytes or a base64 representation.
    fn store_restricted_zone_sync(
        &self,
        event: RestrictedZoneEvidenceEvent<'_>,
    ) -> Result<(), String> {
        let started = std::time::Instant::now();
        let blob = match self.persist_keyframe_blob(event.jpeg_data) {
            Ok(blob) => blob,
            Err(err) => {
                self.state.pipeline_metrics().inc_keyframe_blob_failure();
                self.state
                    .pipeline_metrics()
                    .keyframe_blob_latency_ms
                    .record(started.elapsed().as_millis() as u64);
                return Err(err);
            }
        };
        self.state
            .pipeline_metrics()
            .keyframe_blob_latency_ms
            .record(started.elapsed().as_millis() as u64);
        if blob.created {
            self.state
                .pipeline_metrics()
                .record_keyframe_blob_written(blob.bytes as u64);
        } else {
            self.state.pipeline_metrics().inc_keyframe_blob_reused();
        }
        let payload = json!({
            "session_id": event.session_id,
            "frame_index": event.frame_index,
            "pts_ms": event.pts_ms,
            "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
            "coordinates": event.coordinates,
            "confidence": event.observation.motion_score,
            "assertion": {
                "policy_id": event.policy.policy_id,
                "policy_version": event.policy.policy_version,
                "status": "active",
                "coordinate_space": RESTRICTED_ZONE_COORDINATE_SPACE,
                "zone": event.policy.region,
                "trigger": {
                    "kind": "perceptual_hash_motion",
                    "score": event.observation.motion_score,
                    "threshold": event.observation.threshold,
                    "consecutive_frames": event.observation.consecutive_frames,
                },
                "subject": event.subject,
            },
            "provenance": {
                "device_id": event.policy.device_id,
                "model_id": RESTRICTED_ZONE_MOTION_MODEL,
                "pipeline_generation": event.generation.get(),
            },
            "evidence": {
                "image_ref": blob.image_ref,
                "image_media_type": "image/jpeg",
                "image_bytes": blob.bytes,
                "image_sha256": blob.sha256,
            },
        });

        self.state
            .append_run_event(event.run_id, RESTRICTED_ZONE_ACTIVITY_EVENT, payload)
            .map(|_| ())
    }

    fn store_trigger_sync(&self, event: TriggerAssertionEvent<'_>) -> Result<(), String> {
        let started = std::time::Instant::now();
        let blob = match self.persist_keyframe_blob(event.jpeg_data) {
            Ok(blob) => blob,
            Err(err) => {
                self.state.pipeline_metrics().inc_keyframe_blob_failure();
                self.state
                    .pipeline_metrics()
                    .keyframe_blob_latency_ms
                    .record(started.elapsed().as_millis() as u64);
                return Err(err);
            }
        };
        self.state
            .pipeline_metrics()
            .keyframe_blob_latency_ms
            .record(started.elapsed().as_millis() as u64);
        if blob.created {
            self.state
                .pipeline_metrics()
                .record_keyframe_blob_written(blob.bytes as u64);
        } else {
            self.state.pipeline_metrics().inc_keyframe_blob_reused();
        }
        let payload = json!({
            "session_id": event.session_id,
            "frame_index": event.frame_index,
            "pts_ms": event.pts_ms,
            "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
            "coordinates": event.coordinates,
            "confidence": event.confidence,
            "event_type": event.event_type,
            "trigger": {
                "program_id": event.program.program_id,
                "program_version": event.program.version,
                "isa_version": event.program.isa_version,
                "inputs": {
                    "motion_score": event.motion_score,
                    "novelty_score": event.novelty_score,
                    "confidence": event.confidence,
                },
                "actions": {
                    "event": true,
                    "webhook": event.notify_webhook,
                    "local_output": event.notify_local_output,
                    "capture": "keyframe",
                }
            },
            "provenance": {
                "pipeline_generation": event.generation.get(),
            },
            "evidence": {
                "image_ref": blob.image_ref,
                "image_media_type": "image/jpeg",
                "image_bytes": blob.bytes,
                "image_sha256": blob.sha256,
            },
        });

        let event_kind = format!("trigger.{}", event.event_type);
        let committed = self
            .state
            .append_run_event(event.run_id, &event_kind, payload)?;
        if event.notify_local_output {
            match send_trigger_local_output(&committed, event.generation.get()) {
                Ok(()) => self.state.pipeline_metrics().inc_trigger_local_output(),
                Err(error) => {
                    self.state
                        .pipeline_metrics()
                        .inc_trigger_local_output_failure();
                    tracing::warn!(%error, "trigger local output delivery failed");
                }
            }
        }
        Ok(())
    }
}

#[cfg(unix)]
fn send_trigger_local_output(
    event: &vidarax_core::timeline::TimelineEvent,
    generation: u64,
) -> Result<(), String> {
    let path = std::env::var_os("VIDARAX_TRIGGER_LOCAL_OUTPUT_SOCKET")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "VIDARAX_TRIGGER_LOCAL_OUTPUT_SOCKET is not configured".to_string())?;
    send_trigger_local_output_to(std::path::Path::new(&path), event, generation)
}

#[cfg(unix)]
fn send_trigger_local_output_to(
    path: &std::path::Path,
    event: &vidarax_core::timeline::TimelineEvent,
    generation: u64,
) -> Result<(), String> {
    use std::os::unix::net::UnixDatagram;

    let message = serde_json::to_vec(&json!({
        "event_id": format!("{}:{}", event.run_id, event.seq),
        "run_id": event.run_id,
        "stream_id": event.stream_id,
        "seq": event.seq,
        "pts_ms": event.pts_ms,
        "kind": event.kind,
        "pipeline_generation": generation,
    }))
    .map_err(|error| format!("encode local trigger output: {error}"))?;
    let socket = UnixDatagram::unbound()
        .map_err(|error| format!("create local trigger output socket: {error}"))?;
    socket
        .set_nonblocking(true)
        .map_err(|error| format!("configure local trigger output socket: {error}"))?;
    socket
        .send_to(&message, path)
        .map(|_| ())
        .map_err(|error| format!("send local trigger output: {error}"))
}

#[cfg(not(unix))]
fn send_trigger_local_output(
    _event: &vidarax_core::timeline::TimelineEvent,
    _generation: u64,
) -> Result<(), String> {
    Err("local trigger output requires Unix-domain datagrams".to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::WalEventSink;
    use crate::state::AppState;
    use std::sync::Arc;
    use vidarax_contracts::triggers::compile_trigger;
    use vidarax_core::coordinates::{FrameCoordinates, NormalizedRect};
    use vidarax_core::webrtc::runtime::PipelineGeneration;
    use vidarax_core::webrtc::workers::{
        EventSink, KeyframeEvent, RestrictedZoneEvidenceEvent, TriggerAssertionEvent,
    };
    use vidarax_core::zone::{
        RestrictedZonePolicy, ZoneActivityObservation, ZoneActivityTransition,
        RESTRICTED_ZONE_ACTIVITY_EVENT,
    };

    fn temp_wal() -> std::path::PathBuf {
        // A per-call unique path. A bare nanosecond timestamp collided when two
        // tests ran in the same clock tick under `cargo test`'s parallelism, so
        // they shared one WAL file and clobbered each other's events (flaky
        // event-count assertions). The atomic counter makes every call within a
        // process distinct; the process id separates concurrent test binaries;
        // the timestamp separates process lifetimes, so a crashed test that
        // left its file behind cannot be reopened in append mode by a later run
        // that reuses the pid and restarts the counter at zero.
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "wal-sink-test-{}-{}-{}.wal",
            std::process::id(),
            nanos,
            seq
        ))
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

        let sink = WalEventSink::new(state.clone());
        sink.emit_event_sync(
            run_id,
            "sess-01",
            42,
            1400,
            FrameCoordinates::full_frame(1920, 1080),
            "vlm",
            0.92,
            "a dog is running",
        )
        .expect("emit_event_sync should succeed");

        let events = state
            .read_run_events(run_id)
            .expect("read_run_events failed");
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
    fn emit_event_nonblocking_hands_off_to_timeline_writer() {
        let (state, path) = make_state();
        let run_id = "run-feedface12345678";

        let sink = WalEventSink::new(state.clone());
        let start = std::time::Instant::now();
        sink.emit_event_nonblocking(
            run_id,
            "sess-01",
            3,
            99,
            FrameCoordinates::full_frame(1280, 720),
            "loop_detected",
            0.9,
            "loop detected via perceptual-hash ring buffer",
        )
        .expect("emit_event_nonblocking should enqueue");
        assert!(
            start.elapsed() < std::time::Duration::from_millis(50),
            "nonblocking emit should not wait for durable append"
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let events = loop {
            let events = state
                .read_run_events(run_id)
                .expect("read_run_events failed");
            if !events.is_empty() || std::time::Instant::now() >= deadline {
                break events;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "loop_detected");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn store_keyframe_writes_keyframe_stored_kind() {
        let (state, path) = make_state();
        let run_id = "run-deadbeef00000000";

        let sink = WalEventSink::new(state.clone());
        sink.store_keyframe_sync(KeyframeEvent {
            run_id,
            frame_index: 7,
            pts_ms: 231,
            coordinates: FrameCoordinates::full_frame(1920, 1080),
            event_type: "scene_cut",
            description: "a park scene",
            jpeg_data: b"\xff\xd8\xff\xd9",
        })
        .expect("store_keyframe_sync should succeed");

        let events = state
            .read_run_events(run_id)
            .expect("read_run_events failed");
        assert_eq!(events.len(), 1);
        let ev = &events[0];
        assert_eq!(ev.kind, "keyframe_stored");
        assert!(ev.payload.contains("scene_cut"));
        assert!(ev.payload.contains("a park scene"));
        assert!(ev
            .payload
            .contains("\"coordinate_schema\":\"vidarax.image.v1\""));
        let payload: serde_json::Value =
            serde_json::from_str(&ev.payload).expect("keyframe payload should be valid JSON");
        assert_eq!(payload["coordinates"]["source_extent"]["width"], 1920);
        assert_eq!(payload["coordinates"]["source_extent"]["height"], 1080);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn restricted_zone_event_has_provenance_and_binary_sidecar_reference() {
        let (state, path) = make_state();
        let run_id = "run-zone0000000000";
        let policy = RestrictedZonePolicy {
            policy_id: "loading-bay-east".to_string(),
            policy_version: 4,
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
        };
        let coordinates = FrameCoordinates::resolve(1920, 1080, Some(policy.crop())).unwrap();

        let sink = WalEventSink::new(state.clone());
        sink.store_restricted_zone_sync(RestrictedZoneEvidenceEvent {
            run_id,
            session_id: "sess-zone",
            frame_index: 11,
            pts_ms: 363,
            coordinates,
            generation: PipelineGeneration::new(9),
            policy: &policy,
            observation: ZoneActivityObservation {
                transition: ZoneActivityTransition::Entered,
                motion_score: 0.58,
                threshold: 0.40,
                consecutive_frames: 2,
            },
            subject: None,
            jpeg_data: b"\xff\xd8zone-evidence\xff\xd9",
        })
        .expect("zone evidence should commit");

        let events = state.read_run_events(run_id).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, RESTRICTED_ZONE_ACTIVITY_EVENT);
        let payload: serde_json::Value = serde_json::from_str(&events[0].payload).unwrap();
        assert_eq!(payload["assertion"]["policy_id"], "loading-bay-east");
        assert_eq!(payload["assertion"]["subject"], serde_json::Value::Null);
        assert_eq!(payload["provenance"]["device_id"], "camera-17");
        assert_eq!(payload["provenance"]["pipeline_generation"], 9);
        assert_eq!(payload["provenance"]["model_id"], "vidarax.motion-phash.v1");
        assert_eq!(payload["assertion"]["coordinate_space"], "normalized_image");
        assert_eq!(payload["coordinates"]["resolved_region"]["x"], 480);
        assert!(payload["evidence"]["image_ref"]
            .as_str()
            .unwrap()
            .starts_with("keyframes/blobs/"));
        assert!(payload.get("image_data").is_none());
        assert!(payload.get("data_base64").is_none());
        assert!(!events[0].payload.contains("zone-evidence"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn trigger_event_is_namespaced_away_from_run_lifecycle_kinds() {
        let (state, path) = make_state();
        let program = compile_trigger(
            "trigger lifecycle-guard version 1\nwhen motion_score >= 0.1\nemit run_deleted\ncapture keyframe\nend",
        )
        .unwrap();
        WalEventSink::new(state.clone())
            .store_trigger_sync(TriggerAssertionEvent {
                run_id: "run-trigger-namespace",
                session_id: "sess-trigger",
                frame_index: 1,
                pts_ms: 33,
                coordinates: FrameCoordinates::full_frame(640, 480),
                generation: PipelineGeneration::new(4),
                program: &program,
                event_type: "run_deleted",
                confidence: 0.9,
                motion_score: 0.8,
                novelty_score: 0.7,
                notify_webhook: false,
                notify_local_output: false,
                jpeg_data: b"\xff\xd8trigger\xff\xd9",
            })
            .unwrap();
        let events = state.read_run_events("run-trigger-namespace").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "trigger.run_deleted");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&events[0].payload).unwrap()["event_type"],
            "run_deleted"
        );
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn local_output_is_metadata_only_and_stably_identified() {
        use std::os::unix::net::UnixDatagram;

        let socket_path = temp_wal().with_extension("sock");
        let socket = UnixDatagram::bind(&socket_path).unwrap();
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(1)))
            .unwrap();
        let event = vidarax_core::timeline::TimelineEvent {
            seq: 7,
            run_id: "run-trigger-local".into(),
            stream_id: "camera-2".into(),
            pts_ms: 99,
            kind: "trigger.near_miss".into(),
            payload: r#"{"evidence":{"image_ref":"keyframes/blobs/abc.jpg"}}"#.into(),
        };

        super::send_trigger_local_output_to(&socket_path, &event, PipelineGeneration::new(3).get())
            .unwrap();
        let mut buffer = [0u8; 2048];
        let count = socket.recv(&mut buffer).unwrap();
        let message: serde_json::Value = serde_json::from_slice(&buffer[..count]).unwrap();
        assert_eq!(message["event_id"], "run-trigger-local:7");
        assert_eq!(message["pipeline_generation"], 3);
        assert_eq!(message["kind"], "trigger.near_miss");
        assert!(message.get("payload").is_none());
        assert!(!String::from_utf8_lossy(&buffer[..count]).contains("image_ref"));
        let _ = std::fs::remove_file(socket_path);
    }

    #[test]
    fn multiple_events_accumulate_in_wal() {
        let (state, path) = make_state();
        let run_id = "run-cafebabe12345678";

        let sink = WalEventSink::new(state.clone());
        for i in 0..5u64 {
            sink.emit_event_sync(
                run_id,
                "sess-multi",
                i,
                i * 33,
                FrameCoordinates::full_frame(640, 480),
                "vlm",
                0.8,
                "frame description",
            )
            .unwrap();
        }
        sink.store_keyframe_sync(KeyframeEvent {
            run_id,
            frame_index: 3,
            pts_ms: 99,
            coordinates: FrameCoordinates::full_frame(640, 480),
            event_type: "periodic_keepalive",
            description: "desc",
            jpeg_data: b"",
        })
        .unwrap();

        let events = state
            .read_run_events(run_id)
            .expect("read_run_events failed");
        assert_eq!(events.len(), 6);
        assert!(events.iter().any(|e| e.kind == "keyframe_stored"));
        assert_eq!(events.iter().filter(|e| e.kind == "vlm").count(), 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wal_sink_is_send_sync_and_arc_compatible() {
        let (state, path) = make_state();

        // Verify Arc<dyn EventSink> construction compiles and can be cloned.
        let sink: Arc<dyn EventSink> = WalEventSink::arc(state);
        let _clone = Arc::clone(&sink);

        let _ = std::fs::remove_file(path);
    }
}

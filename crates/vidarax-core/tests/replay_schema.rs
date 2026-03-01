use std::fs;
use std::path::{Path, PathBuf};

use jsonschema::validator_for;
use serde::Deserialize;
use serde_json::{json, Value};
use vidarax_core::gate::{FrameSignal, GateConfig, GateEngine, GateEvent};

#[derive(Debug, Deserialize)]
struct SignalFixture {
    frame_index: u64,
    pts_ms: u64,
    perceptual_hash: u64,
    luma_mean: f32,
    flicker_score: f32,
    ghosting_score: f32,
    noise_variance_score: f32,
}

impl From<SignalFixture> for FrameSignal {
    fn from(value: SignalFixture) -> Self {
        Self {
            frame_index: value.frame_index,
            pts_ms: value.pts_ms,
            perceptual_hash: value.perceptual_hash,
            luma_mean: value.luma_mean,
            flicker_score: value.flicker_score,
            ghosting_score: value.ghosting_score,
            noise_variance_score: value.noise_variance_score,
        }
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn load_json(path: &Path) -> Value {
    let raw = fs::read_to_string(path).expect("read json file");
    serde_json::from_str(&raw).expect("parse json")
}

fn fnv_events(events: &[GateEvent]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for event in events {
        for b in event
            .event_type
            .as_code()
            .bytes()
            .chain([b':'])
            .chain(event.reason_code.as_str().bytes())
        {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        for b in event.frame_index.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }
    hash
}

trait GateEventCode {
    fn as_code(&self) -> &'static str;
}

impl GateEventCode for vidarax_core::gate::GateEventType {
    fn as_code(&self) -> &'static str {
        match self {
            vidarax_core::gate::GateEventType::KeepKeyframe => "keep",
            vidarax_core::gate::GateEventType::SuspectArtifact => "artifact",
            vidarax_core::gate::GateEventType::Skip => "skip",
        }
    }
}

#[test]
fn deterministic_replay_hash_is_stable() {
    let root = repo_root();
    let fixture_path = root.join("fixtures/replay/frame-signals.json");
    let signals: Vec<SignalFixture> =
        serde_json::from_value(load_json(&fixture_path)).expect("signals fixture");
    let signals: Vec<FrameSignal> = signals.into_iter().map(Into::into).collect();
    let config = GateConfig::default();

    let run_once = || {
        let mut gate = GateEngine::new(config.clone());
        signals
            .iter()
            .copied()
            .map(|s| gate.process(s))
            .collect::<Vec<_>>()
    };

    let first = run_once();
    let second = run_once();
    assert_eq!(first, second, "gate replay must be deterministic");

    let fingerprint = fnv_events(&first);
    assert_eq!(fingerprint, 0xbe0bc19beda6ad48);
}

#[test]
fn schemas_accept_reference_fixtures() {
    let root = repo_root();
    let processing_schema = load_json(&root.join("schemas/processing-config.schema.json"));
    let frame_schema = load_json(&root.join("schemas/frame-metadata.schema.json"));
    let processing_instance = load_json(&root.join("fixtures/replay/processing-config.valid.json"));
    let frame_instance = load_json(&root.join("fixtures/replay/frame-metadata.valid.json"));

    let processing_validator = validator_for(&processing_schema).expect("processing schema");
    let frame_validator = validator_for(&frame_schema).expect("frame schema");
    assert!(
        processing_validator.is_valid(&processing_instance),
        "processing fixture is invalid"
    );
    assert!(
        frame_validator.is_valid(&frame_instance),
        "frame fixture is invalid"
    );
}

#[test]
fn schemas_reject_missing_required_fields() {
    let root = repo_root();
    let frame_schema = load_json(&root.join("schemas/frame-metadata.schema.json"));
    let validator = validator_for(&frame_schema).expect("frame schema");

    let invalid = json!({
        "run_id": "run-local-0001",
        "stream_id": "stream-0"
    });

    let errors = validator
        .iter_errors(&invalid)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        !errors.is_empty(),
        "invalid fixture must fail schema validation"
    );
}

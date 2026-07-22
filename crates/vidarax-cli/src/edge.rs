use std::fs;
use std::io::Read as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use clap::{Subcommand, ValueEnum};
use ed25519_dalek::{Signer as _, SigningKey, Verifier as _, VerifyingKey};
use fs2::FileExt as _;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const MANIFEST_SCHEMA_VERSION: u32 = 2;
const DEFAULT_MAX_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MIN_FREE_SPACE_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Subcommand, Debug)]
pub(crate) enum EdgeCommands {
    /// Create a signing key pair. Keep the private key off the edge device.
    Keygen {
        #[arg(long)]
        private_key: PathBuf,
        #[arg(long)]
        public_key: PathBuf,
    },
    /// Initialize an edge node with its device identity and update public key.
    Enroll {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        device_id: String,
        #[arg(long)]
        hardware_cohort: String,
        #[arg(long)]
        public_key: PathBuf,
        /// Absolute executable invoked as HOOK ACTION MODEL_PATH RELEASE_ID.
        #[arg(long)]
        activation_hook: PathBuf,
    },
    /// Sign an unsigned model release manifest.
    Sign {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        private_key: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Download, verify, and stage a signed model release.
    Apply {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Poll a desired-manifest URL and apply each new signed release.
    Watch {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        manifest_url: String,
        #[arg(long, default_value_t = 60)]
        interval_secs: u64,
        /// Fetch once and exit; useful for provisioning and health checks.
        #[arg(long)]
        once: bool,
    },
    /// Record candidate health and advance shadow/canary or roll it back.
    Report {
        #[arg(long)]
        state_dir: PathBuf,
        /// Candidate release that produced this health report.
        #[arg(long)]
        release_id: String,
        /// Candidate stage that produced this health report.
        #[arg(long)]
        stage: RolloutStage,
        #[arg(long)]
        samples: u64,
        #[arg(long)]
        success_rate: f64,
        #[arg(long)]
        p95_ms: u64,
        #[arg(long)]
        rss_bytes: u64,
    },
    /// Print local enrollment and release state.
    Status {
        #[arg(long)]
        state_dir: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    uri: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RolloutStage {
    Shadow,
    Canary,
    Active,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Acceptance {
    minimum_samples: u64,
    minimum_success_rate: f64,
    maximum_p95_ms: u64,
    maximum_rss_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelManifest {
    schema_version: u32,
    /// Signed, monotonically increasing sequence for this hardware cohort.
    sequence: u64,
    release_id: String,
    model_id: String,
    hardware_cohort: String,
    stage: RolloutStage,
    artifact: Artifact,
    acceptance: Acceptance,
    #[serde(default)]
    signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceState {
    schema_version: u32,
    device_id: String,
    hardware_cohort: String,
    update_public_key: String,
    activation_hook: PathBuf,
    current_release: Option<String>,
    previous_release: Option<String>,
    candidate: Option<CandidateState>,
    last_rejected_release: Option<String>,
    #[serde(default)]
    highest_release_sequence: u64,
    #[serde(default)]
    pending_transition: Option<PendingTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateState {
    release_id: String,
    stage: RolloutStage,
    manifest_path: PathBuf,
    artifact_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum HookAction {
    StageShadow,
    StageCanary,
    Activate,
    Rollback,
}

impl HookAction {
    const fn as_str(self) -> &'static str {
        match self {
            Self::StageShadow => "stage_shadow",
            Self::StageCanary => "stage_canary",
            Self::Activate => "activate",
            Self::Rollback => "rollback",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingTransition {
    action: HookAction,
    candidate: CandidateState,
}

pub(crate) async fn run(command: EdgeCommands, json: bool) -> Result<(), String> {
    match command {
        EdgeCommands::Keygen {
            private_key,
            public_key,
        } => keygen(&private_key, &public_key, json),
        EdgeCommands::Enroll {
            state_dir,
            device_id,
            hardware_cohort,
            public_key,
            activation_hook,
        } => enroll(
            &state_dir,
            &device_id,
            &hardware_cohort,
            &public_key,
            &activation_hook,
            json,
        ),
        EdgeCommands::Sign {
            manifest,
            private_key,
            output,
        } => sign_manifest(&manifest, &private_key, &output, json),
        EdgeCommands::Apply {
            state_dir,
            manifest,
        } => apply(&state_dir, &manifest, json).await,
        EdgeCommands::Watch {
            state_dir,
            manifest_url,
            interval_secs,
            once,
        } => watch(&state_dir, &manifest_url, interval_secs, once, json).await,
        EdgeCommands::Report {
            state_dir,
            release_id,
            stage,
            samples,
            success_rate,
            p95_ms,
            rss_bytes,
        } => report(
            &state_dir,
            &release_id,
            stage,
            samples,
            success_rate,
            p95_ms,
            rss_bytes,
            json,
        ),
        EdgeCommands::Status { state_dir } => status(&state_dir, json),
    }
}

fn keygen(private_path: &Path, public_path: &Path, json: bool) -> Result<(), String> {
    refuse_existing(private_path)?;
    refuse_existing(public_path)?;
    let signing = SigningKey::generate(&mut OsRng);
    atomic_write(private_path, hex::encode(signing.to_bytes()).as_bytes())?;
    atomic_write(
        public_path,
        hex::encode(signing.verifying_key().to_bytes()).as_bytes(),
    )?;
    print_value(
        json,
        serde_json::json!({"private_key": private_path, "public_key": public_path}),
        "created edge update signing key pair",
    )
}

fn enroll(
    state_dir: &Path,
    device_id: &str,
    hardware_cohort: &str,
    public_key: &Path,
    activation_hook: &Path,
    json: bool,
) -> Result<(), String> {
    let _lock = DeviceLock::acquire(state_dir)?;
    refuse_existing(&state_dir.join("device.json"))?;
    validate_identifier("device_id", device_id)?;
    validate_identifier("hardware_cohort", hardware_cohort)?;
    if !activation_hook.is_absolute() {
        return Err("activation_hook must be an absolute path".to_string());
    }
    let activation_hook = fs::canonicalize(activation_hook).map_err(|error| {
        format!(
            "failed to resolve activation hook {}: {error}",
            activation_hook.display()
        )
    })?;
    if !activation_hook.is_file() {
        return Err("activation_hook must resolve to a regular file".to_string());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if fs::metadata(&activation_hook)
            .map_err(|error| format!("failed to inspect activation hook: {error}"))?
            .permissions()
            .mode()
            & 0o111
            == 0
        {
            return Err("activation_hook must be executable".to_string());
        }
    }
    let public_key = read_hex_32(public_key, "public key")?;
    VerifyingKey::from_bytes(&public_key).map_err(|e| format!("invalid public key: {e}"))?;
    fs::create_dir_all(state_dir.join("releases"))
        .map_err(|e| format!("failed to create {}: {e}", state_dir.display()))?;
    let state = DeviceState {
        schema_version: 1,
        device_id: device_id.to_string(),
        hardware_cohort: hardware_cohort.to_string(),
        update_public_key: hex::encode(public_key),
        activation_hook,
        current_release: None,
        previous_release: None,
        candidate: None,
        last_rejected_release: None,
        highest_release_sequence: 0,
        pending_transition: None,
    };
    write_state(state_dir, &state)?;
    print_value(
        json,
        serde_json::to_value(&state).map_err(|e| e.to_string())?,
        "edge node enrolled",
    )
}

fn sign_manifest(
    manifest_path: &Path,
    private_key: &Path,
    output: &Path,
    json: bool,
) -> Result<(), String> {
    let mut manifest = read_manifest(manifest_path)?;
    validate_manifest(&manifest)?;
    let signing = SigningKey::from_bytes(&read_hex_32(private_key, "private key")?);
    manifest.signature.clear();
    let signature = signing.sign(&signing_bytes(&manifest)?);
    manifest.signature = hex::encode(signature.to_bytes());
    atomic_json(output, &manifest)?;
    print_value(
        json,
        serde_json::json!({"release_id": manifest.release_id, "output": output}),
        "signed model release manifest",
    )
}

async fn apply(state_dir: &Path, manifest_path: &Path, json: bool) -> Result<(), String> {
    let _lock = DeviceLock::acquire(state_dir)?;
    let mut state = read_state(state_dir)?;
    recover_pending_transition(state_dir, &mut state)?;
    prune_releases(state_dir, &state)?;
    let manifest = read_manifest(manifest_path)?;
    validate_manifest(&manifest)?;
    verify_manifest(&manifest, &state.update_public_key)?;
    if manifest.hardware_cohort != state.hardware_cohort {
        return Err(format!(
            "manifest hardware cohort {} does not match enrolled cohort {}",
            manifest.hardware_cohort, state.hardware_cohort
        ));
    }
    let release_dir = state_dir.join("releases").join(&manifest.release_id);
    let artifact_path = release_dir.join("model.bin");
    let stored_manifest = release_dir.join("manifest.json");
    if stored_manifest.exists() {
        let stored = read_manifest(&stored_manifest)?;
        if signing_bytes(&stored)? != signing_bytes(&manifest)? {
            return Err(format!(
                "release_id {} is already bound to a different signed manifest",
                manifest.release_id
            ));
        }
    }
    if state.current_release.as_deref() == Some(&manifest.release_id)
        || state
            .candidate
            .as_ref()
            .is_some_and(|candidate| candidate.release_id == manifest.release_id)
    {
        return print_value(
            json,
            serde_json::json!({"release_id": manifest.release_id, "unchanged": true}),
            "desired model release is already present",
        );
    }
    if state.last_rejected_release.as_deref() == Some(&manifest.release_id) {
        return print_value(
            json,
            serde_json::json!({"release_id": manifest.release_id, "unchanged": true, "state": "rejected"}),
            "desired model release was already rejected",
        );
    }
    if manifest.sequence <= state.highest_release_sequence {
        return Err(format!(
            "release sequence {} is not newer than device high-water mark {}",
            manifest.sequence, state.highest_release_sequence
        ));
    }

    if stored_manifest.exists() {
        verify_artifact_file(&manifest.artifact, &artifact_path)?;
    } else {
        if release_dir.exists() {
            return Err(format!(
                "release_id {} has an incomplete existing release directory",
                manifest.release_id
            ));
        }
        reserve_artifact_space(state_dir, manifest.artifact.size_bytes)?;
        let staging_dir = state_dir.join("releases").join(format!(
            ".{}.staging-{}",
            manifest.release_id,
            std::process::id()
        ));
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir).map_err(|error| {
                format!(
                    "failed to clear stale release staging directory {}: {error}",
                    staging_dir.display()
                )
            })?;
        }
        fs::create_dir_all(&staging_dir)
            .map_err(|e| format!("failed to create {}: {e}", staging_dir.display()))?;
        let staged_artifact = staging_dir.join("model.bin");
        let staged_manifest = staging_dir.join("manifest.json");
        let staged = async {
            download_verified_artifact(&manifest.artifact, &staged_artifact).await?;
            atomic_json(&staged_manifest, &manifest)?;
            fs::rename(&staging_dir, &release_dir).map_err(|error| {
                format!(
                    "failed to publish verified release {}: {error}",
                    manifest.release_id
                )
            })?;
            sync_directory(
                release_dir
                    .parent()
                    .ok_or_else(|| "release directory has no parent".to_string())?,
            )
        }
        .await;
        if staged.is_err() {
            let _ = fs::remove_dir_all(&staging_dir);
        }
        staged?;
    }

    if let Some(candidate) = state.candidate.clone() {
        perform_transition(
            state_dir,
            &mut state,
            PendingTransition {
                action: HookAction::Rollback,
                candidate,
            },
        )?;
    }

    let stage = manifest.stage;
    let action = match stage {
        RolloutStage::Shadow => HookAction::StageShadow,
        RolloutStage::Canary => HookAction::StageCanary,
        RolloutStage::Active => HookAction::Activate,
    };
    // Commit the anti-replay high-water mark and recoverable transition in one
    // atomic state replacement before invoking external code. A crash can
    // replay this exact transition, but can never make an older signed release
    // eligible again.
    state.highest_release_sequence = manifest.sequence;
    state.pending_transition = Some(PendingTransition {
        action,
        candidate: CandidateState {
            release_id: manifest.release_id.clone(),
            stage,
            manifest_path: stored_manifest,
            artifact_path,
        },
    });
    write_state(state_dir, &state)?;
    complete_or_rollback_pending_transition(state_dir, &mut state)?;
    prune_releases(state_dir, &state)?;
    print_value(
        json,
        serde_json::json!({"release_id": manifest.release_id, "stage": stage, "verified": true}),
        "verified and staged model release",
    )
}

async fn watch(
    state_dir: &Path,
    manifest_url: &str,
    interval_secs: u64,
    once: bool,
    json: bool,
) -> Result<(), String> {
    if interval_secs == 0 || interval_secs > 86_400 {
        return Err("interval_secs must be in [1, 86400]".to_string());
    }
    loop {
        let bytes = read_bounded_uri(manifest_url, 64 * 1024).await?;
        serde_json::from_slice::<ModelManifest>(&bytes)
            .map_err(|e| format!("invalid desired manifest: {e}"))?;
        let path = state_dir.join(format!("desired-manifest-{}.json", std::process::id()));
        atomic_write(&path, &bytes)?;
        let applied = apply(state_dir, &path, json).await;
        let _ = fs::remove_file(&path);
        applied?;
        if once {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    state_dir: &Path,
    expected_release_id: &str,
    expected_stage: RolloutStage,
    samples: u64,
    success_rate: f64,
    p95_ms: u64,
    rss_bytes: u64,
    json: bool,
) -> Result<(), String> {
    let _lock = DeviceLock::acquire(state_dir)?;
    validate_identifier("release_id", expected_release_id)?;
    if !(0.0..=1.0).contains(&success_rate) || !success_rate.is_finite() {
        return Err("success_rate must be finite and in [0, 1]".to_string());
    }
    let mut state = read_state(state_dir)?;
    recover_pending_transition(state_dir, &mut state)?;
    let candidate = state
        .candidate
        .as_ref()
        .cloned()
        .ok_or_else(|| "there is no staged candidate".to_string())?;
    if candidate.release_id != expected_release_id || candidate.stage != expected_stage {
        return Err(format!(
            "stale health report for {} {:?}; current candidate is {} {:?}",
            expected_release_id, expected_stage, candidate.release_id, candidate.stage
        ));
    }
    let manifest = read_manifest(&candidate.manifest_path)?;
    let accepted = samples >= manifest.acceptance.minimum_samples
        && success_rate >= manifest.acceptance.minimum_success_rate
        && p95_ms <= manifest.acceptance.maximum_p95_ms
        && rss_bytes <= manifest.acceptance.maximum_rss_bytes;
    let release_id = candidate.release_id.clone();
    let outcome = if !accepted {
        perform_transition(
            state_dir,
            &mut state,
            PendingTransition {
                action: HookAction::Rollback,
                candidate,
            },
        )?;
        "rolled_back"
    } else if candidate.stage == RolloutStage::Shadow {
        let mut candidate = candidate;
        candidate.stage = RolloutStage::Canary;
        perform_transition(
            state_dir,
            &mut state,
            PendingTransition {
                action: HookAction::StageCanary,
                candidate,
            },
        )?;
        "advanced_to_canary"
    } else {
        let mut candidate = candidate;
        candidate.stage = RolloutStage::Active;
        perform_transition(
            state_dir,
            &mut state,
            PendingTransition {
                action: HookAction::Activate,
                candidate,
            },
        )?;
        "activated"
    };
    prune_releases(state_dir, &state)?;
    print_value(
        json,
        serde_json::json!({"release_id": release_id, "outcome": outcome, "accepted": accepted}),
        outcome,
    )
}

fn perform_transition(
    state_dir: &Path,
    state: &mut DeviceState,
    transition: PendingTransition,
) -> Result<(), String> {
    state.pending_transition = Some(transition);
    write_state(state_dir, state)?;
    complete_or_rollback_pending_transition(state_dir, state)
}

fn complete_or_rollback_pending_transition(
    state_dir: &Path,
    state: &mut DeviceState,
) -> Result<(), String> {
    match complete_pending_transition(state_dir, state) {
        Ok(()) => Ok(()),
        Err(error) => {
            let Some(pending) = state.pending_transition.clone() else {
                return Err(error);
            };
            if pending.action == HookAction::Rollback {
                return Err(error);
            }

            let failed_action = pending.action;
            state.pending_transition = Some(PendingTransition {
                action: HookAction::Rollback,
                candidate: pending.candidate,
            });
            write_state(state_dir, state)?;
            match complete_pending_transition(state_dir, state) {
                Ok(()) => Err(format!(
                    "{} transition failed and was rolled back: {error}",
                    failed_action.as_str()
                )),
                Err(rollback_error) => Err(format!(
                    "{} transition failed: {error}; rollback also failed: {rollback_error}",
                    failed_action.as_str()
                )),
            }
        }
    }
}

fn recover_pending_transition(state_dir: &Path, state: &mut DeviceState) -> Result<(), String> {
    if state.pending_transition.is_some() {
        complete_or_rollback_pending_transition(state_dir, state)?;
    }
    Ok(())
}

fn complete_pending_transition(state_dir: &Path, state: &mut DeviceState) -> Result<(), String> {
    let transition = state
        .pending_transition
        .clone()
        .ok_or_else(|| "there is no pending edge transition".to_string())?;
    run_transition_hook(
        state,
        transition.action,
        &transition.candidate.artifact_path,
        &transition.candidate.release_id,
    )?;

    match transition.action {
        HookAction::StageShadow | HookAction::StageCanary => {
            state.candidate = Some(transition.candidate);
        }
        HookAction::Activate => {
            write_current_pointer(state_dir, &transition.candidate.artifact_path)?;
            state.previous_release = state.current_release.take();
            state.current_release = Some(transition.candidate.release_id);
            state.candidate = None;
        }
        HookAction::Rollback => {
            state.candidate = None;
            state.last_rejected_release = Some(transition.candidate.release_id);
        }
    }
    state.pending_transition = None;
    write_state(state_dir, state)
}

fn reserve_artifact_space(state_dir: &Path, artifact_bytes: u64) -> Result<(), String> {
    let maximum = std::env::var("VIDARAX_EDGE_MAX_ARTIFACT_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| "VIDARAX_EDGE_MAX_ARTIFACT_BYTES must be an integer".to_string())
        })
        .transpose()?
        .unwrap_or(DEFAULT_MAX_ARTIFACT_BYTES);
    if maximum == 0 || artifact_bytes > maximum {
        return Err(format!(
            "signed artifact size {artifact_bytes} exceeds device limit {maximum}"
        ));
    }
    let available = fs2::available_space(state_dir)
        .map_err(|error| format!("failed to inspect edge storage capacity: {error}"))?;
    let required = artifact_bytes.saturating_add(MIN_FREE_SPACE_HEADROOM_BYTES);
    if available < required {
        return Err(format!(
            "insufficient edge storage: {available} bytes available, {required} required"
        ));
    }
    Ok(())
}

fn prune_releases(state_dir: &Path, state: &DeviceState) -> Result<(), String> {
    let releases = state_dir.join("releases");
    let mut keep = std::collections::HashSet::new();
    if let Some(release) = &state.current_release {
        keep.insert(release.as_str());
    }
    if let Some(release) = &state.previous_release {
        keep.insert(release.as_str());
    }
    if let Some(candidate) = &state.candidate {
        keep.insert(candidate.release_id.as_str());
    }
    if let Some(pending) = &state.pending_transition {
        keep.insert(pending.candidate.release_id.as_str());
    }
    for entry in fs::read_dir(&releases)
        .map_err(|error| format!("failed to inspect {}: {error}", releases.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to inspect release entry: {error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if keep.contains(name) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            fs::remove_dir_all(&path)
                .map_err(|error| format!("failed to prune {}: {error}", path.display()))?;
        }
    }
    sync_directory(&releases)
}

fn run_transition_hook(
    state: &DeviceState,
    action: HookAction,
    artifact_path: &Path,
    release_id: &str,
) -> Result<(), String> {
    let mut command = std::process::Command::new(&state.activation_hook);
    command
        .arg(action.as_str())
        .arg(artifact_path)
        .arg(release_id)
        .env("VIDARAX_EDGE_ACTION", action.as_str())
        .env("VIDARAX_EDGE_MODEL_PATH", artifact_path)
        .env("VIDARAX_EDGE_RELEASE_ID", release_id);
    if let Some(current_release) = &state.current_release {
        command.env("VIDARAX_EDGE_CURRENT_RELEASE", current_release);
    }
    let current_model = state.current_release.as_ref().and_then(|release| {
        artifact_path
            .parent()
            .and_then(Path::parent)
            .map(|releases| releases.join(release).join("model.bin"))
    });
    if let Some(current_model) = current_model {
        command.env("VIDARAX_EDGE_CURRENT_MODEL", current_model);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start edge transition hook: {error}"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => return Err(format!("activation hook exited with {status}")),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("activation hook exceeded 30 seconds".to_string());
            }
            Err(error) => return Err(format!("edge transition hook wait failed: {error}")),
        }
    }
}

struct DeviceLock(fs::File);

impl DeviceLock {
    fn acquire(state_dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(state_dir)
            .map_err(|error| format!("failed to create {}: {error}", state_dir.display()))?;
        let file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(state_dir.join("device.lock"))
            .map_err(|error| format!("failed to open edge state lock: {error}"))?;
        file.try_lock_exclusive()
            .map_err(|_| "another edge state operation is already running".to_string())?;
        Ok(Self(file))
    }
}

impl Drop for DeviceLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn status(state_dir: &Path, json: bool) -> Result<(), String> {
    let state = read_state(state_dir)?;
    print_value(
        json,
        serde_json::to_value(&state).map_err(|e| e.to_string())?,
        &format!(
            "device {}: current={}, candidate={}",
            state.device_id,
            state.current_release.as_deref().unwrap_or("none"),
            state
                .candidate
                .as_ref()
                .map_or("none", |candidate| candidate.release_id.as_str())
        ),
    )
}

fn validate_manifest(manifest: &ModelManifest) -> Result<(), String> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported manifest schema {}",
            manifest.schema_version
        ));
    }
    validate_identifier("release_id", &manifest.release_id)?;
    if manifest.sequence == 0 {
        return Err("manifest sequence must be greater than zero".to_string());
    }
    if manifest.stage != RolloutStage::Shadow {
        return Err("new model releases must begin in the shadow stage".to_string());
    }
    if manifest.model_id.trim().is_empty() || manifest.hardware_cohort.trim().is_empty() {
        return Err("model_id and hardware_cohort cannot be empty".to_string());
    }
    let sha_is_valid = hex::decode(&manifest.artifact.sha256).is_ok_and(|bytes| bytes.len() == 32);
    if manifest.artifact.size_bytes == 0
        || !sha_is_valid
        || manifest.acceptance.minimum_samples == 0
        || !(0.0..=1.0).contains(&manifest.acceptance.minimum_success_rate)
        || !manifest.acceptance.minimum_success_rate.is_finite()
        || manifest.acceptance.maximum_p95_ms == 0
        || manifest.acceptance.maximum_rss_bytes == 0
    {
        return Err("manifest contains invalid artifact or acceptance bounds".to_string());
    }
    Ok(())
}

fn verify_manifest(manifest: &ModelManifest, public_key_hex: &str) -> Result<(), String> {
    let public_bytes: [u8; 32] = hex::decode(public_key_hex)
        .map_err(|e| format!("invalid enrolled public key: {e}"))?
        .try_into()
        .map_err(|_| "enrolled public key must contain 32 bytes".to_string())?;
    let public = VerifyingKey::from_bytes(&public_bytes)
        .map_err(|e| format!("invalid enrolled public key: {e}"))?;
    let signature_bytes: [u8; 64] = hex::decode(&manifest.signature)
        .map_err(|e| format!("invalid manifest signature: {e}"))?
        .try_into()
        .map_err(|_| "manifest signature must contain 64 bytes".to_string())?;
    let signature = ed25519_dalek::Signature::from_bytes(&signature_bytes);
    let mut unsigned = manifest.clone();
    unsigned.signature.clear();
    public
        .verify(&signing_bytes(&unsigned)?, &signature)
        .map_err(|_| "manifest signature verification failed".to_string())
}

fn signing_bytes(manifest: &ModelManifest) -> Result<Vec<u8>, String> {
    serde_json::to_vec(manifest).map_err(|e| format!("failed to encode manifest: {e}"))
}

async fn read_bounded_uri(uri: &str, maximum_bytes: usize) -> Result<Vec<u8>, String> {
    if let Some(path) = uri.strip_prefix("file://") {
        if !Path::new(path).is_absolute() {
            return Err("file artifact uri must contain an absolute path".to_string());
        }
        let mut file =
            fs::File::open(path).map_err(|e| format!("failed to read artifact {path}: {e}"))?;
        let mut bytes = Vec::with_capacity(maximum_bytes.min(64 * 1024));
        std::io::Read::by_ref(&mut file)
            .take(maximum_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("failed to read artifact {path}: {e}"))?;
        if bytes.len() > maximum_bytes {
            return Err(format!("remote payload exceeds {maximum_bytes} bytes"));
        }
        return Ok(bytes);
    }
    if !uri.starts_with("https://") {
        return Err("artifact uri must use https:// or file://".to_string());
    }
    let response = https_client()?
        .get(uri)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("artifact download failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("artifact download failed: {e}"))?;
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(format!("remote payload exceeds {maximum_bytes} bytes"));
    }
    let mut response = response;
    let mut bytes = Vec::with_capacity(maximum_bytes.min(64 * 1024));
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("artifact download failed: {e}"))?
    {
        if bytes.len().saturating_add(chunk.len()) > maximum_bytes {
            return Err(format!("remote payload exceeds {maximum_bytes} bytes"));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn download_verified_artifact(artifact: &Artifact, destination: &Path) -> Result<(), String> {
    let temp = destination.with_extension(format!("download-{}", std::process::id()));
    let result = async {
        let mut output = open_private_file(&temp)?;
        let mut digest = Sha256::new();
        let mut received = 0u64;
        if let Some(path) = artifact.uri.strip_prefix("file://") {
            if !Path::new(path).is_absolute() {
                return Err("file artifact uri must contain an absolute path".to_string());
            }
            let mut input =
                fs::File::open(path).map_err(|e| format!("failed to read artifact {path}: {e}"))?;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let count = input
                    .read(&mut buffer)
                    .map_err(|e| format!("failed to read artifact {path}: {e}"))?;
                if count == 0 {
                    break;
                }
                received = received.saturating_add(count as u64);
                if received > artifact.size_bytes {
                    return Err("artifact exceeds its signed size".to_string());
                }
                digest.update(&buffer[..count]);
                output
                    .write_all(&buffer[..count])
                    .map_err(|e| format!("failed to write {}: {e}", temp.display()))?;
            }
        } else {
            if !artifact.uri.starts_with("https://") {
                return Err("artifact uri must use https:// or file://".to_string());
            }
            let mut response = https_client()?
                .get(&artifact.uri)
                .timeout(std::time::Duration::from_secs(120))
                .send()
                .await
                .map_err(|e| format!("artifact download failed: {e}"))?
                .error_for_status()
                .map_err(|e| format!("artifact download failed: {e}"))?;
            if response
                .content_length()
                .is_some_and(|length| length != artifact.size_bytes)
            {
                return Err("artifact content-length does not match its signed size".to_string());
            }
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|e| format!("artifact download failed: {e}"))?
            {
                received = received.saturating_add(chunk.len() as u64);
                if received > artifact.size_bytes {
                    return Err("artifact exceeds its signed size".to_string());
                }
                digest.update(&chunk);
                output
                    .write_all(&chunk)
                    .map_err(|e| format!("failed to write {}: {e}", temp.display()))?;
            }
        }
        if received != artifact.size_bytes {
            return Err(format!(
                "artifact size mismatch: expected {}, received {received}",
                artifact.size_bytes
            ));
        }
        if hex::encode(digest.finalize()) != artifact.sha256.to_ascii_lowercase() {
            return Err("artifact sha256 mismatch".to_string());
        }
        output
            .sync_all()
            .map_err(|e| format!("failed to persist {}: {e}", temp.display()))?;
        drop(output);
        fs::rename(&temp, destination)
            .map_err(|e| format!("failed to replace {}: {e}", destination.display()))
    }
    .await;
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn verify_artifact_file(artifact: &Artifact, path: &Path) -> Result<(), String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("failed to open stored artifact {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            format!("failed to read stored artifact {}: {error}", path.display())
        })?;
        if count == 0 {
            break;
        }
        size = size.saturating_add(count as u64);
        if size > artifact.size_bytes {
            return Err("stored artifact exceeds its signed size".to_string());
        }
        digest.update(&buffer[..count]);
    }
    if size != artifact.size_bytes {
        return Err("stored artifact size does not match its signed size".to_string());
    }
    if hex::encode(digest.finalize()) != artifact.sha256.to_ascii_lowercase() {
        return Err("stored artifact sha256 mismatch".to_string());
    }
    Ok(())
}

fn https_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("failed to build edge download client: {error}"))
}

fn read_manifest(path: &Path) -> Result<ModelManifest, String> {
    read_json(path, "model manifest", 64 * 1024)
}

fn read_state(state_dir: &Path) -> Result<DeviceState, String> {
    read_json(&state_dir.join("device.json"), "device state", 256 * 1024)
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    label: &str,
    maximum_bytes: usize,
) -> Result<T, String> {
    let mut file = fs::File::open(path)
        .map_err(|e| format!("failed to read {label} {}: {e}", path.display()))?;
    let mut bytes = Vec::with_capacity(maximum_bytes.min(64 * 1024));
    std::io::Read::by_ref(&mut file)
        .take(maximum_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read {label} {}: {e}", path.display()))?;
    if bytes.len() > maximum_bytes {
        return Err(format!(
            "{label} {} exceeds {maximum_bytes} bytes",
            path.display()
        ));
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid {label} {}: {e}", path.display()))
}

fn write_state(state_dir: &Path, state: &DeviceState) -> Result<(), String> {
    atomic_json(&state_dir.join("device.json"), state)
}

fn write_current_pointer(state_dir: &Path, artifact_path: &Path) -> Result<(), String> {
    atomic_write(
        &state_dir.join("current-model"),
        artifact_path.to_string_lossy().as_bytes(),
    )
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = open_private_file(&temp)?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("failed to persist {}: {e}", temp.display()))?;
    fs::rename(&temp, path).map_err(|e| format!("failed to replace {}: {e}", path.display()))?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), String> {
    fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to persist directory {}: {error}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn open_private_file(path: &Path) -> Result<fs::File, String> {
    let mut options = fs::OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options
        .open(path)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn read_hex_32(path: &Path, label: &str) -> Result<[u8; 32], String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {label} {}: {e}", path.display()))?;
    hex::decode(text.trim())
        .map_err(|e| format!("invalid {label}: {e}"))?
        .try_into()
        .map_err(|_| format!("{label} must contain 32 bytes"))
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{label} must be 1-128 URL-safe characters"));
    }
    Ok(())
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        Err(format!("refusing to overwrite {}", path.display()))
    } else {
        Ok(())
    }
}

fn print_value(json: bool, value: serde_json::Value, message: &str) -> Result<(), String> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
        );
    } else {
        println!("{message}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "vidarax-edge-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn manifest_for(artifact: &Path, release_id: &str, cohort: &str) -> ModelManifest {
        let bytes = fs::read(artifact).unwrap();
        ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            sequence: 1,
            release_id: release_id.to_string(),
            model_id: "detector-v1".to_string(),
            hardware_cohort: cohort.to_string(),
            stage: RolloutStage::Shadow,
            artifact: Artifact {
                uri: format!("file://{}", artifact.display()),
                sha256: hex::encode(Sha256::digest(&bytes)),
                size_bytes: bytes.len() as u64,
            },
            acceptance: Acceptance {
                minimum_samples: 10,
                minimum_success_rate: 0.95,
                maximum_p95_ms: 50,
                maximum_rss_bytes: 512 * 1024 * 1024,
            },
            signature: String::new(),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn signed_shadow_release_advances_through_canary_to_active() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = sidecar("rollout");
        let private = root.join("control/private.key");
        let public = root.join("control/public.key");
        let hook = root.join("control/activate");
        let state_dir = root.join("device");
        keygen(&private, &public, false).unwrap();
        fs::write(
            &hook,
            "#!/bin/sh\nprintf '%s|%s|%s\\n' \"$1\" \"$2\" \"$3\" >> \"$0.called\"\n",
        )
        .unwrap();
        fs::set_permissions(&hook, fs::Permissions::from_mode(0o700)).unwrap();
        enroll(&state_dir, "edge-01", "jetson-orin", &public, &hook, false).unwrap();

        let artifact = root.join("model.safetensors");
        fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        fs::write(&artifact, b"model-v1").unwrap();
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        atomic_json(
            &unsigned,
            &manifest_for(&artifact, "release-001", "jetson-orin"),
        )
        .unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();
        apply(&state_dir, &signed, false).await.unwrap();

        let stale = report(
            &state_dir,
            "release-stale",
            RolloutStage::Shadow,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap_err();
        assert!(stale.contains("stale health report"));
        report(
            &state_dir,
            "release-001",
            RolloutStage::Shadow,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();
        assert_eq!(
            read_state(&state_dir).unwrap().candidate.unwrap().stage,
            RolloutStage::Canary
        );
        report(
            &state_dir,
            "release-001",
            RolloutStage::Canary,
            20,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();
        let state = read_state(&state_dir).unwrap();
        assert_eq!(state.current_release.as_deref(), Some("release-001"));
        assert!(state.candidate.is_none());
        assert!(state.pending_transition.is_none());
        assert!(state_dir.join("current-model").exists());
        let activation = fs::read_to_string(hook.with_extension("called")).unwrap();
        assert!(activation
            .lines()
            .next()
            .unwrap()
            .starts_with("stage_shadow|"));
        assert!(activation.contains("stage_canary|"));
        assert!(activation.contains("activate|"));
        assert!(activation.contains("model.bin"));
        assert!(activation.contains("release-001"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_stage_transition_runs_rollback_before_rejecting_candidate() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = sidecar("transition-rollback");
        let private = root.join("control/private.key");
        let public = root.join("control/public.key");
        let hook = root.join("control/activate");
        let state_dir = root.join("device");
        keygen(&private, &public, false).unwrap();
        fs::write(
            &hook,
            "#!/bin/sh\nprintf '%s\\n' \"$1\" >> \"$0.called\"\n[ \"$1\" != stage_canary ]\n",
        )
        .unwrap();
        fs::set_permissions(&hook, fs::Permissions::from_mode(0o700)).unwrap();
        enroll(
            &state_dir,
            "edge-rollback",
            "jetson-orin",
            &public,
            &hook,
            false,
        )
        .unwrap();

        let artifact = root.join("model.bin");
        fs::write(&artifact, b"model-rollback").unwrap();
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        atomic_json(
            &unsigned,
            &manifest_for(&artifact, "release-rollback", "jetson-orin"),
        )
        .unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();
        apply(&state_dir, &signed, false).await.unwrap();

        let error = report(
            &state_dir,
            "release-rollback",
            RolloutStage::Shadow,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap_err();
        assert!(error.contains("was rolled back"));
        let state = read_state(&state_dir).unwrap();
        assert!(state.candidate.is_none());
        assert!(state.pending_transition.is_none());
        assert_eq!(
            state.last_rejected_release.as_deref(),
            Some("release-rollback")
        );
        let calls = fs::read_to_string(hook.with_extension("called")).unwrap();
        assert_eq!(
            calls.lines().collect::<Vec<_>>(),
            vec!["stage_shadow", "stage_canary", "rollback"]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn modified_signed_manifest_is_rejected() {
        let root = sidecar("tamper");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-02",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        let manifest = ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            sequence: 1,
            release_id: "release-002".to_string(),
            model_id: "detector-v2".to_string(),
            hardware_cohort: "jetson-orin".to_string(),
            stage: RolloutStage::Shadow,
            artifact: Artifact {
                uri: "file:///tmp/not-read-before-signature-check".to_string(),
                sha256: "0".repeat(64),
                size_bytes: 1,
            },
            acceptance: Acceptance {
                minimum_samples: 1,
                minimum_success_rate: 0.5,
                maximum_p95_ms: 100,
                maximum_rss_bytes: 1,
            },
            signature: String::new(),
        };
        atomic_json(&unsigned, &manifest).unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();
        let mut changed = read_manifest(&signed).unwrap();
        changed.model_id = "tampered".to_string();
        atomic_json(&signed, &changed).unwrap();
        assert!(apply(&state_dir, &signed, false)
            .await
            .unwrap_err()
            .contains("signature verification failed"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn hardware_cohort_is_enforced_before_download() {
        let root = sidecar("cohort");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        let artifact = root.join("model.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&artifact, b"model").unwrap();
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-03",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        atomic_json(
            &unsigned,
            &manifest_for(&artifact, "release-003", "x86-cuda"),
        )
        .unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();
        fs::remove_file(&artifact).unwrap();

        assert!(apply(&state_dir, &signed, false)
            .await
            .unwrap_err()
            .contains("does not match enrolled cohort"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejected_release_is_not_downloaded_again() {
        let root = sidecar("rejected");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        let artifact = root.join("model.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&artifact, b"model").unwrap();
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-04",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        atomic_json(
            &unsigned,
            &manifest_for(&artifact, "release-004", "jetson-orin"),
        )
        .unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();
        apply(&state_dir, &signed, false).await.unwrap();
        report(
            &state_dir,
            "release-004",
            RolloutStage::Shadow,
            1,
            0.1,
            500,
            1024 * 1024 * 1024,
            false,
        )
        .unwrap();
        fs::remove_file(&artifact).unwrap();

        apply(&state_dir, &signed, false).await.unwrap();
        let state = read_state(&state_dir).unwrap();
        assert_eq!(state.last_rejected_release.as_deref(), Some("release-004"));
        assert!(state.candidate.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn older_signed_release_cannot_replace_a_newer_release() {
        let root = sidecar("anti-rollback");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        let artifact_a = root.join("model-a.bin");
        let artifact_b = root.join("model-b.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&artifact_a, b"model-a").unwrap();
        fs::write(&artifact_b, b"model-b").unwrap();
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-anti-rollback",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();

        let first = manifest_for(&artifact_a, "release-a", "jetson-orin");
        let first_unsigned = root.join("first-unsigned.json");
        let first_signed = root.join("first-signed.json");
        atomic_json(&first_unsigned, &first).unwrap();
        sign_manifest(&first_unsigned, &private, &first_signed, false).unwrap();
        apply(&state_dir, &first_signed, false).await.unwrap();
        report(
            &state_dir,
            "release-a",
            RolloutStage::Shadow,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();
        report(
            &state_dir,
            "release-a",
            RolloutStage::Canary,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();

        let mut second = manifest_for(&artifact_b, "release-b", "jetson-orin");
        second.sequence = 2;
        let second_unsigned = root.join("second-unsigned.json");
        let second_signed = root.join("second-signed.json");
        atomic_json(&second_unsigned, &second).unwrap();
        sign_manifest(&second_unsigned, &private, &second_signed, false).unwrap();
        apply(&state_dir, &second_signed, false).await.unwrap();
        report(
            &state_dir,
            "release-b",
            RolloutStage::Shadow,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();
        report(
            &state_dir,
            "release-b",
            RolloutStage::Canary,
            10,
            0.99,
            40,
            128 * 1024 * 1024,
            false,
        )
        .unwrap();

        let error = apply(&state_dir, &first_signed, false).await.unwrap_err();
        assert!(error.contains("high-water mark"));
        let state = read_state(&state_dir).unwrap();
        assert_eq!(state.current_release.as_deref(), Some("release-b"));
        assert_eq!(state.highest_release_sequence, 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn release_id_cannot_be_rebound_to_different_signed_content() {
        let root = sidecar("release-binding");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        let artifact_a = root.join("model-a.bin");
        let artifact_b = root.join("model-b.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&artifact_a, b"model-a").unwrap();
        fs::write(&artifact_b, b"model-b").unwrap();
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-release-binding",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();

        let first = manifest_for(&artifact_a, "release-fixed", "jetson-orin");
        let first_unsigned = root.join("first-unsigned.json");
        let first_signed = root.join("first-signed.json");
        atomic_json(&first_unsigned, &first).unwrap();
        sign_manifest(&first_unsigned, &private, &first_signed, false).unwrap();
        apply(&state_dir, &first_signed, false).await.unwrap();

        let mut rebound = manifest_for(&artifact_b, "release-fixed", "jetson-orin");
        rebound.sequence = 2;
        let rebound_unsigned = root.join("rebound-unsigned.json");
        let rebound_signed = root.join("rebound-signed.json");
        atomic_json(&rebound_unsigned, &rebound).unwrap();
        sign_manifest(&rebound_unsigned, &private, &rebound_signed, false).unwrap();
        let error = apply(&state_dir, &rebound_signed, false).await.unwrap_err();
        assert!(error.contains("already present") || error.contains("already bound"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn failed_download_can_retry_the_same_signed_release() {
        let root = sidecar("download-retry");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        let artifact = root.join("model.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&artifact, b"correct-model").unwrap();
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-download-retry",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();
        let manifest = manifest_for(&artifact, "release-retry", "jetson-orin");
        let unsigned = root.join("unsigned.json");
        let signed = root.join("signed.json");
        atomic_json(&unsigned, &manifest).unwrap();
        sign_manifest(&unsigned, &private, &signed, false).unwrap();

        fs::write(&artifact, b"broken").unwrap();
        assert!(apply(&state_dir, &signed, false).await.is_err());
        assert!(!state_dir.join("releases/release-retry").exists());

        fs::write(&artifact, b"correct-model").unwrap();
        apply(&state_dir, &signed, false).await.unwrap();
        assert_eq!(
            read_state(&state_dir)
                .unwrap()
                .candidate
                .unwrap()
                .release_id,
            "release-retry"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn enrollment_refuses_to_replace_an_existing_device_identity() {
        let root = sidecar("enrollment-identity");
        let private = root.join("private.key");
        let public = root.join("public.key");
        let state_dir = root.join("device");
        keygen(&private, &public, false).unwrap();
        enroll(
            &state_dir,
            "edge-original",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap();

        let error = enroll(
            &state_dir,
            "edge-replacement",
            "jetson-orin",
            &public,
            Path::new("/usr/bin/true"),
            false,
        )
        .unwrap_err();
        assert!(error.contains("refusing to overwrite"));
        assert_eq!(read_state(&state_dir).unwrap().device_id, "edge-original");
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn signing_keys_are_private_on_disk() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = sidecar("permissions");
        let private = root.join("private.key");
        let public = root.join("public.key");
        keygen(&private, &public, false).unwrap();
        assert_eq!(
            fs::metadata(&private).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(&public).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::remove_dir_all(root).unwrap();
    }
}

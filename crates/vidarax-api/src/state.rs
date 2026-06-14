use arc_swap::ArcSwap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use serde_json::Value;
use vidarax_contracts::lifecycle::StreamState;
use vidarax_core::ingest::pipeline::{create_pipeline, DecodePipeline, PipelineBackend};
use vidarax_core::provider::InferenceProvider;
use vidarax_core::tiered_vlm::DistillationConfig;
use vidarax_core::timeline::{append_event, read_all_events, TimelineEvent};
use vidarax_core::webrtc::session::WebRtcSession;

use crate::ids::{parse_run_sequence, random_run_id};
use crate::inference_metrics::InferenceMetrics;
use crate::security::SecurityPolicy;
use crate::spacetime_client::SpacetimeClient;
use crate::tenant_labels::{LabelMapResult, TenantLabelMaps};
use vidarax_core::metrics::PipelineMetrics;
use vidarax_core::webrtc::session::WebRtcConfig;

/// Maximum number of concurrent WebRTC sessions to prevent memory exhaustion.
const MAX_WEBRTC_SESSIONS: usize = 100;

/// Stream id used for single-stream runs when no explicit stream is attached.
const DEFAULT_STREAM_ID: &str = "stream-0";

/// In-memory store for active WebRTC sessions, keyed by session ID.
/// Each entry stores the owning principal alongside the session.
type SessionMap = Arc<RwLock<HashMap<String, (String, Arc<WebRtcSession>)>>>;

#[derive(Clone)]
pub struct AppState {
    run_seq: Arc<AtomicU64>,
    request_seq: Arc<AtomicU64>,
    event_seq: Arc<AtomicU64>,
    wal_path: Arc<PathBuf>,
    ingest_file_roots: Arc<Vec<PathBuf>>,
    provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    decode_pipeline: Arc<dyn DecodePipeline>,
    security_policy: Arc<SecurityPolicy>,
    inference_metrics: Arc<InferenceMetrics>,
    pipeline_metrics: Arc<PipelineMetrics>,
    distillation_config: Arc<DistillationConfig>,
    run_registry: Arc<ArcSwap<RunRegistry>>,
    tenant_label_maps: Arc<TenantLabelMaps>,
    stream_ttl_secs: u64,
    active_stream_limit: usize,
    spacetime_client: Option<SpacetimeClient>,
    /// Active WebRTC peer connections indexed by session ID.
    sessions: SessionMap,
    /// WebRTC configuration (STUN/TURN servers, token rate limit).
    webrtc_config: WebRtcConfig,
}

struct AppStateConfig {
    wal_path: PathBuf,
    provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    security_policy: SecurityPolicy,
    stream_ttl_secs: u64,
    active_stream_limit: usize,
}

impl AppStateConfig {
    fn for_tests(wal_path: PathBuf) -> Self {
        Self {
            wal_path,
            provider: None,
            security_policy: SecurityPolicy::from_config_for_tests(),
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
        }
    }

    fn build(self) -> AppState {
        AppState {
            run_seq: Arc::new(AtomicU64::new(0)),
            request_seq: Arc::new(AtomicU64::new(0)),
            event_seq: Arc::new(AtomicU64::new(0)),
            wal_path: Arc::new(self.wal_path),
            ingest_file_roots: Arc::new(default_test_ingest_roots()),
            provider: self.provider,
            decode_pipeline: default_test_decode_pipeline(),
            security_policy: Arc::new(self.security_policy),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
            distillation_config: Arc::new(DistillationConfig::default()),
            run_registry: Arc::new(ArcSwap::from_pointee(RunRegistry::default())),
            tenant_label_maps: Arc::new(TenantLabelMaps::default()),
            stream_ttl_secs: self.stream_ttl_secs.max(1),
            active_stream_limit: self.active_stream_limit.max(1),
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            webrtc_config: WebRtcConfig::default(),
        }
    }
}

impl AppState {
    pub fn from_wal(
        wal_path: PathBuf,
        ingest_file_roots: Vec<PathBuf>,
        provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
        decode_pipeline: Arc<dyn DecodePipeline>,
        security_policy: SecurityPolicy,
        stream_ttl_secs: u64,
        active_stream_limit: usize,
        webrtc_config: WebRtcConfig,
        distillation: DistillationConfig,
    ) -> Result<Self, String> {
        let existing_events = read_all_events(&wal_path).map_err(|err| err.to_string())?;
        let run_registry = Arc::new(ArcSwap::from(build_run_registry(&existing_events)));
        let tenant_label_maps = Arc::new(TenantLabelMaps::from_env()?);
        let max_run_seq = existing_events.iter().fold(0u64, |acc, event| {
            let from_legacy = parse_run_sequence(&event.run_id).unwrap_or(0);
            acc.max(from_legacy)
        });
        let run_count = existing_events
            .iter()
            .filter(|event| event.kind == "run_created")
            .count() as u64;
        let max_event_seq = existing_events
            .iter()
            .map(|event| event.seq)
            .max()
            .unwrap_or(0);

        Ok(Self {
            run_seq: Arc::new(AtomicU64::new(run_count.max(max_run_seq))),
            request_seq: Arc::new(AtomicU64::new(0)),
            event_seq: Arc::new(AtomicU64::new(max_event_seq)),
            wal_path: Arc::new(wal_path),
            ingest_file_roots: Arc::new(ingest_file_roots),
            provider,
            decode_pipeline,
            security_policy: Arc::new(security_policy),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
            distillation_config: Arc::new(distillation),
            run_registry,
            tenant_label_maps,
            stream_ttl_secs,
            active_stream_limit: active_stream_limit.max(1),
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            webrtc_config,
        })
    }

    pub fn with_wal_for_tests(wal_path: PathBuf) -> Self {
        AppStateConfig::for_tests(wal_path).build()
    }

    pub fn with_wal_for_tests_and_endpoints(
        wal_path: PathBuf,
        provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    ) -> Self {
        Self::with_wal_for_tests_full(
            wal_path,
            provider,
            SecurityPolicy::from_config_for_tests(),
        )
    }

    pub fn with_wal_for_tests_full(
        wal_path: PathBuf,
        provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
        security_policy: SecurityPolicy,
    ) -> Self {
        AppStateConfig {
            provider,
            security_policy,
            ..AppStateConfig::for_tests(wal_path)
        }
        .build()
    }

    pub fn with_wal_for_tests_runtime(
        wal_path: PathBuf,
        provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
        security_policy: SecurityPolicy,
        stream_ttl_secs: u64,
        active_stream_limit: usize,
    ) -> Self {
        AppStateConfig {
            provider,
            security_policy,
            stream_ttl_secs,
            active_stream_limit,
            ..AppStateConfig::for_tests(wal_path)
        }
        .build()
    }

    /// Attach a `SpacetimeClient` to this state (builder pattern).
    ///
    /// ```no_run
    /// use vidarax_api::spacetime_client::SpacetimeClient;
    /// # let state: vidarax_api::AppState = todo!();
    /// let state = state.with_spacetime_client(
    ///     SpacetimeClient::new("http://127.0.0.1:3000", "vidarax")
    /// );
    /// ```
    pub fn with_spacetime_client(mut self, client: SpacetimeClient) -> Self {
        self.spacetime_client = Some(client);
        self
    }

    /// Return the attached `SpacetimeClient`, if any.
    pub fn spacetime_client(&self) -> Option<&SpacetimeClient> {
        self.spacetime_client.as_ref()
    }

    // -----------------------------------------------------------------------
    // WebRTC session management
    // -----------------------------------------------------------------------

    /// Insert a new WebRTC session bound to the given principal.
    ///
    /// Returns `false` if the session ID already exists (collision) or the
    /// global session limit has been reached.
    pub async fn insert_session(
        &self,
        sess_id: String,
        principal: String,
        session: Arc<WebRtcSession>,
    ) -> bool {
        let mut map = self.sessions.write().await;
        if map.len() >= MAX_WEBRTC_SESSIONS {
            return false;
        }
        if map.contains_key(&sess_id) {
            return false;
        }
        map.insert(sess_id, (principal, session));
        true
    }

    /// Look up a WebRTC session by ID.  Returns `None` if not found.
    pub async fn get_session(&self, sess_id: &str) -> Option<(String, Arc<WebRtcSession>)> {
        self.sessions.read().await.get(sess_id).cloned()
    }

    /// Remove and return a WebRTC session.  Dropping the returned `Arc`
    /// (or ignoring the return value) triggers peer connection cleanup.
    pub async fn remove_session(&self, sess_id: &str) -> Option<(String, Arc<WebRtcSession>)> {
        self.sessions.write().await.remove(sess_id)
    }

    /// Number of active WebRTC sessions.
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    pub fn next_run_id(&self) -> String {
        // Sequence counter is still incremented for monotonic ordering; it is
        // no longer used as an ID fallback (M-11).
        self.run_seq.fetch_add(1, Ordering::AcqRel);
        random_run_id()
    }

    pub fn next_request_id(&self) -> String {
        let seq = self.request_seq.fetch_add(1, Ordering::AcqRel) + 1;
        format!("req-{seq:016x}")
    }

    pub fn append_run_event(
        &self,
        run_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<TimelineEvent, String> {
        self.append_run_event_for_stream(run_id, DEFAULT_STREAM_ID, kind, payload)
    }

    pub fn append_run_event_for_stream(
        &self,
        run_id: &str,
        stream_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<TimelineEvent, String> {
        let seq = self.event_seq.fetch_add(1, Ordering::AcqRel) + 1;
        let event = TimelineEvent {
            seq,
            run_id: run_id.to_owned(),
            stream_id: stream_id.to_owned(),
            pts_ms: now_epoch_ms(),
            kind: kind.to_owned(),
            payload: payload.to_string(),
        };
        append_event(self.wal_path.as_ref(), &event).map_err(|err| err.to_string())?;
        self.apply_event_to_registry(&event);
        Ok(event)
    }

    pub async fn append_run_event_async(
        &self,
        run_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<TimelineEvent, String> {
        self.append_run_event_for_stream_async(run_id, DEFAULT_STREAM_ID, kind, payload)
            .await
    }

    pub async fn append_run_event_for_stream_async(
        &self,
        run_id: &str,
        stream_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<TimelineEvent, String> {
        let seq = self.event_seq.fetch_add(1, Ordering::AcqRel) + 1;
        let event = TimelineEvent {
            seq,
            run_id: run_id.to_owned(),
            stream_id: stream_id.to_owned(),
            pts_ms: now_epoch_ms(),
            kind: kind.to_owned(),
            payload: payload.to_string(),
        };
        let wal_path = Arc::clone(&self.wal_path);
        let event_for_write = event.clone();
        tokio::task::spawn_blocking(move || append_event(wal_path.as_ref(), &event_for_write))
            .await
            .map_err(|err| format!("timeline append worker join failure: {err}"))?
            .map_err(|err| err.to_string())?;
        self.apply_event_to_registry(&event);
        Ok(event)
    }

    pub fn read_run_events(&self, run_id: &str) -> Result<Vec<TimelineEvent>, String> {
        // TODO(perf): Full WAL scan per request. A per-run index (run_id → file
        // offset range) would make this O(1) seek instead of O(total events).
        let events = read_all_events(self.wal_path.as_ref()).map_err(|err| err.to_string())?;
        Ok(events
            .into_iter()
            .filter(|event| event.run_id == run_id)
            .collect())
    }

    pub async fn read_run_events_async(&self, run_id: &str) -> Result<Vec<TimelineEvent>, String> {
        let wal_path = Arc::clone(&self.wal_path);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let events = read_all_events(wal_path.as_ref()).map_err(|err| err.to_string())?;
            Ok(events
                .into_iter()
                .filter(|event| event.run_id == run_id)
                .collect::<Vec<_>>())
        })
        .await
        .map_err(|err| format!("timeline read worker join failure: {err}"))?
    }

    pub fn read_all_events(&self) -> Result<Vec<TimelineEvent>, String> {
        read_all_events(self.wal_path.as_ref()).map_err(|err| err.to_string())
    }

    pub async fn read_all_events_async(&self) -> Result<Vec<TimelineEvent>, String> {
        let wal_path = Arc::clone(&self.wal_path);
        tokio::task::spawn_blocking(move || {
            read_all_events(wal_path.as_ref()).map_err(|err| err.to_string())
        })
        .await
        .map_err(|err| format!("timeline read worker join failure: {err}"))?
    }

    pub fn metrics_snapshot(&self) -> (u64, u64) {
        let runs = self.run_seq.load(Ordering::Acquire);
        let events = self.event_seq.load(Ordering::Acquire);
        (runs, events)
    }

    pub fn provider(&self) -> Option<&Arc<dyn InferenceProvider + Send + Sync>> {
        self.provider.as_ref()
    }

    pub fn decode_pipeline(&self) -> Arc<dyn DecodePipeline> {
        Arc::clone(&self.decode_pipeline)
    }

    pub fn ingest_file_roots(&self) -> &[PathBuf] {
        self.ingest_file_roots.as_ref().as_slice()
    }

    pub fn security_policy(&self) -> &SecurityPolicy {
        self.security_policy.as_ref()
    }

    pub fn inference_metrics(&self) -> &InferenceMetrics {
        self.inference_metrics.as_ref()
    }

    pub fn pipeline_metrics(&self) -> &PipelineMetrics {
        self.pipeline_metrics.as_ref()
    }

    /// Return the raw `Arc` for cases that need to move the metrics into
    /// a background thread (e.g. WHIP drain task).
    pub fn pipeline_metrics_arc(&self) -> &std::sync::Arc<PipelineMetrics> {
        &self.pipeline_metrics
    }

    /// Return the WebRTC configuration (STUN/TURN servers, token rate limit).
    pub fn webrtc_config(&self) -> &WebRtcConfig {
        &self.webrtc_config
    }

    pub fn distillation_config(&self) -> &DistillationConfig {
        self.distillation_config.as_ref()
    }

    pub fn map_event_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_event(tenant_id, label)
    }

    pub fn map_object_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_object(tenant_id, label)
    }

    pub fn run_runtime_snapshot(&self, run_id: &str, now_ms: u64) -> Option<RunRuntimeSnapshot> {
        let registry = self.run_registry.load();
        let summary = registry.runs.get(run_id)?.snapshot();
        if !summary.created {
            return None;
        }
        let ttl_ms = self.stream_ttl_secs.saturating_mul(1000);
        Some(RunRuntimeSnapshot {
            principal_key: summary.principal_key.to_string(),
            state: apply_expiry(summary.state, summary.last_activity_ms, now_ms, ttl_ms),
            last_activity_ms: summary.last_activity_ms,
        })
    }

    pub fn count_active_runs_for_principal(&self, principal_key: &str, now_ms: u64) -> usize {
        let registry = self.run_registry.load();
        let Some(run_ids) = registry.by_principal.get(principal_key) else {
            return 0;
        };
        let ttl_ms = self.stream_ttl_secs.saturating_mul(1000);
        run_ids
            .iter()
            .filter_map(|run_id| registry.runs.get(run_id))
            .filter(|summary| {
                let summary = summary.snapshot();
                !apply_expiry(summary.state, summary.last_activity_ms, now_ms, ttl_ms).is_terminal()
            })
            .count()
    }

    pub fn stream_ttl_secs(&self) -> u64 {
        self.stream_ttl_secs
    }

    pub fn active_stream_limit(&self) -> usize {
        self.active_stream_limit
    }

    fn apply_event_to_registry(&self, event: &TimelineEvent) {
        if event.kind == "run_created" {
            self.run_registry
                .rcu(|current| Arc::new(current.with_structural_event(event)));
            return;
        }

        let registry = self.run_registry.load();
        if let Some(summary) = registry.runs.get(&event.run_id) {
            summary.apply_event(event);
            return;
        }
        drop(registry);

        self.run_registry
            .rcu(|current| Arc::new(current.with_structural_event(event)));
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn default_test_ingest_roots() -> Vec<PathBuf> {
    let tmp = std::env::temp_dir();
    vec![tmp.canonicalize().unwrap_or(tmp)]
}

fn default_test_decode_pipeline() -> Arc<dyn DecodePipeline> {
    create_pipeline(PipelineBackend::CpuFfmpeg)
}

#[derive(Clone)]
pub struct RunRuntimeSnapshot {
    pub principal_key: String,
    pub state: StreamState,
    pub last_activity_ms: u64,
}

#[derive(Clone, Default)]
struct RunRegistry {
    runs: HashMap<String, Arc<RunState>>,
    by_principal: HashMap<Arc<str>, HashSet<String>>,
}

impl RunRegistry {
    fn with_structural_event(&self, event: &TimelineEvent) -> Self {
        let mut next = self.clone();
        next.apply_structural_event(event);
        next
    }

    fn apply_structural_event(&mut self, event: &TimelineEvent) {
        let before = self
            .runs
            .get(&event.run_id)
            .map(|entry| entry.snapshot())
            .unwrap_or_else(RunSummary::default_public);
        let mut after = before.clone();
        after.apply_event(event);

        if !after.created {
            self.runs
                .insert(event.run_id.clone(), Arc::new(RunState::from_summary(after)));
            return;
        }

        if before.created && before.principal_key != after.principal_key {
            if let Some(set) = self.by_principal.get_mut(&*before.principal_key) {
                set.remove(&event.run_id);
                if set.is_empty() {
                    self.by_principal.remove(&*before.principal_key);
                }
            }
        }
        self.by_principal
            .entry(after.principal_key.clone())
            .or_default()
            .insert(event.run_id.clone());
        self.runs
            .insert(event.run_id.clone(), Arc::new(RunState::from_summary(after)));
    }
}

struct RunState {
    created: AtomicBool,
    principal_key: ArcSwap<Arc<str>>,
    state: AtomicU8,
    last_activity_ms: AtomicU64,
}

impl RunState {
    fn from_summary(summary: RunSummary) -> Self {
        Self {
            created: AtomicBool::new(summary.created),
            principal_key: ArcSwap::from(Arc::new(summary.principal_key)),
            state: AtomicU8::new(encode_stream_state(summary.state)),
            last_activity_ms: AtomicU64::new(summary.last_activity_ms),
        }
    }

    fn apply_event(&self, event: &TimelineEvent) {
        if let Some(next) = transition_state(event.kind.as_str()) {
            self.state
                .store(encode_stream_state(next), Ordering::Release);
        }
        self.last_activity_ms
            .fetch_max(event.pts_ms, Ordering::AcqRel);
    }

    fn snapshot(&self) -> RunSummary {
        RunSummary {
            created: self.created.load(Ordering::Acquire),
            principal_key: Arc::clone(&*self.principal_key.load_full()),
            state: decode_stream_state(self.state.load(Ordering::Acquire)),
            last_activity_ms: self.last_activity_ms.load(Ordering::Acquire),
        }
    }
}

#[derive(Clone)]
struct RunSummary {
    created: bool,
    principal_key: Arc<str>,
    state: StreamState,
    last_activity_ms: u64,
}

impl RunSummary {
    fn default_public() -> Self {
        Self {
            created: false,
            principal_key: Arc::from("public"),
            state: StreamState::Pending,
            last_activity_ms: 0,
        }
    }

    fn apply_event(&mut self, event: &TimelineEvent) {
        if event.kind == "run_created" {
            self.created = true;
            if let Some(principal) = principal_key_from_payload(&event.payload) {
                self.principal_key = principal;
            }
        }
        if let Some(next) = transition_state(event.kind.as_str()) {
            self.state = next;
        }
        self.last_activity_ms = self.last_activity_ms.max(event.pts_ms);
    }
}

fn build_run_registry(events: &[TimelineEvent]) -> Arc<RunRegistry> {
    let mut registry = RunRegistry::default();
    for event in events {
        registry.apply_structural_event(event);
    }
    Arc::new(registry)
}

fn principal_key_from_payload(raw: &str) -> Option<Arc<str>> {
    serde_json::from_str::<Value>(raw).ok().and_then(|payload| {
        payload
            .get("principal_key")
            .and_then(|value| value.as_str())
            .map(Arc::from)
    })
}

fn transition_state(event_kind: &str) -> Option<StreamState> {
    match event_kind {
        "run_created" => Some(StreamState::Pending),
        "ingest_received"
        | "analysis_generated"
        | "inference_completed"
        | "keepalive_refreshed" => Some(StreamState::Processing),
        "run_completed" => Some(StreamState::Completed),
        "run_failed" => Some(StreamState::Failed),
        "stop_requested" => Some(StreamState::Cancelled),
        _ => None,
    }
}

fn encode_stream_state(state: StreamState) -> u8 {
    match state {
        StreamState::Pending => 0,
        StreamState::Processing => 1,
        StreamState::Completed => 2,
        StreamState::Failed => 3,
        StreamState::Cancelled => 4,
        StreamState::Expired => 5,
    }
}

fn decode_stream_state(raw: u8) -> StreamState {
    match raw {
        1 => StreamState::Processing,
        2 => StreamState::Completed,
        3 => StreamState::Failed,
        4 => StreamState::Cancelled,
        5 => StreamState::Expired,
        _ => StreamState::Pending,
    }
}

fn apply_expiry(
    state: StreamState,
    last_activity_ms: u64,
    now_ms: u64,
    ttl_ms: u64,
) -> StreamState {
    if state.is_terminal() || ttl_ms == 0 || last_activity_ms == 0 {
        return state;
    }
    if now_ms.saturating_sub(last_activity_ms) > ttl_ms {
        StreamState::Expired
    } else {
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::thread;

    fn event(seq: u64, kind: &str, pts_ms: u64, payload: Value) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: "run-1".to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms,
            kind: kind.to_string(),
            payload: payload.to_string(),
        }
    }

    #[test]
    fn registry_keeps_same_map_snapshot_for_existing_run_event() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(
            format!("vidarax-state-test-{}.wal", std::process::id()),
        ));
        state.apply_event_to_registry(&event(
            1,
            "run_created",
            100,
            json!({"principal_key": "tenant-a"}),
        ));
        let before = state.run_registry.load_full();

        state.apply_event_to_registry(&event(2, "analysis_generated", 150, json!({})));
        let after = state.run_registry.load_full();

        assert!(
            Arc::ptr_eq(&before, &after),
            "existing-run events must update run atomics without replacing the registry map"
        );
        let snapshot = state.run_runtime_snapshot("run-1", 150).unwrap();
        assert_eq!(snapshot.principal_key, "tenant-a");
        assert_eq!(snapshot.state, StreamState::Processing);
        assert_eq!(snapshot.last_activity_ms, 150);
    }

    #[test]
    fn append_run_event_for_stream_persists_stream_id() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(
            format!("vidarax-state-stream-test-{}.wal", std::process::id()),
        ));

        state
            .append_run_event_for_stream("run-1", "camera-west", "analysis_generated", json!({}))
            .unwrap();

        let events = state.read_run_events("run-1").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].stream_id, "camera-west");
    }

    #[test]
    fn concurrent_keyframes_do_not_clobber_completed_state() {
        const KEYFRAME_THREADS: usize = 16;
        const KEYFRAMES_PER_THREAD: usize = 10_000;

        let run = Arc::new(RunState::from_summary(RunSummary {
            created: true,
            principal_key: Arc::from("tenant-a"),
            state: StreamState::Processing,
            last_activity_ms: 100,
        }));
        let start = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::with_capacity(KEYFRAME_THREADS + 1);

        for worker in 0..KEYFRAME_THREADS {
            let run = Arc::clone(&run);
            let start = Arc::clone(&start);
            let ready = Arc::clone(&ready);
            threads.push(thread::spawn(move || {
                ready.fetch_add(1, AtomicOrdering::AcqRel);
                while !start.load(AtomicOrdering::Acquire) {
                    thread::yield_now();
                }
                for i in 0..KEYFRAMES_PER_THREAD {
                    run.apply_event(&event(
                        1_000 + (worker * KEYFRAMES_PER_THREAD + i) as u64,
                        "keyframe_stored",
                        200 + i as u64,
                        json!({}),
                    ));
                }
            }));
        }

        let completed_run = Arc::clone(&run);
        let completed_start = Arc::clone(&start);
        let completed_ready = Arc::clone(&ready);
        threads.push(thread::spawn(move || {
            completed_ready.fetch_add(1, AtomicOrdering::AcqRel);
            while !completed_start.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            completed_run.apply_event(&event(2, "run_completed", 500, json!({})));
        }));

        while ready.load(AtomicOrdering::Acquire) != KEYFRAME_THREADS + 1 {
            thread::yield_now();
        }
        start.store(true, AtomicOrdering::Release);

        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(run.snapshot().state, StreamState::Completed);
    }
}

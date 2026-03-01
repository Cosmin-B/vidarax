use arc_swap::ArcSwap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use serde_json::Value;
use vidarax_contracts::lifecycle::StreamState;
use vidarax_core::provider::ProviderEndpoints;
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
    inference_endpoints: Option<ProviderEndpoints>,
    security_policy: Arc<SecurityPolicy>,
    inference_metrics: Arc<InferenceMetrics>,
    pipeline_metrics: Arc<PipelineMetrics>,
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

impl AppState {
    pub fn from_wal(
        wal_path: PathBuf,
        ingest_file_roots: Vec<PathBuf>,
        inference_endpoints: Option<ProviderEndpoints>,
        security_policy: SecurityPolicy,
        stream_ttl_secs: u64,
        active_stream_limit: usize,
        webrtc_config: WebRtcConfig,
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
            inference_endpoints,
            security_policy: Arc::new(security_policy),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
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
        Self {
            run_seq: Arc::new(AtomicU64::new(0)),
            request_seq: Arc::new(AtomicU64::new(0)),
            event_seq: Arc::new(AtomicU64::new(0)),
            wal_path: Arc::new(wal_path),
            ingest_file_roots: Arc::new(default_test_ingest_roots()),
            inference_endpoints: None,
            security_policy: Arc::new(SecurityPolicy::from_config_for_tests()),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
            run_registry: Arc::new(ArcSwap::from_pointee(RunRegistry::default())),
            tenant_label_maps: Arc::new(TenantLabelMaps::default()),
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            webrtc_config: WebRtcConfig::default(),
        }
    }

    pub fn with_wal_for_tests_and_endpoints(
        wal_path: PathBuf,
        inference_endpoints: Option<ProviderEndpoints>,
    ) -> Self {
        Self::with_wal_for_tests_full(
            wal_path,
            inference_endpoints,
            SecurityPolicy::from_config_for_tests(),
        )
    }

    pub fn with_wal_for_tests_full(
        wal_path: PathBuf,
        inference_endpoints: Option<ProviderEndpoints>,
        security_policy: SecurityPolicy,
    ) -> Self {
        Self {
            run_seq: Arc::new(AtomicU64::new(0)),
            request_seq: Arc::new(AtomicU64::new(0)),
            event_seq: Arc::new(AtomicU64::new(0)),
            wal_path: Arc::new(wal_path),
            ingest_file_roots: Arc::new(default_test_ingest_roots()),
            inference_endpoints,
            security_policy: Arc::new(security_policy),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
            run_registry: Arc::new(ArcSwap::from_pointee(RunRegistry::default())),
            tenant_label_maps: Arc::new(TenantLabelMaps::default()),
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            webrtc_config: WebRtcConfig::default(),
        }
    }

    pub fn with_wal_for_tests_runtime(
        wal_path: PathBuf,
        inference_endpoints: Option<ProviderEndpoints>,
        security_policy: SecurityPolicy,
        stream_ttl_secs: u64,
        active_stream_limit: usize,
    ) -> Self {
        Self {
            run_seq: Arc::new(AtomicU64::new(0)),
            request_seq: Arc::new(AtomicU64::new(0)),
            event_seq: Arc::new(AtomicU64::new(0)),
            wal_path: Arc::new(wal_path),
            ingest_file_roots: Arc::new(default_test_ingest_roots()),
            inference_endpoints,
            security_policy: Arc::new(security_policy),
            inference_metrics: Arc::new(InferenceMetrics::new()),
            pipeline_metrics: Arc::new(PipelineMetrics::new()),
            run_registry: Arc::new(ArcSwap::from_pointee(RunRegistry::default())),
            tenant_label_maps: Arc::new(TenantLabelMaps::default()),
            stream_ttl_secs: stream_ttl_secs.max(1),
            active_stream_limit: active_stream_limit.max(1),
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            webrtc_config: WebRtcConfig::default(),
        }
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
        let seq = self.event_seq.fetch_add(1, Ordering::AcqRel) + 1;
        let event = TimelineEvent {
            seq,
            run_id: run_id.to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms: now_epoch_ms(),
            kind: kind.to_string(),
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
        let seq = self.event_seq.fetch_add(1, Ordering::AcqRel) + 1;
        let event = TimelineEvent {
            seq,
            run_id: run_id.to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms: now_epoch_ms(),
            kind: kind.to_string(),
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

    pub fn inference_endpoints(&self) -> Option<&ProviderEndpoints> {
        self.inference_endpoints.as_ref()
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

    pub fn map_event_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_event(tenant_id, label)
    }

    pub fn map_object_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_object(tenant_id, label)
    }

    pub fn run_runtime_snapshot(&self, run_id: &str, now_ms: u64) -> Option<RunRuntimeSnapshot> {
        let registry = self.run_registry.load();
        let summary = registry.runs.get(run_id)?;
        if !summary.created {
            return None;
        }
        let ttl_ms = self.stream_ttl_secs.saturating_mul(1000);
        Some(RunRuntimeSnapshot {
            principal_key: summary.principal_key.clone(),
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
        self.run_registry.rcu(|current| {
            let mut next = (**current).clone();
            next.apply_event(event);
            Arc::new(next)
        });
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

#[derive(Clone)]
pub struct RunRuntimeSnapshot {
    pub principal_key: String,
    pub state: StreamState,
    pub last_activity_ms: u64,
}

#[derive(Clone, Default)]
struct RunRegistry {
    runs: HashMap<String, RunSummary>,
    by_principal: HashMap<String, HashSet<String>>,
}

impl RunRegistry {
    fn apply_event(&mut self, event: &TimelineEvent) {
        let run_id = event.run_id.clone();
        let entry = self
            .runs
            .entry(run_id.clone())
            .or_insert_with(RunSummary::default_public);
        let was_created = entry.created;
        let previous_principal = entry.principal_key.clone();

        if event.kind == "run_created" {
            entry.created = true;
            if let Some(principal) = principal_key_from_payload(&event.payload) {
                entry.principal_key = principal;
            }
        }

        entry.state = transition_state(entry.state, event.kind.as_str());
        entry.last_activity_ms = entry.last_activity_ms.max(event.pts_ms);

        if !entry.created {
            return;
        }

        if was_created && previous_principal != entry.principal_key {
            if let Some(set) = self.by_principal.get_mut(&previous_principal) {
                set.remove(&run_id);
                if set.is_empty() {
                    self.by_principal.remove(&previous_principal);
                }
            }
        }
        self.by_principal
            .entry(entry.principal_key.clone())
            .or_default()
            .insert(run_id);
    }
}

#[derive(Clone)]
struct RunSummary {
    created: bool,
    principal_key: String,
    state: StreamState,
    last_activity_ms: u64,
}

impl RunSummary {
    fn default_public() -> Self {
        Self {
            created: false,
            principal_key: "public".to_string(),
            state: StreamState::Pending,
            last_activity_ms: 0,
        }
    }
}

fn build_run_registry(events: &[TimelineEvent]) -> Arc<RunRegistry> {
    let mut registry = RunRegistry::default();
    for event in events {
        registry.apply_event(event);
    }
    Arc::new(registry)
}

fn principal_key_from_payload(raw: &str) -> Option<String> {
    serde_json::from_str::<Value>(raw).ok().and_then(|payload| {
        payload
            .get("principal_key")
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    })
}

fn transition_state(current: StreamState, event_kind: &str) -> StreamState {
    match event_kind {
        "run_created" => StreamState::Pending,
        "ingest_received"
        | "analysis_generated"
        | "inference_completed"
        | "keepalive_refreshed" => StreamState::Processing,
        "run_completed" => StreamState::Completed,
        "run_failed" => StreamState::Failed,
        "stop_requested" => StreamState::Cancelled,
        _ => current,
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

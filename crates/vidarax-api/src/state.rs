use arc_swap::ArcSwap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, RwLock};

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
/// Each entry stores the owning principal, run ID, and live session.
type SessionEntry = (String, Arc<str>, Arc<WebRtcSession>);
type SessionMap = Arc<RwLock<HashMap<String, SessionEntry>>>;
type ReclaimedSessionEntry = (String, Arc<str>);
type ReclaimedSessionMap = Arc<RwLock<ReclaimedSessions>>;
type StreamReservations = Arc<Mutex<HashMap<String, usize>>>;

/// WHIP DELETE remains idempotent for watcher-reclaimed sessions within this
/// window. Older random session IDs are tombstones only; retaining them forever
/// would make memory grow with all historical disconnects.
const RECLAIMED_SESSION_TTL_MS: u64 = 10 * 60 * 1000;
const RECLAIMED_SESSION_MAX_ENTRIES: usize = 1024;

const RUN_DELETE_LIVE: u8 = 0;
const RUN_DELETE_APPEND_IN_FLIGHT: u8 = 1;
const RUN_DELETE_DELETED: u8 = 2;

#[derive(Default)]
struct ReclaimedSessions {
    entries: HashMap<String, ReclaimedSessionRecord>,
    order: VecDeque<String>,
}

struct ReclaimedSessionRecord {
    principal: String,
    run_id: Arc<str>,
    reclaimed_at_ms: u64,
}

impl ReclaimedSessions {
    fn insert(&mut self, sess_id: String, principal: String, run_id: Arc<str>, now_ms: u64) {
        self.prune_expired(now_ms);
        if self.entries.contains_key(&sess_id) {
            self.order.retain(|existing| existing != &sess_id);
        }
        self.order.push_back(sess_id.clone());
        self.entries.insert(
            sess_id,
            ReclaimedSessionRecord {
                principal,
                run_id,
                reclaimed_at_ms: now_ms,
            },
        );
        self.enforce_cap();
    }

    fn get(&mut self, sess_id: &str, now_ms: u64) -> Option<ReclaimedSessionEntry> {
        self.prune_expired(now_ms);
        self.entries
            .get(sess_id)
            .map(|entry| (entry.principal.clone(), Arc::clone(&entry.run_id)))
    }

    fn remove(&mut self, sess_id: &str) {
        self.entries.remove(sess_id);
        self.order.retain(|existing| existing != sess_id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn prune_expired(&mut self, now_ms: u64) {
        while let Some(sess_id) = self.order.front() {
            let Some(entry) = self.entries.get(sess_id) else {
                self.order.pop_front();
                continue;
            };
            if now_ms.saturating_sub(entry.reclaimed_at_ms) <= RECLAIMED_SESSION_TTL_MS {
                break;
            }
            let sess_id = self.order.pop_front().expect("front checked above");
            self.entries.remove(&sess_id);
        }
    }

    fn enforce_cap(&mut self) {
        while self.entries.len() > RECLAIMED_SESSION_MAX_ENTRIES {
            let Some(sess_id) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&sess_id);
        }
    }
}

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
    stream_reservations: StreamReservations,
    tenant_label_maps: Arc<TenantLabelMaps>,
    stream_ttl_secs: u64,
    active_stream_limit: usize,
    spacetime_client: Option<SpacetimeClient>,
    /// Active WebRTC peer connections indexed by session ID.
    sessions: SessionMap,
    /// Recently reclaimed WHIP sessions, retained so DELETE remains idempotent
    /// after a peer-state watcher has already removed the live session entry.
    reclaimed_sessions: ReclaimedSessionMap,
    /// WebRTC configuration (STUN/TURN servers, token rate limit).
    webrtc_config: WebRtcConfig,
}

#[must_use]
pub struct StreamSlotGuard {
    reservations: StreamReservations,
    principal_key: String,
}

impl Drop for StreamSlotGuard {
    fn drop(&mut self) {
        let mut reservations = self
            .reservations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(count) = reservations.get_mut(&self.principal_key) {
            if *count <= 1 {
                reservations.remove(&self.principal_key);
            } else {
                *count -= 1;
            }
        }
    }
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
            stream_reservations: Arc::new(Mutex::new(HashMap::new())),
            tenant_label_maps: Arc::new(TenantLabelMaps::default()),
            stream_ttl_secs: self.stream_ttl_secs.max(1),
            active_stream_limit: self.active_stream_limit.max(1),
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            reclaimed_sessions: Arc::new(RwLock::new(ReclaimedSessions::default())),
            webrtc_config: WebRtcConfig::default(),
        }
    }
}

impl AppState {
    // WAL restoration receives distinct dependencies needed to rebuild AppState.
    #[allow(clippy::too_many_arguments)]
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
            stream_reservations: Arc::new(Mutex::new(HashMap::new())),
            tenant_label_maps,
            stream_ttl_secs,
            active_stream_limit: active_stream_limit.max(1),
            spacetime_client: None,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            reclaimed_sessions: Arc::new(RwLock::new(ReclaimedSessions::default())),
            webrtc_config,
        })
    }

    pub fn with_wal_for_tests(wal_path: PathBuf) -> Self {
        AppStateConfig::for_tests(wal_path).build()
    }

    pub fn with_wal_for_tests_requiring_api_keys(
        wal_path: PathBuf,
        api_keys: Vec<String>,
    ) -> Self {
        Self::with_wal_for_tests_full(
            wal_path,
            None,
            SecurityPolicy::from_test_policy(true, api_keys, false, false, vec![]),
        )
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

    #[cfg(test)]
    pub(crate) fn with_tenant_label_maps_for_tests(
        mut self,
        maps: crate::tenant_labels::TenantLabelMaps,
    ) -> Self {
        self.tenant_label_maps = Arc::new(maps);
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
        run_id: Arc<str>,
        session: Arc<WebRtcSession>,
    ) -> bool {
        let mut map = self.sessions.write().await;
        if map.len() >= MAX_WEBRTC_SESSIONS {
            return false;
        }
        if map.contains_key(&sess_id) {
            return false;
        }
        map.insert(sess_id.clone(), (principal, run_id, session));
        drop(map);
        self.reclaimed_sessions.write().await.remove(&sess_id);
        true
    }

    /// Look up a WebRTC session by ID.  Returns `None` if not found.
    pub async fn get_session(&self, sess_id: &str) -> Option<SessionEntry> {
        self.sessions.read().await.get(sess_id).cloned()
    }

    /// Remove and return a WebRTC session only if it still belongs to `run_id`.
    ///
    /// The write lock makes session removal the atomic ownership transfer for
    /// reclaim paths: exactly one caller can win cleanup for a given session.
    pub(crate) async fn remove_session_for_run(
        &self,
        sess_id: &str,
        run_id: &str,
    ) -> Option<SessionEntry> {
        let mut map = self.sessions.write().await;
        let (principal, existing_run_id, _session) = map.get(sess_id)?;
        if &**existing_run_id != run_id {
            return None;
        }
        self.reclaimed_sessions.write().await.insert(
            sess_id.to_string(),
            principal.clone(),
            Arc::clone(existing_run_id),
            now_epoch_ms(),
        );
        map.remove(sess_id)
    }

    pub(crate) async fn get_reclaimed_session(
        &self,
        sess_id: &str,
    ) -> Option<ReclaimedSessionEntry> {
        self.reclaimed_sessions
            .write()
            .await
            .get(sess_id, now_epoch_ms())
    }

    #[cfg(test)]
    pub(crate) async fn hold_reclaimed_sessions_write_for_tests(&self) -> impl Drop {
        Arc::clone(&self.reclaimed_sessions).write_owned().await
    }

    /// Number of active WebRTC sessions.
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    pub fn next_run_id(&self) -> String {
        // Sequence counter is still incremented for monotonic ordering.
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
        if kind == "run_deleted" {
            return self
                .append_run_deleted_for_stream_idempotent(run_id, stream_id, payload)
                .map(|result| result.event);
        }

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
        if kind == "run_deleted" {
            return self
                .append_run_deleted_for_stream_idempotent_async(run_id, stream_id, payload)
                .await
                .map(|result| result.event);
        }

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

    pub(crate) async fn append_run_deleted_idempotent_async(
        &self,
        run_id: &str,
        payload: Value,
    ) -> Result<bool, String> {
        self.append_run_deleted_for_stream_idempotent_async(run_id, DEFAULT_STREAM_ID, payload)
            .await
            .map(|result| result.appended)
    }

    fn append_run_deleted_for_stream_idempotent(
        &self,
        run_id: &str,
        stream_id: &str,
        payload: Value,
    ) -> Result<RunDeletedAppend, String> {
        loop {
            match self.begin_run_deleted_append(run_id) {
                RunDeleteClaim::Claimed(run) => {
                    let guard = RunDeleteAppendGuard::new(run);
                    let event =
                        self.new_timeline_event(run_id, stream_id, "run_deleted", &payload);
                    if let Err(err) = append_event(self.wal_path.as_ref(), &event) {
                        return Err(err.to_string());
                    }
                    self.apply_event_to_registry(&event);
                    guard.commit();
                    return Ok(RunDeletedAppend {
                        event,
                        appended: true,
                    });
                }
                RunDeleteClaim::AlreadyDeleted => {
                    return Ok(RunDeletedAppend {
                        event: self.synthetic_run_deleted_event(run_id, stream_id, &payload),
                        appended: false,
                    });
                }
                RunDeleteClaim::InFlight(run) => run.wait_delete_append_blocking(),
                RunDeleteClaim::Missing => {
                    let event =
                        self.new_timeline_event(run_id, stream_id, "run_deleted", &payload);
                    append_event(self.wal_path.as_ref(), &event)
                        .map_err(|err| err.to_string())?;
                    self.apply_event_to_registry(&event);
                    return Ok(RunDeletedAppend {
                        event,
                        appended: true,
                    });
                }
            }
        }
    }

    async fn append_run_deleted_for_stream_idempotent_async(
        &self,
        run_id: &str,
        stream_id: &str,
        payload: Value,
    ) -> Result<RunDeletedAppend, String> {
        loop {
            match self.begin_run_deleted_append(run_id) {
                RunDeleteClaim::Claimed(run) => {
                    let event =
                        self.new_timeline_event(run_id, stream_id, "run_deleted", &payload);
                    let state = self.clone();
                    let append_task = tokio::spawn(async move {
                        let guard = RunDeleteAppendGuard::new(run);
                        let wal_path = Arc::clone(&state.wal_path);
                        let event_for_write = event.clone();
                        let write_result = tokio::task::spawn_blocking(move || {
                            append_event(wal_path.as_ref(), &event_for_write)
                        })
                        .await
                        .map_err(|err| format!("timeline append worker join failure: {err}"))?
                        .map_err(|err| err.to_string());

                        write_result?;

                        state.apply_event_to_registry(&event);
                        guard.commit();
                        Ok(RunDeletedAppend {
                            event,
                            appended: true,
                        })
                    });

                    return append_task
                        .await
                        .map_err(|err| format!("timeline append coordinator join failure: {err}"))?;
                }
                RunDeleteClaim::AlreadyDeleted => {
                    return Ok(RunDeletedAppend {
                        event: self.synthetic_run_deleted_event(run_id, stream_id, &payload),
                        appended: false,
                    });
                }
                RunDeleteClaim::InFlight(run) => run.wait_delete_append().await,
                RunDeleteClaim::Missing => {
                    let event =
                        self.new_timeline_event(run_id, stream_id, "run_deleted", &payload);
                    let wal_path = Arc::clone(&self.wal_path);
                    let event_for_write = event.clone();
                    tokio::task::spawn_blocking(move || {
                        append_event(wal_path.as_ref(), &event_for_write)
                    })
                    .await
                    .map_err(|err| format!("timeline append worker join failure: {err}"))?
                    .map_err(|err| err.to_string())?;
                    self.apply_event_to_registry(&event);
                    return Ok(RunDeletedAppend {
                        event,
                        appended: true,
                    });
                }
            }
        }
    }

    fn begin_run_deleted_append(&self, run_id: &str) -> RunDeleteClaim {
        let registry = self.run_registry.load();
        let Some(run) = registry.runs.get(run_id).cloned() else {
            return RunDeleteClaim::Missing;
        };
        match run.begin_delete_append() {
            RunDeleteState::Claimed => RunDeleteClaim::Claimed(run),
            RunDeleteState::AlreadyDeleted => RunDeleteClaim::AlreadyDeleted,
            RunDeleteState::InFlight => RunDeleteClaim::InFlight(run),
        }
    }

    fn new_timeline_event(
        &self,
        run_id: &str,
        stream_id: &str,
        kind: &str,
        payload: &Value,
    ) -> TimelineEvent {
        let seq = self.event_seq.fetch_add(1, Ordering::AcqRel) + 1;
        TimelineEvent {
            seq,
            run_id: run_id.to_owned(),
            stream_id: stream_id.to_owned(),
            pts_ms: now_epoch_ms(),
            kind: kind.to_owned(),
            payload: payload.to_string(),
        }
    }

    fn synthetic_run_deleted_event(
        &self,
        run_id: &str,
        stream_id: &str,
        payload: &Value,
    ) -> TimelineEvent {
        TimelineEvent {
            seq: self.event_seq.load(Ordering::Acquire),
            run_id: run_id.to_owned(),
            stream_id: stream_id.to_owned(),
            pts_ms: now_epoch_ms(),
            kind: "run_deleted".to_owned(),
            payload: payload.to_string(),
        }
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
        if !summary.created || summary.deleted {
            return None;
        }
        let ttl_ms = self.stream_ttl_secs.saturating_mul(1000);
        Some(RunRuntimeSnapshot {
            principal_key: summary.principal_key.to_string(),
            state: apply_expiry(summary.state, summary.last_activity_ms, now_ms, ttl_ms),
            last_activity_ms: summary.last_activity_ms,
        })
    }

    pub(crate) fn run_is_deleted(&self, run_id: &str) -> bool {
        let registry = self.run_registry.load();
        registry
            .runs
            .get(run_id)
            .map(|summary| summary.snapshot().deleted)
            .unwrap_or(false)
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
                !summary.deleted
                    && !apply_expiry(summary.state, summary.last_activity_ms, now_ms, ttl_ms)
                        .is_terminal()
            })
            .count()
    }

    /// Reserve a per-principal stream slot before persisting a new run.
    ///
    /// The committed active count and in-flight reservations are checked under
    /// one lock so concurrent creators cannot all pass the same snapshot. The
    /// returned guard releases the reservation once the run is durably
    /// registered and visible to active-run accounting.
    pub fn try_reserve_stream_slot(
        &self,
        principal_key: &str,
        now_ms: u64,
    ) -> Option<StreamSlotGuard> {
        let mut reservations = self
            .stream_reservations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let committed = self.count_active_runs_for_principal(principal_key, now_ms);
        let reserved = *reservations.get(principal_key).unwrap_or(&0);
        if committed.saturating_add(reserved) >= self.active_stream_limit {
            return None;
        }

        *reservations
            .entry(principal_key.to_string())
            .or_insert(0) += 1;
        Some(StreamSlotGuard {
            reservations: Arc::clone(&self.stream_reservations),
            principal_key: principal_key.to_string(),
        })
    }

    pub fn stream_ttl_secs(&self) -> u64 {
        self.stream_ttl_secs
    }

    pub fn active_stream_limit(&self) -> usize {
        self.active_stream_limit
    }

    fn apply_event_to_registry(&self, event: &TimelineEvent) {
        if matches!(event.kind.as_str(), "run_created" | "run_deleted") {
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

        if before.created
            && (!after.created || after.deleted || before.principal_key != after.principal_key)
        {
            if let Some(set) = self.by_principal.get_mut(&*before.principal_key) {
                set.remove(&event.run_id);
                if set.is_empty() {
                    self.by_principal.remove(&*before.principal_key);
                }
            }
        }

        if !after.created || after.deleted {
            self.runs
                .insert(event.run_id.clone(), Arc::new(RunState::from_summary(after)));
            return;
        }

        self.by_principal
            .entry(after.principal_key.clone())
            .or_default()
            .insert(event.run_id.clone());
        self.runs
            .insert(event.run_id.clone(), Arc::new(RunState::from_summary(after)));
    }
}

struct RunDeletedAppend {
    event: TimelineEvent,
    appended: bool,
}

enum RunDeleteClaim {
    Claimed(Arc<RunState>),
    AlreadyDeleted,
    InFlight(Arc<RunState>),
    Missing,
}

enum RunDeleteState {
    Claimed,
    AlreadyDeleted,
    InFlight,
}

struct RunDeleteAppendGuard {
    run: Arc<RunState>,
    committed: bool,
}

impl RunDeleteAppendGuard {
    fn new(run: Arc<RunState>) -> Self {
        Self {
            run,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.run.commit_delete_append();
        self.committed = true;
    }
}

impl Drop for RunDeleteAppendGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.run.rollback_delete_append();
        }
    }
}

struct RunState {
    created: AtomicBool,
    delete_state: AtomicU8,
    delete_notify: Notify,
    principal_key: ArcSwap<Arc<str>>,
    state: AtomicU8,
    last_activity_ms: AtomicU64,
}

impl RunState {
    fn from_summary(summary: RunSummary) -> Self {
        Self {
            created: AtomicBool::new(summary.created),
            delete_state: AtomicU8::new(if summary.deleted {
                RUN_DELETE_DELETED
            } else {
                RUN_DELETE_LIVE
            }),
            delete_notify: Notify::new(),
            principal_key: ArcSwap::from(Arc::new(summary.principal_key)),
            state: AtomicU8::new(encode_stream_state(summary.state)),
            last_activity_ms: AtomicU64::new(summary.last_activity_ms),
        }
    }

    fn apply_event(&self, event: &TimelineEvent) {
        if event.kind == "run_deleted" {
            self.commit_delete_append();
        }
        if let Some(next) = transition_state(event.kind.as_str()) {
            self.state
                .store(encode_stream_state(next), Ordering::Release);
        }
        self.last_activity_ms
            .fetch_max(event.pts_ms, Ordering::AcqRel);
    }

    fn begin_delete_append(&self) -> RunDeleteState {
        match self.delete_state.compare_exchange(
            RUN_DELETE_LIVE,
            RUN_DELETE_APPEND_IN_FLIGHT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => RunDeleteState::Claimed,
            Err(RUN_DELETE_DELETED) => RunDeleteState::AlreadyDeleted,
            Err(RUN_DELETE_APPEND_IN_FLIGHT) => RunDeleteState::InFlight,
            Err(_) => RunDeleteState::InFlight,
        }
    }

    fn commit_delete_append(&self) {
        self.delete_state
            .store(RUN_DELETE_DELETED, Ordering::Release);
        self.delete_notify.notify_waiters();
    }

    fn rollback_delete_append(&self) {
        let _ = self.delete_state.compare_exchange(
            RUN_DELETE_APPEND_IN_FLIGHT,
            RUN_DELETE_LIVE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.delete_notify.notify_waiters();
    }

    async fn wait_delete_append(&self) {
        loop {
            let notified = self.delete_notify.notified();
            if self.delete_state.load(Ordering::Acquire) != RUN_DELETE_APPEND_IN_FLIGHT {
                return;
            }
            notified.await;
        }
    }

    fn wait_delete_append_blocking(&self) {
        while self.delete_state.load(Ordering::Acquire) == RUN_DELETE_APPEND_IN_FLIGHT {
            std::thread::yield_now();
        }
    }

    fn snapshot(&self) -> RunSummary {
        RunSummary {
            created: self.created.load(Ordering::Acquire),
            deleted: self.delete_state.load(Ordering::Acquire) == RUN_DELETE_DELETED,
            principal_key: Arc::clone(&*self.principal_key.load_full()),
            state: decode_stream_state(self.state.load(Ordering::Acquire)),
            last_activity_ms: self.last_activity_ms.load(Ordering::Acquire),
        }
    }
}

#[derive(Clone)]
struct RunSummary {
    created: bool,
    deleted: bool,
    principal_key: Arc<str>,
    state: StreamState,
    last_activity_ms: u64,
}

impl RunSummary {
    fn default_public() -> Self {
        Self {
            created: false,
            deleted: false,
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
        if event.kind == "run_deleted" {
            self.deleted = true;
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
    fn stream_slot_reservations_enforce_limit_before_registry_writes() {
        let state = AppState::with_wal_for_tests_runtime(
            std::env::temp_dir().join(format!(
                "vidarax-state-reservation-limit-{}.wal",
                std::process::id()
            )),
            None,
            SecurityPolicy::from_config_for_tests(),
            3600,
            2,
        );
        let principal = "tenant-a";
        let now_ms = now_epoch_ms();

        let mut guards = Vec::new();
        for _ in 0..state.active_stream_limit() {
            guards.push(
                state
                    .try_reserve_stream_slot(principal, now_ms)
                    .expect("reservation should fit under the per-principal limit"),
            );
        }

        assert!(state.try_reserve_stream_slot(principal, now_ms).is_none());
        drop(guards.pop().expect("held reservation"));
        assert!(state.try_reserve_stream_slot(principal, now_ms).is_some());
        drop(guards);
        assert!(state.try_reserve_stream_slot(principal, now_ms).is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_run_deleted_appends_are_idempotent() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-delete-idempotent-{}.wal",
            std::process::id()
        )));
        let principal = "tenant-a";

        state
            .append_run_event(
                "run-1",
                "run_created",
                json!({ "principal_key": principal }),
            )
            .unwrap();
        assert_eq!(state.count_active_runs_for_principal(principal, now_epoch_ms()), 1);

        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let mut tasks = Vec::new();
        for worker in 0..8 {
            let state = state.clone();
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                state
                    .append_run_deleted_idempotent_async(
                        "run-1",
                        json!({ "worker": worker }),
                    )
                    .await
            }));
        }

        let mut appended = 0;
        for task in tasks {
            if task.await.unwrap().unwrap() {
                appended += 1;
            }
        }

        let events = state.read_run_events("run-1").unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        assert_eq!(appended, 1);
        assert_eq!(state.count_active_runs_for_principal(principal, now_epoch_ms()), 0);
        assert!(state.run_runtime_snapshot("run-1", now_epoch_ms()).is_none());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_run_deleted_append_releases_claim_for_retry() {
        use std::io::Read;

        let dir = std::env::temp_dir().join(format!(
            "vidarax-state-delete-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let wal_path = dir.join("timeline.wal");
        let state = AppState::with_wal_for_tests(wal_path.clone());
        state
            .append_run_event(
                "run-1",
                "run_created",
                json!({ "principal_key": "tenant-a" }),
            )
            .unwrap();

        std::fs::remove_file(&wal_path).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(&wal_path)
            .status()
            .expect("mkfifo should run");
        assert!(status.success(), "mkfifo failed: {status}");

        let delete_state = state.clone();
        let delete_task = tokio::spawn(async move {
            delete_state
                .append_run_deleted_idempotent_async("run-1", json!({ "reason": "cancelled" }))
                .await
        });

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let registry = state.run_registry.load();
                let run = registry.runs.get("run-1").expect("run exists");
                if run.delete_state.load(Ordering::Acquire) == RUN_DELETE_APPEND_IN_FLIGHT {
                    return;
                }
                drop(registry);
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("delete append should reach in-flight claim");

        delete_task.abort();
        let _ = delete_task.await;

        let mut fifo_reader = std::fs::OpenOptions::new()
            .read(true)
            .open(&wal_path)
            .expect("open fifo reader");
        let mut discarded = String::new();
        fifo_reader
            .read_to_string(&mut discarded)
            .expect("read cancelled append");
        drop(fifo_reader);
        std::fs::remove_file(&wal_path).unwrap();

        let appended = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            state.append_run_deleted_idempotent_async("run-1", json!({ "reason": "retry" })),
        )
        .await
        .expect("retry after cancelled delete must not wait forever")
        .expect("retry delete append should succeed");
        assert!(!appended, "cancelled caller must not force a duplicate tombstone");
        assert!(state.run_runtime_snapshot("run-1", now_epoch_ms()).is_none());
    }

    #[tokio::test]
    async fn reclaimed_sessions_are_bounded_across_historical_disconnects() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-reclaimed-bound-{}.wal",
            std::process::id()
        )));

        for i in 0..(RECLAIMED_SESSION_MAX_ENTRIES + 16) {
            let sess_id = format!("sess-reclaimed-{i:04}");
            let run_id: Arc<str> = Arc::from(format!("run-reclaimed-{i:04}"));
            assert!(
                state
                    .insert_session(
                        sess_id.clone(),
                        "tenant-a".to_string(),
                        Arc::clone(&run_id),
                        Arc::new(WebRtcSession::new_for_tests()),
                    )
                    .await
            );
            state
                .remove_session_for_run(&sess_id, &run_id)
                .await
                .expect("session should be reclaimed");
        }

        assert_eq!(
            state.reclaimed_sessions.read().await.len(),
            RECLAIMED_SESSION_MAX_ENTRIES
        );
        assert!(state.get_reclaimed_session("sess-reclaimed-0000").await.is_none());
    }

    #[test]
    fn concurrent_keyframes_do_not_clobber_completed_state() {
        const KEYFRAME_THREADS: usize = 16;
        const KEYFRAMES_PER_THREAD: usize = 10_000;

        let run = Arc::new(RunState::from_summary(RunSummary {
            created: true,
            deleted: false,
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

use arc_swap::ArcSwap;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot, Notify};

use serde_json::Value;

/// Count of detached timeline events dropped because the writer queue was full.
/// The non-blocking emit intentionally never blocks the frame path, so a full
/// queue drops the event rather than applying backpressure. That drop is silent
/// to the caller by design, but it must not be silent to an operator: every
/// drop increments this counter and logs a warning carrying the running total,
/// so the loss is visible rather than invisible.
static DROPPED_DETACHED_TIMELINE_EVENTS: AtomicU64 = AtomicU64::new(0);

use vidarax_contracts::lifecycle::StreamState;
use vidarax_core::admission::{AdmissionLimits, InferenceAdmission};
use vidarax_core::ingest::pipeline::{create_pipeline, DecodePipeline, PipelineBackend};
use vidarax_core::novelty::LiveNoveltyConfig;
use vidarax_core::provider::{AdmittedProvider, InferenceProvider};
#[cfg(test)]
use vidarax_core::timeline::append_event;
use vidarax_core::timeline::{read_all_events, TimelineEvent, WalWriter};
use vidarax_core::webrtc::resources::MediaSessionResources;
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
type SessionMap = DashMap<String, SessionEntry>;
type MediaReservationMap = DashMap<String, MediaResourcePermit>;
type ReclaimedSessionEntry = (String, Arc<str>);
type ReclaimedSessionMap = Arc<Mutex<ReclaimedSessions>>;
type StreamReservations = Arc<DashMap<String, usize>>;

/// WHIP DELETE remains idempotent for watcher-reclaimed sessions within this
/// window. Older random session IDs are tombstones only; retaining them forever
/// would make memory grow with all historical disconnects.
const RECLAIMED_SESSION_TTL_MS: u64 = 10 * 60 * 1000;
const RECLAIMED_SESSION_MAX_ENTRIES: usize = 1024;

const RUN_DELETE_LIVE: u8 = 0;
const RUN_DELETE_APPEND_IN_FLIGHT: u8 = 1;
const RUN_DELETE_DELETED: u8 = 2;

/// How many of a run's most recent events stay mirrored in memory. A client
/// polling with an advancing `from_seq` cursor is served straight from this
/// ring; only a cursor older than the ring's oldest retained event falls back
/// to a full WAL scan. Sized to hold well over a minute of keyframe and
/// structural events at typical rates so steady polling never misses the ring.
const RUN_EVENT_TAIL_CAP: usize = 256;
/// How many of the most recently deleted runs keep their registry entry so a
/// repeated DELETE stays idempotent (served as AlreadyDeleted with no new WAL
/// write). Beyond this, the oldest deleted entries are forgotten; a DELETE of
/// one then falls through to the Missing path and appends a fresh run_deleted,
/// which is acceptable for long-forgotten runs. Sized well above any realistic
/// re-DELETE window.
const DELETED_RUN_RETENTION_CAP: usize = 4096;
/// How many runs keep a warm in-memory event tail. Beyond this, the least
/// recently appended run's warm tail is dropped. Its subsequent reads fall
/// back to the durable WAL scan, which is always correct but slower. This is
/// sized well above any realistic concurrent live-run count so normal
/// operation never evicts.
const WARM_RUN_TAIL_CAP: usize = 1024;
const TIMELINE_WRITER_QUEUE_CAP: usize = 1024;

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
struct MediaResourceBudget {
    inner: Arc<MediaResourceBudgetInner>,
}

struct MediaResourceBudgetInner {
    memory_limit_bytes: u64,
    worker_thread_limit: usize,
    reserved_bytes: AtomicU64,
    reserved_worker_threads: AtomicUsize,
    rejected_total: AtomicU64,
    last_rejection_timestamp_seconds: AtomicU64,
}

impl MediaResourceBudget {
    fn new(memory_limit_bytes: u64, worker_thread_limit: usize) -> Self {
        Self {
            inner: Arc::new(MediaResourceBudgetInner {
                memory_limit_bytes,
                worker_thread_limit,
                reserved_bytes: AtomicU64::new(0),
                reserved_worker_threads: AtomicUsize::new(0),
                rejected_total: AtomicU64::new(0),
                last_rejection_timestamp_seconds: AtomicU64::new(0),
            }),
        }
    }

    fn try_reserve(&self, resources: MediaSessionResources) -> Option<MediaResourcePermit> {
        if self
            .inner
            .reserved_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                reserved
                    .checked_add(resources.reserved_bytes)
                    .filter(|next| *next <= self.inner.memory_limit_bytes)
            })
            .is_err()
        {
            self.record_rejection();
            return None;
        }

        if self
            .inner
            .reserved_worker_threads
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                reserved
                    .checked_add(resources.worker_threads)
                    .filter(|next| *next <= self.inner.worker_thread_limit)
            })
            .is_err()
        {
            self.inner
                .reserved_bytes
                .fetch_sub(resources.reserved_bytes, Ordering::AcqRel);
            self.record_rejection();
            return None;
        }

        Some(MediaResourcePermit {
            budget: Arc::clone(&self.inner),
            resources,
        })
    }

    fn record_rejection(&self) {
        self.inner.rejected_total.fetch_add(1, Ordering::Relaxed);
        self.inner.last_rejection_timestamp_seconds.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
            Ordering::Relaxed,
        );
    }

    fn render_prometheus(&self) -> String {
        format!(
            "vidarax_media_capacity_memory_limit_bytes {}\n\
             vidarax_media_capacity_memory_reserved_bytes {}\n\
             vidarax_media_capacity_worker_thread_limit {}\n\
             vidarax_media_capacity_worker_threads_reserved {}\n\
             vidarax_media_capacity_rejections_total {}\n\
             vidarax_media_capacity_last_rejection_timestamp_seconds {}\n",
            self.inner.memory_limit_bytes,
            self.inner.reserved_bytes.load(Ordering::Relaxed),
            self.inner.worker_thread_limit,
            self.inner.reserved_worker_threads.load(Ordering::Relaxed),
            self.inner.rejected_total.load(Ordering::Relaxed),
            self.inner
                .last_rejection_timestamp_seconds
                .load(Ordering::Relaxed),
        )
    }
}

pub(crate) struct MediaResourcePermit {
    budget: Arc<MediaResourceBudgetInner>,
    resources: MediaSessionResources,
}

impl Drop for MediaResourcePermit {
    fn drop(&mut self) {
        self.budget
            .reserved_bytes
            .fetch_sub(self.resources.reserved_bytes, Ordering::AcqRel);
        self.budget
            .reserved_worker_threads
            .fetch_sub(self.resources.worker_threads, Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    run_seq: AtomicU64,
    request_seq: AtomicU64,
    pipeline_generation_seq: AtomicU64,
    wal_path: Arc<PathBuf>,
    ingest_file_roots: Vec<PathBuf>,
    provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    decode_pipeline: Arc<dyn DecodePipeline>,
    security_policy: SecurityPolicy,
    // Arc-wrapped so a background WHIP worker thread can hold its own handle
    // (as an `InferenceObserver`) and record tiered inference outcomes
    // without borrowing from AppState.
    inference_metrics: Arc<InferenceMetrics>,
    inference_admission: Arc<InferenceAdmission>,
    pipeline_metrics: Arc<PipelineMetrics>,
    novelty_config: LiveNoveltyConfig,
    run_registry: Arc<RunRegistry>,
    timeline_tx: mpsc::Sender<TimelineCommand>,
    timeline_snapshot: Arc<ArcSwap<RingSnapshot>>,
    #[cfg(test)]
    timeline_test_control: Arc<TimelineWriterTestControl>,
    stream_reservations: StreamReservations,
    tenant_label_maps: TenantLabelMaps,
    stream_ttl_secs: u64,
    active_stream_limit: usize,
    spacetime_client: Option<SpacetimeClient>,
    /// Active WebRTC peer connections indexed by session ID.
    sessions: SessionMap,
    /// Exact global admission count for `sessions`. The DashMap length is an
    /// observation, not an atomic reservation, so it cannot enforce the cap
    /// under concurrent inserts by itself.
    session_slots: AtomicUsize,
    media_budget: MediaResourceBudget,
    media_reservations: MediaReservationMap,
    /// Join deadline for pipeline generations, derived at startup from the
    /// configured backend fallback count, admission wait, and novelty
    /// embedding timeout.
    media_join_deadline: std::time::Duration,
    /// Recently reclaimed WHIP sessions, retained so DELETE remains idempotent
    /// after a peer-state watcher has already removed the live session entry.
    reclaimed_sessions: ReclaimedSessionMap,
    /// WebRTC configuration (STUN/TURN servers, token rate limit).
    webrtc_config: WebRtcConfig,
}

impl Deref for AppState {
    type Target = AppStateInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[must_use]
pub struct StreamSlotGuard {
    reservations: StreamReservations,
    principal_key: String,
}

impl Drop for StreamSlotGuard {
    fn drop(&mut self) {
        // Release the principal's slot under a single shard lock so a concurrent
        // reservation for the same principal cannot race the decrement. Holding the
        // entry across the check and the remove keeps the count and its presence in
        // step, and dropping the entry when it reaches zero keeps the map bounded to
        // principals that actually hold reservations.
        if let Entry::Occupied(mut slot) = self.reservations.entry(self.principal_key.clone()) {
            if *slot.get() <= 1 {
                slot.remove();
            } else {
                *slot.get_mut() -= 1;
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
    inference_admission_limits: AdmissionLimits,
    media_memory_budget_bytes: u64,
    media_worker_thread_budget: usize,
}

impl AppStateConfig {
    fn for_tests(wal_path: PathBuf) -> Self {
        Self {
            wal_path,
            provider: None,
            security_policy: SecurityPolicy::from_config_for_tests(),
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            inference_admission_limits: AdmissionLimits {
                global_in_flight: 8,
                per_principal_in_flight: 4,
                global_waiters: 128,
                wait_timeout: std::time::Duration::from_secs(5),
            },
            media_memory_budget_bytes: u64::MAX,
            media_worker_thread_budget: usize::MAX,
        }
    }

    fn build(self) -> AppState {
        let wal_path = Arc::new(self.wal_path);
        let wal = WalWriter::open(wal_path.as_ref()).expect("timeline WAL should open");
        let run_registry = Arc::new(RunRegistry::default());
        let timeline_snapshot = Arc::new(ArcSwap::from_pointee(RingSnapshot::default()));
        #[cfg(test)]
        let timeline_test_control = Arc::new(TimelineWriterTestControl::default());
        let timeline_tx = spawn_timeline_writer(
            wal,
            Arc::clone(&run_registry),
            Arc::clone(&timeline_snapshot),
            0,
            HashMap::new(),
            #[cfg(test)]
            Arc::clone(&timeline_test_control),
        );
        AppState {
            inner: Arc::new(AppStateInner {
                run_seq: AtomicU64::new(0),
                request_seq: AtomicU64::new(0),
                pipeline_generation_seq: AtomicU64::new(0),
                wal_path,
                ingest_file_roots: default_test_ingest_roots(),
                provider: self.provider,
                decode_pipeline: default_test_decode_pipeline(),
                security_policy: self.security_policy,
                inference_metrics: Arc::new(InferenceMetrics::new()),
                inference_admission: Arc::new(
                    InferenceAdmission::new(self.inference_admission_limits)
                        .expect("test admission limits are valid"),
                ),
                pipeline_metrics: Arc::new(PipelineMetrics::new()),
                novelty_config: LiveNoveltyConfig::default(),
                run_registry,
                timeline_tx,
                timeline_snapshot,
                #[cfg(test)]
                timeline_test_control,
                stream_reservations: Arc::new(DashMap::new()),
                tenant_label_maps: TenantLabelMaps::default(),
                stream_ttl_secs: self.stream_ttl_secs.max(1),
                active_stream_limit: self.active_stream_limit.max(1),
                spacetime_client: None,
                sessions: DashMap::new(),
                session_slots: AtomicUsize::new(0),
                media_budget: MediaResourceBudget::new(
                    self.media_memory_budget_bytes,
                    self.media_worker_thread_budget,
                ),
                media_reservations: DashMap::new(),
                reclaimed_sessions: Arc::new(Mutex::new(ReclaimedSessions::default())),
                media_join_deadline: vidarax_core::webrtc::runtime::supervise_join_deadline_from(
                    &vidarax_core::webrtc::runtime::JoinDeadlineInputs {
                        max_serial_inference_attempts: 1,
                        admission_wait_ms: self.inference_admission_limits.wait_timeout.as_millis()
                            as u64,
                        novelty_embedding_timeout_ms: LiveNoveltyConfig::default()
                            .embedding_timeout_ms,
                    },
                ),
                webrtc_config: WebRtcConfig::default(),
            }),
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
        novelty: LiveNoveltyConfig,
        inference_admission_limits: AdmissionLimits,
        media_memory_budget_bytes: u64,
        media_worker_thread_budget: usize,
        inference_backend_count: usize,
    ) -> Result<Self, String> {
        let novelty_embedding_timeout_ms = novelty.embedding_timeout_ms;
        let admission_wait_ms = inference_admission_limits.wait_timeout.as_millis() as u64;
        let existing_events = read_all_events(&wal_path).map_err(|err| err.to_string())?;
        let run_registry = build_run_registry(&existing_events);
        let initial_tails = build_event_tails(&existing_events);
        let timeline_snapshot = Arc::new(ArcSwap::from_pointee(RingSnapshot::from_tails(
            &initial_tails,
            existing_events
                .iter()
                .map(|event| event.seq)
                .max()
                .unwrap_or(0),
        )));
        let tenant_label_maps = TenantLabelMaps::from_env()?;
        let max_run_seq = existing_events.iter().fold(0u64, |acc, event| {
            let from_legacy = parse_run_sequence(&event.run_id).unwrap_or(0);
            acc.max(from_legacy)
        });
        let run_count = existing_events
            .iter()
            .filter(|event| event.kind == "run_created")
            .count() as u64;
        let max_seq = existing_events
            .iter()
            .map(|event| event.seq)
            .max()
            .unwrap_or(0);
        let wal_path = Arc::new(wal_path);
        let wal = WalWriter::open(wal_path.as_ref()).map_err(|err| err.to_string())?;
        #[cfg(test)]
        let timeline_test_control = Arc::new(TimelineWriterTestControl::default());
        let timeline_tx = spawn_timeline_writer(
            wal,
            Arc::clone(&run_registry),
            Arc::clone(&timeline_snapshot),
            max_seq,
            initial_tails,
            #[cfg(test)]
            Arc::clone(&timeline_test_control),
        );

        Ok(Self {
            inner: Arc::new(AppStateInner {
                run_seq: AtomicU64::new(run_count.max(max_run_seq)),
                request_seq: AtomicU64::new(0),
                pipeline_generation_seq: AtomicU64::new(0),
                wal_path,
                ingest_file_roots,
                provider,
                decode_pipeline,
                security_policy,
                inference_metrics: Arc::new(InferenceMetrics::new()),
                inference_admission: Arc::new(
                    InferenceAdmission::new(inference_admission_limits)
                        .map_err(ToString::to_string)?,
                ),
                pipeline_metrics: Arc::new(PipelineMetrics::new()),
                novelty_config: novelty,
                run_registry,
                timeline_tx,
                timeline_snapshot,
                #[cfg(test)]
                timeline_test_control,
                stream_reservations: Arc::new(DashMap::new()),
                tenant_label_maps,
                stream_ttl_secs,
                active_stream_limit: active_stream_limit.max(1),
                spacetime_client: None,
                sessions: DashMap::new(),
                session_slots: AtomicUsize::new(0),
                media_budget: MediaResourceBudget::new(
                    media_memory_budget_bytes,
                    media_worker_thread_budget,
                ),
                media_reservations: DashMap::new(),
                reclaimed_sessions: Arc::new(Mutex::new(ReclaimedSessions::default())),
                media_join_deadline: vidarax_core::webrtc::runtime::supervise_join_deadline_from(
                    &vidarax_core::webrtc::runtime::JoinDeadlineInputs {
                        max_serial_inference_attempts: inference_backend_count as u64,
                        admission_wait_ms,
                        novelty_embedding_timeout_ms,
                    },
                ),
                webrtc_config,
            }),
        })
    }

    pub fn with_wal_for_tests(wal_path: PathBuf) -> Self {
        AppStateConfig::for_tests(wal_path).build()
    }

    pub fn with_wal_for_tests_requiring_api_keys(wal_path: PathBuf, api_keys: Vec<String>) -> Self {
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
        Self::with_wal_for_tests_full(wal_path, provider, SecurityPolicy::from_config_for_tests())
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

    #[cfg(test)]
    pub(crate) fn set_timeline_append_failure_for_tests(&self, fail: bool) {
        self.timeline_test_control.set_failure(fail);
    }

    #[cfg(test)]
    pub(crate) fn pause_timeline_appends_for_tests(&self) {
        self.timeline_test_control.pause();
    }

    #[cfg(test)]
    pub(crate) fn wait_until_timeline_writer_paused_for_tests(&self) {
        self.timeline_test_control.wait_until_paused();
    }

    #[cfg(test)]
    pub(crate) fn resume_timeline_appends_for_tests(&self) {
        self.timeline_test_control.resume();
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
        Arc::get_mut(&mut self.inner)
            .expect("AppState builder mutation requires an unshared state")
            .spacetime_client = Some(client);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_tenant_label_maps_for_tests(
        mut self,
        maps: crate::tenant_labels::TenantLabelMaps,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("AppState test mutation requires an unshared state")
            .tenant_label_maps = maps;
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
    #[cfg(test)]
    pub fn insert_session(
        &self,
        sess_id: String,
        principal: String,
        run_id: Arc<str>,
        session: Arc<WebRtcSession>,
    ) -> bool {
        self.insert_session_reserved(sess_id, principal, run_id, session, None)
    }

    pub(crate) fn insert_session_with_media_reservation(
        &self,
        sess_id: String,
        principal: String,
        run_id: Arc<str>,
        session: Arc<WebRtcSession>,
        reservation: MediaResourcePermit,
    ) -> bool {
        self.insert_session_reserved(sess_id, principal, run_id, session, Some(reservation))
    }

    fn insert_session_reserved(
        &self,
        sess_id: String,
        principal: String,
        run_id: Arc<str>,
        session: Arc<WebRtcSession>,
        reservation: Option<MediaResourcePermit>,
    ) -> bool {
        if self
            .session_slots
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < MAX_WEBRTC_SESSIONS).then_some(active + 1)
            })
            .is_err()
        {
            return false;
        }
        // Claim the id under one shard lock so two inserts for the same id cannot
        // both win.
        match self.sessions.entry(sess_id.clone()) {
            Entry::Occupied(_) => {
                self.session_slots.fetch_sub(1, Ordering::AcqRel);
                return false;
            }
            Entry::Vacant(slot) => {
                slot.insert((principal, run_id, session));
            }
        }
        if let Some(reservation) = reservation {
            self.media_reservations.insert(sess_id.clone(), reservation);
        }
        self.reclaimed_sessions
            .lock()
            .expect("reclaimed-session registry lock poisoned")
            .remove(&sess_id);
        true
    }

    /// Release the media reservation for a session. Called by the pipeline
    /// supervisor after its join loop returns, and skipped on a forced
    /// shutdown because detached threads still hold their memory.
    pub(crate) fn release_media_reservation(&self, sess_id: &str) {
        self.media_reservations.remove(sess_id);
    }

    pub(crate) fn media_join_deadline(&self) -> std::time::Duration {
        self.media_join_deadline
    }

    /// Close the live WebRTC session that owns `run_id`, if any. The peer
    /// watcher then reclaims the session and the supervisor tears the
    /// pipeline down, so a REST stop or delete stops live work instead of
    /// only recording an event. With `preserve_history` the mark is stored
    /// on the session itself before the close, so exactly the reclaim that
    /// this close triggers skips the run_deleted tombstone and the stopped
    /// run stays readable. The mark is set once and never cleared.
    pub(crate) fn close_live_session_for_run(&self, run_id: &str, preserve_history: bool) {
        for entry in self.sessions.iter() {
            let (_principal, existing_run_id, session) = entry.value();
            if &**existing_run_id == run_id {
                if preserve_history {
                    session.mark_preserve_history();
                }
                session.close();
                break;
            }
        }
    }

    /// Look up a WebRTC session by ID.  Returns `None` if not found.
    pub fn get_session(&self, sess_id: &str) -> Option<SessionEntry> {
        self.sessions
            .get(sess_id)
            .map(|entry| entry.value().clone())
    }

    /// Remove and return a WebRTC session only if it still belongs to `run_id`.
    ///
    /// `remove_if` evaluates ownership and removes under one shard lock, so the
    /// removal is the atomic ownership transfer for reclaim paths: exactly one
    /// caller can win cleanup for a given session. The reclaim record is written
    /// right after the winning removal.
    pub(crate) fn remove_session_for_run(
        &self,
        sess_id: &str,
        run_id: &str,
    ) -> Option<SessionEntry> {
        let (_id, entry) = self
            .sessions
            .remove_if(sess_id, |_, (_, existing_run_id, _)| {
                &**existing_run_id == run_id
            })?;
        self.session_slots.fetch_sub(1, Ordering::AcqRel);
        // The media reservation is NOT released here. Session removal happens
        // on peer teardown, while the generation's OS threads may still be
        // joining. The supervisor releases the permit after the join loop
        // finishes, so the budget cannot be re-admitted while the old
        // generation's threads are still running.
        let (principal, existing_run_id, _session) = &entry;
        self.reclaimed_sessions
            .lock()
            .expect("reclaimed-session registry lock poisoned")
            .insert(
                sess_id.to_string(),
                principal.clone(),
                Arc::clone(existing_run_id),
                now_epoch_ms(),
            );
        Some(entry)
    }

    pub(crate) fn try_reserve_media_resources(
        &self,
        resources: MediaSessionResources,
    ) -> Option<MediaResourcePermit> {
        self.media_budget.try_reserve(resources)
    }

    pub fn render_media_capacity_prometheus(&self) -> String {
        self.media_budget.render_prometheus()
    }

    pub(crate) fn get_reclaimed_session(&self, sess_id: &str) -> Option<ReclaimedSessionEntry> {
        self.reclaimed_sessions
            .lock()
            .expect("reclaimed-session registry lock poisoned")
            .get(sess_id, now_epoch_ms())
    }

    /// Number of active WebRTC sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
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

    pub fn next_pipeline_generation(&self) -> vidarax_core::webrtc::runtime::PipelineGeneration {
        let value = self
            .pipeline_generation_seq
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        vidarax_core::webrtc::runtime::PipelineGeneration::new(value)
    }

    pub fn append_run_event(
        &self,
        run_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<TimelineEvent, String> {
        self.append_run_event_for_stream(run_id, DEFAULT_STREAM_ID, kind, payload)
    }

    pub fn append_run_event_nonblocking(
        &self,
        run_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<(), String> {
        self.append_run_event_for_stream_nonblocking(run_id, DEFAULT_STREAM_ID, kind, payload)
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

        self.append_timeline_sync(TimelineAppendRequest::new(
            run_id, stream_id, kind, payload, None,
        ))
    }

    pub fn append_run_event_for_stream_nonblocking(
        &self,
        run_id: &str,
        stream_id: &str,
        kind: &str,
        payload: Value,
    ) -> Result<(), String> {
        if kind == "run_deleted" {
            return Err("run_deleted requires confirmed append".to_string());
        }

        self.append_timeline_nonblocking(TimelineAppendRequest::new(
            run_id, stream_id, kind, payload, None,
        ))
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

        self.append_timeline_async(TimelineAppendRequest::new(
            run_id, stream_id, kind, payload, None,
        ))
        .await
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
                    let event = self.append_timeline_sync(TimelineAppendRequest::new(
                        run_id,
                        stream_id,
                        "run_deleted",
                        payload,
                        Some(guard),
                    ))?;
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
                    let event = self.append_timeline_sync(TimelineAppendRequest::new(
                        run_id,
                        stream_id,
                        "run_deleted",
                        payload,
                        None,
                    ))?;
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
                    let guard = RunDeleteAppendGuard::new(run);
                    let event = self
                        .append_timeline_async(TimelineAppendRequest::new(
                            run_id,
                            stream_id,
                            "run_deleted",
                            payload,
                            Some(guard),
                        ))
                        .await?;
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
                RunDeleteClaim::InFlight(run) => run.wait_delete_append().await,
                RunDeleteClaim::Missing => {
                    let event = self
                        .append_timeline_async(TimelineAppendRequest::new(
                            run_id,
                            stream_id,
                            "run_deleted",
                            payload,
                            None,
                        ))
                        .await?;
                    return Ok(RunDeletedAppend {
                        event,
                        appended: true,
                    });
                }
            }
        }
    }

    fn begin_run_deleted_append(&self, run_id: &str) -> RunDeleteClaim {
        let Some(run) = self
            .run_registry
            .runs
            .get(run_id)
            .map(|entry| Arc::clone(&entry))
        else {
            return RunDeleteClaim::Missing;
        };
        match run.begin_delete_append() {
            RunDeleteState::Claimed => RunDeleteClaim::Claimed(run),
            RunDeleteState::AlreadyDeleted => RunDeleteClaim::AlreadyDeleted,
            RunDeleteState::InFlight => RunDeleteClaim::InFlight(run),
        }
    }

    fn append_timeline_sync(
        &self,
        request: TimelineAppendRequest,
    ) -> Result<TimelineEvent, String> {
        // A worker detached by a forced shutdown can outlive its run's
        // deletion. Never append ordinary events after the tombstone.
        if self.run_is_deleted(&request.run_id) {
            return Err(format!("run {} is deleted", request.run_id));
        }
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let mut command = TimelineCommand::Append {
            request: Some(request),
            reply: TimelineReply::Sync(reply_tx),
        };
        loop {
            match self.timeline_tx.try_send(command) {
                Ok(()) => break,
                Err(mpsc::error::TrySendError::Full(returned)) => {
                    command = returned;
                    std::thread::yield_now();
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err("timeline writer is closed".to_string());
                }
            }
        }
        reply_rx
            .recv()
            .map_err(|err| format!("timeline writer reply failure: {err}"))?
    }

    fn append_timeline_nonblocking(&self, request: TimelineAppendRequest) -> Result<(), String> {
        match self
            .timeline_tx
            .try_send(TimelineCommand::AppendDetached { request })
        {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                let dropped = DROPPED_DETACHED_TIMELINE_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::warn!(
                    dropped_total = dropped,
                    "timeline writer queue full; dropping detached event"
                );
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err("timeline writer is closed".to_string())
            }
        }
    }

    async fn append_timeline_async(
        &self,
        request: TimelineAppendRequest,
    ) -> Result<TimelineEvent, String> {
        // Cancellation boundary: before `send` completes, the command remains
        // owned by this future and is not appended. After `send` completes, the
        // single-owner writer owns the command and will finish it even if the
        // caller disappears while waiting for the reply. Callers that may retry
        // a state transition need their own idempotency rule; run deletion uses
        // `append_run_deleted_for_stream_idempotent_async` for exactly that.
        let (reply_tx, reply_rx) = oneshot::channel();
        self.timeline_tx
            .send(TimelineCommand::Append {
                request: Some(request),
                reply: TimelineReply::Async(reply_tx),
            })
            .await
            .map_err(|_| "timeline writer is closed".to_string())?;
        reply_rx
            .await
            .map_err(|err| format!("timeline writer reply failure: {err}"))?
    }

    fn synthetic_run_deleted_event(
        &self,
        run_id: &str,
        stream_id: &str,
        payload: &Value,
    ) -> TimelineEvent {
        TimelineEvent {
            seq: self.timeline_snapshot.load().max_seq,
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

    /// Read a run's events with `seq >= from_seq`, served from the in-memory
    /// tail when it still holds that range and falling back to a WAL scan only
    /// on a cold ring or a cursor older than the tail. The result is filtered to
    /// `seq >= from_seq` and ordered by `seq` whichever source answers, so a
    /// polling client sees the same cursor order on a warm hit and a cold miss.
    pub async fn read_run_events_from(
        &self,
        run_id: &str,
        from_seq: u64,
    ) -> Result<Vec<TimelineEvent>, String> {
        if let Some(events) = self.timeline_snapshot.load().events_since(run_id, from_seq) {
            return Ok(events);
        }
        let mut events = self.read_run_events_async(run_id).await?;
        events.retain(|event| event.seq >= from_seq);
        events.sort_by_key(|event| event.seq);
        Ok(events)
    }

    pub fn metrics_snapshot(&self) -> (u64, u64) {
        let runs = self.run_seq.load(Ordering::Acquire);
        let events = self.timeline_snapshot.load().max_seq;
        (runs, events)
    }

    pub fn provider(&self) -> Option<&Arc<dyn InferenceProvider + Send + Sync>> {
        self.provider.as_ref()
    }

    pub fn admitted_provider(
        &self,
        principal: &str,
    ) -> Option<Arc<dyn InferenceProvider + Send + Sync>> {
        self.provider.as_ref().map(|provider| {
            Arc::new(AdmittedProvider::new(
                Arc::clone(provider),
                Arc::clone(&self.inference_admission),
                principal.into(),
            )) as Arc<dyn InferenceProvider + Send + Sync>
        })
    }

    pub fn decode_pipeline(&self) -> Arc<dyn DecodePipeline> {
        Arc::clone(&self.decode_pipeline)
    }

    pub fn ingest_file_roots(&self) -> &[PathBuf] {
        self.ingest_file_roots.as_slice()
    }

    pub fn security_policy(&self) -> &SecurityPolicy {
        &self.security_policy
    }

    pub fn inference_metrics(&self) -> &InferenceMetrics {
        &self.inference_metrics
    }

    pub fn inference_admission(&self) -> &InferenceAdmission {
        &self.inference_admission
    }

    /// Return the raw `Arc` for cases that need to move the metrics into
    /// a background thread (e.g. WHIP VLM/clip worker wiring) as an
    /// `InferenceObserver`.
    pub fn inference_metrics_arc(&self) -> &Arc<InferenceMetrics> {
        &self.inference_metrics
    }

    pub fn pipeline_metrics(&self) -> &PipelineMetrics {
        &self.pipeline_metrics
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

    pub fn novelty_config(&self) -> &LiveNoveltyConfig {
        &self.novelty_config
    }

    pub(crate) fn keyframe_blob_root(&self) -> PathBuf {
        self.wal_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("keyframes")
            .join("blobs")
    }

    pub fn map_event_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_event(tenant_id, label)
    }

    pub fn map_object_label(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.tenant_label_maps.map_object(tenant_id, label)
    }

    pub fn run_runtime_snapshot(&self, run_id: &str, now_ms: u64) -> Option<RunRuntimeSnapshot> {
        let summary = self.run_registry.runs.get(run_id)?.snapshot();
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
        self.run_registry
            .runs
            .get(run_id)
            .map(|summary| summary.snapshot().deleted)
            .unwrap_or(false)
    }

    pub fn count_active_runs_for_principal(&self, principal_key: &str, now_ms: u64) -> usize {
        let Some(run_ids) = self.run_registry.by_principal.get(principal_key) else {
            return 0;
        };
        let ttl_ms = self.stream_ttl_secs.saturating_mul(1000);
        run_ids
            .iter()
            .filter_map(|run_id| self.run_registry.runs.get(run_id))
            .filter(|summary| {
                let summary = summary.snapshot();
                // Trust the run entry, not just the index bucket: only count a run
                // that still names this principal. The index and the entry are
                // locked independently now, so this rejects any run whose bucket
                // membership has drifted from its live principal.
                summary.principal_key.as_ref() == principal_key
                    && !summary.deleted
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
        // Hold the principal's shard entry across the check and the increment so
        // two creators for the same principal cannot both pass the same snapshot.
        // The committed active count comes from the run registry, a different map,
        // so reading it here does not re-enter this one.
        let entry = self.stream_reservations.entry(principal_key.to_string());
        let reserved = match &entry {
            Entry::Occupied(slot) => *slot.get(),
            Entry::Vacant(_) => 0,
        };
        let committed = self.count_active_runs_for_principal(principal_key, now_ms);
        if committed.saturating_add(reserved) >= self.active_stream_limit {
            return None;
        }

        match entry {
            Entry::Occupied(mut slot) => *slot.get_mut() += 1,
            Entry::Vacant(slot) => {
                slot.insert(1);
            }
        }
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

    #[cfg(test)]
    fn apply_event_to_registry(&self, event: &TimelineEvent) {
        self.run_registry.apply_appended_event(event);
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

#[derive(Default)]
struct RingSnapshot {
    tails: HashMap<String, Arc<VecDeque<TimelineEvent>>>,
    max_seq: u64,
}

impl RingSnapshot {
    fn from_tails(tails: &HashMap<String, Arc<VecDeque<TimelineEvent>>>, max_seq: u64) -> Self {
        Self {
            tails: tails.clone(),
            max_seq,
        }
    }

    fn events_since(&self, run_id: &str, from_seq: u64) -> Option<Vec<TimelineEvent>> {
        let tail = self.tails.get(run_id)?;
        let oldest = tail.front()?.seq;
        if from_seq < oldest {
            return None;
        }
        Some(
            tail.iter()
                .filter(|event| event.seq >= from_seq)
                .cloned()
                .collect(),
        )
    }
}

struct TimelineAppendRequest {
    run_id: String,
    stream_id: String,
    kind: String,
    payload: Value,
    delete_guard: Option<RunDeleteAppendGuard>,
}

impl TimelineAppendRequest {
    fn new(
        run_id: &str,
        stream_id: &str,
        kind: &str,
        payload: Value,
        delete_guard: Option<RunDeleteAppendGuard>,
    ) -> Self {
        Self {
            run_id: run_id.to_owned(),
            stream_id: stream_id.to_owned(),
            kind: kind.to_owned(),
            payload,
            delete_guard,
        }
    }
}

enum TimelineReply {
    Async(oneshot::Sender<Result<TimelineEvent, String>>),
    Sync(std::sync::mpsc::Sender<Result<TimelineEvent, String>>),
}

impl TimelineReply {
    fn send(self, result: Result<TimelineEvent, String>) {
        match self {
            TimelineReply::Async(reply) => {
                let _ = reply.send(result);
            }
            TimelineReply::Sync(reply) => {
                let _ = reply.send(result);
            }
        }
    }
}

enum TimelineCommand {
    Append {
        request: Option<TimelineAppendRequest>,
        reply: TimelineReply,
    },
    AppendDetached {
        request: TimelineAppendRequest,
    },
}

#[cfg(test)]
#[derive(Default)]
struct TimelineWriterTestState {
    fail_appends: bool,
    pause_appends: bool,
    writer_paused: bool,
}

#[cfg(test)]
#[derive(Default)]
struct TimelineWriterTestControl {
    state: std::sync::Mutex<TimelineWriterTestState>,
    changed: std::sync::Condvar,
}

#[cfg(test)]
impl TimelineWriterTestControl {
    fn before_append(&self) -> Result<(), String> {
        let mut state = self.state.lock().expect("timeline test control lock");
        while state.pause_appends {
            state.writer_paused = true;
            self.changed.notify_all();
            state = self
                .changed
                .wait(state)
                .expect("timeline test control wait");
        }
        state.writer_paused = false;
        if state.fail_appends {
            Err("injected timeline WAL append failure".to_string())
        } else {
            Ok(())
        }
    }

    fn set_failure(&self, fail: bool) {
        self.state
            .lock()
            .expect("timeline test control lock")
            .fail_appends = fail;
    }

    fn pause(&self) {
        self.state
            .lock()
            .expect("timeline test control lock")
            .pause_appends = true;
    }

    fn wait_until_paused(&self) {
        let state = self.state.lock().expect("timeline test control lock");
        let (_state, timeout) = self
            .changed
            .wait_timeout_while(state, std::time::Duration::from_secs(2), |state| {
                !state.writer_paused
            })
            .expect("timeline test control wait");
        assert!(!timeout.timed_out(), "timeline writer did not pause");
    }

    fn resume(&self) {
        let mut state = self.state.lock().expect("timeline test control lock");
        state.pause_appends = false;
        self.changed.notify_all();
    }
}

struct TimelineWriter {
    wal: WalWriter,
    registry: Arc<RunRegistry>,
    snapshot: Arc<ArcSwap<RingSnapshot>>,
    next_seq: u64,
    tails: HashMap<String, Arc<VecDeque<TimelineEvent>>>,
    tail_recency: HashMap<String, u64>,
    tail_touch_tick: u64,
    #[cfg(test)]
    test_control: Arc<TimelineWriterTestControl>,
}

impl TimelineWriter {
    fn run(mut self, mut rx: mpsc::Receiver<TimelineCommand>) {
        while let Some(command) = rx.blocking_recv() {
            match command {
                TimelineCommand::Append { request, reply } => {
                    let result = self.append(request.expect("timeline append request present"));
                    reply.send(result);
                }
                TimelineCommand::AppendDetached { request } => {
                    if let Err(err) = self.append(request) {
                        tracing::warn!(%err, "detached timeline append failed");
                    }
                }
            }
        }
    }

    fn append(&mut self, mut request: TimelineAppendRequest) -> Result<TimelineEvent, String> {
        // Checked here, at dequeue time, because an ordinary event can sit in
        // the queue behind a run_deleted for the same run. The API-level check
        // cannot see that ordering, this one can.
        if request.kind != "run_deleted"
            && self
                .registry
                .runs
                .get(&request.run_id)
                .map(|summary| summary.snapshot().deleted)
                .unwrap_or(false)
        {
            return Err(format!("run {} is deleted", request.run_id));
        }
        self.next_seq = self.next_seq.saturating_add(1);
        let event = TimelineEvent {
            seq: self.next_seq,
            run_id: request.run_id,
            stream_id: request.stream_id,
            pts_ms: now_epoch_ms(),
            kind: request.kind,
            payload: request.payload.to_string(),
        };
        if let Err(err) = self.append_wal(&event) {
            self.next_seq = self.next_seq.saturating_sub(1);
            return Err(err);
        }

        self.registry.apply_appended_event(&event);
        match event.kind.as_str() {
            "run_deleted" => {
                self.tails.remove(&event.run_id);
                self.tail_recency.remove(&event.run_id);
            }
            _ => {
                let is_new_run = !self.tails.contains_key(&event.run_id);
                if is_new_run && self.tails.len() >= WARM_RUN_TAIL_CAP {
                    evict_least_recent_tail(&mut self.tails, &mut self.tail_recency);
                }
                // Snapshots may still hold the previous tail, so append by
                // replacing only this run's bounded tail and sharing all others.
                let mut tail = self
                    .tails
                    .get(&event.run_id)
                    .map(|existing| existing.as_ref().clone())
                    .unwrap_or_default();
                insert_event_by_seq(&mut tail, &event);
                self.tails.insert(event.run_id.clone(), Arc::new(tail));
                self.tail_touch_tick = self.tail_touch_tick.saturating_add(1);
                record_tail_touch(&mut self.tail_recency, &event.run_id, self.tail_touch_tick);
            }
        }
        if let Some(guard) = request.delete_guard.take() {
            guard.commit();
        }
        self.publish();
        Ok(event)
    }

    fn append_wal(&mut self, event: &TimelineEvent) -> Result<(), String> {
        // Tests inject failures and pauses here so rollback and cancellation
        // behavior can be exercised without changing the persistent file path.
        #[cfg(test)]
        self.test_control.before_append()?;
        self.wal.append(event).map_err(|err| err.to_string())
    }

    fn publish(&self) {
        self.snapshot.store(Arc::new(RingSnapshot::from_tails(
            &self.tails,
            self.next_seq,
        )));
    }
}

fn spawn_timeline_writer(
    wal: WalWriter,
    registry: Arc<RunRegistry>,
    snapshot: Arc<ArcSwap<RingSnapshot>>,
    next_seq: u64,
    tails: HashMap<String, Arc<VecDeque<TimelineEvent>>>,
    #[cfg(test)] test_control: Arc<TimelineWriterTestControl>,
) -> mpsc::Sender<TimelineCommand> {
    let (tx, rx) = mpsc::channel(TIMELINE_WRITER_QUEUE_CAP);
    let tail_recency = tails
        .iter()
        .map(|(run_id, tail)| (run_id.clone(), tail.back().map_or(0, |event| event.seq)))
        .collect();
    let writer = TimelineWriter {
        wal,
        registry,
        snapshot,
        next_seq,
        tails,
        tail_recency,
        tail_touch_tick: next_seq,
        #[cfg(test)]
        test_control,
    };
    std::thread::Builder::new()
        .name("vidarax-timeline-writer".to_string())
        .spawn(move || writer.run(rx))
        .expect("timeline writer thread should spawn");
    tx
}

#[derive(Clone)]
pub struct RunRuntimeSnapshot {
    pub principal_key: String,
    pub state: StreamState,
    pub last_activity_ms: u64,
}

/// Sharded per-run registry. Both maps carry their own locking, so adding or
/// deleting a run only touches that run's shard plus its principal's index
/// bucket; unrelated runs keep processing events with no contention and nothing
/// gets copied. The per-run state itself lives in atomics inside `RunState`, so
/// once a run exists its ongoing events never come back through here.
#[derive(Default)]
struct RunRegistry {
    runs: DashMap<String, Arc<RunState>>,
    by_principal: DashMap<Arc<str>, HashSet<String>>,
    /// Structural registry writes are owned by the timeline writer thread, so
    /// this FIFO's mutex has no writer contention. The mutex only permits the
    /// registry to remain shared with concurrent readers.
    deleted_run_order: Mutex<VecDeque<String>>,
}

impl RunRegistry {
    /// Register the effect of a create/delete (or an out-of-order first event)
    /// for one run. Every lock here is taken and released before the next map is
    /// touched: the `by_principal` bucket is settled with no `runs` lock held,
    /// and the run entry is written with no `by_principal` lock held. That keeps
    /// the write order the mirror of the reader in `count_active_runs_for_principal`
    /// (which walks `by_principal` then `runs`) so the two can never deadlock.
    ///
    /// Same-run structural events do not overlap in practice: a run is created by
    /// a single owner and deleted through the `RunState` compare-exchange claim,
    /// which lets exactly one caller reach this point. Distinct runs are free to
    /// arrive together and land on their own shards.
    fn apply_structural_event(&self, event: &TimelineEvent) {
        let before = self
            .runs
            .get(&event.run_id)
            .map(|entry| entry.snapshot())
            .unwrap_or_else(RunSummary::default_public);
        let mut after = before.clone();
        after.apply_event(event);
        let freshly_deleted = !before.deleted && after.deleted;

        if before.created
            && (!after.created || after.deleted || before.principal_key != after.principal_key)
        {
            let stale = &*before.principal_key;
            if let Some(mut set) = self.by_principal.get_mut(stale) {
                set.remove(&event.run_id);
            }
            // Re-checks emptiness under the bucket lock, so a run that slipped
            // back in between the drop above and here keeps its bucket.
            self.by_principal.remove_if(stale, |_, set| set.is_empty());
        }

        if after.created && !after.deleted {
            self.by_principal
                .entry(after.principal_key.clone())
                .or_default()
                .insert(event.run_id.clone());
        }

        self.runs.insert(
            event.run_id.clone(),
            Arc::new(RunState::from_summary(after)),
        );

        if freshly_deleted {
            let mut deleted_run_order = self
                .deleted_run_order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            deleted_run_order.push_back(event.run_id.clone());
            if deleted_run_order.len() > DELETED_RUN_RETENTION_CAP {
                let oldest = deleted_run_order
                    .pop_front()
                    .expect("deleted run FIFO exceeded its cap");
                self.runs
                    .remove_if(&oldest, |_, run| run.snapshot().deleted);
            }
        }
    }

    /// Apply one durable event to the shared structural registry. The timeline
    /// writer owns the event rings, so this method only mutates run metadata and
    /// per-run state needed by other handlers.
    fn apply_appended_event(&self, event: &TimelineEvent) {
        match event.kind.as_str() {
            "run_created" => {
                self.apply_structural_event(event);
            }
            "run_deleted" => {
                self.apply_structural_event(event);
            }
            _ => {
                // A run we already know about carries its whole state in per-run
                // atomics, so the common path just nudges those in place and
                // touches no other run.
                if let Some(run) = self.runs.get(&event.run_id) {
                    run.apply_event(event);
                } else {
                    // First sight of a run without its create event (out-of-order
                    // replay); fall back to the structural path to register it.
                    self.apply_structural_event(event);
                }
            }
        }
    }
}

/// Place `event` in a per-run tail so the ring stays ordered by `seq` and
/// bounded to [`RUN_EVENT_TAIL_CAP`]. Near-sorted arrivals take the push-back
/// fast path; a late or out-of-order event is inserted at its seq and an exact
/// duplicate is ignored. Trimming always drops the smallest seq, so the tail
/// keeps the run's highest-seq events and its front is the oldest it can serve.
fn insert_event_by_seq(tail: &mut VecDeque<TimelineEvent>, event: &TimelineEvent) {
    if tail.back().is_none_or(|back| event.seq > back.seq) {
        tail.push_back(event.clone());
    } else {
        match tail.binary_search_by(|held| held.seq.cmp(&event.seq)) {
            Ok(_) => return,
            Err(pos) => tail.insert(pos, event.clone()),
        }
    }

    while tail.len() > RUN_EVENT_TAIL_CAP {
        tail.pop_front();
    }
}

/// Refresh an existing run without allocating another owned map key. Only a
/// newly warm run needs to allocate its recency key.
fn record_tail_touch(recency: &mut HashMap<String, u64>, run_id: &str, tick: u64) {
    if let Some(last_touch) = recency.get_mut(run_id) {
        *last_touch = tick;
    } else {
        recency.insert(run_id.to_owned(), tick);
    }
}

/// Remove only the coldest warm-read accelerator. The WAL, sequence counter,
/// and run registry remain untouched because they own durable data and run
/// state. Published snapshots may retain the removed tail through its `Arc`.
fn evict_least_recent_tail<T>(tails: &mut HashMap<String, T>, recency: &mut HashMap<String, u64>) {
    let Some(run_id) = recency
        .iter()
        .min_by_key(|(_, tick)| **tick)
        .map(|(run_id, _)| run_id.clone())
    else {
        debug_assert!(tails.is_empty());
        return;
    };

    let removed_tail = tails.remove(&run_id).is_some();
    let removed_recency = recency.remove(&run_id).is_some();
    debug_assert!(removed_tail && removed_recency);
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
    let registry = RunRegistry::default();
    for event in events {
        registry.apply_structural_event(event);
    }
    Arc::new(registry)
}

fn build_event_tails(events: &[TimelineEvent]) -> HashMap<String, Arc<VecDeque<TimelineEvent>>> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|event| event.seq);
    let mut tails: HashMap<String, VecDeque<TimelineEvent>> = HashMap::new();
    let mut recency = HashMap::new();
    for event in sorted {
        if event.kind == "run_deleted" {
            tails.remove(&event.run_id);
            recency.remove(&event.run_id);
        } else {
            let is_new_run = !tails.contains_key(&event.run_id);
            if is_new_run && tails.len() >= WARM_RUN_TAIL_CAP {
                evict_least_recent_tail(&mut tails, &mut recency);
            }
            insert_event_by_seq(tails.entry(event.run_id.clone()).or_default(), &event);
            record_tail_touch(&mut recency, &event.run_id, event.seq);
        }
    }
    tails
        .into_iter()
        .map(|(run_id, tail)| (run_id, Arc::new(tail)))
        .collect()
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

    fn media_resources(bytes: u64, threads: usize) -> MediaSessionResources {
        MediaSessionResources {
            worker_threads: threads,
            reserved_bytes: bytes,
            rtp_queue_bytes: bytes,
            decoded_frame_bytes: 0,
            jpeg_payload_bytes: 0,
            scratch_bytes: 0,
            sidecar_bytes: 0,
        }
    }

    #[test]
    fn media_budget_reserves_both_dimensions_and_releases_on_drop() {
        let budget = MediaResourceBudget::new(100, 4);
        let first = budget.try_reserve(media_resources(60, 2)).unwrap();
        assert!(budget.try_reserve(media_resources(50, 1)).is_none());
        assert!(budget.try_reserve(media_resources(10, 3)).is_none());
        drop(first);
        assert!(budget.try_reserve(media_resources(100, 4)).is_some());

        let metrics = budget.render_prometheus();
        assert!(metrics.contains("vidarax_media_capacity_rejections_total 2"));
    }

    #[test]
    fn concurrent_media_budget_admission_never_exceeds_limit() {
        const CONTENDERS: usize = 16;
        let budget = MediaResourceBudget::new(1_000, 4);
        let start = Arc::new(std::sync::Barrier::new(CONTENDERS + 1));
        let release = Arc::new(AtomicBool::new(false));
        let (results_tx, results_rx) = std::sync::mpsc::channel();
        let mut threads = Vec::with_capacity(CONTENDERS);

        for _ in 0..CONTENDERS {
            let budget = budget.clone();
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let results_tx = results_tx.clone();
            threads.push(thread::spawn(move || {
                start.wait();
                let permit = budget.try_reserve(media_resources(100, 1));
                results_tx.send(permit.is_some()).unwrap();
                while !release.load(AtomicOrdering::Acquire) {
                    thread::yield_now();
                }
                drop(permit);
            }));
        }
        drop(results_tx);
        start.wait();

        let admitted = (0..CONTENDERS)
            .map(|_| results_rx.recv().unwrap())
            .filter(|admitted| *admitted)
            .count();
        assert_eq!(admitted, 4);
        assert_eq!(
            budget.inner.reserved_worker_threads.load(Ordering::Acquire),
            4
        );
        assert_eq!(budget.inner.reserved_bytes.load(Ordering::Acquire), 400);

        release.store(true, AtomicOrdering::Release);
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(
            budget.inner.reserved_worker_threads.load(Ordering::Acquire),
            0
        );
        assert_eq!(budget.inner.reserved_bytes.load(Ordering::Acquire), 0);
    }

    fn event(seq: u64, kind: &str, pts_ms: u64, payload: Value) -> TimelineEvent {
        event_for_run(seq, "run-1", kind, pts_ms, payload)
    }

    fn event_for_run(
        seq: u64,
        run_id: &str,
        kind: &str,
        pts_ms: u64,
        payload: Value,
    ) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: run_id.to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms,
            kind: kind.to_string(),
            payload: payload.to_string(),
        }
    }

    #[test]
    fn registry_keeps_same_map_snapshot_for_existing_run_event() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-state-test-{}.wal", std::process::id())),
        );
        state.apply_event_to_registry(&event(
            1,
            "run_created",
            100,
            json!({"principal_key": "tenant-a"}),
        ));
        let before = Arc::clone(&state.run_registry.runs.get("run-1").expect("run exists"));

        state.apply_event_to_registry(&event(2, "analysis_generated", 150, json!({})));
        let after = Arc::clone(&state.run_registry.runs.get("run-1").expect("run exists"));

        // Same `RunState` pointer before and after means the per-event write went
        // straight to the run's atomics: it neither allocated a replacement entry
        // nor rebuilt any map, which is the whole point of dropping the old
        // clone-the-registry write path.
        assert!(
            Arc::ptr_eq(&before, &after),
            "existing-run events must update run atomics in place without replacing the entry"
        );
        let snapshot = state.run_runtime_snapshot("run-1", 150).unwrap();
        assert_eq!(snapshot.principal_key, "tenant-a");
        assert_eq!(snapshot.state, StreamState::Processing);
        assert_eq!(snapshot.last_activity_ms, 150);
    }

    #[test]
    fn tail_insert_orders_by_seq_and_dedups() {
        let mut tail = VecDeque::new();
        for seq in [1u64, 2, 5] {
            insert_event_by_seq(&mut tail, &event(seq, "analysis_generated", seq, json!({})));
        }
        // A late, out-of-order arrival lands at its seq, not at the back.
        insert_event_by_seq(&mut tail, &event(3, "analysis_generated", 3, json!({})));
        // An exact-seq repeat is ignored rather than double-counted.
        insert_event_by_seq(&mut tail, &event(3, "analysis_generated", 3, json!({})));

        let seqs: Vec<u64> = tail.iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3, 5]);
    }

    #[test]
    fn tail_eviction_keeps_highest_seqs() {
        let mut tail = VecDeque::new();
        for seq in 1..=(RUN_EVENT_TAIL_CAP as u64 + 5) {
            insert_event_by_seq(&mut tail, &event(seq, "analysis_generated", seq, json!({})));
        }
        assert_eq!(tail.len(), RUN_EVENT_TAIL_CAP);
        assert_eq!(tail.front().unwrap().seq, 6);
        assert_eq!(tail.back().unwrap().seq, RUN_EVENT_TAIL_CAP as u64 + 5);

        // An arrival older than everything retained never displaces the window.
        insert_event_by_seq(&mut tail, &event(1, "analysis_generated", 1, json!({})));
        assert_eq!(tail.len(), RUN_EVENT_TAIL_CAP);
        assert_eq!(tail.front().unwrap().seq, 6);
    }

    #[test]
    fn snapshot_serves_warm_cursor_and_defers_cold() {
        let mut tails = HashMap::new();
        let mut tail = VecDeque::new();
        for seq in 1..=4 {
            insert_event_by_seq(&mut tail, &event(seq, "analysis_generated", seq, json!({})));
        }
        tails.insert("run-1".to_string(), Arc::new(tail));
        let snapshot = RingSnapshot::from_tails(&tails, 4);

        let served = snapshot
            .events_since("run-1", 2)
            .expect("warm cursor served");
        assert_eq!(
            served.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(snapshot.events_since("run-1", 99).unwrap().len(), 0);
        assert!(snapshot.events_since("run-1", 0).is_none());
        assert!(snapshot.events_since("missing", 1).is_none());
    }

    #[test]
    fn snapshot_eviction_defers_when_cursor_predates_window() {
        let mut tails = HashMap::new();
        let mut tail = VecDeque::new();
        for seq in 1..=(RUN_EVENT_TAIL_CAP as u64 + 3) {
            insert_event_by_seq(&mut tail, &event(seq, "analysis_generated", seq, json!({})));
        }
        tails.insert("run-1".to_string(), Arc::new(tail));
        let snapshot = RingSnapshot::from_tails(&tails, RUN_EVENT_TAIL_CAP as u64 + 3);

        assert!(snapshot.events_since("run-1", 1).is_none());
        assert!(snapshot.events_since("run-1", 4).is_some());
    }

    #[test]
    fn publishing_append_reuses_unchanged_run_tail_arc() {
        let wal_path =
            std::env::temp_dir().join(format!("vidarax-state-cow-tail-{}.wal", std::process::id()));
        std::fs::remove_file(&wal_path).ok();
        let snapshot = Arc::new(ArcSwap::from_pointee(RingSnapshot::default()));
        let mut writer = TimelineWriter {
            wal: WalWriter::open(&wal_path).unwrap(),
            registry: Arc::new(RunRegistry::default()),
            snapshot: Arc::clone(&snapshot),
            next_seq: 0,
            tails: HashMap::new(),
            tail_recency: HashMap::new(),
            tail_touch_tick: 0,
            test_control: Arc::new(TimelineWriterTestControl::default()),
        };

        writer
            .append(TimelineAppendRequest::new(
                "run-1",
                "stream-0",
                "analysis_generated",
                json!({}),
                None,
            ))
            .unwrap();
        let first_snapshot = snapshot.load();
        let first_run_tail = Arc::clone(first_snapshot.tails.get("run-1").unwrap());
        drop(first_snapshot);

        writer
            .append(TimelineAppendRequest::new(
                "run-2",
                "stream-0",
                "analysis_generated",
                json!({}),
                None,
            ))
            .unwrap();

        let second_snapshot = snapshot.load();
        let second_run_tail = Arc::clone(second_snapshot.tails.get("run-1").unwrap());
        assert!(Arc::ptr_eq(&first_run_tail, &second_run_tail));

        std::fs::remove_file(wal_path).ok();
    }

    #[tokio::test]
    async fn warm_run_tail_lru_caps_and_falls_back_to_wal() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-warm-tail-cap-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = AppState::with_wal_for_tests(wal_path.clone());

        for run in 0..(WARM_RUN_TAIL_CAP + 2) {
            state
                .append_run_event(&format!("run-{run}"), "run_created", json!({}))
                .unwrap();
        }

        let snapshot = state.timeline_snapshot.load();
        assert_eq!(snapshot.tails.len(), WARM_RUN_TAIL_CAP);
        assert!(!snapshot.tails.contains_key("run-0"));
        assert!(!snapshot.tails.contains_key("run-1"));
        drop(snapshot);

        let cold_events = state.read_run_events_from("run-0", 0).await.unwrap();
        assert_eq!(cold_events.len(), 1);
        assert_eq!(cold_events[0].run_id, "run-0");
        assert_eq!(cold_events[0].kind, "run_created");

        std::fs::remove_file(wal_path).ok();
    }

    #[test]
    fn warm_run_tail_lru_refreshes_recently_appended_run() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-warm-tail-refresh-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = AppState::with_wal_for_tests(wal_path.clone());

        for run in 0..WARM_RUN_TAIL_CAP {
            state
                .append_run_event(&format!("run-{run}"), "run_created", json!({}))
                .unwrap();
        }
        state
            .append_run_event("run-0", "analysis_generated", json!({}))
            .unwrap();
        state
            .append_run_event(
                &format!("run-{WARM_RUN_TAIL_CAP}"),
                "run_created",
                json!({}),
            )
            .unwrap();

        let snapshot = state.timeline_snapshot.load();
        assert_eq!(snapshot.tails.len(), WARM_RUN_TAIL_CAP);
        assert!(snapshot.tails.contains_key("run-0"));
        assert!(!snapshot.tails.contains_key("run-1"));

        std::fs::remove_file(wal_path).ok();
    }

    #[test]
    fn run_deleted_removes_tail_and_recency_entry() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-warm-tail-delete-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let snapshot = Arc::new(ArcSwap::from_pointee(RingSnapshot::default()));
        let mut writer = TimelineWriter {
            wal: WalWriter::open(&wal_path).unwrap(),
            registry: Arc::new(RunRegistry::default()),
            snapshot,
            next_seq: 0,
            tails: HashMap::new(),
            tail_recency: HashMap::new(),
            tail_touch_tick: 0,
            test_control: Arc::new(TimelineWriterTestControl::default()),
        };

        writer
            .append(TimelineAppendRequest::new(
                "run-1",
                "stream-0",
                "run_created",
                json!({}),
                None,
            ))
            .unwrap();
        assert!(writer.tails.contains_key("run-1"));
        assert!(writer.tail_recency.contains_key("run-1"));

        writer
            .append(TimelineAppendRequest::new(
                "run-1",
                "stream-0",
                "run_deleted",
                json!({}),
                None,
            ))
            .unwrap();
        assert!(!writer.tails.contains_key("run-1"));
        assert!(!writer.tail_recency.contains_key("run-1"));

        std::fs::remove_file(wal_path).ok();
    }

    #[test]
    fn build_event_tails_forgets_deleted_run() {
        let events = vec![
            event(1, "run_created", 1, json!({"principal_key": "public"})),
            event(2, "analysis_generated", 2, json!({})),
            event(3, "run_deleted", 3, json!({})),
        ];

        let tails = build_event_tails(&events);

        assert!(!tails.contains_key("run-1"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_same_run_appends_never_skip_a_seq() {
        const TASKS: u64 = 4;
        const PER: u64 = 60;
        let total = (TASKS * PER) as usize;
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-same-run-race-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = Arc::new(AppState::with_wal_for_tests(wal_path));

        let writers: Vec<_> = (0..TASKS)
            .map(|task| {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    let stream = format!("stream-{task}");
                    for _ in 0..PER {
                        state
                            .append_run_event_for_stream_async(
                                "run-1",
                                &stream,
                                "analysis_generated",
                                json!({}),
                            )
                            .await
                            .expect("append failed");
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        let reader = {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let mut cursor = 1u64;
                let mut consumed = 0usize;
                let start = std::time::Instant::now();
                while consumed < total {
                    assert!(
                        start.elapsed() < std::time::Duration::from_secs(30),
                        "reader stalled at cursor {cursor}, consumed {consumed}/{total}"
                    );
                    let batch = state.read_run_events_from("run-1", cursor).await.unwrap();
                    if batch.is_empty() {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    for ev in batch {
                        assert_eq!(ev.seq, cursor, "reader served seq {} out of order", ev.seq);
                        cursor += 1;
                        consumed += 1;
                    }
                }
            })
        };

        for writer in writers {
            writer.await.expect("writer panicked");
        }
        reader
            .await
            .expect("reader saw a skipped or out-of-order seq");

        std::fs::remove_file(state.wal_path.as_ref()).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_fallback_reads_never_skip_a_seq() {
        // Drive read_run_events_from against racing async appends. The writer
        // serializes seq assignment, WAL append, and snapshot publication, so the
        // reader must see only contiguous seqs as it advances its cursor.
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-fallback-race-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = Arc::new(AppState::with_wal_for_tests(wal_path));

        const TASKS: u64 = 8;
        const PER: u64 = 30;
        let total = (TASKS * PER) as usize;

        let writers: Vec<_> = (0..TASKS)
            .map(|task| {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    let stream = format!("stream-{task}");
                    for _ in 0..PER {
                        state
                            .append_run_event_for_stream_async(
                                "run-1",
                                &stream,
                                "analysis_generated",
                                json!({}),
                            )
                            .await
                            .expect("append failed");
                    }
                })
            })
            .collect();

        let reader = {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let mut cursor = 1u64;
                let mut consumed = 0usize;
                let start = std::time::Instant::now();
                while consumed < total {
                    assert!(
                        start.elapsed() < std::time::Duration::from_secs(30),
                        "reader stalled at cursor {cursor}, consumed {consumed}/{total}"
                    );
                    let batch = state.read_run_events_from("run-1", cursor).await.unwrap();
                    if batch.is_empty() {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    for ev in batch {
                        assert_eq!(ev.seq, cursor, "reader served seq {} out of order", ev.seq);
                        cursor += 1;
                        consumed += 1;
                    }
                }
            })
        };

        for writer in writers {
            writer.await.expect("writer panicked");
        }
        reader
            .await
            .expect("reader saw a skipped or out-of-order seq");

        std::fs::remove_file(state.wal_path.as_ref()).ok();
    }

    #[tokio::test]
    async fn read_run_events_from_matches_wal_for_same_cursor() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-tail-cursor-{}.wal",
            std::process::id()
        )));
        state
            .append_run_event("run-1", "run_created", json!({"principal_key": "public"}))
            .unwrap();
        for _ in 0..5 {
            state
                .append_run_event("run-1", "analysis_generated", json!({}))
                .unwrap();
        }

        // A warm cursor serves exactly the WAL tail for the same from_seq.
        let from_seq = 3;
        let memory = state.read_run_events_from("run-1", from_seq).await.unwrap();
        let wal: Vec<_> = state
            .read_run_events_async("run-1")
            .await
            .unwrap()
            .into_iter()
            .filter(|e| e.seq >= from_seq)
            .collect();
        assert_eq!(memory, wal);
        assert!(memory.iter().all(|e| e.seq >= from_seq));
    }

    #[tokio::test]
    async fn fallback_filters_and_sorts_wal_events() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-fallback-sort-{}.wal",
            std::process::id()
        )));
        for seq in [4, 2, 3, 1] {
            append_event(
                state.wal_path.as_ref(),
                &event(seq, "analysis_generated", seq, json!({})),
            )
            .unwrap();
        }

        let served = state.read_run_events_from("run-1", 3).await.unwrap();
        assert_eq!(served.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![3, 4]);

        std::fs::remove_file(state.wal_path.as_ref()).ok();
    }

    #[test]
    fn append_run_event_for_stream_persists_stream_id() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-stream-test-{}.wal",
            std::process::id()
        )));

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
    async fn ordinary_append_queued_behind_tombstone_is_rejected() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-tombstone-race-{}.wal",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&wal_path);
        let state = AppState::with_wal_for_tests(wal_path.clone());

        state
            .append_run_event("run_race", "run_created", serde_json::json!({}))
            .expect("create run");

        // Pause the writer, then queue the tombstone and an ordinary event so
        // the ordinary event sits in the queue behind the deletion. The old
        // API-level guard could not see this ordering.
        state.pause_timeline_appends_for_tests();
        let delete_state = state.clone();
        let delete_task = tokio::task::spawn_blocking(move || {
            delete_state.append_run_event("run_race", "run_deleted", serde_json::json!({}))
        });
        state.wait_until_timeline_writer_paused_for_tests();
        state.append_run_event_for_stream_nonblocking(
            "run_race",
            "stream-0",
            "note",
            serde_json::json!({"text": "late"}),
        );
        state.resume_timeline_appends_for_tests();

        delete_task
            .await
            .expect("join delete task")
            .expect("tombstone append");

        // Drain the writer with one more synchronous command, then check the
        // late ordinary event was rejected at dequeue time.
        assert!(state
            .append_run_event("run_race", "note", serde_json::json!({}))
            .is_err());
        let events = read_all_events(&wal_path).expect("read events");
        assert!(
            events
                .iter()
                .filter(|event| event.run_id == "run_race")
                .all(|event| event.kind != "note"),
            "no ordinary event may land after the tombstone"
        );
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn deleted_run_registry_retention_cap_evicts_only_old_tombstones() {
        const EXTRA_DELETIONS: usize = 7;
        let registry = RunRegistry::default();
        let live_run_id = "run-live";
        registry.apply_structural_event(&event_for_run(
            1,
            live_run_id,
            "run_created",
            1,
            json!({ "principal_key": "tenant-a" }),
        ));

        let mut seq = 2;
        for i in 0..(DELETED_RUN_RETENTION_CAP + EXTRA_DELETIONS) {
            let run_id = format!("run-deleted-{i:05}");
            registry.apply_structural_event(&event_for_run(
                seq,
                &run_id,
                "run_created",
                seq,
                json!({ "principal_key": "tenant-a" }),
            ));
            seq += 1;
            registry.apply_structural_event(&event_for_run(
                seq,
                &run_id,
                "run_deleted",
                seq,
                json!({}),
            ));
            seq += 1;
        }

        assert_eq!(
            registry
                .runs
                .iter()
                .filter(|run| run.snapshot().deleted)
                .count(),
            DELETED_RUN_RETENTION_CAP
        );
        assert_eq!(registry.runs.len(), DELETED_RUN_RETENTION_CAP + 1);
        for i in 0..EXTRA_DELETIONS {
            assert!(!registry.runs.contains_key(&format!("run-deleted-{i:05}")));
        }
        assert!(registry.runs.get(live_run_id).is_some_and(|run| {
            let run = run.snapshot();
            run.created && !run.deleted
        }));
        assert!(registry
            .by_principal
            .get("tenant-a")
            .is_some_and(|run_ids| run_ids.contains(live_run_id)));
    }

    #[test]
    fn recent_deleted_run_keeps_delete_idempotency() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-recent-delete-idempotency-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = AppState::with_wal_for_tests(wal_path.clone());
        state
            .append_run_event(
                "run-recent",
                "run_created",
                json!({ "principal_key": "tenant-a" }),
            )
            .unwrap();

        let first = state
            .append_run_deleted_for_stream_idempotent("run-recent", "stream-0", json!({}))
            .unwrap();
        let second = state
            .append_run_deleted_for_stream_idempotent("run-recent", "stream-0", json!({}))
            .unwrap();

        assert!(first.appended);
        assert!(!second.appended);
        assert_eq!(
            state
                .read_run_events("run-recent")
                .unwrap()
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        drop(state);
        std::fs::remove_file(wal_path).ok();
    }

    #[test]
    fn evicted_deleted_run_appends_delete_again_via_missing_path() {
        let wal_path = std::env::temp_dir().join(format!(
            "vidarax-state-evicted-delete-boundary-{}.wal",
            std::process::id()
        ));
        std::fs::remove_file(&wal_path).ok();
        let state = AppState::with_wal_for_tests(wal_path.clone());
        state.apply_event_to_registry(&event_for_run(
            1,
            "run-evicted",
            "run_created",
            1,
            json!({ "principal_key": "tenant-a" }),
        ));
        state.apply_event_to_registry(&event_for_run(
            2,
            "run-evicted",
            "run_deleted",
            2,
            json!({}),
        ));
        for i in 0..DELETED_RUN_RETENTION_CAP {
            state.apply_event_to_registry(&event_for_run(
                i as u64 + 3,
                &format!("run-newer-deleted-{i:05}"),
                "run_deleted",
                i as u64 + 3,
                json!({}),
            ));
        }
        assert!(!state.run_registry.runs.contains_key("run-evicted"));

        let repeated = state
            .append_run_deleted_for_stream_idempotent(
                "run-evicted",
                "stream-0",
                json!({ "reason": "late retry" }),
            )
            .unwrap();

        assert!(repeated.appended);
        assert_eq!(
            state
                .read_run_events("run-evicted")
                .unwrap()
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        drop(state);
        std::fs::remove_file(wal_path).ok();
    }

    #[test]
    fn registry_replay_bounds_deleted_runs_and_keeps_live_runs() {
        const EXTRA_DELETIONS: usize = 5;
        let mut events = Vec::with_capacity((DELETED_RUN_RETENTION_CAP + EXTRA_DELETIONS) * 2 + 2);
        let mut seq = 1u64;
        for i in 0..(DELETED_RUN_RETENTION_CAP + EXTRA_DELETIONS) {
            let run_id = format!("run-replay-deleted-{i:05}");
            events.push(event_for_run(
                seq,
                &run_id,
                "run_created",
                seq,
                json!({ "principal_key": "tenant-a" }),
            ));
            seq += 1;
            events.push(event_for_run(seq, &run_id, "run_deleted", seq, json!({})));
            seq += 1;
        }
        for run_id in ["run-replay-live-a", "run-replay-live-b"] {
            events.push(event_for_run(
                seq,
                run_id,
                "run_created",
                seq,
                json!({ "principal_key": "tenant-a" }),
            ));
            seq += 1;
        }

        let registry = build_run_registry(&events);

        assert_eq!(registry.runs.len(), DELETED_RUN_RETENTION_CAP + 2);
        assert!(!registry.runs.contains_key("run-replay-deleted-00000"));
        let live_runs = registry.by_principal.get("tenant-a").unwrap();
        assert_eq!(live_runs.len(), 2);
        assert!(live_runs.contains("run-replay-live-a"));
        assert!(live_runs.contains("run-replay-live-b"));
        drop(live_runs);
        let most_recent_deleted = format!(
            "run-replay-deleted-{:05}",
            DELETED_RUN_RETENTION_CAP + EXTRA_DELETIONS - 1
        );
        let run = registry.runs.get(&most_recent_deleted).unwrap();
        assert!(matches!(
            run.begin_delete_append(),
            RunDeleteState::AlreadyDeleted
        ));
    }

    #[test]
    fn stale_deleted_fifo_entry_does_not_evict_recreated_live_run() {
        let registry = RunRegistry::default();
        registry.apply_structural_event(&event_for_run(
            1,
            "run-recreated",
            "run_deleted",
            1,
            json!({}),
        ));

        // Model the old tombstone being forgotten before the identifier is
        // reused. Its stale FIFO entry must not remove the new live lifecycle.
        registry.runs.remove("run-recreated");
        registry.apply_structural_event(&event_for_run(
            2,
            "run-recreated",
            "run_created",
            2,
            json!({ "principal_key": "tenant-a" }),
        ));
        for i in 0..DELETED_RUN_RETENTION_CAP {
            registry.apply_structural_event(&event_for_run(
                i as u64 + 3,
                &format!("run-after-recreate-{i:05}"),
                "run_deleted",
                i as u64 + 3,
                json!({}),
            ));
        }

        assert!(registry.runs.get("run-recreated").is_some_and(|run| {
            let run = run.snapshot();
            run.created && !run.deleted
        }));
        assert!(registry
            .by_principal
            .get("tenant-a")
            .is_some_and(|run_ids| run_ids.contains("run-recreated")));
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
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_epoch_ms()),
            1
        );

        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let mut tasks = Vec::new();
        for worker in 0..8 {
            let state = state.clone();
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                state
                    .append_run_deleted_idempotent_async("run-1", json!({ "worker": worker }))
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
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_epoch_ms()),
            0
        );
        assert!(state
            .run_runtime_snapshot("run-1", now_epoch_ms())
            .is_none());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_run_deleted_append_releases_claim_for_retry() {
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

        state.pause_timeline_appends_for_tests();

        let delete_state = state.clone();
        let delete_task = tokio::spawn(async move {
            delete_state
                .append_run_deleted_idempotent_async("run-1", json!({ "reason": "cancelled" }))
                .await
        });

        state.wait_until_timeline_writer_paused_for_tests();

        delete_task.abort();
        let _ = delete_task.await;
        state.resume_timeline_appends_for_tests();

        let appended = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            state.append_run_deleted_idempotent_async("run-1", json!({ "reason": "retry" })),
        )
        .await
        .expect("retry after cancelled delete must not wait forever")
        .expect("retry delete append should succeed");
        assert!(
            !appended,
            "cancelled caller must not force a duplicate tombstone"
        );
        assert!(state
            .run_runtime_snapshot("run-1", now_epoch_ms())
            .is_none());
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
            assert!(state.insert_session(
                sess_id.clone(),
                "tenant-a".to_string(),
                Arc::clone(&run_id),
                Arc::new(WebRtcSession::new_for_tests()),
            ));
            state
                .remove_session_for_run(&sess_id, &run_id)
                .expect("session should be reclaimed");
        }

        assert_eq!(
            state
                .reclaimed_sessions
                .lock()
                .expect("reclaimed-session registry lock poisoned")
                .len(),
            RECLAIMED_SESSION_MAX_ENTRIES
        );
        assert!(state.get_reclaimed_session("sess-reclaimed-0000").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_session_admission_never_exceeds_global_cap() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-state-session-cap-{}.wal",
            std::process::id()
        )));
        let barrier = Arc::new(tokio::sync::Barrier::new(MAX_WEBRTC_SESSIONS * 2));
        let mut tasks = Vec::with_capacity(MAX_WEBRTC_SESSIONS * 2);

        for i in 0..(MAX_WEBRTC_SESSIONS * 2) {
            let state = state.clone();
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                let session = Arc::new(WebRtcSession::new_for_tests());
                barrier.wait().await;
                state.insert_session(
                    format!("sess-cap-{i:04}"),
                    "tenant-a".to_string(),
                    Arc::from(format!("run-cap-{i:04}")),
                    session,
                )
            }));
        }

        let mut admitted = 0;
        for task in tasks {
            if task.await.expect("admission task should not panic") {
                admitted += 1;
            }
        }
        assert_eq!(admitted, MAX_WEBRTC_SESSIONS);
        assert_eq!(state.session_count(), MAX_WEBRTC_SESSIONS);
        assert_eq!(
            state.session_slots.load(Ordering::Acquire),
            MAX_WEBRTC_SESSIONS
        );
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

    // Measurement harness for the timeline snapshot-publish cost (w3r.12).
    // Run explicitly: cargo test -p vidarax-api --lib --release measure_publish -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure_publish_snapshot_cost_by_run_count() {
        use std::time::Instant;
        for &n in &[1usize, 10, 100, 1000, 5000] {
            let mut tails: HashMap<String, Arc<VecDeque<TimelineEvent>>> = HashMap::new();
            for r in 0..n {
                let mut dq = VecDeque::new();
                for s in 0..8u64 {
                    dq.push_back(event(s + 1, "analysis_generated", s + 1, json!({})));
                }
                tails.insert(format!("run-{r}"), Arc::new(dq));
            }
            for _ in 0..200 {
                std::hint::black_box(RingSnapshot::from_tails(&tails, n as u64));
            }
            let iters = 5000u32;
            let start = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(RingSnapshot::from_tails(&tails, n as u64));
            }
            let per = start.elapsed().as_nanos() as f64 / iters as f64;
            println!("publish from_tails: {n:>5} runs -> {per:>10.0} ns/append");
        }
    }

    // Measurement harness for a real end-to-end sync append (WAL + writer),
    // for scale comparison against the publish cost above.
    #[test]
    #[ignore]
    fn measure_sync_append_latency() {
        use std::time::Instant;
        let wal_path =
            std::env::temp_dir().join(format!("vidarax-bench-append-{}.wal", std::process::id()));
        std::fs::remove_file(&wal_path).ok();
        let state = AppState::with_wal_for_tests(wal_path);
        for i in 0..200u64 {
            state
                .append_run_event(&format!("run-{i}"), "run_created", json!({}))
                .unwrap();
        }
        let iters = 2000u32;
        let start = Instant::now();
        for i in 0..iters {
            state
                .append_run_event("run-0", "analysis_generated", json!({ "i": i }))
                .unwrap();
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("end-to-end sync append (200 runs live): {per:.0} ns/append");
        std::fs::remove_file(state.wal_path.as_ref()).ok();
    }
}

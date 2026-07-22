//! Durable event delivery built on the timeline WAL.
//!
//! The broadcast channel is only a bounded wake-up accelerator. SSE clients
//! and webhook workers recover missed notifications from the WAL by sequence,
//! so neither a slow network peer nor a full notification ring can backpressure
//! the timeline writer.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use hmac::{Hmac, Mac};
use reqwest::redirect;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use vidarax_core::ingest::validate_public_https_url;
use vidarax_core::timeline::{read_events_after, TimelineEvent};

use crate::handlers::load_run_snapshot;
use crate::ids::validate_run_id;
use crate::models::FieldError;
use crate::response::{
    bad_request_error, internal_error, not_found_error, ok, service_unavailable, validation_error,
};
use crate::state::AppState;

const TIMELINE_BROADCAST_CAP: usize = 1024;
const SSE_OUTPUT_CAP: usize = 16;
const REPLAY_BATCH: usize = 256;
const WEBHOOK_COMMAND_CAP: usize = 64;
const WEBHOOK_WAKE_CAP: usize = 1;
const MAX_WEBHOOKS: usize = 64;
const MAX_WEBHOOK_BODY_BYTES: usize = 1024 * 1024;
const MAX_EVENT_FILTERS: usize = 32;
const MAX_FILTER_LEN: usize = 96;
const MAX_DELIVERY_ATTEMPTS: u32 = 3;
const WEBHOOK_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(3);
const WEBHOOK_DELIVERY_LOG: &str = "webhook-delivery.wal";
const LAST_EVENT_ID: &str = "last-event-id";

#[derive(Default)]
pub(crate) struct DeliveryMetrics {
    sse_active: AtomicU64,
    sse_events: AtomicU64,
    sse_replayed: AtomicU64,
    sse_notification_lag: AtomicU64,
    sse_output_stalls: AtomicU64,
    webhook_configured: AtomicU64,
    webhook_attempts: AtomicU64,
    webhook_delivered: AtomicU64,
    webhook_retries: AtomicU64,
    webhook_dead_letters: AtomicU64,
    webhook_wake_coalesced: AtomicU64,
    webhook_notification_lag: AtomicU64,
}

impl DeliveryMetrics {
    pub(crate) fn render_prometheus(&self) -> String {
        format!(
            concat!(
                "vidarax_sse_subscribers_active {}\n",
                "vidarax_sse_events_total {}\n",
                "vidarax_sse_replayed_events_total {}\n",
                "vidarax_sse_notification_lag_total {}\n",
                "vidarax_sse_output_queue_stalls_total {}\n",
                "vidarax_webhooks_configured {}\n",
                "vidarax_webhook_attempts_total {}\n",
                "vidarax_webhook_delivered_total {}\n",
                "vidarax_webhook_retries_total {}\n",
                "vidarax_webhook_dead_letters_total {}\n",
                "vidarax_webhook_wake_coalesced_total {}\n",
                "vidarax_webhook_notification_lag_total {}\n"
            ),
            self.sse_active.load(Ordering::Relaxed),
            self.sse_events.load(Ordering::Relaxed),
            self.sse_replayed.load(Ordering::Relaxed),
            self.sse_notification_lag.load(Ordering::Relaxed),
            self.sse_output_stalls.load(Ordering::Relaxed),
            self.webhook_configured.load(Ordering::Relaxed),
            self.webhook_attempts.load(Ordering::Relaxed),
            self.webhook_delivered.load(Ordering::Relaxed),
            self.webhook_retries.load(Ordering::Relaxed),
            self.webhook_dead_letters.load(Ordering::Relaxed),
            self.webhook_wake_coalesced.load(Ordering::Relaxed),
            self.webhook_notification_lag.load(Ordering::Relaxed),
        )
    }
}

#[derive(Clone)]
pub(crate) struct DeliveryHub {
    events: broadcast::Sender<TimelineEvent>,
    commands: mpsc::Sender<WebhookCommand>,
    metrics: Arc<DeliveryMetrics>,
    webhooks_enabled: bool,
}

impl DeliveryHub {
    pub(crate) fn spawn(
        wal_path: PathBuf,
        existing_events: &[TimelineEvent],
        signing_secret: Option<String>,
    ) -> Self {
        let (events, event_rx) = broadcast::channel(TIMELINE_BROADCAST_CAP);
        let (commands, command_rx) = mpsc::channel(WEBHOOK_COMMAND_CAP);
        let metrics = Arc::new(DeliveryMetrics::default());
        let webhooks_enabled = signing_secret
            .as_ref()
            .is_some_and(|secret| secret.len() >= 32);
        let configs = restore_webhook_configs(existing_events);
        let delivery_log_path = delivery_log_path_for(&wal_path);
        let thread_metrics = Arc::clone(&metrics);
        let thread_secret = signing_secret.filter(|secret| secret.len() >= 32);
        std::thread::Builder::new()
            .name("vidarax-event-delivery".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("delivery runtime should build");
                runtime.block_on(run_coordinator(
                    wal_path,
                    delivery_log_path,
                    configs,
                    thread_secret,
                    event_rx,
                    command_rx,
                    thread_metrics,
                ));
            })
            .expect("delivery thread should spawn");
        Self {
            events,
            commands,
            metrics,
            webhooks_enabled,
        }
    }

    pub(crate) fn event_sender(&self) -> broadcast::Sender<TimelineEvent> {
        self.events.clone()
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<TimelineEvent> {
        self.events.subscribe()
    }

    pub(crate) fn metrics(&self) -> &Arc<DeliveryMetrics> {
        &self.metrics
    }

    async fn reserve(&self, config: WebhookConfig) -> Result<(), String> {
        if !self.webhooks_enabled {
            return Err(
                "webhook delivery requires VIDARAX_WEBHOOK_SECRET with at least 32 bytes"
                    .to_string(),
            );
        }
        let (reply, response) = oneshot::channel();
        self.commands
            .send(WebhookCommand::Reserve { config, reply })
            .await
            .map_err(|_| "webhook coordinator is closed".to_string())?;
        response
            .await
            .map_err(|_| "webhook coordinator dropped its reply".to_string())?
    }

    async fn activate(&self, webhook_id: String, registered_seq: u64) -> Result<(), String> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(WebhookCommand::Activate {
                webhook_id,
                registered_seq,
                reply,
            })
            .await
            .map_err(|_| "webhook coordinator is closed".to_string())?;
        response
            .await
            .map_err(|_| "webhook coordinator dropped its reply".to_string())?
    }

    async fn cancel_reservation(&self, webhook_id: String) {
        let _ = self
            .commands
            .send(WebhookCommand::Remove { webhook_id })
            .await;
    }

    async fn list(&self, run_id: String) -> Result<Vec<WebhookSummary>, String> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(WebhookCommand::List { run_id, reply })
            .await
            .map_err(|_| "webhook coordinator is closed".to_string())?;
        response
            .await
            .map_err(|_| "webhook coordinator dropped its reply".to_string())
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct EventStreamQuery {
    after: Option<u64>,
    kind: Option<String>,
}

pub(crate) async fn stream_events(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
    headers: HeaderMap,
    Query(query): Query<EventStreamQuery>,
) -> Response {
    if !validate_run_id(&run_id) {
        return bad_request_error(
            &state,
            "invalid event stream request",
            vec![FieldError {
                field: "run_id",
                message: "invalid run id".to_string(),
            }],
        )
        .into_response();
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }
    if query
        .kind
        .as_ref()
        .is_some_and(|kind| kind.is_empty() || kind.len() > MAX_FILTER_LEN)
    {
        return validation_error(
            &state,
            "invalid event stream request",
            vec![FieldError {
                field: "kind",
                message: format!("kind must contain 1..={MAX_FILTER_LEN} bytes"),
            }],
        )
        .into_response();
    }
    let header_cursor = match headers.get(LAST_EVENT_ID) {
        Some(raw) => match raw
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
        {
            Some(cursor) => Some(cursor),
            None => {
                return validation_error(
                    &state,
                    "invalid event stream cursor",
                    vec![FieldError {
                        field: "Last-Event-ID",
                        message: "must be an unsigned timeline sequence".to_string(),
                    }],
                )
                .into_response()
            }
        },
        None => None,
    };
    let cursor = header_cursor.or(query.after).unwrap_or(0);
    let mut notifications = state.subscribe_timeline_events();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(SSE_OUTPUT_CAP);
    let producer_state = state.clone();
    tokio::spawn(async move {
        let metrics = Arc::clone(producer_state.delivery_metrics());
        metrics.sse_active.fetch_add(1, Ordering::Relaxed);
        let _guard = ActiveSseGuard {
            metrics: Arc::clone(&metrics),
        };
        let mut cursor = cursor;
        loop {
            match replay_to_sse(
                &producer_state,
                &run_id,
                query.kind.as_deref(),
                &mut cursor,
                &tx,
                &metrics,
            )
            .await
            {
                Ok(true) => continue,
                Ok(false) => {}
                Err(()) => return,
            }

            let notification = tokio::select! {
                _ = tx.closed() => return,
                notification = notifications.recv() => notification,
            };
            match notification {
                Ok(event) => {
                    if event.seq <= cursor {
                        continue;
                    }
                    cursor = event.seq;
                    if event.run_id == run_id
                        && query
                            .kind
                            .as_ref()
                            .is_none_or(|wanted| event.kind == *wanted)
                        && send_sse_event(&tx, &event, &metrics).await.is_err()
                    {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    metrics
                        .sse_notification_lag
                        .fetch_add(skipped, Ordering::Relaxed);
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        )
        .into_response()
}

async fn replay_to_sse(
    state: &AppState,
    run_id: &str,
    kind: Option<&str>,
    cursor: &mut u64,
    output: &mpsc::Sender<Result<Event, Infallible>>,
    metrics: &DeliveryMetrics,
) -> Result<bool, ()> {
    let batch = state
        .read_timeline_after(*cursor, REPLAY_BATCH)
        .await
        .map_err(|err| {
            tracing::warn!(%err, run_id, "SSE WAL replay failed");
        })?;
    if batch.is_empty() {
        return Ok(false);
    }
    let full = batch.len() == REPLAY_BATCH;
    for event in batch {
        *cursor = event.seq;
        if event.run_id != run_id || kind.is_some_and(|wanted| event.kind != wanted) {
            continue;
        }
        send_sse_event(output, &event, metrics).await?;
        metrics.sse_replayed.fetch_add(1, Ordering::Relaxed);
    }
    Ok(full)
}

async fn send_sse_event(
    output: &mpsc::Sender<Result<Event, Infallible>>,
    event: &TimelineEvent,
    metrics: &DeliveryMetrics,
) -> Result<(), ()> {
    match output.try_send(Ok(timeline_sse_event(event))) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(returned)) => {
            metrics.sse_output_stalls.fetch_add(1, Ordering::Relaxed);
            output.send(returned).await.map_err(|_| ())?;
        }
        Err(mpsc::error::TrySendError::Closed(_)) => return Err(()),
    }
    metrics.sse_events.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

struct ActiveSseGuard {
    metrics: Arc<DeliveryMetrics>,
}

impl Drop for ActiveSseGuard {
    fn drop(&mut self) {
        self.metrics.sse_active.fetch_sub(1, Ordering::Relaxed);
    }
}

fn timeline_sse_event(event: &TimelineEvent) -> Event {
    Event::default()
        .id(event.seq.to_string())
        .event(event.kind.clone())
        .json_data(cloud_event(event))
        .expect("timeline event envelope is serializable")
}

fn cloud_event(event: &TimelineEvent) -> Value {
    json!({
        "specversion": "1.0",
        "id": format!("{}:{}", event.run_id, event.seq),
        "source": format!("/v1/runs/{}", event.run_id),
        "type": format!("dev.vidarax.timeline.{}", event.kind),
        "subject": event.stream_id,
        "datacontenttype": "application/json",
        "sequence": event.seq,
        "pts_ms": event.pts_ms,
        "data": serde_json::from_str::<Value>(&event.payload)
            .unwrap_or_else(|_| Value::String(event.payload.clone()))
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateWebhookRequest {
    url: String,
    #[serde(default)]
    event_kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WebhookConfig {
    webhook_id: String,
    run_id: String,
    url: String,
    event_kinds: Vec<String>,
    registered_seq: u64,
}

#[derive(Debug, Clone, Serialize)]
struct WebhookSummary {
    webhook_id: String,
    url: String,
    event_kinds: Vec<String>,
    registered_seq: u64,
    last_terminal_seq: u64,
    delivered: u64,
    dead_letters: u64,
    last_error: Option<String>,
    state: &'static str,
}

#[derive(Clone, Default)]
struct WorkerStatus {
    last_terminal_seq: u64,
    delivered: u64,
    dead_letters: u64,
    last_error: Option<String>,
}

pub(crate) async fn create_webhook(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
    headers: HeaderMap,
    Json(payload): Json<CreateWebhookRequest>,
) -> Response {
    if !validate_run_id(&run_id) {
        return bad_request_error(&state, "invalid webhook request", vec![]).into_response();
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }
    let url = payload.url.trim().to_string();
    if let Err(message) = validate_webhook_url(&url).await {
        return validation_error(
            &state,
            "invalid webhook request",
            vec![FieldError {
                field: "url",
                message,
            }],
        )
        .into_response();
    }
    let event_kinds = match normalize_event_filters(payload.event_kinds) {
        Ok(filters) => filters,
        Err(message) => {
            return validation_error(
                &state,
                "invalid webhook request",
                vec![FieldError {
                    field: "event_kinds",
                    message,
                }],
            )
            .into_response()
        }
    };
    let webhook_id = random_webhook_id();
    let config = WebhookConfig {
        webhook_id: webhook_id.clone(),
        run_id: run_id.clone(),
        url: url.clone(),
        event_kinds: event_kinds.clone(),
        registered_seq: 0,
    };
    if let Err(err) = state.delivery().reserve(config).await {
        return service_unavailable(&state, "webhooks_unavailable", err).into_response();
    }
    let event = match state
        .append_run_event_async(
            &run_id,
            "webhook_registered",
            json!({
                "webhook_id": webhook_id,
                "url": url,
                "event_kinds": event_kinds,
            }),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => {
            state.delivery().cancel_reservation(webhook_id).await;
            return internal_error(&state, format!("failed to persist webhook: {err}"))
                .into_response();
        }
    };
    if let Err(err) = state
        .delivery()
        .activate(webhook_id.clone(), event.seq)
        .await
    {
        return internal_error(&state, format!("failed to activate webhook: {err}"))
            .into_response();
    }
    (
        StatusCode::CREATED,
        Json(json!({
            "request_id": state.next_request_id(),
            "run_id": run_id,
            "webhook_id": webhook_id,
            "url": url,
            "event_kinds": event_kinds,
            "registered_seq": event.seq,
        })),
    )
        .into_response()
}

pub(crate) async fn list_webhooks(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }
    match state.delivery().list(run_id.clone()).await {
        Ok(webhooks) => ok(json!({
            "request_id": state.next_request_id(),
            "run_id": run_id,
            "webhooks": webhooks,
        }))
        .into_response(),
        Err(err) => internal_error(&state, err).into_response(),
    }
}

pub(crate) async fn delete_webhook(
    State(state): State<AppState>,
    AxumPath((run_id, webhook_id)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> Response {
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }
    let webhooks = match state.delivery().list(run_id.clone()).await {
        Ok(webhooks) => webhooks,
        Err(err) => return internal_error(&state, err).into_response(),
    };
    if !webhooks.iter().any(|hook| hook.webhook_id == webhook_id) {
        return not_found_error(
            &state,
            "webhook not found",
            vec![FieldError {
                field: "webhook_id",
                message: "no webhook with this id exists for the run".to_string(),
            }],
        )
        .into_response();
    }
    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "webhook_deleted",
            json!({ "webhook_id": webhook_id }),
        )
        .await
    {
        return internal_error(&state, format!("failed to persist webhook deletion: {err}"))
            .into_response();
    }
    state
        .delivery()
        .cancel_reservation(webhook_id.clone())
        .await;
    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "webhook_id": webhook_id,
        "deleted": true,
    }))
    .into_response()
}

enum WebhookCommand {
    Reserve {
        config: WebhookConfig,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Activate {
        webhook_id: String,
        registered_seq: u64,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Remove {
        webhook_id: String,
    },
    List {
        run_id: String,
        reply: oneshot::Sender<Vec<WebhookSummary>>,
    },
}

struct HookEntry {
    config: WebhookConfig,
    wake: Option<mpsc::Sender<()>>,
    stop: Option<oneshot::Sender<()>>,
    status: Arc<Mutex<WorkerStatus>>,
}

async fn run_coordinator(
    wal_path: PathBuf,
    delivery_log_path: PathBuf,
    configs: Vec<WebhookConfig>,
    secret: Option<String>,
    mut events: broadcast::Receiver<TimelineEvent>,
    mut commands: mpsc::Receiver<WebhookCommand>,
    metrics: Arc<DeliveryMetrics>,
) {
    let delivery_state = load_delivery_state(&delivery_log_path);
    let delivery_log = secret.as_ref().and_then(|_| {
        open_delivery_log(&delivery_log_path)
            .ok()
            .map(|file| Arc::new(Mutex::new(file)))
    });
    let mut hooks = HashMap::new();
    if let (Some(secret), Some(delivery_log)) = (secret.as_ref(), delivery_log.as_ref()) {
        for config in configs.into_iter().take(MAX_WEBHOOKS) {
            let restored = delivery_state
                .get(&config.webhook_id)
                .cloned()
                .unwrap_or_default();
            let entry = spawn_hook_worker(
                config.clone(),
                restored,
                wal_path.clone(),
                Arc::clone(delivery_log),
                Arc::<[u8]>::from(secret.as_bytes()),
                Arc::clone(&metrics),
            );
            hooks.insert(config.webhook_id.clone(), entry);
        }
    }
    metrics
        .webhook_configured
        .store(hooks.len() as u64, Ordering::Relaxed);

    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                match command {
                    WebhookCommand::Reserve { config, reply } => {
                        let result = if secret.is_none() {
                            Err("webhook delivery requires VIDARAX_WEBHOOK_SECRET with at least 32 bytes".to_string())
                        } else if delivery_log.is_none() {
                            Err("webhook delivery log is unavailable".to_string())
                        } else if hooks.len() >= MAX_WEBHOOKS {
                            Err(format!("webhook limit reached ({MAX_WEBHOOKS})"))
                        } else if hooks.contains_key(&config.webhook_id) {
                            Err("webhook id collision".to_string())
                        } else {
                            hooks.insert(config.webhook_id.clone(), HookEntry {
                                config,
                                wake: None,
                                stop: None,
                                status: Arc::new(Mutex::new(WorkerStatus::default())),
                            });
                            metrics.webhook_configured.store(hooks.len() as u64, Ordering::Relaxed);
                            Ok(())
                        };
                        let _ = reply.send(result);
                    }
                    WebhookCommand::Activate { webhook_id, registered_seq, reply } => {
                        let result = match hooks.get_mut(&webhook_id) {
                            Some(entry) if entry.wake.is_none() => {
                                entry.config.registered_seq = registered_seq;
                                let activated = spawn_hook_worker(
                                    entry.config.clone(),
                                    WorkerStatus { last_terminal_seq: registered_seq, ..WorkerStatus::default() },
                                    wal_path.clone(),
                                    Arc::clone(delivery_log.as_ref().expect("reservation required delivery log")),
                                    Arc::<[u8]>::from(secret.as_ref().expect("reservation required secret").as_bytes()),
                                    Arc::clone(&metrics),
                                );
                                *entry = activated;
                                Ok(())
                            }
                            Some(_) => Err("webhook is already active".to_string()),
                            None => Err("webhook reservation was not found".to_string()),
                        };
                        let _ = reply.send(result);
                    }
                    WebhookCommand::Remove { webhook_id } => {
                        if let Some(mut hook) = hooks.remove(&webhook_id) {
                            if let Some(stop) = hook.stop.take() {
                                let _ = stop.send(());
                            }
                        }
                        metrics.webhook_configured.store(hooks.len() as u64, Ordering::Relaxed);
                    }
                    WebhookCommand::List { run_id, reply } => {
                        let mut summaries = hooks.values()
                            .filter(|entry| entry.config.run_id == run_id)
                            .map(hook_summary)
                            .collect::<Vec<_>>();
                        summaries.sort_by(|left, right| left.webhook_id.cmp(&right.webhook_id));
                        let _ = reply.send(summaries);
                    }
                }
            }
            event = events.recv() => {
                match event {
                    Ok(event) => wake_run_hooks(&hooks, &event.run_id, &metrics),
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        metrics.webhook_notification_lag.fetch_add(skipped, Ordering::Relaxed);
                        wake_all_hooks(&hooks, &metrics);
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

fn hook_summary(entry: &HookEntry) -> WebhookSummary {
    let status = entry
        .status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    WebhookSummary {
        webhook_id: entry.config.webhook_id.clone(),
        url: entry.config.url.clone(),
        event_kinds: entry.config.event_kinds.clone(),
        registered_seq: entry.config.registered_seq,
        last_terminal_seq: status.last_terminal_seq,
        delivered: status.delivered,
        dead_letters: status.dead_letters,
        last_error: status.last_error.clone(),
        state: if entry.wake.is_some() {
            "active"
        } else {
            "pending"
        },
    }
}

fn wake_run_hooks(hooks: &HashMap<String, HookEntry>, run_id: &str, metrics: &DeliveryMetrics) {
    for hook in hooks.values().filter(|hook| hook.config.run_id == run_id) {
        wake_hook(hook, metrics);
    }
}

fn wake_all_hooks(hooks: &HashMap<String, HookEntry>, metrics: &DeliveryMetrics) {
    for hook in hooks.values() {
        wake_hook(hook, metrics);
    }
}

fn wake_hook(hook: &HookEntry, metrics: &DeliveryMetrics) {
    let Some(wake) = &hook.wake else { return };
    if matches!(wake.try_send(()), Err(mpsc::error::TrySendError::Full(_))) {
        metrics
            .webhook_wake_coalesced
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn spawn_hook_worker(
    config: WebhookConfig,
    restored: WorkerStatus,
    wal_path: PathBuf,
    delivery_log: Arc<Mutex<File>>,
    secret: Arc<[u8]>,
    metrics: Arc<DeliveryMetrics>,
) -> HookEntry {
    let (wake, mut wake_rx) = mpsc::channel(WEBHOOK_WAKE_CAP);
    let (stop, mut stop_rx) = oneshot::channel();
    let status = Arc::new(Mutex::new(restored));
    let worker_status = Arc::clone(&status);
    let worker_config = config.clone();
    tokio::spawn(async move {
        loop {
            let failed = if let Err(err) = replay_hook(
                &worker_config,
                &wal_path,
                &delivery_log,
                &secret,
                &worker_status,
                &metrics,
            )
            .await
            {
                let mut status = worker_status
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                status.last_error = Some(truncate_error(err));
                true
            } else {
                false
            };
            if failed {
                tokio::select! {
                    _ = &mut stop_rx => return,
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {},
                    wake = wake_rx.recv() => if wake.is_none() { return },
                }
            } else {
                tokio::select! {
                    _ = &mut stop_rx => return,
                    wake = wake_rx.recv() => if wake.is_none() { return },
                }
            }
        }
    });
    HookEntry {
        config,
        wake: Some(wake),
        stop: Some(stop),
        status,
    }
}

async fn replay_hook(
    config: &WebhookConfig,
    wal_path: &Path,
    delivery_log: &Arc<Mutex<File>>,
    secret: &[u8],
    status: &Arc<Mutex<WorkerStatus>>,
    metrics: &DeliveryMetrics,
) -> Result<(), String> {
    loop {
        let cursor = status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last_terminal_seq
            .max(config.registered_seq);
        let path = wal_path.to_path_buf();
        let batch =
            tokio::task::spawn_blocking(move || read_events_after(path, cursor, REPLAY_BATCH))
                .await
                .map_err(|err| format!("webhook WAL replay worker failed: {err}"))?
                .map_err(|err| format!("webhook WAL replay failed: {err}"))?;
        if batch.is_empty() {
            return Ok(());
        }
        let full = batch.len() == REPLAY_BATCH;
        let mut scanned_to = cursor;
        for event in batch {
            scanned_to = event.seq;
            if event.run_id != config.run_id || !webhook_matches(config, &event) {
                continue;
            }
            let envelope = serde_json::to_vec(&cloud_event(&event))
                .map_err(|err| format!("failed to serialize webhook event: {err}"))?;
            let event_id = format!("{}:{}", event.run_id, event.seq);
            let mut last_error = (envelope.len() > MAX_WEBHOOK_BODY_BYTES)
                .then(|| format!("webhook event body exceeds {MAX_WEBHOOK_BODY_BYTES} bytes"));
            let mut delivered = false;
            if last_error.is_none() {
                for attempt in 1..=MAX_DELIVERY_ATTEMPTS {
                    metrics.webhook_attempts.fetch_add(1, Ordering::Relaxed);
                    match deliver_webhook(&config.url, &event_id, &envelope, secret).await {
                        Ok(()) => {
                            delivered = true;
                            break;
                        }
                        Err(err) => {
                            last_error = Some(truncate_error(err));
                            if attempt < MAX_DELIVERY_ATTEMPTS {
                                metrics.webhook_retries.fetch_add(1, Ordering::Relaxed);
                                append_delivery_record(
                                    delivery_log,
                                    &DeliveryRecord {
                                        webhook_id: config.webhook_id.clone(),
                                        seq: event.seq,
                                        state: "retry".to_string(),
                                        error: last_error.clone(),
                                    },
                                )?;
                                tokio::time::sleep(retry_delay(attempt)).await;
                            }
                        }
                    }
                }
            }
            let terminal_state = if delivered {
                "delivered"
            } else {
                "dead_letter"
            };
            append_delivery_record(
                delivery_log,
                &DeliveryRecord {
                    webhook_id: config.webhook_id.clone(),
                    seq: event.seq,
                    state: terminal_state.to_string(),
                    error: last_error.clone(),
                },
            )?;
            let mut current = status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            current.last_terminal_seq = event.seq;
            current.last_error = if delivered { None } else { last_error };
            if delivered {
                current.delivered = current.delivered.saturating_add(1);
                metrics.webhook_delivered.fetch_add(1, Ordering::Relaxed);
            } else {
                current.dead_letters = current.dead_letters.saturating_add(1);
                metrics.webhook_dead_letters.fetch_add(1, Ordering::Relaxed);
            }
        }
        append_delivery_record(
            delivery_log,
            &DeliveryRecord {
                webhook_id: config.webhook_id.clone(),
                seq: scanned_to,
                state: "checkpoint".to_string(),
                error: None,
            },
        )?;
        status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last_terminal_seq = scanned_to;
        if !full {
            return Ok(());
        }
    }
}

fn webhook_matches(config: &WebhookConfig, event: &TimelineEvent) -> bool {
    if event.kind.starts_with("webhook_") {
        return false;
    }
    config.event_kinds.is_empty() || config.event_kinds.iter().any(|kind| kind == &event.kind)
}

async fn deliver_webhook(
    target: &str,
    event_id: &str,
    body: &[u8],
    secret: &[u8],
) -> Result<(), String> {
    let started = Instant::now();
    let parsed =
        reqwest::Url::parse(target).map_err(|err| format!("invalid webhook URL: {err}"))?;
    let validation_url = parsed.clone();
    let addresses = tokio::time::timeout(
        WEBHOOK_ATTEMPT_TIMEOUT,
        tokio::task::spawn_blocking(move || validate_public_https_url(&validation_url)),
    )
    .await
    .map_err(|_| "webhook DNS validation timed out".to_string())?
    .map_err(|err| format!("webhook DNS validation worker failed: {err}"))??;
    let remaining = WEBHOOK_ATTEMPT_TIMEOUT
        .checked_sub(started.elapsed())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| "webhook attempt timed out".to_string())?;
    let host = parsed
        .host_str()
        .ok_or_else(|| "webhook URL is missing a host".to_string())?;
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| "invalid webhook signing secret".to_string())?;
    mac.update(body);
    let signature = format!("v1={}", hex_bytes(&mac.finalize().into_bytes()));
    let client = reqwest::Client::builder()
        .redirect(redirect::Policy::none())
        .resolve_to_addrs(host, &addresses)
        .no_proxy()
        .connect_timeout(remaining)
        .timeout(remaining)
        .build()
        .map_err(|err| format!("failed to build webhook client: {err}"))?;
    let response = client
        .post(parsed)
        .header("content-type", "application/cloudevents+json")
        .header("x-vidarax-event-id", event_id)
        .header("x-vidarax-signature", signature)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|err| format!("webhook request failed: {err}"))?;
    if response.status().is_redirection() {
        return Err("webhook redirect rejected".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("webhook returned HTTP {}", response.status()));
    }
    Ok(())
}

fn retry_delay(attempt: u32) -> Duration {
    Duration::from_millis(match attempt {
        1 => 100,
        _ => 500,
    })
}

async fn validate_webhook_url(raw: &str) -> Result<(), String> {
    if raw.len() > 2048 {
        return Err("webhook URL exceeds 2048 bytes".to_string());
    }
    let url = reqwest::Url::parse(raw).map_err(|err| format!("invalid webhook URL: {err}"))?;
    tokio::time::timeout(
        WEBHOOK_ATTEMPT_TIMEOUT,
        tokio::task::spawn_blocking(move || validate_public_https_url(&url)),
    )
    .await
    .map_err(|_| "webhook DNS validation timed out".to_string())?
    .map_err(|err| format!("webhook DNS validation worker failed: {err}"))??;
    Ok(())
}

fn normalize_event_filters(filters: Vec<String>) -> Result<Vec<String>, String> {
    if filters.len() > MAX_EVENT_FILTERS {
        return Err(format!(
            "at most {MAX_EVENT_FILTERS} event kinds are allowed"
        ));
    }
    let mut seen = HashSet::with_capacity(filters.len());
    let mut normalized = Vec::with_capacity(filters.len());
    for filter in filters {
        let filter = filter.trim().to_string();
        if filter.is_empty() || filter.len() > MAX_FILTER_LEN {
            return Err(format!(
                "each event kind must contain 1..={MAX_FILTER_LEN} bytes"
            ));
        }
        if filter.starts_with("webhook_") {
            return Err("webhook delivery bookkeeping cannot be subscribed to".to_string());
        }
        if seen.insert(filter.clone()) {
            normalized.push(filter);
        }
    }
    normalized.sort();
    Ok(normalized)
}

fn restore_webhook_configs(events: &[TimelineEvent]) -> Vec<WebhookConfig> {
    let mut configs = HashMap::new();
    for event in events {
        let payload = serde_json::from_str::<Value>(&event.payload).unwrap_or(Value::Null);
        match event.kind.as_str() {
            "webhook_registered" => {
                let Some(webhook_id) = payload.get("webhook_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(url) = payload.get("url").and_then(Value::as_str) else {
                    continue;
                };
                let event_kinds = payload
                    .get("event_kinds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                configs.insert(
                    webhook_id.to_string(),
                    WebhookConfig {
                        webhook_id: webhook_id.to_string(),
                        run_id: event.run_id.clone(),
                        url: url.to_string(),
                        event_kinds,
                        registered_seq: event.seq,
                    },
                );
            }
            "webhook_deleted" => {
                if let Some(webhook_id) = payload.get("webhook_id").and_then(Value::as_str) {
                    configs.remove(webhook_id);
                }
            }
            _ => {}
        }
    }
    configs.into_values().collect()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct DeliveryRecord {
    webhook_id: String,
    seq: u64,
    state: String,
    error: Option<String>,
}

fn open_delivery_log(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn delivery_log_path_for(wal_path: &Path) -> PathBuf {
    if wal_path
        .file_name()
        .is_some_and(|name| name == "timeline.wal")
    {
        return wal_path.with_file_name(WEBHOOK_DELIVERY_LOG);
    }
    let filename = wal_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("timeline.wal");
    wal_path.with_file_name(format!("{filename}.webhook-delivery.wal"))
}

fn load_delivery_state(path: &Path) -> HashMap<String, WorkerStatus> {
    let Ok(file) = OpenOptions::new().read(true).open(path) else {
        return HashMap::new();
    };
    let mut states = HashMap::<String, WorkerStatus>::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(record) = serde_json::from_str::<DeliveryRecord>(&line) else {
            continue;
        };
        let state = states.entry(record.webhook_id).or_default();
        match record.state.as_str() {
            "delivered" => {
                state.last_terminal_seq = state.last_terminal_seq.max(record.seq);
                state.delivered = state.delivered.saturating_add(1);
            }
            "dead_letter" => {
                state.last_terminal_seq = state.last_terminal_seq.max(record.seq);
                state.dead_letters = state.dead_letters.saturating_add(1);
                state.last_error = record.error;
            }
            "checkpoint" => state.last_terminal_seq = state.last_terminal_seq.max(record.seq),
            _ => {}
        }
    }
    states
}

fn append_delivery_record(log: &Arc<Mutex<File>>, record: &DeliveryRecord) -> Result<(), String> {
    let encoded = serde_json::to_string(record)
        .map_err(|err| format!("failed to encode webhook delivery state: {err}"))?;
    let mut file = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    writeln!(file, "{encoded}")
        .map_err(|err| format!("failed to append webhook delivery state: {err}"))?;
    file.flush()
        .map_err(|err| format!("failed to flush webhook delivery state: {err}"))
}

fn random_webhook_id() -> String {
    let mut bytes = [0u8; 12];
    getrandom::fill(&mut bytes).expect("OS randomness should be available");
    format!("wh_{}", hex_bytes(&bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn truncate_error(error: String) -> String {
    error.chars().take(256).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("vidarax-delivery-{name}-{nanos}.wal"))
    }

    #[test]
    fn cloud_event_has_stable_identity_and_preserves_reference_payload() {
        let event = TimelineEvent {
            seq: 42,
            run_id: "run_abc".to_string(),
            stream_id: "camera-1".to_string(),
            pts_ms: 1234,
            kind: "keyframe_stored".to_string(),
            payload: r#"{"sha256":"abc","url":"/v1/runs/run_abc/keyframes/abc"}"#.to_string(),
        };
        let envelope = cloud_event(&event);
        assert_eq!(envelope["id"], "run_abc:42");
        assert_eq!(envelope["sequence"], 42);
        assert_eq!(envelope["data"]["sha256"], "abc");
        assert!(!envelope.to_string().contains("base64"));
    }

    #[test]
    fn webhook_filters_are_bounded_and_exclude_bookkeeping() {
        assert!(normalize_event_filters(vec!["webhook_delivered".to_string()]).is_err());
        assert!(normalize_event_filters(vec!["x".repeat(MAX_FILTER_LEN + 1)]).is_err());
        assert_eq!(
            normalize_event_filters(vec!["vlm".into(), "vlm".into(), "gate".into()]).unwrap(),
            vec!["gate".to_string(), "vlm".to_string()]
        );
    }

    #[test]
    fn restores_registered_webhooks_and_applies_deletions() {
        let events = vec![
            TimelineEvent { seq: 1, run_id: "run-a".into(), stream_id: "s".into(), pts_ms: 0, kind: "webhook_registered".into(), payload: r#"{"webhook_id":"wh_a","url":"https://example.com/hook","event_kinds":["vlm"]}"#.into() },
            TimelineEvent { seq: 2, run_id: "run-a".into(), stream_id: "s".into(), pts_ms: 0, kind: "webhook_deleted".into(), payload: r#"{"webhook_id":"wh_a"}"#.into() },
            TimelineEvent { seq: 3, run_id: "run-b".into(), stream_id: "s".into(), pts_ms: 0, kind: "webhook_registered".into(), payload: r#"{"webhook_id":"wh_b","url":"https://example.com/hook","event_kinds":[]}"#.into() },
        ];
        let restored = restore_webhook_configs(&events);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].webhook_id, "wh_b");
        assert_eq!(restored[0].registered_seq, 3);
    }

    #[test]
    fn hmac_signature_is_deterministic() {
        let mut first =
            Hmac::<Sha256>::new_from_slice(b"01234567890123456789012345678901").unwrap();
        first.update(b"payload");
        let mut second =
            Hmac::<Sha256>::new_from_slice(b"01234567890123456789012345678901").unwrap();
        second.update(b"payload");
        assert_eq!(
            first.finalize().into_bytes(),
            second.finalize().into_bytes()
        );
    }

    #[test]
    fn retry_records_do_not_advance_the_durable_cursor() {
        let path = test_path("retry-state");
        let log = Arc::new(Mutex::new(open_delivery_log(&path).unwrap()));
        append_delivery_record(
            &log,
            &DeliveryRecord {
                webhook_id: "wh_a".into(),
                seq: 9,
                state: "retry".into(),
                error: Some("temporary".into()),
            },
        )
        .unwrap();
        let states = load_delivery_state(&path);
        assert_eq!(states["wh_a"].last_terminal_seq, 0);

        append_delivery_record(
            &log,
            &DeliveryRecord {
                webhook_id: "wh_a".into(),
                seq: 9,
                state: "delivered".into(),
                error: None,
            },
        )
        .unwrap();
        drop(log);
        let states = load_delivery_state(&path);
        assert_eq!(states["wh_a"].last_terminal_seq, 9);
        assert_eq!(states["wh_a"].delivered, 1);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sse_replay_is_strictly_after_cursor_and_filtered_by_run() {
        let path = test_path("sse-replay");
        let state = AppState::with_wal_for_tests(path.clone());
        state
            .append_run_event("run-a", "run_created", json!({"principal_key":"public"}))
            .unwrap();
        state.append_run_event("run-b", "other", json!({})).unwrap();
        state
            .append_run_event("run-a", "gate", json!({"value":1}))
            .unwrap();
        state
            .append_run_event("run-a", "vlm", json!({"value":2}))
            .unwrap();

        let metrics = DeliveryMetrics::default();
        let (tx, mut rx) = mpsc::channel(8);
        let mut cursor = 1;
        assert!(
            !replay_to_sse(&state, "run-a", None, &mut cursor, &tx, &metrics)
                .await
                .unwrap()
        );
        assert_eq!(cursor, 4);
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
        assert!(rx.try_recv().is_err());
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn router_exposes_sse_and_rejects_invalid_last_event_id() {
        let path = test_path("sse-route");
        let state = AppState::with_wal_for_tests(path.clone());
        let run_id = "run-0000000000000001";
        state
            .append_run_event(run_id, "run_created", json!({"principal_key":"public"}))
            .unwrap();
        let app = crate::app_router(state.clone());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/runs/{run_id}/events/stream?after=1"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        drop(response);

        let invalid = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/runs/{run_id}/events/stream"))
                    .header(LAST_EVENT_ID, "not-a-sequence")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn webhook_route_rejects_private_targets_before_configuration() {
        let path = test_path("webhook-ssrf");
        let state = AppState::with_wal_for_tests(path.clone());
        let run_id = "run-0000000000000002";
        state
            .append_run_event(run_id, "run_created", json!({"principal_key":"public"}))
            .unwrap();
        let response = crate::app_router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/runs/{run_id}/webhooks"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"url":"https://127.0.0.1/hook"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        drop(state);
        let _ = std::fs::remove_file(path);
    }
}

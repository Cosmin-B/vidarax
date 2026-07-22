//! WHIP (WebRTC-HTTP Ingestion Protocol) Axum handlers.
//!
//! Implements RFC 9725 signalling:
//! - `POST /v1/stream/whip` — SDP offer → answer exchange, returns 201 + Location
//! - `PATCH /v1/stream/whip/{sess_id}` — trickle ICE candidate
//! - `DELETE /v1/stream/whip/{sess_id}` — terminate session
//!
//! The endpoint accepts `Content-Type: application/sdp` for the offer and
//! returns `Content-Type: application/sdp` for the answer per RFC 9725.
//!
//! # Frame pipeline
//!
//! On session creation a bounded `kanal` channel is created.  The ingest task
//! (`session.run(tx, metrics)`) forwards H.264 NAL units through the channel.  Three
//! worker pools form the real-time pipeline:
//!
//! ```text
//! session.run(frame_tx)
//!   ↓ kanal::Receiver<RtpFrame>  (128)
//! spawn_decode_workers     — H.264 → YUV → FrameSignal + JPEG
//!   ↓ kanal::Sender<StreamFrame> (64)
//! spawn_analysis_workers   — gate engine, loop detection
//!   ↓ kanal::Sender<KeyframeWork> (32)
//! spawn_vlm_workers        — VLM inference → EventSink
//! ```

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use vidarax_core::provider::{
    InferenceObserver, InferenceProvider, InferenceRequest, InferenceResult, ProviderError,
    ProviderKind, TokenUsage,
};
use vidarax_core::webrtc::clip::ClipConfig as CoreClipConfig;
use vidarax_core::webrtc::resources::MediaSessionResources;
use vidarax_core::webrtc::runtime::SessionCommand;
use vidarax_core::webrtc::session::{
    PeerConnectionState, WebRtcSession, WebRtcSetupError, RTP_FRAME_QUEUE_CAPACITY,
};
use vidarax_core::webrtc::workers::{
    spawn_pipeline, EventSink, PipelineWiring, WorkerPoolConfig, STREAM_FRAME_QUEUE_CAPACITY,
    VLM_WORK_QUEUE_CAPACITY,
};

use crate::models::AttachStreamRequest;
use crate::state::AppState;
use crate::wal_sink::WalEventSink;

const ATTACH_CONFIG_HEADER: &str = "x-attach-config";
const ATTACH_CONFIG_HEADER_MAX_ENCODED_LEN: usize = 8 * 1024;
const RUN_ID_HEADER: &str = "x-vidarax-run-id";
const WHIP_RECLAIM_INITIAL_BACKOFF: Duration = Duration::from_millis(50);
const WHIP_RECLAIM_MAX_BACKOFF: Duration = Duration::from_secs(1);
const WHIP_CREATE_TOMBSTONE_INLINE_ATTEMPTS: usize = 3;

/// Used when no inference endpoints are configured so the pipeline can still
/// run without VLM inference (frames are decoded and gated, but the VLM step
/// is a no-op).
struct NullInferenceProvider;

impl InferenceProvider for NullInferenceProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vllm
    }

    fn infer(&self, _request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        Ok(InferenceResult {
            provider: ProviderKind::Vllm,
            model: std::sync::Arc::from("null"),
            output_text: "(no inference provider configured)".to_string(),
            fallback_used: false,
            finish_reason: Some("stop".to_string()),
            inference_latency_ms: 0,
            usage: TokenUsage::default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Session ID generation
// ---------------------------------------------------------------------------

/// Generate a random 16-hex-char session ID using the OS RNG.
///
/// Falls back to a timestamp-based ID if the OS RNG is unavailable.
fn new_session_id() -> String {
    let mut bytes = [0u8; 8];
    if getrandom::fill(&mut bytes).is_ok() {
        let mut id = String::with_capacity(5 + 16);
        id.push_str("sess-");
        for b in &bytes {
            id.push(hex_char(b >> 4));
            id.push(hex_char(b & 0x0f));
        }
        return id;
    }
    // Fallback: timestamp + counter avoids collisions in the extremely
    // unlikely event the OS RNG is temporarily unavailable.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    format!("sess-{ts:016x}")
}

#[inline]
fn hex_char(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        _ => (b'a' + (v - 10)) as char,
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Map a WebRtcSession setup failure to a WHIP HTTP response.
/// Unserveable-video offers are client errors (415); negotiation failures are 500.
fn whip_setup_error_response(err: &WebRtcSetupError) -> (StatusCode, &'static str) {
    match err {
        WebRtcSetupError::UnsupportedVideo(_) => (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "offer video cannot be served",
        ),
        WebRtcSetupError::Negotiation(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "WebRTC negotiation failed",
        ),
    }
}

/// `POST /v1/stream/whip`
///
/// Accepts a WebRTC SDP offer (`Content-Type: application/sdp`), negotiates
/// the answer, stores the session, and starts the media ingestion task.
///
/// Response:
/// - `201 Created` with `Content-Type: application/sdp` and SDP answer body
/// - `Location: /v1/stream/whip/{sess_id}` header
///
/// Errors:
/// - `400` — empty or non-UTF-8 SDP offer body
/// - `415` — offer video cannot be served (no live-serveable codec, or multiple video m-sections)
/// - `500` — malformed SDP, rustrtc, or ICE negotiation failure
#[tracing::instrument(name = "whip.offer", skip_all)]
pub async fn whip_offer(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Parse the SDP offer from the raw body.
    let offer_sdp = match std::str::from_utf8(&body) {
        Ok(s) if !s.trim().is_empty() => s,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "expected non-empty UTF-8 SDP offer body",
            )
                .into_response();
        }
    };

    // Log the content-type for debugging; WHIP clients should send
    // "application/sdp" but we accept any body since we only use raw text.
    if let Some(ct) = headers.get(header::CONTENT_TYPE) {
        tracing::debug!("WHIP offer content-type: {:?}", ct);
    }

    let attach_config = match parse_attach_config_header(&headers) {
        Ok(config) => config,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };

    // Create the rustrtc PeerConnection and negotiate the SDP answer.
    let generation = state.next_pipeline_generation();
    let (mut session, answer_sdp, commands) = match WebRtcSession::new_with_generation(
        offer_sdp,
        state.webrtc_config(),
        generation,
    )
    .await
    {
        Ok(parts) => parts,
        Err(e) => {
            tracing::warn!("WHIP offer negotiation failed: {e}");
            let (status, body) = whip_setup_error_response(&e);
            return (status, body).into_response();
        }
    };

    let clip_config = match apply_attach_config(&mut session, attach_config) {
        Ok(config) => config,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };

    start_whip_session(
        state,
        headers,
        Arc::new(session),
        answer_sdp,
        clip_config,
        commands,
    )
    .await
}

async fn start_whip_session(
    state: AppState,
    headers: HeaderMap,
    session: Arc<WebRtcSession>,
    answer_sdp: String,
    clip_config: Option<CoreClipConfig>,
    commands: tokio::sync::mpsc::Receiver<SessionCommand>,
) -> Response {
    let sess_id = new_session_id();
    let principal = state.security_policy().principal_key_from_headers(&headers);

    // Create a session-scoped span so all worker-thread log events are
    // attributed to this WebRTC session.
    let session_span = tracing::info_span!("whip_session", sess_id = %sess_id);

    // Create a run ID for VLM events.
    let run_id: Arc<str> = Arc::from(state.next_run_id().as_str());
    let session_id_arc: Arc<str> = Arc::from(sess_id.as_str());

    let transaction = tokio::spawn(start_whip_session_transaction(
        state,
        sess_id,
        principal,
        session_span,
        run_id,
        session_id_arc,
        session,
        answer_sdp,
        clip_config,
        commands,
    ));

    // The transaction is detached deliberately: WAL append plus session insert
    // must not be cancelled with the HTTP request. If the client disappears
    // after the session becomes visible, the peer-state reclaimer bounds it.
    match transaction.await {
        Ok(Ok(started)) => started.into_response(),
        Ok(Err(err)) => err.into_response(),
        Err(err) => {
            tracing::error!("WHIP creation transaction join failure: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to persist WHIP session",
            )
                .into_response()
        }
    }
}

struct WhipSessionStarted {
    sess_id: String,
    run_id: Arc<str>,
    answer_sdp: String,
}

impl WhipSessionStarted {
    fn into_response(self) -> Response {
        let location = format!("/v1/stream/whip/{}", self.sess_id);
        tracing::info!(
            "WHIP session created sess_id={} answer_candidates={}",
            self.sess_id,
            self.answer_sdp.matches("a=candidate:").count()
        );

        Response::builder()
            .status(StatusCode::CREATED)
            .header(header::CONTENT_TYPE, "application/sdp")
            .header(
                header::LOCATION,
                HeaderValue::from_str(&location).unwrap_or_else(|_| HeaderValue::from_static("/")),
            )
            .header(
                RUN_ID_HEADER,
                HeaderValue::from_str(self.run_id.as_ref())
                    .unwrap_or_else(|_| HeaderValue::from_static("")),
            )
            .body(axum::body::Body::from(self.answer_sdp))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }
}

struct WhipSessionStartError {
    status: StatusCode,
    message: &'static str,
}

impl WhipSessionStartError {
    fn into_response(self) -> Response {
        (self.status, self.message).into_response()
    }
}

// Session startup passes distinct transaction handles and config.
#[allow(clippy::too_many_arguments)]
async fn start_whip_session_transaction(
    state: AppState,
    sess_id: String,
    principal: String,
    session_span: tracing::Span,
    run_id: Arc<str>,
    session_id_arc: Arc<str>,
    session: Arc<WebRtcSession>,
    answer_sdp: String,
    clip_config: Option<CoreClipConfig>,
    commands: tokio::sync::mpsc::Receiver<SessionCommand>,
) -> Result<WhipSessionStarted, WhipSessionStartError> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let _slot = match state.try_reserve_stream_slot(&principal, now_ms) {
        Some(slot) => slot,
        None => {
            // Mirrors the normal run creation path: check before any persistent
            // run_created event so a rejected WHIP offer leaves no active run.
            session.close();
            return Err(WhipSessionStartError {
                status: StatusCode::CONFLICT,
                message: "active stream limit exceeded",
            });
        }
    };

    let mut admission_pool_config = WorkerPoolConfig::from(state.webrtc_config());
    admission_pool_config.max_output_tokens_per_second = session.max_output_tokens_per_second;
    admission_pool_config.crop = session.crop;
    admission_pool_config.restricted_zone = session.restricted_zone.clone();
    let media_resources = MediaSessionResources::for_pipeline(
        &admission_pool_config,
        session.codec,
        clip_config.is_some(),
    );
    let media_reservation = match state.try_reserve_media_resources(media_resources) {
        Some(reservation) => reservation,
        None => {
            tracing::warn!(
                session_id = %sess_id,
                generation = session.generation().get(),
                requested_bytes = media_resources.reserved_bytes,
                requested_worker_threads = media_resources.worker_threads,
                "WHIP media capacity reservation rejected"
            );
            session.close();
            return Err(WhipSessionStartError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "media process capacity exhausted",
            });
        }
    };

    // Register the run before exposing the live session. Reclaim and active-run
    // accounting rely on every visible WHIP session having a durable run_created.
    if let Err(err) = state.append_run_event(
        &run_id,
        "run_created",
        serde_json::json!({
            "principal_key": principal,
            "session_id": sess_id,
            "source": "whip",
        }),
    ) {
        tracing::error!("WHIP run_created append failed for {sess_id}: {err}");
        session.close();
        return Err(WhipSessionStartError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "failed to persist WHIP session",
        });
    }

    // The guard can overlap briefly with the committed run; that only rejects
    // a racing creator early, never admits one beyond the cap.
    // Store the session bound to the requesting principal.
    if !state.insert_session_with_media_reservation(
        sess_id.clone(),
        principal.clone(),
        Arc::clone(&run_id),
        Arc::clone(&session),
        media_reservation,
    ) {
        // Collision or global session limit reached.
        tracing::error!("WHIP session insert failed for {sess_id} (collision or limit)");
        tombstone_created_whip_run_with_request_bound(
            &state,
            &run_id,
            &sess_id,
            "session_insert_failed",
        )
        .await;
        session.close();
        return Err(WhipSessionStartError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "session limit reached or ID collision",
        });
    }

    // From here on the visible session has a reclaimer. There is no await
    // between insert completion and watcher spawn, so caller cancellation cannot
    // expose an ownerless active run.
    spawn_session_reclaimer(
        state.clone(),
        sess_id.clone(),
        Arc::clone(&run_id),
        session.subscribe_peer_state(),
    );

    // Record the new session in pipeline metrics.
    state.pipeline_metrics().inc_sessions_created();

    // ── Channel topology ───────────────────────────────────────────────────
    // RTP frames → decoded stream frames → keyframe work.
    let (frame_tx, frame_rx) =
        kanal::bounded::<vidarax_core::webrtc::session::RtpFrame>(RTP_FRAME_QUEUE_CAPACITY);
    let (stream_tx, stream_rx) =
        kanal::bounded::<vidarax_core::webrtc::workers::StreamFrame>(STREAM_FRAME_QUEUE_CAPACITY);
    let (vlm_tx, vlm_rx) =
        kanal::bounded::<vidarax_core::webrtc::workers::KeyframeWork>(VLM_WORK_QUEUE_CAPACITY);
    let metrics_arc = Arc::clone(state.pipeline_metrics_arc());

    // ── Media ingestion task ───────────────────────────────────────────────
    let run_future = session.run(frame_tx, Arc::clone(&metrics_arc));
    tokio::spawn(run_future);

    // ── EventSink selection ────────────────────────────────────────────────
    // WAL events and raw keyframe blobs stay local; descriptions may be mirrored.
    let event_sink: Arc<dyn EventSink> = WalEventSink::arc(state.clone());

    // ── InferenceProvider selection ────────────────────────────────────────
    // Use the HTTP router when provider endpoints are configured.
    //
    // Tiering stays local-only (single model, no escalation) unless an
    // operator opts in via VIDARAX_WEBRTC_SECOND_PASS_MODEL. That env var is
    // only read once, at process startup, by ServerConfig::from_env and then
    // resolved into a TieredVlmConfig on WebRtcConfig (see
    // crate::config::build_webrtc_vlm_config and crate::build_webrtc_config).
    // Every session below clones that already-resolved value instead of
    // re-reading the environment itself, so a ServerConfig built
    // programmatically (not from env) and passed to crate::run is what
    // actually governs a session's tiering, not whatever the process
    // environment happens to hold when this particular session starts.
    //
    // state.provider() below is built once at process startup by
    // vidarax_core::backends::build_provider_with_model_routing, not the
    // plain fallback chain: it routes a request by its exact `model` field
    // to whichever backend was configured to serve that id, in addition to
    // keeping the ordinary priority fallback for everything else. So
    // run_tiered() swapping `request.model` to the second-pass model id
    // genuinely reaches a different backend WHEN some backend is configured
    // to serve that id (e.g. a `[[backends]] type = "gemini"` entry whose
    // `model` is set to exactly VIDARAX_WEBRTC_SECOND_PASS_MODEL). If no
    // backend is configured for that model id, the request still lands on
    // the fallback chain's primary, which does not serve it; the call fails
    // there and run_tiered() falls back to the first-pass result. So tiering
    // is a safe no-op, never a crash, until a matching backend is configured;
    // see the commented-out gemini block in vidarax.toml for how to opt in.
    let webrtc_config_for_workers = state.webrtc_config().clone();
    let vlm_config = webrtc_config_for_workers.vlm_tiering.clone();
    let max_output_tokens_per_second = session.max_output_tokens_per_second;
    // The session's crop starts from the server default and may have been
    // overridden by the attach request; it wins over the config default.
    let session_crop = session.crop;
    let restricted_zone_for_workers = session.restricted_zone.clone();

    // Capture everything needed by the spawn_blocking closure.
    let run_id_for_workers = Arc::clone(&run_id);
    let session_id_for_workers = Arc::clone(&session_id_arc);
    let event_sink_for_workers = Arc::clone(&event_sink);
    let session_span_for_workers = session_span.clone();
    let metrics_for_workers = Arc::clone(&metrics_arc);
    let metrics_for_supervisor = Arc::clone(&metrics_arc);
    let vlm_config_for_workers = vlm_config.clone();
    let novelty_for_workers = state.novelty_config().clone();
    // Same InferenceMetrics instance /metrics reads from, handed to the
    // worker pipeline as an observer so each tiered VLM pass (keyframe or
    // clip mode) is recorded under the provider that actually served it.
    let observer_for_workers: Option<Arc<dyn InferenceObserver>> =
        Some(Arc::clone(state.inference_metrics_arc()) as Arc<dyn InferenceObserver>);
    let provider = state.admitted_provider(&principal);
    let clip_config_for_workers = clip_config;
    let initial_prompt = session.initial_prompt();
    let initial_guided_json = session.initial_guided_json();
    let generation = session.generation();
    let stopping = session.stopping_flag();
    let state_for_supervisor = state.clone();
    let sess_id_for_permit = sess_id.clone();
    let run_id_for_permit = Arc::clone(&run_id);

    // Worker threads are long-running OS threads (not tokio tasks).
    // Use spawn_blocking as a bridge from async context to the thread spawns;
    // the actual worker threads are spawned inside and run independently.
    //
    // A copy of the session id for the failure log inside the detached task;
    // the wiring below consumes the shared session-id handle.
    let session_id_for_log = sess_id.clone();
    tokio::task::spawn_blocking(move || {
        let vlm_provider: Arc<dyn InferenceProvider + Send + Sync> = match provider {
            Some(p) => p,
            None => Arc::new(NullInferenceProvider),
        };

        // Pool topology and tunables come from the session's WebRtcConfig. The
        // token-rate cap uses the session value, which honours a PATCH that
        // landed before the worker threads started.
        let mut pool_config = WorkerPoolConfig::from(&webrtc_config_for_workers);
        pool_config.max_output_tokens_per_second = max_output_tokens_per_second;
        pool_config.crop = session_crop;
        pool_config.restricted_zone = restricted_zone_for_workers;

        match spawn_pipeline(
            &pool_config,
            PipelineWiring {
                rtp_rx: frame_rx,
                stream_tx,
                stream_rx,
                vlm_tx,
                vlm_rx,
                event_sink: event_sink_for_workers,
                provider: Arc::new(vlm_provider),
                run_id: run_id_for_workers,
                session_id: session_id_for_workers,
                initial_prompt,
                initial_guided_json,
                generation,
                commands,
                stopping,
                vlm_config: vlm_config_for_workers,
                novelty: novelty_for_workers,
                clip_config: clip_config_for_workers,
                codec: session.codec,
                metrics: metrics_for_workers,
                session_span: session_span_for_workers,
                observer: observer_for_workers,
            },
        ) {
            Ok(runtime) => {
                let generation = runtime.generation().get();
                metrics_for_supervisor.pipeline_generation_started();
                tracing::info!(
                    session_id = %session_id_for_log,
                    generation,
                    workers = runtime.worker_count(),
                    "WHIP media pipeline generation started"
                );
                let outcome =
                    runtime.supervise(state_for_supervisor.media_join_deadline(), |fault| {
                        tracing::error!(
                            session_id = %session_id_for_log,
                            generation,
                            stage = fault.stage.as_str(),
                            reason = ?fault.reason,
                            "WHIP media pipeline generation faulted"
                        );
                        session.close();
                    });
                metrics_for_supervisor.pipeline_generation_stopped(outcome);
                match outcome {
                    vidarax_core::webrtc::runtime::PipelineShutdown::JoinDeadline {
                        detached,
                        ..
                    } => {
                        // Detached threads still hold their share of the media
                        // budget. Keep the reservation so new sessions are not
                        // admitted against memory that is still in use.
                        tracing::error!(
                            session_id = %session_id_for_log,
                            generation,
                            detached,
                            "forced shutdown left threads detached; media reservation kept"
                        );
                    }
                    _ => {
                        state_for_supervisor
                            .release_media_generation(&sess_id_for_permit, &run_id_for_permit);
                    }
                }
                tracing::info!(
                    session_id = %session_id_for_log,
                    generation,
                    outcome = ?outcome,
                    "WHIP media pipeline generation stopped"
                );
            }
            Err(err) => {
                metrics_for_supervisor.record_pipeline_start_failure(
                    err.fault,
                    err.join_deadline,
                    err.detached,
                );
                // A worker thread failed to spawn, e.g. OS thread-resource
                // exhaustion (EAGAIN). Close the peer so the reclaimer removes
                // the visible session and tombstones its run.
                tracing::error!(
                    session_id = %session_id_for_log,
                    stage = err.fault.stage.as_str(),
                    error = %err,
                    "WHIP media pipeline failed to start; closing session"
                );
                session.close();
                if err.detached == 0 {
                    state_for_supervisor
                        .release_media_generation(&sess_id_for_permit, &run_id_for_permit);
                } else {
                    // Startup rollback left threads running past its deadline.
                    // They still hold their share of the media budget, so the
                    // reservation stays, same as a forced shutdown.
                    tracing::error!(
                        session_id = %session_id_for_log,
                        detached = err.detached,
                        "startup abort left threads detached; media reservation kept"
                    );
                }
            }
        }
    });

    Ok(WhipSessionStarted {
        sess_id,
        run_id,
        answer_sdp,
    })
}

fn parse_attach_config_header(headers: &HeaderMap) -> Result<Option<AttachStreamRequest>, String> {
    let Some(value) = headers.get(ATTACH_CONFIG_HEADER) else {
        return Ok(None);
    };
    if value.as_bytes().len() > ATTACH_CONFIG_HEADER_MAX_ENCODED_LEN {
        return Err(format!(
            "{ATTACH_CONFIG_HEADER} is too large; use PATCH /v1/stream/whip/{{sess_id}}/prompt for larger prompts"
        ));
    }
    let text = value
        .to_str()
        .map_err(|_| format!("{ATTACH_CONFIG_HEADER} must be base64url-encoded JSON"))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(text.as_bytes())
        .map_err(|err| format!("invalid {ATTACH_CONFIG_HEADER} base64url: {err}"))?;
    let json = std::str::from_utf8(&bytes)
        .map_err(|err| format!("invalid {ATTACH_CONFIG_HEADER} UTF-8 JSON: {err}"))?;
    serde_json::from_str::<AttachStreamRequest>(json)
        .map(Some)
        .map_err(|err| format!("invalid {ATTACH_CONFIG_HEADER}: {err}"))
}

fn apply_attach_config(
    session: &mut WebRtcSession,
    config: Option<AttachStreamRequest>,
) -> Result<Option<CoreClipConfig>, String> {
    let Some(config) = config else {
        return Ok(None);
    };

    if config.restricted_zone.is_some() && config.clip_mode.is_some() {
        return Err("restricted_zone and clip_mode cannot be enabled together".to_string());
    }

    if let Some(zone) = config.restricted_zone {
        zone.validate().map_err(str::to_string)?;
        let zone_crop = zone.crop();
        if let Some(crop) = config.crop {
            crop.validate().map_err(|e| e.to_string())?;
            if crop != zone_crop {
                return Err(
                    "crop must exactly match restricted_zone.region when both are set".to_string(),
                );
            }
        }
        session.crop = Some(zone_crop);
        session.restricted_zone = Some(Arc::new(zone));
    } else if let Some(crop) = config.crop {
        crop.validate().map_err(|e| e.to_string())?;
        session.crop = Some(crop);
    }

    if let Some(prompt) = config.prompt {
        session.set_initial_prompt(prompt);
    }
    if let Some(max) = config.max_output_tokens_per_second {
        session.max_output_tokens_per_second = max;
    }
    if let Some(clip) = config.clip_mode {
        let clip = clip.into_core();
        clip.validate()?;
        return Ok(Some(clip));
    }

    Ok(None)
}

/// `PATCH /v1/stream/whip/{sess_id}`
///
/// Accepts a trickle ICE candidate fragment and forwards it to the
/// appropriate session.
///
/// Body: ICE candidate line, e.g. `candidate:1 1 udp ... typ host`
///
/// Response:
/// - `204 No Content` — candidate accepted (or no-op for end-of-candidates)
/// - `404 Not Found` — unknown session
#[tracing::instrument(name = "whip.ice", skip_all, fields(sess_id))]
pub async fn whip_ice(
    State(state): State<AppState>,
    Path(sess_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    let Some((owner_principal, _run_id, session)) = state.get_session(&sess_id) else {
        tracing::debug!("WHIP ICE unknown session {sess_id}");
        return StatusCode::NOT_FOUND;
    };

    // Verify the caller owns this session.
    let caller = state.security_policy().principal_key_from_headers(&headers);
    if caller != owner_principal {
        tracing::warn!("WHIP ICE sess={sess_id} principal mismatch");
        return StatusCode::FORBIDDEN;
    }

    let candidate_str = match std::str::from_utf8(&body) {
        Ok(s) => s.trim(),
        Err(_) => return StatusCode::BAD_REQUEST,
    };

    // Empty body = end-of-candidates signal; ignore silently.
    if candidate_str.is_empty() {
        return StatusCode::NO_CONTENT;
    }

    // Strip optional "a=" prefix per trickle ICE fragment format.
    let candidate_line = candidate_str.strip_prefix("a=").unwrap_or(candidate_str);

    if let Err(e) = session.add_remote_candidate(candidate_line) {
        tracing::warn!("WHIP ICE sess={sess_id} error: {e}");
        // Return 204 anyway — the connection may still work with other candidates.
    }

    StatusCode::NO_CONTENT
}

/// `DELETE /v1/stream/whip/{sess_id}`
///
/// Terminates the WebRTC session.  Dropping the session handle triggers
/// rustrtc's cleanup sequence.
///
/// Response:
/// - `200 OK` — session terminated
/// - `404 Not Found` — unknown session
/// - `500 Internal Server Error` — persistent cleanup failed; session remains retryable
#[tracing::instrument(name = "whip.terminate", skip_all, fields(sess_id))]
pub async fn whip_terminate(
    State(state): State<AppState>,
    Path(sess_id): Path<String>,
    headers: HeaderMap,
) -> StatusCode {
    // Verify ownership before cleanup. A peer-state watcher may have already
    // reclaimed the live session, so consult the reclaimed-session record too.
    let live_session = state
        .get_session(&sess_id)
        .map(|(principal, run_id, _)| (principal, run_id));
    let reclaimed_session = if live_session.is_none() {
        state
            .get_reclaimed_session(&sess_id)
            .and_then(|(principal, run_id)| {
                state.run_is_deleted(&run_id).then_some((principal, run_id))
            })
    } else {
        None
    };
    let Some((owner_principal, run_id)) = live_session.or(reclaimed_session) else {
        tracing::debug!("WHIP terminate: unknown session {sess_id}");
        return StatusCode::NOT_FOUND;
    };

    let caller = state.security_policy().principal_key_from_headers(&headers);
    if caller != owner_principal {
        tracing::warn!("WHIP terminate sess={sess_id} principal mismatch");
        return StatusCode::FORBIDDEN;
    }

    match terminate_whip_session(&state, &sess_id, &run_id).await {
        Ok(()) => StatusCode::OK,
        Err(err) => {
            // A committed tombstone satisfies the delete's contract even if
            // the reclaim is owned elsewhere. The watcher finishes the
            // session removal in that case.
            if state.run_is_deleted(&run_id) {
                tracing::info!(
                    "WHIP terminate: tombstone committed, cleanup owned elsewhere sess={sess_id}: {err}"
                );
                return StatusCode::OK;
            }
            tracing::error!("WHIP terminate cleanup incomplete sess={sess_id}: {err}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

async fn terminate_whip_session(
    state: &AppState,
    sess_id: &str,
    run_id: &str,
) -> Result<(), String> {
    let state = state.clone();
    let sess_id = sess_id.to_string();
    let run_id = run_id.to_string();

    tokio::spawn(async move { terminate_whip_session_transaction(state, sess_id, run_id).await })
        .await
        .map_err(|err| format!("WHIP terminate transaction join failure: {err}"))?
}

async fn terminate_whip_session_transaction(
    state: AppState,
    sess_id: String,
    run_id: String,
) -> Result<(), String> {
    // The tombstone, the delete marking, and the reclaim live in one
    // cancellation-resistant task. A client can drop the request right after
    // the tombstone commits, and cancelling there would leave a deleted run
    // with a live session and pipeline still running.
    //
    // The delete guarantees its own tombstone, exactly like the REST delete
    // handler, so a stop-claimed reclaim that skips its append cannot lose
    // it. The append is idempotent, so a duplicate from the reclaim is
    // impossible.
    append_whip_run_deleted_once(&state, &run_id, &sess_id, "delete").await?;
    if let Some((_, _, session)) = state.get_session(&sess_id) {
        session.mark_close_disposition_delete();
    }
    reclaim_whip_session_transaction(state, sess_id, run_id, "delete".to_string()).await
}

fn should_reclaim_peer_state(state: PeerConnectionState) -> bool {
    matches!(
        state,
        PeerConnectionState::Disconnected
            | PeerConnectionState::Failed
            | PeerConnectionState::Closed
    )
}

fn spawn_session_reclaimer(
    state: AppState,
    sess_id: String,
    run_id: Arc<str>,
    mut peer_state_rx: tokio::sync::watch::Receiver<PeerConnectionState>,
) {
    tokio::spawn(async move {
        loop {
            let peer_state = *peer_state_rx.borrow_and_update();
            if should_reclaim_peer_state(peer_state) {
                let reason = match peer_state {
                    PeerConnectionState::Disconnected => "peer_disconnected",
                    PeerConnectionState::Failed => "peer_failed",
                    PeerConnectionState::Closed => "peer_closed",
                    _ => "peer_terminal",
                };
                reclaim_whip_session_from_watcher(&state, &sess_id, &run_id, reason).await;
                break;
            }
            if peer_state_rx.changed().await.is_err() {
                reclaim_whip_session_from_watcher(
                    &state,
                    &sess_id,
                    &run_id,
                    "peer_state_channel_closed",
                )
                .await;
                break;
            }
        }
    });
}

async fn reclaim_whip_session(
    state: &AppState,
    sess_id: &str,
    run_id: &str,
    reason: &str,
) -> Result<(), String> {
    let state = state.clone();
    let sess_id = sess_id.to_string();
    let run_id = run_id.to_string();
    let reason = reason.to_string();

    tokio::spawn(
        async move { reclaim_whip_session_transaction(state, sess_id, run_id, reason).await },
    )
    .await
    .map_err(|err| format!("WHIP reclaim transaction join failure: {err}"))?
}

async fn reclaim_whip_session_transaction(
    state: AppState,
    sess_id: String,
    run_id: String,
    reason: String,
) -> Result<(), String> {
    // Claim the reclaim on the session itself so exactly one caller decides
    // the tombstone, and read the disposition only after the claim. That
    // closes the window where a stop mark lands between a read and the
    // append. A stop arriving after the claim is racing a peer-death cleanup
    // and loses, which is right because the peer is already gone. The claim
    // is released only on a failed append, so the watcher retry can claim
    // again and finish the cleanup. Tombstone-then-remove order is kept so a
    // committed tombstone never coexists with a silently dropped session.
    let Some((_principal, existing_run_id, session)) = state.get_session(&sess_id) else {
        return Ok(());
    };
    if *existing_run_id != run_id {
        return Ok(());
    }
    if !session.try_claim_reclaim() {
        // Another reclaimer owns the decision right now. Wait for it to
        // finish (session removed) or fail and release the claim (claim
        // becomes ours). Returning Ok while the session is still visible
        // would strand it: the owner may fail its append after we reported
        // success, and a satisfied caller never retries. The wait is bounded
        // so a stuck owner turns into a retryable error instead of a hang.
        let mut claimed = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if state.get_session(&sess_id).is_none() {
                return Ok(());
            }
            if session.try_claim_reclaim() {
                claimed = true;
                break;
            }
        }
        if !claimed {
            return Err(format!("reclaim of {sess_id} still owned; retry"));
        }
    }

    let preserve_history = session.close_preserves_history();
    if !preserve_history {
        if let Err(err) = append_whip_run_deleted_once(&state, &run_id, &sess_id, &reason).await {
            session.release_reclaim_claim();
            return Err(err);
        }
    }

    if let Some((_principal, _existing_run_id, session)) =
        state.remove_session_for_run(&sess_id, &run_id)
    {
        state.pipeline_metrics().inc_sessions_removed();
        session.close();
        tracing::info!(
            "WHIP session reclaimed sess_id={sess_id} reason={reason} preserve_history={preserve_history}"
        );
    }

    Ok(())
}

async fn reclaim_whip_session_from_watcher(
    state: &AppState,
    sess_id: &str,
    run_id: &str,
    reason: &str,
) {
    reclaim_whip_session_from_watcher_with_backoff(
        state,
        sess_id,
        run_id,
        reason,
        WHIP_RECLAIM_INITIAL_BACKOFF,
        WHIP_RECLAIM_MAX_BACKOFF,
    )
    .await;
}

async fn reclaim_whip_session_from_watcher_with_backoff(
    state: &AppState,
    sess_id: &str,
    run_id: &str,
    reason: &str,
    initial_backoff: Duration,
    max_backoff: Duration,
) {
    let mut backoff = initial_backoff.max(Duration::from_millis(1));
    let max_backoff = max_backoff.max(backoff);

    loop {
        match reclaim_whip_session(state, sess_id, run_id, reason).await {
            Ok(()) => return,
            Err(err) => {
                if watcher_reclaim_terminal(state, sess_id, run_id).await {
                    tracing::info!(
                        "WHIP watcher reclaim already completed run_id={run_id} sess_id={sess_id}"
                    );
                    return;
                }

                tracing::error!(
                    "failed to reclaim WHIP session from watcher run_id={run_id} sess_id={sess_id}; retrying in {:?}: {err}",
                    backoff
                );
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2).min(max_backoff);
            }
        }
    }
}

async fn watcher_reclaim_terminal(state: &AppState, sess_id: &str, run_id: &str) -> bool {
    if let Some((_principal, existing_run_id, _session)) = state.get_session(sess_id) {
        return &*existing_run_id != run_id;
    }

    if state.run_is_deleted(run_id) {
        return true;
    }

    state
        .get_reclaimed_session(sess_id)
        .is_some_and(|(_principal, reclaimed_run_id)| &*reclaimed_run_id == run_id)
}

async fn append_whip_run_deleted_once(
    state: &AppState,
    run_id: &str,
    sess_id: &str,
    reason: &str,
) -> Result<(), String> {
    state
        .append_run_deleted_idempotent_async(
            run_id,
            serde_json::json!({
                "source": "whip",
                "session_id": sess_id,
                "reason": reason,
            }),
        )
        .await
        .map(|_| ())
}

async fn tombstone_created_whip_run_until_success(
    state: &AppState,
    run_id: &str,
    sess_id: &str,
    reason: &str,
) {
    let mut backoff = WHIP_RECLAIM_INITIAL_BACKOFF;
    loop {
        match append_whip_run_deleted_once(state, run_id, sess_id, reason).await {
            Ok(()) => return,
            Err(err) => {
                if state.run_is_deleted(run_id) {
                    return;
                }
                tracing::error!(
                    "failed to tombstone WHIP creation run_id={run_id} sess_id={sess_id}; retrying in {:?}: {err}",
                    backoff
                );
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2).min(WHIP_RECLAIM_MAX_BACKOFF);
            }
        }
    }
}

async fn tombstone_created_whip_run_with_request_bound(
    state: &AppState,
    run_id: &str,
    sess_id: &str,
    reason: &str,
) {
    if tombstone_created_whip_run_with_request_bound_and_backoff(
        state,
        run_id,
        sess_id,
        reason,
        WHIP_CREATE_TOMBSTONE_INLINE_ATTEMPTS,
        WHIP_RECLAIM_INITIAL_BACKOFF,
        WHIP_RECLAIM_MAX_BACKOFF,
    )
    .await
    {
        return;
    }

    spawn_created_whip_tombstone_retry(
        state.clone(),
        run_id.to_string(),
        sess_id.to_string(),
        reason.to_string(),
    );
}

async fn tombstone_created_whip_run_with_request_bound_and_backoff(
    state: &AppState,
    run_id: &str,
    sess_id: &str,
    reason: &str,
    attempts: usize,
    initial_backoff: Duration,
    max_backoff: Duration,
) -> bool {
    let attempts = attempts.max(1);
    let mut backoff = initial_backoff.max(Duration::from_millis(1));
    let max_backoff = max_backoff.max(backoff);

    for attempt in 0..attempts {
        match append_whip_run_deleted_once(state, run_id, sess_id, reason).await {
            Ok(()) => return true,
            Err(err) => {
                if state.run_is_deleted(run_id) {
                    return true;
                }
                tracing::error!(
                    "failed to tombstone WHIP creation run_id={run_id} sess_id={sess_id}; retrying in {:?}: {err}",
                    backoff
                );
                if attempt + 1 < attempts {
                    tokio::time::sleep(backoff).await;
                    backoff = backoff.saturating_mul(2).min(max_backoff);
                }
            }
        }
    }

    false
}

fn spawn_created_whip_tombstone_retry(
    state: AppState,
    run_id: String,
    sess_id: String,
    reason: String,
) {
    tokio::spawn(async move {
        tombstone_created_whip_run_until_success(&state, &run_id, &sess_id, &reason).await;
    });
}

// ---------------------------------------------------------------------------
// Prompt update handler
// ---------------------------------------------------------------------------

/// Request body for `PATCH /v1/stream/whip/{sess_id}/prompt`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdatePromptRequest {
    pub prompt: String,
    /// Optional JSON Schema for structured VLM output.
    pub output_schema: Option<Value>,
}

#[derive(Debug, Serialize)]
struct UpdatePromptResponse {
    session_id: String,
    prompt: String,
    output_schema: Option<Value>,
}

/// `PATCH /v1/stream/whip/{sess_id}/prompt`
///
/// Replaces the VLM analysis prompt for a running WebRTC session and
/// optionally sets a JSON schema for structured output. The update is sent
/// to the live pipeline as a generation-tagged command, and the handler
/// waits up to two seconds for a VLM worker acknowledgement. The worker
/// applies the new values before its next work item, so `200 OK` means the
/// update is actually in effect, not merely queued.
///
/// Body: `{ "prompt": "new prompt text", "output_schema": {...} }`
///
/// Response:
/// - `200 OK` with `{ "session_id": "...", "prompt": "...", "output_schema": ... }`
/// - `404 Not Found` — unknown session
/// - `403 Forbidden` — caller is not the session owner
/// - `409 Conflict` — the session's generation was closed or replaced, so the update was rejected
/// - `503 Service Unavailable` — no acknowledgement within two seconds. The command was discarded rather than applied later, so the caller must retry.
#[tracing::instrument(name = "whip.update_prompt", skip_all, fields(sess_id))]
pub async fn whip_update_prompt(
    State(state): State<AppState>,
    Path(sess_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<UpdatePromptRequest>,
) -> Response {
    let Some((owner_principal, _run_id, session)) = state.get_session(&sess_id) else {
        tracing::debug!("WHIP update_prompt: unknown session {sess_id}");
        return StatusCode::NOT_FOUND.into_response();
    };

    let caller = state.security_policy().principal_key_from_headers(&headers);
    if caller != owner_principal {
        tracing::warn!("WHIP update_prompt sess={sess_id} principal mismatch");
        return StatusCode::FORBIDDEN.into_response();
    }

    let update = session.update_config(
        body.prompt.clone(),
        body.output_schema.as_ref().map(Value::to_string),
    );
    match tokio::time::timeout(Duration::from_secs(2), update).await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            tracing::warn!(sess_id = %sess_id, error = %err, "WHIP prompt update rejected");
            return StatusCode::CONFLICT.into_response();
        }
        Err(_) => {
            tracing::warn!(sess_id = %sess_id, "WHIP prompt update acknowledgement timed out");
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    }
    tracing::info!(
        sess_id = %sess_id,
        has_schema = body.output_schema.is_some(),
        "WHIP prompt updated"
    );

    Json(UpdatePromptResponse {
        session_id: sess_id,
        prompt: body.prompt,
        output_schema: body.output_schema,
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        apply_attach_config, hex_char, new_session_id, parse_attach_config_header,
        reclaim_whip_session, reclaim_whip_session_from_watcher,
        reclaim_whip_session_from_watcher_with_backoff, should_reclaim_peer_state,
        start_whip_session, tombstone_created_whip_run_with_request_bound_and_backoff,
        whip_setup_error_response, whip_terminate, PeerConnectionState, WebRtcSession,
        WebRtcSetupError, ATTACH_CONFIG_HEADER, RUN_ID_HEADER,
    };
    use axum::body::Body;
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tower::ServiceExt;

    use crate::security::SecurityPolicy;
    use crate::state::AppState;

    fn test_pipeline_session() -> (
        Arc<WebRtcSession>,
        tokio::sync::mpsc::Receiver<vidarax_core::webrtc::runtime::SessionCommand>,
    ) {
        let (session, commands) = WebRtcSession::new_for_tests_with_generation(
            vidarax_core::webrtc::runtime::PipelineGeneration::new(1),
        );
        (Arc::new(session), commands)
    }

    #[test]
    fn whip_setup_error_response_classifies_client_vs_server() {
        let (client_status, _) = whip_setup_error_response(&WebRtcSetupError::UnsupportedVideo(
            "offer advertised video but no live-serveable codec".to_string(),
        ));
        assert_eq!(client_status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let (server_status, _) = whip_setup_error_response(&WebRtcSetupError::Negotiation(
            "create_answer: boom".to_string(),
        ));
        assert_eq!(server_status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn session_id_has_sess_prefix() {
        let id = new_session_id();
        assert!(
            id.starts_with("sess-"),
            "expected 'sess-' prefix, got: {id}"
        );
    }

    #[test]
    fn session_id_has_correct_length() {
        let id = new_session_id();
        // "sess-" (5) + 16 hex chars = 21
        assert_eq!(id.len(), 21, "wrong length: {id}");
    }

    #[test]
    fn hex_char_covers_full_nibble_range() {
        assert_eq!(hex_char(0), '0');
        assert_eq!(hex_char(9), '9');
        assert_eq!(hex_char(10), 'a');
        assert_eq!(hex_char(15), 'f');
    }

    #[test]
    fn attach_config_header_reads_base64url_prompt_clip_mode_and_token_cap() {
        let mut headers = HeaderMap::new();
        let json = r#"{"prompt":"watch exits","token-cap":42,"clip_mode":{"target_fps":6,"clip_length_seconds":0.5,"delay_seconds":0.25}}"#;
        let encoded = URL_SAFE_NO_PAD.encode(json.as_bytes());
        headers.insert(
            ATTACH_CONFIG_HEADER,
            HeaderValue::from_str(&encoded).unwrap(),
        );
        let config = parse_attach_config_header(&headers).unwrap().unwrap();
        let clip = config.clip_mode.unwrap().into_core();

        assert_eq!(config.prompt.as_deref(), Some("watch exits"));
        assert_eq!(config.max_output_tokens_per_second, Some(42));
        clip.validate().unwrap();
        assert_eq!(clip.target_fps, 6);
        assert_eq!(clip.clip_length_seconds, 0.5);
        assert_eq!(clip.delay_seconds, 0.25);
    }

    #[tokio::test]
    async fn attach_config_header_decodes_non_ascii_prompt_before_worker_start() {
        let mut headers = HeaderMap::new();
        let prompt = "watch for exit signs 🚪 and 人";
        let json = serde_json::json!({ "prompt": prompt }).to_string();
        let encoded = URL_SAFE_NO_PAD.encode(json.as_bytes());
        headers.insert(
            ATTACH_CONFIG_HEADER,
            HeaderValue::from_str(&encoded).unwrap(),
        );

        let config = parse_attach_config_header(&headers).unwrap();
        let mut session = WebRtcSession::new_for_tests();
        let clip = apply_attach_config(&mut session, config).unwrap();

        assert!(clip.is_none());
        assert_eq!(session.initial_prompt().as_ref(), prompt);
    }

    #[tokio::test]
    async fn restricted_zone_attach_becomes_generation_static_analysis_crop() {
        let raw = r#"{
            "restricted_zone": {
                "policy_id": "loading-bay-east",
                "policy_version": 4,
                "device_id": "camera-17",
                "region": {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.6},
                "enter_motion_score": 0.4,
                "exit_motion_score": 0.15,
                "enter_after_frames": 2,
                "exit_after_frames": 3
            }
        }"#;
        let config: crate::models::AttachStreamRequest = serde_json::from_str(raw).unwrap();
        let mut session = WebRtcSession::new_for_tests();

        let clip = apply_attach_config(&mut session, Some(config)).unwrap();

        assert!(clip.is_none());
        let zone = session
            .restricted_zone
            .as_ref()
            .expect("zone should be fixed");
        assert_eq!(zone.policy_id, "loading-bay-east");
        assert_eq!(session.crop, Some(zone.crop()));
    }

    #[tokio::test]
    async fn restricted_zone_rejects_conflicting_media_modes() {
        let raw = r#"{
            "clip_mode": {"target_fps": 6, "clip_length_seconds": 0.5, "delay_seconds": 0.25},
            "restricted_zone": {
                "policy_id": "loading-bay-east",
                "policy_version": 1,
                "device_id": "camera-17",
                "region": {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.6},
                "enter_motion_score": 0.4,
                "exit_motion_score": 0.15,
                "enter_after_frames": 2,
                "exit_after_frames": 3
            }
        }"#;
        let config: crate::models::AttachStreamRequest = serde_json::from_str(raw).unwrap();
        let mut session = WebRtcSession::new_for_tests();

        let err = apply_attach_config(&mut session, Some(config)).unwrap_err();
        assert_eq!(
            err,
            "restricted_zone and clip_mode cannot be enabled together"
        );
    }

    #[test]
    fn terminal_peer_states_reclaim_whip_session() {
        assert!(should_reclaim_peer_state(PeerConnectionState::Disconnected));
        assert!(should_reclaim_peer_state(PeerConnectionState::Failed));
        assert!(should_reclaim_peer_state(PeerConnectionState::Closed));
        assert!(!should_reclaim_peer_state(PeerConnectionState::Connected));
        assert!(!should_reclaim_peer_state(PeerConnectionState::Connecting));
    }

    #[tokio::test]
    async fn failed_run_created_append_does_not_expose_whip_session() {
        let dir =
            std::env::temp_dir().join(format!("vidarax-whip-create-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let state = AppState::with_wal_for_tests(dir.join("timeline.wal"));
        state.set_timeline_append_failure_for_tests(true);
        let (session, commands) = test_pipeline_session();

        let response = start_whip_session(
            state.clone(),
            HeaderMap::new(),
            Arc::clone(&session),
            "v=0\r\n".to_string(),
            None,
            commands,
        )
        .await;

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(state.session_count(), 0);
        assert_eq!(state.count_active_runs_for_principal("public", now_ms()), 0);
        assert_eq!(session.close_call_count_for_tests(), 1);
    }

    #[tokio::test]
    async fn whip_offer_response_includes_run_id_header() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-run-header-{}.wal",
            std::process::id()
        )));
        let (session, commands) = test_pipeline_session();

        let response = start_whip_session(
            state,
            HeaderMap::new(),
            Arc::clone(&session),
            "v=0\r\n".to_string(),
            None,
            commands,
        )
        .await;
        session.close();

        assert_eq!(response.status(), StatusCode::CREATED);
        let run_id = response
            .headers()
            .get(RUN_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .expect("WHIP offer should expose the server run_id");
        assert!(
            run_id.starts_with("run-"),
            "unexpected run_id header: {run_id}"
        );
    }

    #[tokio::test]
    async fn whip_creation_enforces_active_stream_limit_before_persisting() {
        let state = AppState::with_wal_for_tests_runtime(
            std::env::temp_dir().join(format!("vidarax-whip-limit-{}.wal", std::process::id())),
            None,
            SecurityPolicy::from_config_for_tests(),
            3600,
            2,
        );
        let mut sessions = Vec::new();

        for _ in 0..state.active_stream_limit() {
            let (session, commands) = test_pipeline_session();
            let response = start_whip_session(
                state.clone(),
                HeaderMap::new(),
                Arc::clone(&session),
                "v=0\r\n".to_string(),
                None,
                commands,
            )
            .await;
            assert_eq!(response.status(), StatusCode::CREATED);
            sessions.push(session);
        }
        assert_eq!(state.count_active_runs_for_principal("public", now_ms()), 2);
        assert_eq!(
            state
                .read_all_events()
                .unwrap()
                .iter()
                .filter(|event| event.kind == "run_created")
                .count(),
            2
        );

        let (rejected_session, commands) = test_pipeline_session();
        let response = start_whip_session(
            state.clone(),
            HeaderMap::new(),
            Arc::clone(&rejected_session),
            "v=0\r\n".to_string(),
            None,
            commands,
        )
        .await;

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(state.session_count(), 2);
        assert_eq!(state.count_active_runs_for_principal("public", now_ms()), 2);
        assert_eq!(rejected_session.close_call_count_for_tests(), 1);
        assert_eq!(
            state
                .read_all_events()
                .unwrap()
                .iter()
                .filter(|event| event.kind == "run_created")
                .count(),
            2
        );

        for session in sessions {
            session.close();
        }
    }

    #[tokio::test]
    async fn held_stream_reservations_reject_normal_and_whip_creation() {
        let state = AppState::with_wal_for_tests_runtime(
            std::env::temp_dir().join(format!(
                "vidarax-cross-path-reservation-limit-{}.wal",
                std::process::id()
            )),
            None,
            SecurityPolicy::from_config_for_tests(),
            3600,
            2,
        );
        let mut guards = Vec::new();
        for _ in 0..state.active_stream_limit() {
            guards.push(
                state
                    .try_reserve_stream_slot("public", now_ms())
                    .expect("reservation should fit under the per-principal limit"),
            );
        }

        let app = crate::app_router(state.clone());
        let create_run = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let normal_response = app.oneshot(create_run).await.unwrap();
        assert_eq!(normal_response.status(), StatusCode::CONFLICT);
        let normal_body = normal_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert!(std::str::from_utf8(&normal_body)
            .unwrap()
            .contains("active stream limit exceeded"));

        let (rejected_session, commands) = test_pipeline_session();
        let whip_response = start_whip_session(
            state,
            HeaderMap::new(),
            Arc::clone(&rejected_session),
            "v=0\r\n".to_string(),
            None,
            commands,
        )
        .await;
        assert_eq!(whip_response.status(), StatusCode::CONFLICT);
        let whip_body = whip_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(
            std::str::from_utf8(&whip_body).unwrap(),
            "active stream limit exceeded"
        );
        assert_eq!(rejected_session.close_call_count_for_tests(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn creation_tombstone_failure_detaches_retry_until_wal_recovers() {
        let dir = std::env::temp_dir().join(format!(
            "vidarax-whip-create-tombstone-retry-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let state = AppState::with_wal_for_tests(dir.join("timeline.wal"));
        let run_id = "run-whip-create-tombstone-retry";
        let sess_id = "sess-tombretry001";

        state
            .append_run_event(
                run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": "tenant-a",
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        state.set_timeline_append_failure_for_tests(true);

        let started = std::time::Instant::now();
        let completed_inline = tombstone_created_whip_run_with_request_bound_and_backoff(
            &state,
            run_id,
            sess_id,
            "session_insert_failed",
            1,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(1),
        )
        .await;
        assert!(!completed_inline);
        assert!(
            started.elapsed() < std::time::Duration::from_millis(200),
            "request-path compensation must not wait for persistent WAL failure"
        );

        super::spawn_created_whip_tombstone_retry(
            state.clone(),
            run_id.to_string(),
            sess_id.to_string(),
            "session_insert_failed".to_string(),
        );
        state.set_timeline_append_failure_for_tests(false);

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.run_is_deleted(run_id) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background tombstone retry should finish after WAL recovery");

        let events = state.read_run_events(run_id).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn reclaim_on_disconnect_tombstones_run_and_releases_active_slot_once() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-disconnect-{}.wal",
            std::process::id()
        )));
        let sess_id = "sess-disconnect0001";
        let run_id: Arc<str> = Arc::from("run-whip-disconnect");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            1
        );
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        reclaim_whip_session(&state, sess_id, &run_id, "peer_disconnected")
            .await
            .unwrap();
        reclaim_whip_session(&state, sess_id, &run_id, "peer_disconnected")
            .await
            .expect("second reclaim must be idempotent");

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(deleted, 1);
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.run_runtime_snapshot(&run_id, now_ms()).is_none());
        assert!(state.get_session(sess_id).is_none());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_reclaim_cannot_commit_tombstone_and_leave_live_session() {
        let dir = std::env::temp_dir().join(format!(
            "vidarax-whip-reclaim-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let wal_path = dir.join("timeline.wal");
        let state = AppState::with_wal_for_tests(wal_path.clone());
        let sess_id = "sess-cancel000001";
        let run_id: Arc<str> = Arc::from("run-whip-reclaim-cancel");
        let principal = "tenant-a";
        let session = Arc::new(WebRtcSession::new_for_tests());

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::clone(&session),
        ));

        state.pause_timeline_appends_for_tests();

        let reclaim_state = state.clone();
        let reclaim_run_id = Arc::clone(&run_id);
        let reclaim_task = tokio::spawn(async move {
            reclaim_whip_session(&reclaim_state, sess_id, &reclaim_run_id, "delete").await
        });

        state.wait_until_timeline_writer_paused_for_tests();
        reclaim_task.abort();
        let _ = reclaim_task.await;
        state.resume_timeline_appends_for_tests();

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if state.run_is_deleted(&run_id)
                    && state.session_count() == 0
                    && session.close_call_count_for_tests() == 1
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocked reclaim should commit, remove, and close the live session");

        assert_eq!(state.session_count(), 0);
        assert!(state.get_session(sess_id).is_none());
        assert_eq!(session.close_call_count_for_tests(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_whip_terminate_still_deletes_run_and_removes_session() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-termcancel-{}.wal",
            std::process::id()
        )));
        let sess_id = "sess-termcancel001";
        let run_id: Arc<str> = Arc::from("run-whip-termcancel");
        let principal = "tenant-a";
        let session = Arc::new(WebRtcSession::new_for_tests());

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::clone(&session),
        ));

        // Cancel the terminate exactly while its tombstone append is in
        // flight, like a client dropping the request. The spawned transaction
        // must still finish: run deleted once, session removed.
        state.pause_timeline_appends_for_tests();
        let term_state = state.clone();
        let term_run_id = Arc::clone(&run_id);
        let term_task = tokio::spawn(async move {
            super::terminate_whip_session(&term_state, sess_id, &term_run_id).await
        });
        state.wait_until_timeline_writer_paused_for_tests();
        term_task.abort();
        let _ = term_task.await;
        state.resume_timeline_appends_for_tests();

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.run_is_deleted(&run_id) && state.session_count() == 0 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled terminate must still delete the run and remove the session");

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(deleted, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_overlapping_claimed_stop_reclaim_still_ends_deleted() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-whip-overlap-{}.wal", std::process::id())),
        );
        let sess_id = "sess-overlap000001";
        let run_id: Arc<str> = Arc::from("run-whip-overlap");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        let session = Arc::new(WebRtcSession::new_for_tests());
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::clone(&session),
        ));

        // A stop reclaimer has read preserve=true and holds the claim.
        session.mark_close_disposition_stop();
        assert!(session.try_claim_reclaim());

        // A WHIP delete lands in that window. Its endpoint appends the
        // tombstone itself, so the delete contract holds no matter who wins
        // the session cleanup. The overlapping reclaim call must report a
        // retryable error, not false success.
        super::append_whip_run_deleted_once(&state, &run_id, sess_id, "delete")
            .await
            .unwrap();
        session.mark_close_disposition_delete();
        assert!(reclaim_whip_session(&state, sess_id, &run_id, "delete")
            .await
            .is_err());
        assert!(state.run_is_deleted(&run_id));

        // The stop reclaimer finishes: it read preserve before the delete
        // upgrade, skips its own append, and removes the session. The run
        // stays deleted through the endpoint's tombstone.
        session.release_reclaim_claim();
        reclaim_whip_session(&state, sess_id, &run_id, "peer_failed")
            .await
            .unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(deleted, 1, "the endpoint tombstone must be the only one");
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn whip_delete_after_stop_mark_still_tombstones() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-whip-wdel-{}.wal", std::process::id())),
        );
        let sess_id = "sess-wdel00000001";
        let run_id: Arc<str> = Arc::from("run-whip-wdel");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        let session = Arc::new(WebRtcSession::new_for_tests());
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::clone(&session),
        ));

        // A stop marks the session first, then a WHIP DELETE lands. The
        // delete disposition overrides the stop mark, and the reclaim itself
        // must append the tombstone because WHIP DELETE has no handler-side
        // append of its own.
        session.mark_close_disposition_stop();
        session.mark_close_disposition_delete();
        reclaim_whip_session(&state, sess_id, &run_id, "delete")
            .await
            .unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(
            deleted, 1,
            "WHIP delete must tombstone despite the stop mark"
        );
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stop_marked_session_reclaim_preserves_history() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-whip-stop-{}.wal", std::process::id())),
        );
        let sess_id = "sess-stop00000001";
        let run_id: Arc<str> = Arc::from("run-whip-stop");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        // The stop handler marks the session before closing it. The reclaim
        // that follows must remove the session without tombstoning the run.
        state.close_live_session_for_run(&run_id, true);
        reclaim_whip_session(&state, sess_id, &run_id, "stop")
            .await
            .unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        assert!(
            events.iter().all(|event| event.kind != "run_deleted"),
            "a stop-driven reclaim must not tombstone the run"
        );
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_after_stop_mark_still_tombstones_exactly_once() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-whip-stopdel-{}.wal", std::process::id())),
        );
        let sess_id = "sess-stopdel000001";
        let run_id: Arc<str> = Arc::from("run-whip-stopdel");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        // Stop marks the session, then a REST delete lands. The delete
        // handler appends its own tombstone before closing, so the run must
        // end up deleted exactly once even though the reclaim skips its
        // duplicate append.
        state.close_live_session_for_run(&run_id, true);
        state
            .append_run_event(&run_id, "run_deleted", serde_json::json!({}))
            .unwrap();
        state.close_live_session_for_run(&run_id, false);
        reclaim_whip_session(&state, sess_id, &run_id, "delete")
            .await
            .unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(deleted, 1, "delete must tombstone exactly once");
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test]
    async fn reclaim_on_delete_skips_duplicate_tombstone_when_run_already_deleted() {
        let state = AppState::with_wal_for_tests(
            std::env::temp_dir().join(format!("vidarax-whip-delete-{}.wal", std::process::id())),
        );
        let sess_id = "sess-delete0000001";
        let run_id: Arc<str> = Arc::from("run-whip-delete");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        state
            .append_run_event(&run_id, "run_deleted", serde_json::json!({}))
            .unwrap();
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        reclaim_whip_session(&state, sess_id, &run_id, "delete")
            .await
            .unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        let deleted = events
            .iter()
            .filter(|event| event.kind == "run_deleted")
            .count();
        assert_eq!(deleted, 1);
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.run_runtime_snapshot(&run_id, now_ms()).is_none());
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test]
    async fn reclaim_delete_after_watcher_reclaim_is_idempotent_success() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-watcher-wins-{}.wal",
            std::process::id()
        )));
        let sess_id = "sess-watchwins001";
        let run_id: Arc<str> = Arc::from("run-whip-watcher-wins");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        reclaim_whip_session_from_watcher(&state, sess_id, &run_id, "peer_disconnected").await;

        reclaim_whip_session(&state, sess_id, &run_id, "delete")
            .await
            .expect("DELETE must be idempotent success after watcher cleanup");
        let events = state.read_run_events(&run_id).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test]
    async fn terminate_after_watcher_reclaim_returns_ok_not_not_found() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-delete-after-watch-{}.wal",
            std::process::id()
        )));
        let sess_id = "sess-deletewatch01";
        let run_id: Arc<str> = Arc::from("run-whip-delete-after-watch");
        let principal = "public";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));

        reclaim_whip_session_from_watcher(&state, sess_id, &run_id, "peer_disconnected").await;

        let status = whip_terminate(
            State(state.clone()),
            Path(sess_id.to_string()),
            HeaderMap::new(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.get_session(sess_id).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_delete_and_watcher_reclaim_has_single_terminal_effect() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-whip-concurrent-reclaim-{}.wal",
            std::process::id()
        )));
        let sess_id = "sess-concurrent001";
        let run_id: Arc<str> = Arc::from("run-whip-concurrent-reclaim");
        let principal = "tenant-a";
        let session = Arc::new(WebRtcSession::new_for_tests());

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::clone(&session),
        ));

        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let delete_state = state.clone();
        let delete_run_id = Arc::clone(&run_id);
        let delete_barrier = Arc::clone(&barrier);
        let delete_task = tokio::spawn(async move {
            delete_barrier.wait().await;
            reclaim_whip_session(&delete_state, sess_id, &delete_run_id, "delete").await
        });

        let watcher_state = state.clone();
        let watcher_run_id = Arc::clone(&run_id);
        let watcher_barrier = Arc::clone(&barrier);
        let watcher_task = tokio::spawn(async move {
            watcher_barrier.wait().await;
            reclaim_whip_session_from_watcher(
                &watcher_state,
                sess_id,
                &watcher_run_id,
                "peer_disconnected",
            )
            .await;
        });

        delete_task.await.unwrap().unwrap();
        watcher_task.await.unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.run_runtime_snapshot(&run_id, now_ms()).is_none());
        assert!(state.get_session(sess_id).is_none());
        assert!(state
            .pipeline_metrics()
            .render_prometheus()
            .contains("vidarax_pipeline_sessions_removed_total 1\n"));
        assert_eq!(session.close_call_count_for_tests(), 1);
    }

    #[tokio::test]
    async fn reclaim_delete_surfaces_failed_tombstone_append() {
        let dir =
            std::env::temp_dir().join(format!("vidarax-whip-delete-fail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let state = AppState::with_wal_for_tests(dir.join("timeline.wal"));
        let sess_id = "sess-deletefail001";
        let run_id: Arc<str> = Arc::from("run-whip-delete-fail");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));
        state.set_timeline_append_failure_for_tests(true);

        let result = reclaim_whip_session(&state, sess_id, &run_id, "delete").await;

        assert!(result.is_err(), "DELETE must surface incomplete cleanup");
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            1
        );
        assert!(
            state.get_session(sess_id).is_some(),
            "failed tombstone append must leave the session reachable for retry"
        );
    }

    #[tokio::test]
    async fn reclaim_watcher_retries_transient_tombstone_append_failure_until_cleanup_succeeds() {
        let dir =
            std::env::temp_dir().join(format!("vidarax-whip-watch-retry-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let state = AppState::with_wal_for_tests(dir.join("timeline.wal"));
        let sess_id = "sess-watchretry01";
        let run_id: Arc<str> = Arc::from("run-whip-watch-retry");
        let principal = "tenant-a";

        state
            .append_run_event(
                &run_id,
                "run_created",
                serde_json::json!({
                    "principal_key": principal,
                    "session_id": sess_id,
                    "source": "whip",
                }),
            )
            .unwrap();
        assert!(state.insert_session(
            sess_id.to_string(),
            principal.to_string(),
            Arc::clone(&run_id),
            Arc::new(WebRtcSession::new_for_tests()),
        ));
        state.set_timeline_append_failure_for_tests(true);

        let restore_state = state.clone();
        let restore_task = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            restore_state.set_timeline_append_failure_for_tests(false);
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            reclaim_whip_session_from_watcher_with_backoff(
                &state,
                sess_id,
                &run_id,
                "peer_disconnected",
                std::time::Duration::from_millis(1),
                std::time::Duration::from_millis(10),
            ),
        )
        .await
        .expect("watcher retry should finish after transient WAL recovery");
        restore_task.await.unwrap();

        let events = state.read_run_events(&run_id).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "run_deleted")
                .count(),
            1
        );
        assert_eq!(
            state.count_active_runs_for_principal(principal, now_ms()),
            0
        );
        assert!(state.run_runtime_snapshot(&run_id, now_ms()).is_none());
        assert!(state.get_session(sess_id).is_none());
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

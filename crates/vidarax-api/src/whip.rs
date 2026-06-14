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
//! (`session.run(tx)`) forwards H.264 NAL units through the channel.  Three
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

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use vidarax_core::provider::{InferenceProvider, InferenceRequest, InferenceResult, ProviderError, ProviderKind};
use vidarax_core::tiered_vlm::TieredVlmConfig;
use vidarax_core::webrtc::session::WebRtcSession;
use vidarax_core::webrtc::workers::{
    EventSink, VlmWorkerParams, spawn_analysis_workers, spawn_decode_workers, spawn_vlm_workers,
};

use crate::state::AppState;
use crate::wal_sink::WalEventSink;

// ---------------------------------------------------------------------------
// NullInferenceProvider — used when no provider endpoints are configured
// ---------------------------------------------------------------------------

/// An [`InferenceProvider`] that always returns a placeholder response.
///
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
        })
    }
}

const HEADER_API_KEY: &str = "x-api-key";
const HEADER_TENANT_ID: &str = "x-tenant-id";

/// Derive a principal key from the request headers, matching the scheme used
/// by the main API handlers.
fn principal_key_from_headers(headers: &HeaderMap) -> String {
    if let Some(tid) = headers.get(HEADER_TENANT_ID).and_then(|v| v.to_str().ok()) {
        return format!("tenant:{tid}");
    }
    if let Some(key) = headers.get(HEADER_API_KEY).and_then(|v| v.to_str().ok()) {
        let mut hash = 1469598103934665603u64;
        for b in key.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        return format!("api-key:{hash:016x}");
    }
    "public".to_string()
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
/// - `400` — empty or unparseable SDP offer
/// - `500` — internal rustrtc or ICE failure
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

    // Create the rustrtc PeerConnection and negotiate the SDP answer.
    let (session, answer_sdp) =
        match WebRtcSession::new(offer_sdp, state.webrtc_config()).await {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!("WHIP offer negotiation failed: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "WebRTC negotiation failed",
                )
                    .into_response();
            }
        };

    let session = Arc::new(session);
    let sess_id = new_session_id();
    let principal = principal_key_from_headers(&headers);

    // Create a session-scoped span so all worker-thread log events are
    // attributed to this WebRTC session.
    let session_span = tracing::info_span!("whip_session", sess_id = %sess_id);

    // Store the session bound to the requesting principal.
    if !state
        .insert_session(sess_id.clone(), principal.clone(), Arc::clone(&session))
        .await
    {
        // Collision or global session limit reached.
        tracing::error!("WHIP session insert failed for {sess_id} (collision or limit)");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "session limit reached or ID collision",
        )
            .into_response();
    }

    // Record the new session in pipeline metrics.
    state.pipeline_metrics().inc_sessions_created();

    // Create a run ID so VLM events have a home in the WAL / SpacetimeDB.
    let run_id: Arc<str> = Arc::from(state.next_run_id().as_str());
    let session_id_arc: Arc<str> = Arc::from(sess_id.as_str());

    // Register the run in the WAL so `GET /v1/runs/{id}/events` works.
    let _ = state.append_run_event(
        &run_id,
        "run_created",
        serde_json::json!({
            "principal_key": principal,
            "session_id": sess_id,
            "source": "whip",
        }),
    );

    // ── Channel topology ───────────────────────────────────────────────────
    // RTP frames (128) → decoded stream frames (64) → keyframe work (32).
    let (frame_tx, frame_rx) =
        kanal::bounded::<vidarax_core::webrtc::session::RtpFrame>(128);
    let (stream_tx, stream_rx) =
        kanal::bounded::<vidarax_core::webrtc::workers::StreamFrame>(64);
    let (vlm_tx, vlm_rx) =
        kanal::bounded::<vidarax_core::webrtc::workers::KeyframeWork>(32);

    // ── Media ingestion task ───────────────────────────────────────────────
    let run_future = session.run(frame_tx);
    tokio::spawn(run_future);

    // ── EventSink selection ────────────────────────────────────────────────
    // Prefer SpacetimeDB when the client is configured; fall back to WAL.
    let event_sink: Arc<dyn EventSink> = if let Some(stdb) = state.spacetime_client() {
        Arc::new(stdb.clone())
    } else {
        WalEventSink::arc(state.clone(), run_id.to_string())
    };

    // ── InferenceProvider selection ────────────────────────────────────────
    // Use the HTTP router when provider endpoints are configured.
    let metrics_arc = Arc::clone(state.pipeline_metrics_arc());
    let vlm_config = TieredVlmConfig::default();
    let webrtc_config_for_workers = state.webrtc_config().clone();

    // Capture everything needed by the spawn_blocking closure.
    let run_id_for_workers = Arc::clone(&run_id);
    let session_id_for_workers = Arc::clone(&session_id_arc);
    let event_sink_for_workers = Arc::clone(&event_sink);
    let session_span_for_workers = session_span.clone();
    let metrics_for_workers = Arc::clone(&metrics_arc);
    let vlm_config_for_workers = vlm_config.clone();
    let provider = state.provider().cloned();
    // Share the session's guided_json handle with VLM workers so that
    // PATCH /prompt with output_schema takes effect on the next keyframe
    // without restarting the worker threads.
    let prompt_for_workers = session.prompt_arc();
    let guided_json_for_workers = session.guided_json_arc();

    // Worker threads are long-running OS threads (not tokio tasks).
    // Use spawn_blocking as a bridge from async context to the thread spawns;
    // the actual worker threads are spawned inside and run independently.
    tokio::task::spawn_blocking(move || {
        // ── Decode workers ─────────────────────────────────────────────
        spawn_decode_workers(
            webrtc_config_for_workers.decode_workers,
            frame_rx,
            stream_tx,
            false, // gpu_available — conservative default; no GPU assumed
            Arc::clone(&metrics_for_workers),
            session_span_for_workers.clone(),
        );

        // ── Analysis workers ───────────────────────────────────────────
        spawn_analysis_workers(
            webrtc_config_for_workers.analysis_workers,
            stream_rx,
            vlm_tx,
            None, // clip_tx — normal keyframe mode, not clip mode
            Arc::clone(&event_sink_for_workers),
            Arc::clone(&run_id_for_workers),
            Arc::clone(&session_id_for_workers),
            Arc::clone(&prompt_for_workers),
            Arc::clone(&metrics_for_workers),
            session_span_for_workers.clone(),
        );

        // ── VLM workers ────────────────────────────────────────────────
        let guided_json = guided_json_for_workers;
        let vlm_provider: Arc<dyn InferenceProvider + Send + Sync> = match provider {
            Some(p) => p,
            None => Arc::new(NullInferenceProvider),
        };
        spawn_vlm_workers(VlmWorkerParams {
            workers: webrtc_config_for_workers.vlm_workers,
            vlm_rx,
            provider: Arc::new(vlm_provider),
            stdb: event_sink_for_workers,
            config: vlm_config_for_workers,
            metrics: metrics_for_workers,
            session_span: session_span_for_workers,
            max_output_tokens_per_second: webrtc_config_for_workers.max_output_tokens_per_second,
            guided_json: Arc::clone(&guided_json),
            training_store: None,
            distillation: vidarax_core::tiered_vlm::DistillationConfig::default(),
        });
    });

    // Build the 201 Created response with SDP answer.
    let location = format!("/v1/stream/whip/{sess_id}");
    tracing::info!(
        "WHIP session created sess_id={sess_id} answer_candidates={}",
        answer_sdp.matches("a=candidate:").count()
    );

    Response::builder()
        .status(StatusCode::CREATED)
        .header(header::CONTENT_TYPE, "application/sdp")
        .header(
            header::LOCATION,
            HeaderValue::from_str(&location).unwrap_or_else(|_| HeaderValue::from_static("/")),
        )
        .body(axum::body::Body::from(answer_sdp))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
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
    let Some((owner_principal, session)) = state.get_session(&sess_id).await else {
        tracing::debug!("WHIP ICE unknown session {sess_id}");
        return StatusCode::NOT_FOUND;
    };

    // Verify the caller owns this session.
    let caller = principal_key_from_headers(&headers);
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
    let candidate_line = candidate_str
        .strip_prefix("a=")
        .unwrap_or(candidate_str);

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
#[tracing::instrument(name = "whip.terminate", skip_all, fields(sess_id))]
pub async fn whip_terminate(
    State(state): State<AppState>,
    Path(sess_id): Path<String>,
    headers: HeaderMap,
) -> StatusCode {
    // Verify ownership before removing — peek first.
    let Some((owner_principal, _)) = state.get_session(&sess_id).await else {
        tracing::debug!("WHIP terminate: unknown session {sess_id}");
        return StatusCode::NOT_FOUND;
    };

    let caller = principal_key_from_headers(&headers);
    if caller != owner_principal {
        tracing::warn!("WHIP terminate sess={sess_id} principal mismatch");
        return StatusCode::FORBIDDEN;
    }

    match state.remove_session(&sess_id).await {
        Some((_principal, session)) => {
            state.pipeline_metrics().inc_sessions_removed();
            tracing::info!("WHIP session terminated sess_id={sess_id}");
            match Arc::try_unwrap(session) {
                Ok(s) => s.terminate(),
                Err(_arc) => {}
            }
            StatusCode::OK
        }
        None => {
            tracing::debug!("WHIP terminate: unknown session {sess_id}");
            StatusCode::NOT_FOUND
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt update handler
// ---------------------------------------------------------------------------

/// Request body for `PATCH /v1/stream/whip/{sess_id}/prompt`.
#[derive(Debug, Deserialize)]
pub struct UpdatePromptRequest {
    pub prompt: String,
    /// Optional JSON schema for guided/structured VLM output.
    ///
    /// When present, the schema is passed as `guided_json` to VLM inference
    /// requests and `max_tokens` is bumped to 1024 to accommodate structured
    /// output.  Set to `null` or omit to revert to free-text inference.
    pub output_schema: Option<String>,
}

#[derive(Debug, Serialize)]
struct UpdatePromptResponse {
    session_id: String,
    prompt: String,
    output_schema: Option<String>,
}

/// `PATCH /v1/stream/whip/{sess_id}/prompt`
///
/// Replaces the VLM analysis prompt for a running WebRTC session and
/// optionally sets a JSON schema for structured output.
/// The new prompt and schema are used by VLM workers on the next keyframe.
///
/// Body: `{ "prompt": "new prompt text", "output_schema": "{...}" }`
///
/// Response:
/// - `200 OK` with `{ "session_id": "...", "prompt": "...", "output_schema": ... }`
/// - `404 Not Found` — unknown session
/// - `403 Forbidden` — caller is not the session owner
#[tracing::instrument(name = "whip.update_prompt", skip_all, fields(sess_id))]
pub async fn whip_update_prompt(
    State(state): State<AppState>,
    Path(sess_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<UpdatePromptRequest>,
) -> Response {
    let Some((owner_principal, session)) = state.get_session(&sess_id).await else {
        tracing::debug!("WHIP update_prompt: unknown session {sess_id}");
        return StatusCode::NOT_FOUND.into_response();
    };

    let caller = principal_key_from_headers(&headers);
    if caller != owner_principal {
        tracing::warn!("WHIP update_prompt sess={sess_id} principal mismatch");
        return StatusCode::FORBIDDEN.into_response();
    }

    session.update_prompt(body.prompt.clone());
    session.update_guided_json(body.output_schema.clone());
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
    use super::{hex_char, new_session_id};

    #[test]
    fn session_id_has_sess_prefix() {
        let id = new_session_id();
        assert!(id.starts_with("sess-"), "expected 'sess-' prefix, got: {id}");
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
}

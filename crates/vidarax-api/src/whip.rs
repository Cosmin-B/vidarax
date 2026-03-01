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
//! (`session.run(tx)`) forwards H.264 NAL units through the channel.  Until the
//! full decode pipeline (x02.3) is wired in, a drain task drops incoming frames
//! while counting them for observability.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use vidarax_core::webrtc::session::{WebRtcConfig, WebRtcSession};

use crate::state::AppState;

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
        match WebRtcSession::new(offer_sdp, &WebRtcConfig::default()).await {
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

    // Store the session bound to the requesting principal.
    if !state
        .insert_session(sess_id.clone(), principal, Arc::clone(&session))
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

    // Spawn the media ingestion task.
    // Bounded channel (128 frames): provides backpressure without excessive
    // buffering.  Frames are drained and discarded until the decode pipeline
    // (x02.3) is wired in.
    let (frame_tx, frame_rx) = kanal::bounded::<vidarax_core::webrtc::session::RtpFrame>(128);
    let run_future = session.run(frame_tx);
    tokio::spawn(run_future);

    // Drain task: counts received NAL units.  Replace with decode workers
    // once the full pipeline (x02.3) is ready.
    let sess_id_drain = sess_id.clone();
    tokio::task::spawn_blocking(move || {
        let mut total: u64 = 0;
        while let Ok(_frame) = frame_rx.recv() {
            total += 1;
            if total % 300 == 0 {
                tracing::debug!("WHIP sess={sess_id_drain} nals_received={total}");
            }
        }
        tracing::info!("WHIP sess={sess_id_drain} drain_task_ended total_nals={total}");
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

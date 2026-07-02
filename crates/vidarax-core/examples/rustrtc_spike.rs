//! Spike: prove rustrtc can receive a WebRTC stream and deliver H.264 NAL bytes.
//!
//! Run:
//!   cargo run --example rustrtc_spike -p vidarax-core
//!
//! Then open `examples/spike_test.html` in a browser (served from any local HTTP server,
//! e.g. `python3 -m http.server 8090 --directory crates/vidarax-core/examples`).
//!
//! Click "Start Screen Share", copy the SDP offer that appears, then click "Send Offer".
//! The spike will print SDP answer to stdout, first-32-byte dumps of each NAL unit, and
//! a running frame count every 30 NAL units received.
//!
//! API surface confirmed from source:
//!   - `PeerConnection::new(RtcConfiguration::default())` — creates a peer connection.
//!   - `pc.set_remote_description(SessionDescription)` — sets the browser offer.
//!   - `pc.create_answer() / pc.set_local_description()` — produces the answer SDP.
//!   - `pc.subscribe_ice_candidates()` — watch channel; wait for at least one candidate.
//!   - `pc.recv().await` → `PeerConnectionEvent::Track(transceiver)`.
//!   - `transceiver.receiver().track()` → `Arc<dyn MediaStreamTrack>`.
//!   - `track.recv().await` → `MediaSample::Video(VideoFrame)`.
//!   - `VideoFrame.data` → raw H.264 NAL payload bytes (`bytes::Bytes`).
//!   - `VideoFrame.is_last_packet` → true on RTP marker bit (last packet of a frame).
//!   - `VideoFrame.payload_type` → negotiated RTP payload type (typically 96 = H.264).
//!   - The library uses `H264Depacketizer` internally: FU-A fragments are reassembled
//!     before delivery, STAP-A is split into individual NAL units.
//!
//! Verdict: PASS — raw H.264 NAL bytes are directly accessible via `VideoFrame.data`.
//! The bytes arrive depacketized (single NAL per VideoFrame), ready for openh264 / NVDEC.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use rustrtc::{
    media::{MediaKind, MediaSample, MediaStreamTrack},
    peer_connection::{PeerConnection, PeerConnectionEvent},
    RtcConfiguration, SdpType, SessionDescription,
};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Shared state: hold the single PeerConnection so the ICE-candidate endpoint
// can find it by session ID.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    session: Arc<RwLock<Option<PeerConnection>>>,
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// Minimal index — just enough to confirm the server is up.
async fn handle_index() -> impl IntoResponse {
    (
        StatusCode::OK,
        "rustrtc spike — POST /offer with {\"sdp\": \"...\", \"type\": \"offer\"}\n",
    )
}

/// WHIP-style SDP exchange: receive a JSON offer, return a JSON answer.
///
/// Expected request body:
/// ```json
/// { "type": "offer", "sdp": "<browser SDP offer string>" }
/// ```
///
/// Response:
/// ```json
/// { "type": "answer", "sdp": "<rustrtc SDP answer string>" }
/// ```
async fn handle_offer(
    State(state): State<AppState>,
    body: axum::extract::Json<serde_json::Value>,
) -> impl IntoResponse {
    let sdp_str = match body.get("sdp").and_then(|v| v.as_str()) {
        Some(s) => s.to_owned(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": "missing 'sdp' field"})),
            );
        }
    };

    // --- Create PeerConnection ---
    let config = RtcConfiguration::default();
    let pc = PeerConnection::new(config);

    // --- Parse and apply the browser offer ---
    let offer = match SessionDescription::parse(SdpType::Offer, &sdp_str) {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("SDP parse error: {e}");
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": format!("SDP parse error: {e}")})),
            );
        }
    };

    if let Err(e) = pc.set_remote_description(offer).await {
        tracing::warn!("set_remote_description: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": format!("set_remote_description: {e}")})),
        );
    }

    // --- Wait up to 3 s for the first local ICE candidate ---
    {
        let mut ice_rx = pc.subscribe_ice_candidates();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(3), ice_rx.recv()).await;
    }

    // --- Build and set the answer ---
    let answer = match pc.create_answer().await {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("create_answer: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": format!("create_answer: {e}")})),
            );
        }
    };

    if let Err(e) = pc.set_local_description(answer) {
        tracing::warn!("set_local_description: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": format!("set_local_description: {e}")})),
        );
    }

    let local_desc = pc.local_description().expect("just set local description");
    let answer_sdp = local_desc.to_sdp_string();

    tracing::info!(
        "Answer SDP ({} candidates):\n{}",
        answer_sdp.matches("a=candidate:").count(),
        answer_sdp
    );

    // --- Spawn the track-ingestion task ---
    let nal_counter: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let nal_counter_clone = Arc::clone(&nal_counter);
    let pc_clone = pc.clone();

    tokio::spawn(async move {
        ingest_tracks(pc_clone, nal_counter_clone).await;
    });

    // --- Persist the PC so /ice can reach it ---
    {
        let mut slot = state.session.write().await;
        *slot = Some(pc);
    }

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "type": "answer",
            "sdp": answer_sdp,
        })),
    )
}

/// Trickle ICE: accept a candidate line from the browser and add it.
///
/// Body: `{ "candidate": "candidate:..." }`
async fn handle_ice(
    State(state): State<AppState>,
    body: axum::extract::Json<serde_json::Value>,
) -> impl IntoResponse {
    let candidate_str = match body.get("candidate").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_owned(),
        _ => return StatusCode::OK, // end-of-candidates signal
    };

    let slot = state.session.read().await;
    if let Some(pc) = slot.as_ref() {
        match rustrtc::transports::ice::IceCandidate::from_sdp(&candidate_str) {
            Ok(c) => {
                if let Err(e) = pc.add_ice_candidate(c) {
                    tracing::warn!("add_ice_candidate: {e}");
                }
            }
            Err(e) => tracing::warn!("parse ICE candidate: {e}"),
        }
    }

    StatusCode::OK
}

// ---------------------------------------------------------------------------
// Media ingestion loop
// ---------------------------------------------------------------------------

/// Poll `PeerConnectionEvent`s; for every video track, spawn a receive loop
/// that prints NAL metadata and counts frames.
async fn ingest_tracks(pc: PeerConnection, nal_counter: Arc<AtomicU64>) {
    while let Some(event) = pc.recv().await {
        match event {
            PeerConnectionEvent::Track(transceiver) => {
                let receiver = match transceiver.receiver() {
                    Some(r) => r,
                    None => continue,
                };
                let track = receiver.track();

                if track.kind() != MediaKind::Video {
                    tracing::info!("audio track received (skipping)");
                    continue;
                }

                tracing::info!(
                    "video track received  id={}  kind={:?}",
                    track.id(),
                    track.kind()
                );
                println!("[spike] video track connected  id={}", track.id());

                let counter = Arc::clone(&nal_counter);
                tokio::spawn(async move {
                    receive_video(track, counter).await;
                });
            }
            PeerConnectionEvent::DataChannel(_) => {}
        }
    }
    tracing::info!("PeerConnection event loop ended");
}

/// Receive loop: pull VideoFrames from the track, print NAL metadata.
async fn receive_video(
    track: Arc<dyn rustrtc::media::MediaStreamTrack>,
    nal_counter: Arc<AtomicU64>,
) {
    loop {
        match track.recv().await {
            Ok(MediaSample::Video(frame)) => {
                let n = nal_counter.fetch_add(1, Ordering::Relaxed) + 1;

                // Print first 32 bytes of this NAL so we can see the H.264 start code / header.
                let preview_len = frame.data.len().min(32);
                let preview = &frame.data[..preview_len];

                // Every NAL: log basic stats.
                if n <= 5 || n % 30 == 0 {
                    let nal_type = frame.data.first().map(|b| b & 0x1F).unwrap_or(0);
                    println!(
                        "[spike] NAL #{n:>6}  pts={:>10}  bytes={:>6}  nal_type={nal_type:>2}  \
                         is_last={:>5}  pt={:?}  preview={preview:02X?}",
                        frame.rtp_timestamp,
                        frame.data.len(),
                        frame.is_last_packet,
                        frame.payload_type,
                    );
                }

                // Every 30 NAL units, print a summary.
                if n % 30 == 0 {
                    println!("[spike] --- {n} NAL units received so far ---");
                }
            }
            Ok(MediaSample::Audio(_)) => {
                // Should not happen on a video track, but handle gracefully.
            }
            Err(e) => {
                println!("[spike] track ended: {e:?}");
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // rustls needs a crypto provider installed before any TLS handshake.
    rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider()).ok();

    tracing_subscriber::fmt()
        .with_env_filter("info,rustrtc=debug")
        .init();

    let state = AppState {
        session: Arc::new(RwLock::new(None)),
    };

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/offer", post(handle_offer))
        .route("/ice", post(handle_ice))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8765));
    println!("[spike] listening on http://{addr}");
    println!("[spike] open examples/spike_test.html, click Start Screen Share, then Send Offer");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

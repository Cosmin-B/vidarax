//! WebRTC session wrapper — rustrtc peer connection for inbound video streams.
//!
//! [`WebRtcSession`] manages the full lifecycle of a single WebRTC peer
//! connection: SDP offer/answer negotiation, trickle ICE, and media ingestion.
//! Video payload bytes are forwarded through a [`kanal`] channel to the
//! downstream decode workers. The answer advertises one selected video codec
//! from the offer, and the same codec drives depacketization and decode.
//!
//! # Example
//!
//! ```ignore
//! use vidarax_core::webrtc::session::{WebRtcConfig, WebRtcSession};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialise the TLS crypto provider once at process startup.
//!     rustls::crypto::CryptoProvider::install_default(
//!         rustls::crypto::ring::default_provider(),
//!     ).ok();
//!
//!     let offer_sdp = "v=0\r\n..."; // browser SDP offer
//!     let (session, answer) = WebRtcSession::new(offer_sdp, &WebRtcConfig::default())
//!         .await
//!         .unwrap();
//!
//!     let (frame_tx, frame_rx) = kanal::bounded(128);
//!     let metrics = std::sync::Arc::new(vidarax_core::metrics::PipelineMetrics::new());
//!     tokio::spawn(async move { session.run(frame_tx, metrics).await });
//!
//!     // consume frames on another thread:
//!     std::thread::spawn(move || {
//!         while let Ok(frame) = frame_rx.recv() {
//!             println!("seq={} pts={}ms payload={}", frame.seq, frame.pts_ms, frame.nals.len());
//!         }
//!     });
//! }
//! ```

use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    Arc,
};

pub use rustrtc::peer_connection::PeerConnectionState;
use rustrtc::{
    media::{MediaKind, MediaSample, MediaStreamTrack},
    peer_connection::{PeerConnection, PeerConnectionEvent},
    IceServer, RtcConfiguration, SdpType, SessionDescription,
};

use crate::crop::CropRegion;
use crate::gate::GateConfig;
use crate::metrics::PipelineMetrics;
use crate::tiered_vlm::TieredVlmConfig;
use crate::webrtc::decode::{
    count_audio_media_sections, count_video_media_sections, select_answer_video_codec_for_offer,
    VideoCodec,
};
use crate::webrtc::recycle::{RecycledBytes, VecPool};
use crate::webrtc::runtime::{
    PipelineGeneration, SessionCommand, SessionControl, SessionControlError,
};
use crate::zone::RestrictedZonePolicy;

/// Annex B start code prepended to every H.264 or H.265 NAL unit.
///
/// rustrtc delivers H.264 NAL payloads **without** start codes; openh264 and
/// ffmpeg expect them prepended.  H.265 / HEVC is wrapped the same way for
/// ffmpeg sidecar input once depacketized. VP8 payloads are passed through
/// unchanged.
const ANNEX_B_START: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
pub const RTP_FRAME_QUEUE_CAPACITY: usize = 128;
/// Maximum compressed access-unit payload admitted into the owned RTP queue.
/// This turns the queue's item bound into a byte bound as well. Oversized
/// frames are shed before the pipeline-owned copy is made.
pub const MAX_RTP_ACCESS_UNIT_BYTES: usize = 2 * 1024 * 1024;

pub fn rtp_nal_pool_slots(decode_workers: usize) -> usize {
    RTP_FRAME_QUEUE_CAPACITY + crate::webrtc::workers::per_stream_decode_workers(decode_workers) + 1
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single video access unit ready for the decode pipeline.
///
/// For H.264 and H.265 / HEVC, `nals` always begins with the 4-byte Annex B
/// start code `00 00 00 01` followed by the raw NAL data.
/// For VP8, `nals` contains the raw VP8 bitstream payload exactly as
/// delivered by rustrtc (no framing added).
///
/// The `seq` counter is per-session and monotonically increasing — it is
/// NOT the RTP sequence number.
#[derive(Debug, Clone)]
pub struct RtpFrame {
    /// Video payload bytes.
    ///
    /// - H.264 and H.265 / HEVC: Annex B encoded (starts with `00 00 00 01`).
    /// - VP8: raw bitstream payload.
    pub nals: RecycledBytes,
    /// Presentation timestamp derived from the 90 kHz RTP clock (in ms).
    pub pts_ms: u64,
    /// Per-session monotonically increasing sequence number.
    pub seq: u64,
    /// Codec of the video track this frame originated from.
    pub codec: VideoCodec,
}

/// TURN server credentials for ICE relay negotiation.
#[derive(Debug, Clone, Default)]
pub struct TurnServer {
    /// TURN server URL, e.g. `"turn:turn.example.com:3478"`.
    pub url: String,
    /// TURN username.
    pub username: String,
    /// TURN credential (shared secret or password).
    pub credential: String,
}

/// Configuration for a [`WebRtcSession`].
#[derive(Debug, Clone)]
pub struct WebRtcConfig {
    /// STUN server URIs used during ICE gathering.
    ///
    /// Defaults to `["stun:stun.l.google.com:19302"]`.
    pub stun_servers: Vec<String>,
    /// Optional TURN relay servers for NAT traversal.
    pub turn_servers: Vec<TurnServer>,
    /// Maximum VLM output tokens per second for this session (backpressure).
    pub max_output_tokens_per_second: u32,
    /// Requested decode worker threads per session.
    ///
    /// One ordered media stream must be decoded by exactly one stateful decoder.
    /// Values above 1 are accepted for API compatibility but clamped at worker
    /// spawn time; decode parallelism is across sessions, not within a stream.
    pub decode_workers: usize,
    /// Number of analysis worker threads per session.
    pub analysis_workers: usize,
    /// Requested VLM worker threads per session.
    ///
    /// Keyframe analysis carries sequential dedup and temporal context, so
    /// values above 1 are accepted for API compatibility but clamped per stream.
    pub vlm_workers: usize,
    /// Target decode width in pixels for the software or hardware decoder.
    pub decode_width: u32,
    /// Target decode height in pixels.
    pub decode_height: u32,
    /// Whether the decoder may use a hardware backend when one is available.
    pub gpu_available: bool,
    /// Gate-engine thresholds the analysis workers apply to each frame.
    pub gate_config: GateConfig,
    /// Perceptual-hash Hamming-distance threshold for treating frames as the same screen.
    pub loop_hamming_threshold: u32,
    /// Repeat count within the window that marks a stream as looping.
    pub loop_repeat_threshold: usize,
    /// Resolved tiered-VLM routing (first-pass model, optional escalation
    /// model, confidence threshold) that every WHIP session on this config
    /// should use for keyframe inference.
    ///
    /// This is resolved once, at startup, from whatever built the enclosing
    /// `ServerConfig` (see `vidarax-api`'s `build_webrtc_config`), and then
    /// cloned per session. A session must never re-derive this from the
    /// process environment: doing so would let a caller who built a
    /// `ServerConfig` programmatically (not from env) have their tiering
    /// choice silently ignored in favor of whatever the environment happens
    /// to hold when a session starts.
    pub vlm_tiering: TieredVlmConfig,
    /// Optional region of interest applied to every decoded frame before the
    /// gate and JPEG encoder see it, so only that part of the screen is
    /// analyzed. Session default; a per-attach crop on the stream request
    /// overrides it. `None` analyzes the whole frame.
    pub crop: Option<CropRegion>,
    /// Optional device-level activity policy used when an attach request does
    /// not replace it.
    pub restricted_zone: Option<Arc<RestrictedZonePolicy>>,
}

/// Failure from [`WebRtcSession::new`].
#[derive(Debug)]
pub enum WebRtcSetupError {
    /// The offer cannot be served as-is: it advertises video with no
    /// live-serveable codec, or it has more than one video m-section.
    /// A client error — the peer must change its offer.
    UnsupportedVideo(String),
    /// SDP parse, ICE, or answer-generation failure inside the stack.
    Negotiation(String),
}

impl std::fmt::Display for WebRtcSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebRtcSetupError::UnsupportedVideo(m) | WebRtcSetupError::Negotiation(m) => {
                f.write_str(m)
            }
        }
    }
}

impl std::error::Error for WebRtcSetupError {}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            max_output_tokens_per_second: 128,
            decode_workers: 1,
            analysis_workers: 1,
            vlm_workers: 1,
            decode_width: 1920,
            decode_height: 1080,
            gpu_available: false,
            gate_config: GateConfig::default(),
            loop_hamming_threshold: 6,
            loop_repeat_threshold: 3,
            vlm_tiering: TieredVlmConfig::default(),
            crop: None,
            restricted_zone: None,
        }
    }
}

/// Durable outcome chosen for a live session when its peer is reclaimed.
///
/// The numeric order is intentional: an explicit run delete overrides every
/// other outcome, and an explicit REST stop overrides a concurrent graceful
/// WHIP resource termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SessionCloseDisposition {
    Fault = 0,
    Complete = 1,
    Stop = 2,
    Delete = 3,
}

/// An active WebRTC peer connection managing a single inbound media stream.
///
/// # Thread-safety
///
/// `WebRtcSession` is `Send` and `Sync` (the inner [`PeerConnection`] is
/// reference-counted).  Multiple handles may be held simultaneously — the
/// connection closes when all handles are dropped.
pub struct WebRtcSession {
    pc: PeerConnection,
    /// Why this session is being closed. Monotonic ordering lets an explicit
    /// stop or delete override a graceful transport close before the winning
    /// reclaimer commits the terminal event.
    close_disposition: AtomicU8,
    /// Reclaim decision ownership. See try_claim_reclaim.
    reclaim_claimed: AtomicBool,
    /// Configuration used when the worker generation starts. Live updates move
    /// through `control`; workers own the accepted values after startup.
    initial_prompt: Arc<str>,
    initial_guided_json: Option<Arc<str>>,
    control: SessionControl,
    /// Token output rate cap (tokens/s) for backpressure in VLM workers.
    pub max_output_tokens_per_second: u32,
    /// Effective region of interest for this session's decode workers. Starts
    /// from the server default and can be overridden per attach. `None` analyzes
    /// the whole frame.
    pub crop: Option<CropRegion>,
    /// Optional generation-static restricted-zone activity policy. The API
    /// validates this before worker startup; replacing it requires a new
    /// generation so in-flight frames cannot cross policy versions.
    pub restricted_zone: Option<Arc<RestrictedZonePolicy>>,
    /// Video codec negotiated from the SDP offer.
    pub codec: VideoCodec,
    rtp_nal_pool_slots: usize,
    /// Monotonic shutdown latch owned by the session. rustrtc's peer-state
    /// watch is not monotonic toward a terminal state: after close() publishes
    /// Closed, a DTLS handshake that just won a scheduling race can still
    /// publish Connected over it, so run() cannot rely on that watch alone to
    /// latch shutdown. close() raises this before closing the peer, and run()
    /// checks it first each iteration, so once set it can neither be missed nor
    /// overwritten.
    shutdown: tokio::sync::watch::Sender<bool>,
    /// Holds a receiver open so `shutdown` retains its value even before run()
    /// subscribes. Without a live receiver a `send(true)` would not be stored,
    /// and a close() that precedes run()'s first poll would be lost.
    _shutdown_rx: tokio::sync::watch::Receiver<bool>,
    #[cfg(any(test, debug_assertions))]
    close_calls: AtomicU64,
}

/// True when the session must stop ingesting: either the session-owned
/// monotonic shutdown latch is set, or rustrtc reports a terminal peer state.
/// Split out from run()'s loop so the stop decision is unit-testable without
/// driving a live peer connection.
fn should_stop(shutdown: bool, peer: PeerConnectionState) -> bool {
    shutdown
        || matches!(
            peer,
            PeerConnectionState::Closed
                | PeerConnectionState::Failed
                | PeerConnectionState::Disconnected
        )
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl WebRtcSession {
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn new_for_tests() -> Self {
        Self::new_for_tests_with_generation(PipelineGeneration::INITIAL).0
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn new_for_tests_with_generation(
        generation: PipelineGeneration,
    ) -> (Self, tokio::sync::mpsc::Receiver<SessionCommand>) {
        let (shutdown, _shutdown_rx) = tokio::sync::watch::channel(false);
        let (control, commands) = SessionControl::channel(generation);
        (
            Self {
                pc: PeerConnection::new(Default::default()),
                close_disposition: AtomicU8::new(0),
                reclaim_claimed: AtomicBool::new(false),
                initial_prompt: Arc::from(""),
                initial_guided_json: None,
                control,
                max_output_tokens_per_second: 128,
                crop: None,
                restricted_zone: None,
                codec: VideoCodec::H264,
                rtp_nal_pool_slots: rtp_nal_pool_slots(1),
                shutdown,
                _shutdown_rx,
                #[cfg(any(test, debug_assertions))]
                close_calls: AtomicU64::new(0),
            },
            commands,
        )
    }

    /// Create a new session from a browser SDP offer.
    ///
    /// 1. Parses `offer_sdp` and applies it as the remote description.
    /// 2. Waits up to 3 s for the first local ICE candidate (so host
    ///    candidates are embedded in the answer).
    /// 3. Creates and applies the local answer.
    ///
    /// The answer is pinned to one live-serveable video codec selected from
    /// the offer: H.264, H.265, or VP8 when built. The selected codec also
    /// drives the depacketizer factory and decode routing.
    ///
    /// Returns `(session, answer_sdp)`.  The caller must send `answer_sdp`
    /// back to the browser to complete the signalling exchange.
    ///
    /// # Errors
    ///
    /// Returns [`WebRtcSetupError::UnsupportedVideo`] for offers whose video
    /// cannot be served: no live-serveable codec, or multiple video m-sections.
    /// Returns [`WebRtcSetupError::Negotiation`] for SDP parse, ICE, or
    /// answer-generation failures.
    ///
    /// # Prerequisites
    ///
    /// The TLS crypto provider must be installed before calling this:
    /// ```ignore
    /// rustls::crypto::CryptoProvider::install_default(
    ///     rustls::crypto::ring::default_provider()
    /// ).ok();
    /// ```
    pub async fn new(
        offer_sdp: &str,
        config: &WebRtcConfig,
    ) -> Result<(Self, String), WebRtcSetupError> {
        let (session, answer, _commands) =
            Self::new_with_generation(offer_sdp, config, PipelineGeneration::INITIAL).await?;
        Ok((session, answer))
    }

    /// Create a session and the single command receiver for its live pipeline.
    /// The receiver must be moved into exactly one worker generation.
    pub async fn new_with_generation(
        offer_sdp: &str,
        config: &WebRtcConfig,
        generation: PipelineGeneration,
    ) -> Result<(Self, String, tokio::sync::mpsc::Receiver<SessionCommand>), WebRtcSetupError> {
        // Select before handing SDP to rustrtc so the answer, depacketizer,
        // and decode routing use the same codec.
        let selected = select_answer_video_codec_for_offer(offer_sdp);
        let video_sections = count_video_media_sections(offer_sdp);
        // A single global answer video capability is installed below, so more than
        // one video m-section cannot be answered correctly (rustrtc would apply that
        // one capability to a section that did not offer that codec). WHIP ingest is
        // single-stream; reject multi-video offers rather than emit an invalid answer.
        if video_sections > 1 {
            return Err(WebRtcSetupError::UnsupportedVideo(
                "offer has multiple video m-sections, which are not supported".to_string(),
            ));
        }
        // A video m-section we cannot serve (an unsupported/unrecognized codec such as
        // AV1, a DON-signaling H.265, or VP8 without the `vp8` feature) must be
        // rejected: otherwise `new` would build a session whose answerer video receiver
        // uses rustrtc's default H.264 depacketizer, and a peer that sent video RTP
        // regardless would be depacketized and decode-routed as the wrong codec.
        if selected.is_none() && video_sections > 0 {
            return Err(WebRtcSetupError::UnsupportedVideo(
                "offer advertised video but no live-serveable codec".to_string(),
            ));
        }
        let codec = match selected {
            Some(sel) => sel.codec,
            None => VideoCodec::from_sdp(offer_sdp),
        };

        let mut rtc_config = RtcConfiguration::default();
        for url in &config.stun_servers {
            rtc_config
                .ice_servers
                .push(IceServer::new(vec![url.clone()]));
        }
        for turn in &config.turn_servers {
            rtc_config.ice_servers.push(
                IceServer::new(vec![turn.url.clone()])
                    .with_credential(turn.username.clone(), turn.credential.clone()),
            );
        }
        if let Some(sel) = selected {
            // rustrtc cannot carry H.264 fmtp here. Its depacketizer accepts
            // single-NAL, STAP-A, and FU-A, then we add Annex B before decode.
            rtc_config.media_capabilities = Some(rustrtc::config::MediaCapabilities {
                video: vec![rustrtc::config::VideoCapability {
                    payload_type: sel.payload_type,
                    codec_name: codec.rtpmap_name().to_string(),
                    clock_rate: sel.clock_rate,
                    rtcp_fbs: rustrtc::config::VideoCapability::default().rtcp_fbs,
                }],
                // Keep audio as one default capability to match the prior None
                // answer and avoid advertising a codec missing from the offer.
                audio: vec![rustrtc::config::AudioCapability::default()],
                application: rustrtc::config::MediaCapabilities::default().application,
            });
        }
        rtc_config.depacketizer_strategy = rustrtc::config::DepacketizerStrategy {
            factory: Arc::new(crate::webrtc::depacketize::VidaraxDepacketizerFactory::new(
                codec,
            )),
        };
        let pc = PeerConnection::new(rtc_config);

        // rustrtc's answerer path builds the inbound receiver from the remote
        // offer without attaching our depacketizer factory, so it would fall back
        // to rustrtc's default H.264 depacketizer and mis-parse H.265 (and VP8)
        // RTP. Pre-creating a recvonly video transceiver makes the offer's video
        // m-section reuse this receiver, which `add_transceiver` builds with the
        // configured factory. Only add it when a video codec was selected, so an
        // audio-only offer does not gain a spurious unmatched video m-line.
        if selected.is_some() {
            pc.add_transceiver(
                rustrtc::MediaKind::Video,
                rustrtc::TransceiverDirection::RecvOnly,
            );
        }

        // rustrtc's answerer matches each offered audio m-section to a local
        // audio transceiver. With none, it falls through to an implicit
        // transceiver that mirrors the offered `sendonly` back as `sendonly` —
        // an answer RFC 3264 disallows for a sendonly offer. Pre-create one
        // recvonly audio transceiver per offered audio section so each is
        // matched and answered `recvonly`. The count is zero for an audio-less
        // offer, so none is added. vidarax does not consume audio: the default
        // audio depacketizer handles any audio RTP and `run` forwards only
        // video frames.
        for _ in 0..count_audio_media_sections(offer_sdp) {
            pc.add_transceiver(
                rustrtc::MediaKind::Audio,
                rustrtc::TransceiverDirection::RecvOnly,
            );
        }

        let offer = SessionDescription::parse(SdpType::Offer, offer_sdp)
            .map_err(|e| WebRtcSetupError::Negotiation(format!("SDP parse: {e}")))?;

        pc.set_remote_description(offer)
            .await
            .map_err(|e| WebRtcSetupError::Negotiation(format!("set_remote_description: {e}")))?;

        // Wait briefly for the first local ICE candidate so that the answer
        // SDP contains usable host candidates.  Trickle ICE will add more.
        {
            let mut ice_rx = pc.subscribe_ice_candidates();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(3), ice_rx.recv()).await;
        }

        let answer = pc
            .create_answer()
            .await
            .map_err(|e| WebRtcSetupError::Negotiation(format!("create_answer: {e}")))?;

        pc.set_local_description(answer)
            .map_err(|e| WebRtcSetupError::Negotiation(format!("set_local_description: {e}")))?;

        // set_local_description just succeeded, so rustrtc should hand back the
        // description here. It is the dependency's state to report, though, so a
        // missing one is surfaced as a negotiation error rather than aborting the
        // whole process over another crate's postcondition.
        let local = pc.local_description().ok_or_else(|| {
            WebRtcSetupError::Negotiation(
                "local description missing after set_local_description".to_string(),
            )
        })?;
        let answer_sdp = local.to_sdp_string();

        let (shutdown, _shutdown_rx) = tokio::sync::watch::channel(false);
        let (control, commands) = SessionControl::channel(generation);
        Ok((
            Self {
                pc,
                close_disposition: AtomicU8::new(0),
                reclaim_claimed: AtomicBool::new(false),
                initial_prompt: Arc::from(""),
                initial_guided_json: None,
                control,
                max_output_tokens_per_second: config.max_output_tokens_per_second,
                crop: config.crop,
                restricted_zone: config.restricted_zone.clone(),
                codec,
                rtp_nal_pool_slots: rtp_nal_pool_slots(config.decode_workers),
                shutdown,
                _shutdown_rx,
                #[cfg(any(test, debug_assertions))]
                close_calls: AtomicU64::new(0),
            },
            answer_sdp,
            commands,
        ))
    }

    /// Forward a trickle ICE candidate received from the browser.
    ///
    /// `line` is the raw candidate attribute value, e.g.
    /// `"candidate:1 1 udp 2113937151 192.168.1.5 54321 typ host"`.
    /// The `a=` prefix must **not** be included.
    ///
    /// Empty or null end-of-candidates signals should be filtered by the
    /// caller before invoking this method.
    pub fn add_remote_candidate(&self, line: &str) -> Result<(), String> {
        let candidate = rustrtc::transports::ice::IceCandidate::from_sdp(line)
            .map_err(|e| format!("ICE candidate parse: {e}"))?;
        self.pc
            .add_ice_candidate(candidate)
            .map_err(|e| format!("add_ice_candidate: {e}"))
    }

    /// Drive the peer connection event loop, forwarding video access units
    /// through `frame_tx` until the connection closes or the sender is dropped.
    ///
    /// Returns an **owned** future (no lifetime ties to `self`) that can be
    /// directly passed to [`tokio::spawn`].  Internally clones the
    /// `PeerConnection` handle so the caller may store `WebRtcSession` in a
    /// shared structure while the task runs concurrently.
    ///
    /// - The codec selected for the answer is also used here. H.264 is
    ///   depacketized by rustrtc. VP8 is depacketized in-crate when built, and
    ///   H.265 by the in-crate HEVC RTP depacketizer. Annex B start codes
    ///   (`00 00 00 01`) are **prepended** to H.264 and H.265 / HEVC payloads
    ///   before sending. VP8 payloads are passed through.
    /// - Audio tracks are silently ignored.
    /// - For each video track a Tokio task is spawned; all share the same
    ///   atomic sequence counter.
    ///
    /// The `frame_tx` channel should have a capacity of at least 32 frames
    /// to absorb burst traffic without the track-receive tasks stalling.
    ///
    /// # Notes on blocking
    ///
    /// RTP ingress feeds a bounded, lossless, ordered queue into the single
    /// decoder. When decode/analysis is slower than ingress, the Tokio task
    /// awaits channel capacity and yields the runtime worker; sustained
    /// overload backpressures the WebRTC media layer, where jitter buffering,
    /// NACKs, and keyframe requests can handle real-time RTP loss without
    /// corrupting the stateful decoder.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (tx, rx) = kanal::bounded(128);
    /// let metrics = std::sync::Arc::new(vidarax_core::metrics::PipelineMetrics::new());
    /// tokio::spawn(session.run(tx, metrics));
    /// ```
    pub fn run(
        &self,
        frame_tx: kanal::Sender<RtpFrame>,
        metrics: Arc<PipelineMetrics>,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        // Clone the PeerConnection so the returned future owns everything it
        // needs and has no lifetime dependency on `self`.
        let pc = self.pc.clone();
        let seq_counter = Arc::new(AtomicU64::new(0));
        let codec = self.codec;
        let rtp_nal_pool_slots = self.rtp_nal_pool_slots;
        let control = self.control.clone();
        // Subscribe to the shutdown latch when run() is called, not at first
        // poll. A close() between this call and the first poll bumps the latch,
        // so the loop's top borrow sees it (and changed() is ready); a close()
        // that already happened is retained by the keepalive receiver and read
        // here as the current value.
        let mut shutdown = self.shutdown.subscribe();

        async move {
            // run() owns the per-track ingestion tasks and gives them a separate
            // monotonic stop signal. A local close() does not wake an in-flight
            // pc.recv() or track.recv(), so every blocking receive/send is raced
            // against this signal and each task is joined during teardown.
            let mut peer_state = pc.subscribe_peer_state();
            let mut track_tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();
            let (track_stop, _) = tokio::sync::watch::channel(false);

            loop {
                // Decide before waiting. peer_state is subscribed inside this
                // future, so a terminal state published before the first poll is
                // already-seen on that fresh receiver and its changed() never
                // fires; the borrow here is what catches it. The shutdown latch
                // is subscribed when run() is called and is monotonic, so a
                // close() can be neither lost nor overwritten by a late rustrtc
                // republish (which the peer-state watch alone permits). A signal
                // raised after this borrow makes the matching changed() ready in
                // the usual way.
                if should_stop(
                    *shutdown.borrow_and_update(),
                    *peer_state.borrow_and_update(),
                ) {
                    break;
                }

                tokio::select! {
                    // Err means the session (and its shutdown sender) is gone;
                    // a non-Err change re-checks the latch at the top.
                    res = shutdown.changed() => {
                        if res.is_err() {
                            break;
                        }
                    }
                    // Err means the peer-state sender was dropped, i.e. the
                    // connection is gone; a non-Err change re-checks at the top.
                    changed = peer_state.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                    event = pc.recv() => {
                        let Some(event) = event else { break };
                        match event {
                            PeerConnectionEvent::Track(transceiver) => {
                                let receiver = match transceiver.receiver() {
                                    Some(r) => r,
                                    None => continue,
                                };
                                let track = receiver.track();

                                if track.kind() != MediaKind::Video {
                                    continue;
                                }

                                let tx = frame_tx.clone();
                                let seq = Arc::clone(&seq_counter);
                                let metrics = Arc::clone(&metrics);
                                let mut stop = track_stop.subscribe();

                                track_tasks.push(tokio::spawn(async move {
                                    let nals_pool = VecPool::with_slots(rtp_nal_pool_slots);
                                    loop {
                                        let sample = tokio::select! {
                                            biased;
                                            changed = stop.changed() => {
                                                if changed.is_err() || *stop.borrow() {
                                                    break;
                                                }
                                                continue;
                                            }
                                            sample = track.recv() => sample,
                                        };

                                        match sample {
                                            Ok(MediaSample::Video(frame)) => {
                                                if frame.data.len() > MAX_RTP_ACCESS_UNIT_BYTES {
                                                    metrics.inc_frames_dropped();
                                                    tracing::warn!(
                                                        payload_bytes = frame.data.len(),
                                                        limit_bytes = MAX_RTP_ACCESS_UNIT_BYTES,
                                                        "dropping oversized RTP video access unit"
                                                    );
                                                    continue;
                                                }
                                                let nal_seq =
                                                    seq.fetch_add(1, Ordering::Relaxed);

                                                let nals = frame_payload_to_nals(
                                                    codec,
                                                    &frame.data,
                                                    &nals_pool,
                                                );

                                                // RTP timestamp is on a 90 kHz clock → ms.
                                                let pts_ms = frame.rtp_timestamp as u64 / 90;

                                                let rtp_frame = RtpFrame {
                                                    nals,
                                                    pts_ms,
                                                    seq: nal_seq,
                                                    codec,
                                                };

                                                let queued = tokio::select! {
                                                    biased;
                                                    changed = stop.changed() => {
                                                        if changed.is_err() || *stop.borrow() {
                                                            break;
                                                        }
                                                        continue;
                                                    }
                                                    queued = enqueue_rtp_frame_lossless(
                                                        &tx,
                                                        rtp_frame,
                                                        &metrics,
                                                    ) => queued,
                                                };
                                                if !queued {
                                                    break;
                                                }
                                            }
                                            Ok(MediaSample::Audio(_)) => {
                                                // Unexpected on a video track; ignore.
                                            }
                                            Err(_) => break,
                                        }
                                    }
                                }));
                            }
                            PeerConnectionEvent::DataChannel(_) => {
                                // Not used; ignore.
                            }
                        }
                    }
                }
            }

            // The connection is going away. Stop and join the track tasks so
            // each frame_tx clone drops in a known place. Together with
            // frame_tx dropping here, the decode worker downstream sees its
            // channel close and exits. An in-flight frame may be discarded at
            // this explicit teardown boundary; active-session sends remain
            // lossless and backpressured.
            track_stop.send_replace(true);
            for handle in track_tasks {
                let _ = handle.await;
            }
            // Mark the generation as stopping before this future drops its
            // final frame_tx. That makes the downstream channel closure a
            // clean shutdown signal rather than an unexpected worker exit.
            control.stop();
        }
    }

    /// Set the prompt before the session is shared or its workers are started.
    pub fn set_initial_prompt(&mut self, text: String) {
        self.initial_prompt = Arc::from(text);
    }

    pub fn initial_prompt(&self) -> Arc<str> {
        Arc::clone(&self.initial_prompt)
    }

    pub fn initial_guided_json(&self) -> Option<Arc<str>> {
        self.initial_guided_json.clone()
    }

    pub fn generation(&self) -> PipelineGeneration {
        self.control.generation()
    }

    /// Send a live configuration update and wait until the active generation
    /// has accepted it. A closed generation rejects the command.
    pub async fn update_config(
        &self,
        prompt: String,
        guided_json: Option<String>,
    ) -> Result<(), SessionControlError> {
        self.control
            .update_config(
                Arc::from(prompt),
                guided_json.map(|schema| Arc::from(schema.as_str())),
            )
            .await
    }

    pub fn stopping_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.control.stopping_flag()
    }

    /// Mark a normal WHIP resource termination as a completed run.
    pub fn mark_close_disposition_complete(&self) {
        self.close_disposition
            .fetch_max(SessionCloseDisposition::Complete as u8, Ordering::AcqRel);
    }

    /// Ask the reclaim of this session to keep the run's history (REST stop).
    pub fn mark_close_disposition_stop(&self) {
        self.close_disposition
            .fetch_max(SessionCloseDisposition::Stop as u8, Ordering::AcqRel);
    }

    /// Record that a delete requested this close. Overrides a stop mark, so a
    /// delete that lands after a stop still tombstones the run.
    pub fn mark_close_disposition_delete(&self) {
        self.close_disposition
            .fetch_max(SessionCloseDisposition::Delete as u8, Ordering::AcqRel);
    }

    /// Read by the reclaimer only after it owns the reclaim claim.
    pub fn close_disposition(&self) -> SessionCloseDisposition {
        match self.close_disposition.load(Ordering::Acquire) {
            1 => SessionCloseDisposition::Complete,
            2 => SessionCloseDisposition::Stop,
            3 => SessionCloseDisposition::Delete,
            _ => SessionCloseDisposition::Fault,
        }
    }

    /// Claim the right to decide this session's reclaim. Exactly one caller
    /// wins. The claim is released only when a tombstone append fails, so a
    /// watcher retry can claim again and finish the cleanup.
    pub fn try_claim_reclaim(&self) -> bool {
        self.reclaim_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn release_reclaim_claim(&self) {
        self.reclaim_claimed.store(false, Ordering::Release);
    }

    /// Subscribe to rustrtc peer state changes for lifecycle cleanup.
    pub fn subscribe_peer_state(&self) -> tokio::sync::watch::Receiver<PeerConnectionState> {
        self.pc.subscribe_peer_state()
    }

    /// Close the peer connection while shared session handles may still exist.
    pub fn close(&self) {
        #[cfg(any(test, debug_assertions))]
        self.close_calls.fetch_add(1, Ordering::Relaxed);
        // Raise the monotonic shutdown latch before closing the peer, so run()
        // observes it even if rustrtc republishes a non-terminal state over the
        // Closed this triggers.
        self.control.stop();
        let _ = self.shutdown.send(true);
        self.pc.close();
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn close_call_count_for_tests(&self) -> u64 {
        self.close_calls.load(Ordering::Relaxed)
    }

    /// Close the peer connection.
    ///
    /// After this call any active `run` task will exit on its next poll cycle
    /// as the underlying `PeerConnection` cleans up.
    pub fn terminate(self) {
        self.close();
        drop(self);
    }
}

async fn enqueue_rtp_frame_lossless(
    tx: &kanal::Sender<RtpFrame>,
    frame: RtpFrame,
    _metrics: &PipelineMetrics,
) -> bool {
    tx.as_async().send(frame).await.is_ok()
}

fn frame_payload_to_nals(codec: VideoCodec, payload: &[u8], pool: &VecPool) -> RecycledBytes {
    let mut nals = pool.acquire();
    match codec {
        VideoCodec::H264 | VideoCodec::H265 => {
            // ffmpeg h264/hevc demuxers expect Annex B start codes. H.264 is
            // depacketized by rustrtc; H.265 by the in-crate HEVC RTP depacketizer.
            nals.reserve(ANNEX_B_START.len() + payload.len());
            nals.extend_from_slice(&ANNEX_B_START);
            nals.extend_from_slice(payload);
        }
        VideoCodec::Vp8 => {
            nals.reserve(payload.len());
            nals.extend_from_slice(payload);
        }
    }
    pool.recycle(nals)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        enqueue_rtp_frame_lossless, frame_payload_to_nals, rtp_nal_pool_slots, RtpFrame,
        WebRtcConfig, WebRtcSession, WebRtcSetupError, ANNEX_B_START, RTP_FRAME_QUEUE_CAPACITY,
    };
    use crate::webrtc::decode::VideoCodec;
    use crate::webrtc::recycle::{RecycledBytes, VecPool};
    use std::sync::Arc;

    #[test]
    fn rtp_nal_pool_covers_queue_workers_and_blocked_sender() {
        assert_eq!(rtp_nal_pool_slots(2), RTP_FRAME_QUEUE_CAPACITY + 1 + 1);
        assert_eq!(rtp_nal_pool_slots(0), RTP_FRAME_QUEUE_CAPACITY + 1 + 1);
    }

    /// The WHIP answerer receive path must depacketize with the codec-selected
    /// in-crate depacketizer, not rustrtc's default H.264 one. Drives
    /// `WebRtcSession::new` with an H.265 offer, then feeds an HEVC Aggregation
    /// Packet through the accepted video receiver's depacketizer and asserts it
    /// is un-aggregated into an HEVC access unit. rustrtc's default H.264
    /// depacketizer would pass the raw packet bytes through unchanged.
    #[tokio::test]
    async fn answerer_video_receiver_uses_selected_depacketizer() {
        // create_answer generates a DTLS certificate, which needs a rustls
        // crypto provider. Idempotent; ignore the error if already installed.
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        // A minimal but complete WHIP-style offer advertising H.265 only.
        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0\r\n\
a=msid-semantic: WMS\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 H265/90000\r\n";

        let (session, _answer) = WebRtcSession::new(offer, &WebRtcConfig::default())
            .await
            .expect("WebRtcSession::new should accept the H.265 offer");

        // The pre-created recvonly video transceiver carries the receiver built
        // with our depacketizer factory.
        let video = session
            .pc
            .get_transceivers()
            .into_iter()
            .find(|t| t.kind() == rustrtc::MediaKind::Video)
            .expect("a video transceiver");
        let receiver = video.receiver().expect("a video receiver");
        let mut depacketizer = receiver
            .depacketizer_factory
            .create(rustrtc::media::MediaKind::Video);

        // HEVC Aggregation Packet (RFC 7798 nal_type 48) carrying two NALs:
        // VPS (type 32) = 0x40,0x01,0xaa and SPS (type 33) = 0x42,0x01,0xbb,
        // each prefixed with its 16-bit big-endian size (3).
        let ap = vec![
            0x60, 0x01, // AP payload header
            0x00, 0x03, 0x40, 0x01, 0xaa, // VPS NAL
            0x00, 0x03, 0x42, 0x01, 0xbb, // SPS NAL
        ];
        let mut header = rustrtc::rtp::RtpHeader::new(96, 1, 100, 12345);
        header.marker = true; // end of access unit -> emit a sample
        let packet = rustrtc::rtp::RtpPacket::new(header, ap);
        let addr = std::net::SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            1234,
        );

        let samples = depacketizer
            .push(packet, 90_000, addr, rustrtc::media::MediaKind::Video)
            .expect("push should not error");
        assert_eq!(samples.len(), 1, "one access unit expected");
        let data = match samples.into_iter().next().unwrap() {
            rustrtc::media::MediaSample::Video(frame) => frame.data.to_vec(),
            rustrtc::media::MediaSample::Audio(_) => panic!("expected a video sample"),
        };

        // H265Depacketizer un-aggregates the AP into the two NALs joined by an
        // Annex B start code. The default H.264 depacketizer would instead emit
        // the raw AP bytes [0x60,0x01,0x00,0x03,...] unchanged.
        assert_eq!(
            data,
            vec![0x40, 0x01, 0xaa, 0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0xbb],
        );
    }

    #[tokio::test]
    async fn answer_audio_section_is_recvonly_not_sendonly() {
        // create_answer needs a rustls crypto provider; idempotent install.
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        // A WHIP-style offer with audio (mid:0) and H.264 video (mid:1), both
        // a=sendonly (browser -> us).
        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0 1\r\n\
a=msid-semantic: WMS\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:111 opus/48000/2\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:1\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 H264/90000\r\n";

        let (_session, answer) = WebRtcSession::new(offer, &WebRtcConfig::default())
            .await
            .expect("video+audio offer should be accepted");

        // BUNDLE requires every offered m-line be answered, so the audio
        // m-section must remain. Extract its block and assert its direction is
        // recvonly. Before the audio transceiver existed, rustrtc mirrored the
        // offered sendonly back as sendonly here, an RFC 3264 violation.
        let audio_block = answer
            .split("m=")
            .find(|section| section.starts_with("audio"))
            .expect("answer retains the audio m-section");
        assert!(
            audio_block.contains("a=recvonly"),
            "audio m-section should be answered recvonly, got:\n{audio_block}"
        );
        assert!(
            !audio_block.contains("a=sendonly"),
            "audio m-section must not be answered sendonly, got:\n{audio_block}"
        );
    }

    #[tokio::test]
    async fn answer_multiple_audio_sections_all_recvonly() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        // Two audio sections (mid:0, mid:1) then H.264 video (mid:2), all sendonly.
        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0 1 2\r\n\
a=msid-semantic: WMS\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:111 opus/48000/2\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 110\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:1\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:110 opus/48000/2\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:2\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 H264/90000\r\n";

        let (_session, answer) = WebRtcSession::new(offer, &WebRtcConfig::default())
            .await
            .expect("two-audio + video offer should be accepted");

        // Both audio sections must be answered recvonly; no section may be
        // answered sendonly.
        let audio_blocks: Vec<&str> = answer
            .split("m=")
            .filter(|section| section.starts_with("audio"))
            .collect();
        assert_eq!(
            audio_blocks.len(),
            2,
            "answer should retain both audio m-sections"
        );
        for block in &audio_blocks {
            assert!(
                block.contains("a=recvonly"),
                "each audio m-section should be answered recvonly, got:\n{block}"
            );
            assert!(
                !block.contains("a=sendonly"),
                "no audio m-section may be answered sendonly, got:\n{block}"
            );
        }
    }

    #[tokio::test]
    async fn new_rejects_h265_offer_with_don() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0\r\n\
a=msid-semantic: WMS\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 H265/90000\r\n\
a=fmtp:96 sprop-max-don-diff=1\r\n";

        let err = match WebRtcSession::new(offer, &WebRtcConfig::default()).await {
            Ok(_) => panic!("DON-signaling H.265 should be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                WebRtcSetupError::UnsupportedVideo(ref m)
                    if m == "offer advertised video but no live-serveable codec"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[tokio::test]
    async fn new_rejects_av1_only_offer() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0\r\n\
a=msid-semantic: WMS\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 AV1/90000\r\n";

        let err = match WebRtcSession::new(offer, &WebRtcConfig::default()).await {
            Ok(_) => panic!("AV1-only video should be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                WebRtcSetupError::UnsupportedVideo(ref m)
                    if m == "offer advertised video but no live-serveable codec"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[tokio::test]
    async fn new_rejects_multiple_video_sections() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0 1\r\n\
a=msid-semantic: WMS\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 H264/90000\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 97\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:1\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:97 H264/90000\r\n";

        let err = match WebRtcSession::new(offer, &WebRtcConfig::default()).await {
            Ok(_) => panic!("multiple video m-sections should be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                WebRtcSetupError::UnsupportedVideo(ref m)
                    if m == "offer has multiple video m-sections, which are not supported"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[tokio::test]
    async fn new_rejects_audio_plus_av1_video() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0 1\r\n\
a=msid-semantic: WMS\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:111 opus/48000/2\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:1\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 AV1/90000\r\n";

        let err = match WebRtcSession::new(offer, &WebRtcConfig::default()).await {
            Ok(_) => panic!("audio plus AV1-only video should be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                WebRtcSetupError::UnsupportedVideo(ref m)
                    if m == "offer advertised video but no live-serveable codec"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[tokio::test]
    async fn new_accepts_audio_only_offer() {
        let _ = rustls::crypto::CryptoProvider::install_default(
            rustls::crypto::aws_lc_rs::default_provider(),
        );

        let offer = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0\r\n\
a=msid-semantic: WMS\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=sendonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:111 opus/48000/2\r\n";

        WebRtcSession::new(offer, &WebRtcConfig::default())
            .await
            .expect("audio-only offer should not be rejected");
    }

    #[test]
    fn rtp_frame_annex_b_layout_h264() {
        // Simulate what run() produces for H.264: start code + payload.
        let payload = vec![0x65, 0xB8, 0x00]; // IDR NAL header byte
        let pool = VecPool::with_slots(1);
        let nals = frame_payload_to_nals(VideoCodec::H264, &payload, &pool);

        assert_eq!(&nals[..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&nals[4..], &payload[..]);
    }

    #[test]
    fn rtp_frame_vp8_no_annex_b() {
        // VP8 payloads should be forwarded without Annex B start codes.
        let payload = RecycledBytes::from(vec![0x30, 0x01, 0x02, 0x03]); // synthetic VP8 bytes
        let frame = RtpFrame {
            nals: payload.clone(),
            pts_ms: 0,
            seq: 0,
            codec: VideoCodec::Vp8,
        };
        // No Annex B prefix — first byte should be the raw VP8 payload byte.
        assert_eq!(frame.nals[0], 0x30);
        assert_eq!(&frame.nals[..], &payload[..]);
    }

    #[test]
    fn frame_payload_to_nals_h265_annex_b_vp8_passthrough() {
        let pool = VecPool::with_slots(1);
        let h265_payload = vec![0x26, 0x01, 0x02, 0x03]; // synthetic HEVC bytes
        let h265_nals = frame_payload_to_nals(VideoCodec::H265, &h265_payload, &pool);

        assert!(h265_nals.starts_with(&ANNEX_B_START));
        assert_eq!(&h265_nals[ANNEX_B_START.len()..], &h265_payload[..]);

        drop(h265_nals);

        let vp8_payload = vec![0x30, 0x01, 0x02, 0x03]; // synthetic VP8 bytes
        let vp8_nals = frame_payload_to_nals(VideoCodec::Vp8, &vp8_payload, &pool);

        assert_eq!(&vp8_nals[..], &vp8_payload[..]);
    }

    #[test]
    fn rtp_frame_pts_conversion() {
        // 90 kHz RTP timestamp → milliseconds.
        let rtp_ts: u32 = 90_000; // 1 second at 90 kHz
        let pts_ms = rtp_ts as u64 / 90;
        assert_eq!(pts_ms, 1_000);

        let rtp_ts2: u32 = 45_000; // 0.5 s
        let pts_ms2 = rtp_ts2 as u64 / 90;
        assert_eq!(pts_ms2, 500);
    }

    #[test]
    fn webrtc_config_default_constructs() {
        let cfg = WebRtcConfig::default();
        assert_eq!(cfg.stun_servers, vec!["stun:stun.l.google.com:19302"]);
        assert!(cfg.turn_servers.is_empty());
        assert_eq!(cfg.max_output_tokens_per_second, 128);
        assert_eq!(cfg.decode_workers, 1);
        assert_eq!(cfg.analysis_workers, 1);
        assert_eq!(cfg.vlm_workers, 1);
    }

    #[test]
    fn webrtc_config_custom_stun() {
        let cfg = WebRtcConfig {
            stun_servers: vec!["stun:stun.example.com:3478".to_string()],
            turn_servers: vec![super::TurnServer {
                url: "turn:turn.example.com:3478".to_string(),
                username: "user".to_string(),
                credential: "pass".to_string(),
            }],
            max_output_tokens_per_second: 64,
            decode_workers: 3,
            analysis_workers: 2,
            vlm_workers: 4,
            ..Default::default()
        };
        assert_eq!(cfg.stun_servers.len(), 1);
        assert_eq!(cfg.turn_servers.len(), 1);
        assert_eq!(cfg.max_output_tokens_per_second, 64);
        assert_eq!(cfg.decode_workers, 3);
        assert_eq!(cfg.analysis_workers, 2);
        assert_eq!(cfg.vlm_workers, 4);
    }

    #[test]
    fn rtp_frame_is_clone_and_debug() {
        let frame = RtpFrame {
            nals: vec![0x00, 0x00, 0x00, 0x01, 0x65].into(),
            pts_ms: 33,
            seq: 0,
            codec: VideoCodec::H264,
        };
        let cloned = frame.clone();
        assert_eq!(frame.seq, cloned.seq);
        assert_eq!(frame.codec, cloned.codec);
        let _ = format!("{cloned:?}");
    }

    #[test]
    fn rtp_frame_vp8_is_clone_and_debug() {
        let frame = RtpFrame {
            nals: vec![0x30, 0x01].into(),
            pts_ms: 100,
            seq: 5,
            codec: VideoCodec::Vp8,
        };
        let cloned = frame.clone();
        assert_eq!(cloned.codec, VideoCodec::Vp8);
        let _ = format!("{cloned:?}");
    }

    #[tokio::test]
    async fn full_rtp_queue_backpressures_without_dropping_or_blocking_tokio() {
        let (tx, rx) = kanal::bounded::<RtpFrame>(1);
        let metrics = Arc::new(crate::metrics::PipelineMetrics::new());

        assert!(
            enqueue_rtp_frame_lossless(
                &tx,
                RtpFrame {
                    nals: vec![0x30].into(),
                    pts_ms: 0,
                    seq: 0,
                    codec: VideoCodec::Vp8,
                },
                &metrics,
            )
            .await
        );

        let tx_for_task = tx.clone();
        let metrics_for_task = Arc::clone(&metrics);
        let mut blocked_send = tokio::spawn(async move {
            enqueue_rtp_frame_lossless(
                &tx_for_task,
                RtpFrame {
                    nals: vec![0x31].into(),
                    pts_ms: 33,
                    seq: 1,
                    codec: VideoCodec::Vp8,
                },
                &metrics_for_task,
            )
            .await
        });

        tokio::task::yield_now().await;
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), &mut blocked_send)
                .await
                .is_err()
        );
        assert_eq!(metrics.frames_dropped_total(), 0);

        let retained = rx.try_recv().unwrap().unwrap();
        assert_eq!(retained.seq, 0);
        let sent = blocked_send.await.unwrap();
        assert!(sent);
        let second = rx.try_recv().unwrap().unwrap();
        assert_eq!(second.seq, 1);
        assert!(rx.try_recv().unwrap().is_none());
    }

    #[tokio::test]
    async fn prompt_and_guided_json_update_is_acknowledged_by_generation() {
        let generation = crate::webrtc::runtime::PipelineGeneration::new(9);
        let (session, mut commands) = WebRtcSession::new_for_tests_with_generation(generation);
        let update = tokio::spawn(async move {
            session
                .update_config(
                    "describe safety events".to_string(),
                    Some(r#"{"type":"object"}"#.to_string()),
                )
                .await
        });
        tokio::task::yield_now().await;
        assert!(!update.is_finished());

        let mut prompt: Arc<str> = Arc::from("");
        let mut schema = None;
        crate::webrtc::runtime::apply_pending_session_commands(
            &mut commands,
            generation,
            &mut prompt,
            &mut schema,
        );

        assert_eq!(prompt.as_ref(), "describe safety events");
        assert_eq!(schema.as_deref(), Some(r#"{"type":"object"}"#));
        assert_eq!(update.await.unwrap(), Ok(()));
    }

    /// The stop decision must latch on the session shutdown flag regardless of
    /// peer state, and otherwise only on a terminal peer state. Driven directly
    /// so the full truth table is covered without a live peer, whose transient
    /// states are not deterministically reachable from a test.
    #[test]
    fn should_stop_latches_on_shutdown_and_terminal_peer_states() {
        use super::should_stop;
        use rustrtc::peer_connection::PeerConnectionState as S;

        // A raised shutdown latch stops the loop for every peer state, including
        // a Connected that a late rustrtc republish could put back after close().
        for peer in [
            S::New,
            S::Connecting,
            S::Connected,
            S::Disconnected,
            S::Failed,
            S::Closed,
        ] {
            assert!(should_stop(true, peer), "shutdown must win over {peer:?}");
        }

        // Without shutdown, only the terminal peer states stop it.
        assert!(should_stop(false, S::Closed));
        assert!(should_stop(false, S::Failed));
        assert!(should_stop(false, S::Disconnected));
        assert!(!should_stop(false, S::New));
        assert!(!should_stop(false, S::Connecting));
        assert!(!should_stop(false, S::Connected));
    }

    /// The latch must retain a close() that happens before anything subscribes
    /// through run(). The session holds a keepalive receiver for exactly this:
    /// without a live receiver, send(true) would not store the value, and a
    /// later subscribe() would read the initial false. Checked directly against
    /// the sender so it does not depend on rustrtc scheduling any state.
    /// Async only because new_for_tests() builds a rustrtc peer, which spawns.
    #[tokio::test]
    async fn shutdown_latch_retains_a_close_before_any_run_subscription() {
        let session = WebRtcSession::new_for_tests();

        // Close before run() (and its subscribe) is ever called.
        session.close();

        // A fresh subscriber, as run()'s next call would create, reads the
        // retained latch value rather than the initial false.
        let mut rx = session.shutdown.subscribe();
        assert!(
            *rx.borrow_and_update(),
            "a close() before any run subscription must still be visible"
        );
    }

    /// A close that lands after run() is called but before its future is first
    /// polled must still terminate run() and release the RTP sender. This is the
    /// spawn-then-fail ordering: the run task is created, then the session is
    /// closed before the task is scheduled. peer_state is subscribed inside the
    /// future, so the Closed rustrtc publishes is already-seen by the first
    /// poll and its changed() never fires; the top-of-loop check is what stops
    /// the loop. Without it run() would park on changed()/pc.recv() forever
    /// while holding the root frame_tx, and the decode worker downstream would
    /// never see its channel close.
    #[tokio::test]
    async fn run_returns_when_closed_before_its_first_poll() {
        let session = WebRtcSession::new_for_tests();

        let (frame_tx, frame_rx) = kanal::bounded::<RtpFrame>(RTP_FRAME_QUEUE_CAPACITY);
        let metrics = Arc::new(crate::metrics::PipelineMetrics::new());

        // Build the future (which subscribes to the latch and takes the RTP
        // sender) but do not poll it, then close. This mirrors the run task
        // being created and the session then closed before it is scheduled.
        let run = session.run(frame_tx, metrics);
        session.close();

        // run() must observe the already-raised latch on its first poll and
        // return, rather than parking on changed()/pc.recv() forever.
        tokio::time::timeout(std::time::Duration::from_secs(5), run)
            .await
            .expect("run() must return after a close that precedes its first poll");

        // run() owned the only RTP sender; once it returns that sender is
        // dropped, so the decode worker's receiver now reports the channel
        // closed instead of blocking on a peer that will never send frames.
        assert!(
            frame_rx.try_recv().is_err(),
            "the RTP receiver should disconnect once run() drops its frame_tx"
        );
    }
}

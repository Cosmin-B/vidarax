//! WebRTC session wrapper — rustrtc peer connection for inbound video streams.
//!
//! [`WebRtcSession`] manages the full lifecycle of a single WebRTC peer
//! connection: SDP offer/answer negotiation, trickle ICE, and media ingestion.
//! Video payload bytes are forwarded through a [`kanal`] channel to the
//! downstream decode workers.  Both H.264 and VP8 codecs are supported; the
//! active codec is detected from the SDP offer and tagged on every [`RtpFrame`]
//! so the correct decode backend is chosen automatically.
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
//!     tokio::spawn(async move { session.run(frame_tx).await });
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
    atomic::{AtomicU64, Ordering},
    Arc,
};

use arc_swap::{ArcSwap, ArcSwapOption};
use rustrtc::{
    IceServer, RtcConfiguration, SdpType, SessionDescription,
    media::{MediaKind, MediaSample, MediaStreamTrack},
    peer_connection::{PeerConnection, PeerConnectionEvent},
};

use crate::webrtc::decode::VideoCodec;
use crate::webrtc::recycle::{RecycledBytes, VecPool};

/// Annex B start code prepended to every H.264 NAL unit.
///
/// rustrtc delivers H.264 NAL payloads **without** start codes; openh264 and
/// ffmpeg expect them prepended.  VP8 payloads are passed through unchanged
/// (no start code wrapping).
const ANNEX_B_START: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
pub const RTP_FRAME_QUEUE_CAPACITY: usize = 128;

pub fn rtp_nal_pool_slots(decode_workers: usize) -> usize {
    RTP_FRAME_QUEUE_CAPACITY + decode_workers.max(1) + 1
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single video access unit ready for the decode pipeline.
///
/// For H.264, `nals` always begins with the 4-byte Annex B start code
/// `00 00 00 01` followed by the raw NAL data.
/// For VP8, `nals` contains the raw VP8 bitstream payload exactly as
/// delivered by rustrtc (no framing added).
///
/// The `seq` counter is per-session and monotonically increasing — it is
/// NOT the RTP sequence number.
#[derive(Debug, Clone)]
pub struct RtpFrame {
    /// Video payload bytes.
    ///
    /// - H.264: Annex B encoded (starts with `00 00 00 01`).
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
    /// Number of decode worker threads per session.
    pub decode_workers: usize,
    /// Number of analysis worker threads per session.
    pub analysis_workers: usize,
    /// Number of VLM worker threads per session.
    pub vlm_workers: usize,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            max_output_tokens_per_second: 128,
            decode_workers: 2,
            analysis_workers: 1,
            vlm_workers: 2,
        }
    }
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
    /// Dynamic VLM prompt; updated via `PATCH /v1/stream/whip/{sess_id}/prompt`.
    pub prompt: Arc<ArcSwap<Arc<str>>>,
    /// Optional JSON schema string for guided/structured VLM output.
    ///
    /// When `Some`, passed as `guided_json` to the VLM inference request and
    /// `max_tokens` is bumped to 1024 to accommodate structured output.
    /// Updated via `PATCH /v1/stream/whip/{sess_id}/prompt` with `output_schema`.
    pub guided_json: Arc<ArcSwapOption<Arc<str>>>,
    /// Token output rate cap (tokens/s) for backpressure in VLM workers.
    pub max_output_tokens_per_second: u32,
    /// Video codec negotiated from the SDP offer (H.264 or VP8).
    pub codec: VideoCodec,
    rtp_nal_pool_slots: usize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl WebRtcSession {
    /// Create a new session from a browser SDP offer.
    ///
    /// 1. Parses `offer_sdp` and applies it as the remote description.
    /// 2. Waits up to 3 s for the first local ICE candidate (so host
    ///    candidates are embedded in the answer).
    /// 3. Creates and applies the local answer.
    ///
    /// Returns `(session, answer_sdp)`.  The caller must send `answer_sdp`
    /// back to the browser to complete the signalling exchange.
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string on SDP parse failure, ICE
    /// negotiation error, or answer-generation failure.
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
    ) -> Result<(Self, String), String> {
        // Detect codec before handing the SDP to rustrtc so that even if the
        // peer connection transforms the SDP we still know what was offered.
        let codec = VideoCodec::from_sdp(offer_sdp);

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
        let pc = PeerConnection::new(rtc_config);

        let offer = SessionDescription::parse(SdpType::Offer, offer_sdp)
            .map_err(|e| format!("SDP parse: {e}"))?;

        pc.set_remote_description(offer)
            .await
            .map_err(|e| format!("set_remote_description: {e}"))?;

        // Wait briefly for the first local ICE candidate so that the answer
        // SDP contains usable host candidates.  Trickle ICE will add more.
        {
            let mut ice_rx = pc.subscribe_ice_candidates();
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                ice_rx.recv(),
            )
            .await;
        }

        let answer = pc
            .create_answer()
            .await
            .map_err(|e| format!("create_answer: {e}"))?;

        pc.set_local_description(answer)
            .map_err(|e| format!("set_local_description: {e}"))?;

        let local = pc.local_description().expect("local description was just set");
        let answer_sdp = local.to_sdp_string();

        Ok((
            Self {
                pc,
                prompt: Arc::new(ArcSwap::from(Arc::new(Arc::from("")))),
                guided_json: Arc::new(ArcSwapOption::from(None::<Arc<Arc<str>>>)),
                max_output_tokens_per_second: config.max_output_tokens_per_second,
                codec,
                rtp_nal_pool_slots: rtp_nal_pool_slots(config.decode_workers),
            },
            answer_sdp,
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

    /// Drive the peer connection event loop, forwarding H.264 NAL units
    /// through `frame_tx` until the connection closes or the sender is dropped.
    ///
    /// Returns an **owned** future (no lifetime ties to `self`) that can be
    /// directly passed to [`tokio::spawn`].  Internally clones the
    /// `PeerConnection` handle so the caller may store `WebRtcSession` in a
    /// shared structure while the task runs concurrently.
    ///
    /// - Annex B start codes (`00 00 00 01`) are **prepended** to every NAL
    ///   before sending — rustrtc delivers raw NAL payloads without them.
    /// - Audio tracks are silently ignored.
    /// - For each video track a Tokio task is spawned; all share the same
    ///   atomic sequence counter.
    ///
    /// The `frame_tx` channel should have a capacity of at least 32 frames
    /// to absorb burst traffic without the track-receive tasks stalling.
    ///
    /// # Notes on blocking
    ///
    /// `kanal::Sender::send` is synchronous.  For a bounded channel that is
    /// not full the call returns essentially instantly.  When the channel is
    /// full the call blocks briefly — this provides natural backpressure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (tx, rx) = kanal::bounded(128);
    /// tokio::spawn(session.run(tx));
    /// ```
    pub fn run(
        &self,
        frame_tx: kanal::Sender<RtpFrame>,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        // Clone the PeerConnection so the returned future owns everything it
        // needs and has no lifetime dependency on `self`.
        let pc = self.pc.clone();
        let seq_counter = Arc::new(AtomicU64::new(0));
        let codec = self.codec;
        let rtp_nal_pool_slots = self.rtp_nal_pool_slots;

        async move {
            while let Some(event) = pc.recv().await {
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

                        tokio::spawn(async move {
                            let nals_pool = VecPool::with_slots(rtp_nal_pool_slots);
                            loop {
                                match track.recv().await {
                                    Ok(MediaSample::Video(frame)) => {
                                        let nal_seq = seq.fetch_add(1, Ordering::Relaxed);

                                        let nals = frame_payload_to_nals(codec, &frame.data, &nals_pool);

                                        // RTP timestamp is on a 90 kHz clock → ms.
                                        let pts_ms = frame.rtp_timestamp as u64 / 90;

                                        let rtp_frame = RtpFrame {
                                            nals,
                                            pts_ms,
                                            seq: nal_seq,
                                            codec,
                                        };

                                        if tx.send(rtp_frame).is_err() {
                                            // Receiver dropped — stop this track loop.
                                            break;
                                        }
                                    }
                                    Ok(MediaSample::Audio(_)) => {
                                        // Unexpected on a video track; ignore.
                                    }
                                    Err(_) => break,
                                }
                            }
                        });
                    }
                    PeerConnectionEvent::DataChannel(_) => {
                        // Not used; ignore.
                    }
                }
            }
        }
    }

    /// Update the VLM analysis prompt for this session.
    ///
    /// Called by `PATCH /v1/stream/whip/{sess_id}/prompt`.  The new prompt is
    /// picked up by analysis workers on the next keyframe decision.
    pub fn update_prompt(&self, text: String) {
        self.prompt.store(Arc::new(Arc::from(text)));
    }

    /// Read the current VLM prompt (empty string means use the default).
    ///
    /// Clones only the `Arc<str>` pointer (pointer-width), not the string data.
    pub fn read_prompt(&self) -> Arc<str> {
        Arc::clone(&*self.prompt.load_full())
    }

    /// Clone the prompt handle for sharing with analysis workers.
    pub fn prompt_arc(&self) -> Arc<ArcSwap<Arc<str>>> {
        Arc::clone(&self.prompt)
    }

    /// Update the guided JSON schema for structured VLM output.
    ///
    /// Pass `None` to disable structured output and revert to free-text.
    /// Called by `PATCH /v1/stream/whip/{sess_id}/prompt` when `output_schema`
    /// is present in the request body.
    pub fn update_guided_json(&self, schema: Option<String>) {
        self.guided_json
            .store(schema.map(|schema| Arc::new(Arc::from(schema))));
    }

    /// Clone the guided-JSON handle for sharing with VLM worker threads.
    pub fn guided_json_arc(&self) -> Arc<ArcSwapOption<Arc<str>>> {
        Arc::clone(&self.guided_json)
    }

    /// Close the peer connection.
    ///
    /// After this call any active `run` task will exit on its next poll cycle
    /// as the underlying `PeerConnection` cleans up.
    pub fn terminate(self) {
        // Dropping self drops our PeerConnection handle; rustrtc closes the
        // connection when all handles are released.
        drop(self);
    }
}

fn frame_payload_to_nals(codec: VideoCodec, payload: &[u8], pool: &VecPool) -> RecycledBytes {
    let mut nals = pool.acquire();
    match codec {
        VideoCodec::H264 => {
            // rustrtc delivers H.264 NAL payload without Annex B start codes.
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
        frame_payload_to_nals, rtp_nal_pool_slots, RtpFrame, WebRtcConfig, WebRtcSession,
        RTP_FRAME_QUEUE_CAPACITY,
    };
    use crate::webrtc::decode::VideoCodec;
    use crate::webrtc::recycle::{RecycledBytes, VecPool};
    use std::sync::Arc;

    #[test]
    fn rtp_nal_pool_covers_queue_workers_and_blocked_sender() {
        assert_eq!(rtp_nal_pool_slots(2), RTP_FRAME_QUEUE_CAPACITY + 2 + 1);
        assert_eq!(rtp_nal_pool_slots(0), RTP_FRAME_QUEUE_CAPACITY + 1 + 1);
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
        assert_eq!(cfg.decode_workers, 2);
        assert_eq!(cfg.analysis_workers, 1);
        assert_eq!(cfg.vlm_workers, 2);
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
        let _ = format!("{:?}", cloned);
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
        let _ = format!("{:?}", cloned);
    }

    #[tokio::test]
    async fn prompt_and_guided_json_read_latest_updates() {
        let session = WebRtcSession {
            pc: rustrtc::peer_connection::PeerConnection::new(Default::default()),
            prompt: Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from("")))),
            guided_json: Arc::new(arc_swap::ArcSwapOption::from(None::<Arc<Arc<str>>>)),
            max_output_tokens_per_second: 128,
            codec: VideoCodec::H264,
            rtp_nal_pool_slots: super::rtp_nal_pool_slots(2),
        };

        session.update_prompt("describe safety events".to_string());
        session.update_guided_json(Some(r#"{"type":"object"}"#.to_string()));

        assert_eq!(&*session.read_prompt(), "describe safety events");
        assert_eq!(
            session
                .guided_json_arc()
                .load_full()
                .as_ref()
                .map(|schema| schema.as_ref().as_ref()),
            Some(r#"{"type":"object"}"#)
        );
    }
}

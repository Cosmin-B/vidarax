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
    atomic::{AtomicU64, Ordering},
    Arc,
};

use arc_swap::{ArcSwap, ArcSwapOption};
pub use rustrtc::peer_connection::PeerConnectionState;
use rustrtc::{
    media::{MediaKind, MediaSample, MediaStreamTrack},
    peer_connection::{PeerConnection, PeerConnectionEvent},
    IceServer, RtcConfiguration, SdpType, SessionDescription,
};

use crate::metrics::PipelineMetrics;
use crate::webrtc::decode::{select_answer_video_codec_for_offer, VideoCodec};
use crate::webrtc::recycle::{RecycledBytes, VecPool};

/// Annex B start code prepended to every H.264 or H.265 NAL unit.
///
/// rustrtc delivers H.264 NAL payloads **without** start codes; openh264 and
/// ffmpeg expect them prepended.  H.265 / HEVC is wrapped the same way for
/// ffmpeg sidecar input once depacketized. VP8 payloads are passed through
/// unchanged.
const ANNEX_B_START: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
pub const RTP_FRAME_QUEUE_CAPACITY: usize = 128;

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
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            max_output_tokens_per_second: 128,
            decode_workers: 1,
            analysis_workers: 1,
            vlm_workers: 1,
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
    /// Video codec negotiated from the SDP offer.
    pub codec: VideoCodec,
    rtp_nal_pool_slots: usize,
    #[cfg(any(test, debug_assertions))]
    close_calls: AtomicU64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl WebRtcSession {
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn new_for_tests() -> Self {
        Self {
            pc: PeerConnection::new(Default::default()),
            prompt: Arc::new(ArcSwap::from(Arc::new(Arc::from("")))),
            guided_json: Arc::new(ArcSwapOption::from(None::<Arc<Arc<str>>>)),
            max_output_tokens_per_second: 128,
            codec: VideoCodec::H264,
            rtp_nal_pool_slots: rtp_nal_pool_slots(1),
            #[cfg(any(test, debug_assertions))]
            close_calls: AtomicU64::new(0),
        }
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
    pub async fn new(offer_sdp: &str, config: &WebRtcConfig) -> Result<(Self, String), String> {
        // Select before handing SDP to rustrtc so the answer, depacketizer,
        // and decode routing use the same codec.
        let selected = select_answer_video_codec_for_offer(offer_sdp);
        // An offer that advertised a recognized video codec we cannot serve (a
        // DON-signaling H.265, or VP8 without the `vp8` feature) must be rejected:
        // otherwise `new` would build a session whose answerer video receiver uses
        // rustrtc's default H.264 depacketizer, and a peer that sends video RTP
        // regardless would be depacketized and decode-routed as the wrong codec.
        if selected.is_none() && !VideoCodec::offered_video_codecs(offer_sdp).is_empty() {
            return Err("offer advertised video but no live-serveable codec".to_string());
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

        let offer = SessionDescription::parse(SdpType::Offer, offer_sdp)
            .map_err(|e| format!("SDP parse: {e}"))?;

        pc.set_remote_description(offer)
            .await
            .map_err(|e| format!("set_remote_description: {e}"))?;

        // Wait briefly for the first local ICE candidate so that the answer
        // SDP contains usable host candidates.  Trickle ICE will add more.
        {
            let mut ice_rx = pc.subscribe_ice_candidates();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(3), ice_rx.recv()).await;
        }

        let answer = pc
            .create_answer()
            .await
            .map_err(|e| format!("create_answer: {e}"))?;

        pc.set_local_description(answer)
            .map_err(|e| format!("set_local_description: {e}"))?;

        let local = pc
            .local_description()
            .expect("local description was just set");
        let answer_sdp = local.to_sdp_string();

        Ok((
            Self {
                pc,
                prompt: Arc::new(ArcSwap::from(Arc::new(Arc::from("")))),
                guided_json: Arc::new(ArcSwapOption::from(None::<Arc<Arc<str>>>)),
                max_output_tokens_per_second: config.max_output_tokens_per_second,
                codec,
                rtp_nal_pool_slots: rtp_nal_pool_slots(config.decode_workers),
                #[cfg(any(test, debug_assertions))]
                close_calls: AtomicU64::new(0),
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
                        let metrics = Arc::clone(&metrics);

                        tokio::spawn(async move {
                            let nals_pool = VecPool::with_slots(rtp_nal_pool_slots);
                            loop {
                                match track.recv().await {
                                    Ok(MediaSample::Video(frame)) => {
                                        let nal_seq = seq.fetch_add(1, Ordering::Relaxed);

                                        let nals =
                                            frame_payload_to_nals(codec, &frame.data, &nals_pool);

                                        // RTP timestamp is on a 90 kHz clock → ms.
                                        let pts_ms = frame.rtp_timestamp as u64 / 90;

                                        let rtp_frame = RtpFrame {
                                            nals,
                                            pts_ms,
                                            seq: nal_seq,
                                            codec,
                                        };

                                        if !enqueue_rtp_frame_lossless(&tx, rtp_frame, &metrics)
                                            .await
                                        {
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

    /// Subscribe to rustrtc peer state changes for lifecycle cleanup.
    pub fn subscribe_peer_state(&self) -> tokio::sync::watch::Receiver<PeerConnectionState> {
        self.pc.subscribe_peer_state()
    }

    /// Close the peer connection while shared session handles may still exist.
    pub fn close(&self) {
        #[cfg(any(test, debug_assertions))]
        self.close_calls.fetch_add(1, Ordering::Relaxed);
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
        WebRtcConfig, WebRtcSession, ANNEX_B_START, RTP_FRAME_QUEUE_CAPACITY,
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
        assert_eq!(err, "offer advertised video but no live-serveable codec");
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
    async fn prompt_and_guided_json_read_latest_updates() {
        let session = WebRtcSession {
            pc: rustrtc::peer_connection::PeerConnection::new(Default::default()),
            prompt: Arc::new(arc_swap::ArcSwap::from(Arc::new(Arc::from("")))),
            guided_json: Arc::new(arc_swap::ArcSwapOption::from(None::<Arc<Arc<str>>>)),
            max_output_tokens_per_second: 128,
            codec: VideoCodec::H264,
            rtp_nal_pool_slots: super::rtp_nal_pool_slots(2),
            close_calls: std::sync::atomic::AtomicU64::new(0),
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

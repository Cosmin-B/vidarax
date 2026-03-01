//! WebRTC session wrapper — rustrtc peer connection for inbound H.264 streams.
//!
//! [`WebRtcSession`] manages the full lifecycle of a single WebRTC peer
//! connection: SDP offer/answer negotiation, trickle ICE, and media ingestion.
//! Decoded H.264 NAL units are forwarded through a [`kanal`] channel to the
//! downstream decode workers.
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
//!             println!("seq={} pts={}ms nals={}", frame.seq, frame.pts_ms, frame.nals.len());
//!         }
//!     });
//! }
//! ```

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use rustrtc::{
    RtcConfiguration, SdpType, SessionDescription,
    media::{MediaKind, MediaSample, MediaStreamTrack},
    peer_connection::{PeerConnection, PeerConnectionEvent},
};

/// Annex B start code prepended to every NAL unit.
///
/// rustrtc delivers H.264 NAL payloads **without** start codes; openh264 and
/// ffmpeg expect them prepended.
const ANNEX_B_START: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single H.264 NAL unit ready for the decode pipeline.
///
/// `nals` always begins with the 4-byte Annex B start code `00 00 00 01`
/// followed by the raw NAL data.  The `seq` counter is per-session and
/// monotonically increasing — it is NOT the RTP sequence number.
#[derive(Debug, Clone)]
pub struct RtpFrame {
    /// H.264 NAL bytes with Annex B start code prepended.
    pub nals: Vec<u8>,
    /// Presentation timestamp derived from the 90 kHz RTP clock (in ms).
    pub pts_ms: u64,
    /// Per-session monotonically increasing sequence number.
    pub seq: u64,
}

/// Configuration for a [`WebRtcSession`].
///
/// Currently empty; fields will be added for STUN server URIs, ICE gather
/// timeouts, etc.
#[derive(Debug, Clone, Default)]
pub struct WebRtcConfig {
    // Future: stun_servers: Vec<String>, gather_timeout_secs: u64, …
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
        _config: &WebRtcConfig,
    ) -> Result<(Self, String), String> {
        let pc = PeerConnection::new(RtcConfiguration::default());

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

        Ok((Self { pc }, answer_sdp))
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
                            loop {
                                match track.recv().await {
                                    Ok(MediaSample::Video(frame)) => {
                                        let nal_seq = seq.fetch_add(1, Ordering::Relaxed);

                                        // rustrtc delivers NAL payload WITHOUT Annex B
                                        // start codes.  Prepend 0x00 0x00 0x00 0x01.
                                        let mut nals = Vec::with_capacity(4 + frame.data.len());
                                        nals.extend_from_slice(&ANNEX_B_START);
                                        nals.extend_from_slice(&frame.data);

                                        // RTP timestamp is on a 90 kHz clock → ms.
                                        let pts_ms = frame.rtp_timestamp as u64 / 90;

                                        let rtp_frame = RtpFrame {
                                            nals,
                                            pts_ms,
                                            seq: nal_seq,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{RtpFrame, WebRtcConfig};

    #[test]
    fn rtp_frame_annex_b_layout() {
        // Simulate what run() produces: start code + payload.
        let payload = vec![0x65, 0xB8, 0x00]; // IDR NAL header byte
        let mut nals = Vec::with_capacity(4 + payload.len());
        nals.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        nals.extend_from_slice(&payload);

        assert_eq!(&nals[..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&nals[4..], &payload[..]);
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
        let _cfg = WebRtcConfig::default();
    }

    #[test]
    fn rtp_frame_is_clone_and_debug() {
        let frame = RtpFrame {
            nals: vec![0x00, 0x00, 0x00, 0x01, 0x65],
            pts_ms: 33,
            seq: 0,
        };
        let cloned = frame.clone();
        assert_eq!(frame.seq, cloned.seq);
        let _ = format!("{:?}", cloned);
    }
}

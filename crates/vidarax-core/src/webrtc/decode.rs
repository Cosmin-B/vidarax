//! Video decode backends: GPU via ffmpeg NVDEC or CPU via openh264 / ffmpeg sidecar.
//!
//! Supports H.264 and VP8 codecs.  The backend is selected at construction time
//! based on [`DecoderConfig`]:
//!
//! - **GPU path (`NvDec`)**: spawns a long-lived ffmpeg sidecar with
//!   `-hwaccel auto`, which handles both H.264 (NVDEC) and VP8 transparently.
//!   Reads back planar YUV420 from stdout.
//! - **Software H.264 path (`Software`)**: uses openh264 in-process.  Lower
//!   latency on machines without CUDA; only valid for H.264 input.
//! - **Software VP8 path (`FfmpegSw`)**: spawns a long-lived ffmpeg sidecar
//!   without hardware acceleration for VP8 streams on machines without a GPU.
//!   Uses `libvpx` inside ffmpeg, so no additional Rust crates are needed.
//!
//! # Example
//!
//! ```no_run
//! use vidarax_core::webrtc::decode::{Decoder, DecoderConfig, VideoCodec};
//!
//! let config = DecoderConfig {
//!     gpu_available: false,
//!     codec: VideoCodec::H264,
//!     width: 1280,
//!     height: 720,
//! };
//! let mut decoder = Decoder::new(&config);
//! // Feed raw NAL / VP8 bytes:
//! // let frame = decoder.decode(&payload_bytes).unwrap();
//! ```

use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;

use openh264::formats::YUVSource;

/// Video codec carried by the WebRTC track.
///
/// Detected from the SDP offer (presence of `"VP8"` or `"H264"` media attributes)
/// before the peer connection is established and propagated through the pipeline
/// so the correct decode backend is selected per-session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoCodec {
    /// ITU-T H.264 / AVC — default when the offer codec cannot be determined.
    #[default]
    H264,
    /// Google VP8 — negotiated by rustrtc when the browser prefers it.
    Vp8,
}

impl VideoCodec {
    /// Parse the raw SDP offer string and return the first video codec found.
    ///
    /// Looks for `VP8` or `H264` (case-insensitive) in the `a=rtpmap:` lines of
    /// the video media section.  Falls back to [`VideoCodec::H264`] when neither
    /// is found so that existing sessions without explicit codec info continue to
    /// work unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use vidarax_core::webrtc::decode::VideoCodec;
    ///
    /// let sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 VP8/90000\r\n";
    /// assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::Vp8);
    ///
    /// let sdp2 = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 H264/90000\r\n";
    /// assert_eq!(VideoCodec::from_sdp(sdp2), VideoCodec::H264);
    /// ```
    pub fn from_sdp(sdp: &str) -> Self {
        // Walk the SDP looking for video m= sections, then scan rtpmap lines.
        let mut in_video = false;
        for line in sdp.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("m=") {
                in_video = trimmed.starts_with("m=video");
                continue;
            }
            if !in_video {
                continue;
            }
            // a=rtpmap:<pt> <codec>/<clock>[/<channels>]
            if let Some(rest) = trimmed.strip_prefix("a=rtpmap:") {
                let codec_part = rest.split_once(' ').map(|(_, v)| v).unwrap_or(rest);
                let codec_name = codec_part.split('/').next().unwrap_or("").to_ascii_uppercase();
                match codec_name.as_str() {
                    "VP8" => return VideoCodec::Vp8,
                    "H264" => return VideoCodec::H264,
                    _ => {}
                }
            }
        }
        // Default: assume H.264 so existing sessions without explicit codec info work.
        VideoCodec::H264
    }

    /// ffmpeg input format flag for this codec (used by the ffmpeg sidecar).
    fn ffmpeg_input_format(self) -> &'static str {
        match self {
            VideoCodec::H264 => "h264",
            VideoCodec::Vp8 => "vp8",
        }
    }
}

/// A planar YUV 4:2:0 frame with packed (non-strided) plane buffers.
///
/// `y.len() == (width * height) as usize`
/// `u.len() == v.len() == (width / 2 * height / 2) as usize`
#[derive(Debug, Clone)]
pub struct YuvFrame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Controls which decode backend [`Decoder::new`] selects.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// When `true`, the NVDEC GPU path is used; otherwise software decode.
    pub gpu_available: bool,
    /// Video codec negotiated with the browser peer.
    pub codec: VideoCodec,
    /// Frame width passed to the NVDEC ffmpeg pipeline.  Ignored for openh264.
    pub width: u32,
    /// Frame height passed to the NVDEC ffmpeg pipeline.  Ignored for openh264.
    pub height: u32,
}

impl DecoderConfig {
    /// Auto-detect GPU availability by probing `nvidia-smi`.
    ///
    /// Falls back to CPU (`gpu_available: false`) if the binary is not found
    /// or returns a non-zero exit status.  Defaults to 1920×1080 for NVDEC and
    /// [`VideoCodec::H264`] as the codec (callers should override `codec` from
    /// the SDP offer).
    pub fn auto_detect() -> Self {
        let gpu_available = Command::new(crate::ingest::nvidia_smi_path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        Self {
            gpu_available,
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
        }
    }
}

/// Errors returned by [`Decoder::decode`].
///
/// The common hot-path case is [`DecodeError::Buffered`] — the codec is still
/// accumulating input (SPS/PPS NALs, incomplete VP8 frame).  It carries no
/// heap allocation, so callers that pattern-match on `Err(_) => continue` pay
/// zero allocation cost per buffered NAL.
#[derive(Debug)]
pub enum DecodeError {
    /// The codec is buffering input; no frame is available yet.
    /// This is normal for H.264 SPS/PPS/IDR sequences and VP8 headers.
    /// Callers should feed the next NAL and retry.
    Buffered,
    /// The ffmpeg reader thread has exited (process terminated or pipe closed).
    ReaderExited,
    /// Writing payload bytes to the ffmpeg stdin pipe failed.
    WriteError(std::io::Error),
    /// Flushing the ffmpeg stdin pipe failed.
    FlushError(std::io::Error),
    /// openh264 reported a hard decode error (bad bitstream, invalid state).
    SoftwareDecode(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Buffered => f.write_str("codec buffering input (no frame yet)"),
            DecodeError::ReaderExited => f.write_str("ffmpeg reader thread exited"),
            DecodeError::WriteError(e) => write!(f, "ffmpeg write: {e}"),
            DecodeError::FlushError(e) => write!(f, "ffmpeg flush: {e}"),
            DecodeError::SoftwareDecode(e) => write!(f, "openh264 decode error: {e}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::WriteError(e) | DecodeError::FlushError(e) => Some(e),
            _ => None,
        }
    }
}

/// Multi-codec video decoder backed by either NVDEC, openh264, or a software
/// ffmpeg sidecar.
///
/// The ffmpeg pipe paths (`NvDec`, `FfmpegSw`) use a dedicated reader thread
/// to avoid deadlocks: H.264 commonly requires multiple NAL units (SPS, PPS,
/// IDR) before ffmpeg can produce a frame, so writing and reading must happen
/// concurrently.  The reader thread continuously reads complete YUV frames from
/// ffmpeg stdout and sends them through an `mpsc` channel.
pub enum Decoder {
    /// GPU: long-lived ffmpeg sidecar using `-hwaccel auto` (~0.5 ms/frame).
    NvDec {
        child: Child,
        stdin: ChildStdin,
        frame_rx: mpsc::Receiver<YuvFrame>,
        width: u32,
        height: u32,
    },
    /// CPU: openh264 in-process decoder (~2–5 ms/frame on ARM).
    Software {
        decoder: openh264::decoder::Decoder,
    },
    /// CPU: long-lived ffmpeg sidecar without hardware acceleration.
    FfmpegSw {
        child: Child,
        stdin: ChildStdin,
        frame_rx: mpsc::Receiver<YuvFrame>,
        width: u32,
        height: u32,
    },
}

/// Spawn a background thread that continuously reads YUV420 frames from
/// ffmpeg stdout and sends them to the returned receiver.
///
/// Plane buffers (`y`, `u`, `v`) are allocated once before the loop and
/// reused across frames, avoiding ~93 MB/s of repeated heap allocation at
/// 1080p/30 fps.  Each [`YuvFrame`] sent on the channel owns a clone of the
/// plane data so the scratch buffers can be refilled immediately.
fn spawn_frame_reader(
    mut stdout: BufReader<ChildStdout>,
    width: u32,
    height: u32,
) -> mpsc::Receiver<YuvFrame> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        // Pre-allocate scratch buffers once; reused for every frame.
        let mut y = vec![0u8; y_size];
        let mut u = vec![0u8; uv_size];
        let mut v = vec![0u8; uv_size];
        loop {
            if stdout.read_exact(&mut y).is_err() {
                break; // ffmpeg closed stdout (process exited)
            }
            if stdout.read_exact(&mut u).is_err() {
                break;
            }
            if stdout.read_exact(&mut v).is_err() {
                break;
            }
            let frame = YuvFrame { y: y.clone(), u: u.clone(), v: v.clone(), width, height };
            if tx.send(frame).is_err() {
                break; // receiver dropped
            }
        }
    });
    rx
}

impl Decoder {
    /// Create a decoder.
    ///
    /// Selects the backend based on `config`:
    ///
    /// | `gpu_available` | `codec`          | Backend     |
    /// |-----------------|------------------|-------------|
    /// | `true`          | H.264 or VP8     | `NvDec`     |
    /// | `false`         | H.264            | `Software`  |
    /// | `false`         | VP8              | `FfmpegSw`  |
    ///
    /// # Panics
    ///
    /// Panics if the selected backend cannot be initialised (ffmpeg not found
    /// for `NvDec`/`FfmpegSw`, or openh264 library init failure for `Software`).
    pub fn new(config: &DecoderConfig) -> Self {
        if config.gpu_available {
            Self::new_nvdec(config.codec, config.width, config.height)
        } else {
            match config.codec {
                VideoCodec::H264 => Self::new_software(),
                VideoCodec::Vp8 => Self::new_ffmpeg_sw(config.codec, config.width, config.height),
            }
        }
    }

    fn new_software() -> Self {
        let decoder =
            openh264::decoder::Decoder::new().expect("openh264 decoder initialisation failed");
        Decoder::Software { decoder }
    }

    /// Spawn an ffmpeg sidecar using `-hwaccel auto` so the same process
    /// handles both H.264 (NVDEC) and VP8 (GPU VP8 decode) without needing to
    /// hard-code the codec decoder name.
    fn new_nvdec(codec: VideoCodec, width: u32, height: u32) -> Self {
        let input_fmt = codec.ffmpeg_input_format();
        let mut child = Command::new(crate::ingest::ffmpeg_path())
            .args([
                "-hwaccel",
                "auto",
                "-f",
                input_fmt,
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                &format!("{width}x{height}"),
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("ffmpeg NVDEC spawn failed — is ffmpeg with NVDEC support available?");

        let stdin = child.stdin.take().expect("ffmpeg stdin missing");
        let stdout = BufReader::new(child.stdout.take().expect("ffmpeg stdout missing"));
        let frame_rx = spawn_frame_reader(stdout, width, height);

        Decoder::NvDec {
            child,
            stdin,
            frame_rx,
            width,
            height,
        }
    }

    /// Spawn a software-only ffmpeg sidecar for VP8 decode on CPU.
    ///
    /// Uses ffmpeg's bundled `libvpx` decoder; no GPU or special codec flags
    /// required.  The `-s` scaling hint is not passed because VP8 streams carry
    /// their dimensions in-band; ffmpeg reads them from the bitstream headers.
    fn new_ffmpeg_sw(codec: VideoCodec, width: u32, height: u32) -> Self {
        let input_fmt = codec.ffmpeg_input_format();
        let mut child = Command::new(crate::ingest::ffmpeg_path())
            .args([
                "-f",
                input_fmt,
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("ffmpeg software VP8 spawn failed — is ffmpeg with libvpx available?");

        let stdin = child.stdin.take().expect("ffmpeg stdin missing");
        let stdout = BufReader::new(child.stdout.take().expect("ffmpeg stdout missing"));
        let frame_rx = spawn_frame_reader(stdout, width, height);

        Decoder::FfmpegSw {
            child,
            stdin,
            frame_rx,
            width,
            height,
        }
    }

    /// Decode raw video payload bytes into a packed YUV420 frame.
    ///
    /// - For `NvDec` and `FfmpegSw`: writes `payload` to the ffmpeg stdin pipe
    ///   and then checks the reader thread's channel for a decoded frame.
    ///   H.264 streams commonly need several NAL units (SPS, PPS, IDR) before
    ///   ffmpeg produces a frame, so this returns `Err` (with a "buffered" msg)
    ///   for NALs that don't yet produce output — callers should keep feeding
    ///   NALs and ignore those errors (same contract as the openh264 path).
    /// - For `Software` (H.264 only): passes `payload` directly to openh264.
    ///   Returns `Err` if openh264 reports no frame (e.g. SPS/PPS only).
    pub fn decode(&mut self, payload: &[u8]) -> Result<YuvFrame, DecodeError> {
        match self {
            Decoder::NvDec {
                stdin, frame_rx, ..
            }
            | Decoder::FfmpegSw {
                stdin, frame_rx, ..
            } => {
                // Write the NAL/payload to ffmpeg.  The reader thread
                // independently reads decoded frames from stdout.
                stdin.write_all(payload).map_err(DecodeError::WriteError)?;
                stdin.flush().map_err(DecodeError::FlushError)?;

                // Try to receive a frame without blocking.  If ffmpeg hasn't
                // produced one yet (e.g. still buffering SPS/PPS) we return
                // `DecodeError::Buffered` — zero allocation — the caller feeds
                // the next NAL and retries.
                match frame_rx.try_recv() {
                    Ok(frame) => Ok(frame),
                    Err(mpsc::TryRecvError::Empty) => Err(DecodeError::Buffered),
                    Err(mpsc::TryRecvError::Disconnected) => Err(DecodeError::ReaderExited),
                }
            }
            Decoder::Software { decoder } => Self::decode_software(decoder, payload),
        }
    }

    fn decode_software(
        decoder: &mut openh264::decoder::Decoder,
        nals: &[u8],
    ) -> Result<YuvFrame, DecodeError> {
        let maybe_yuv = decoder
            .decode(nals)
            .map_err(|e| DecodeError::SoftwareDecode(format!("{e}")))?;

        // openh264 returns None when the codec is accumulating parameter sets
        // (SPS/PPS) or partial slices — this is benign buffering, not an error.
        // Hard decode failures surface as Err(...) above (→ SoftwareDecode).
        let yuv = maybe_yuv.ok_or(DecodeError::Buffered)?;

        // dimensions() returns (width, height) in pixels.
        // strides() returns (y_stride, u_stride, v_stride) bytes-per-row.
        // The plane slices may be padded; we copy only the active pixels.
        let (width, height) = yuv.dimensions();
        let (y_stride, u_stride, v_stride) = yuv.strides();

        let uv_width = width / 2;
        let uv_height = height / 2;

        let mut y_packed = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row * y_stride;
            y_packed.extend_from_slice(&yuv.y()[start..start + width]);
        }

        let mut u_packed = Vec::with_capacity(uv_width * uv_height);
        for row in 0..uv_height {
            let start = row * u_stride;
            u_packed.extend_from_slice(&yuv.u()[start..start + uv_width]);
        }

        let mut v_packed = Vec::with_capacity(uv_width * uv_height);
        for row in 0..uv_height {
            let start = row * v_stride;
            v_packed.extend_from_slice(&yuv.v()[start..start + uv_width]);
        }

        Ok(YuvFrame {
            y: y_packed,
            u: u_packed,
            v: v_packed,
            width: width as u32,
            height: height as u32,
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        match self {
            Decoder::NvDec { child, .. } | Decoder::FfmpegSw { child, .. } => {
                // Best-effort kill; ignore errors during shutdown.
                let _ = child.kill();
            }
            Decoder::Software { .. } => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::VideoCodec;

    #[test]
    fn detects_vp8_from_sdp_rtpmap() {
        let sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 VP8/90000\r\n";
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::Vp8);
    }

    #[test]
    fn detects_h264_from_sdp_rtpmap() {
        let sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 H264/90000\r\n";
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::H264);
    }

    #[test]
    fn defaults_to_h264_when_no_codec_in_sdp() {
        let sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\n";
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::H264);
    }

    #[test]
    fn ignores_audio_rtpmap_for_codec_detection() {
        // VP8 appears only in an audio section — should not match.
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 VP8/48000\r\n", // nonsense but tests section scoping
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H264/90000\r\n",
        );
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::H264);
    }

    #[test]
    fn vp8_preferred_when_listed_before_h264_in_video_section() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );
        // first rtpmap wins
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::Vp8);
    }

    #[test]
    fn ffmpeg_input_format_strings_are_correct() {
        assert_eq!(VideoCodec::H264.ffmpeg_input_format(), "h264");
        assert_eq!(VideoCodec::Vp8.ffmpeg_input_format(), "vp8");
    }

    #[test]
    fn video_codec_default_is_h264() {
        assert_eq!(VideoCodec::default(), VideoCodec::H264);
    }
}

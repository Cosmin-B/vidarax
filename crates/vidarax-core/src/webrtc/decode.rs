//! H.264 decode backends: GPU via ffmpeg NVDEC or CPU via openh264.
//!
//! Selects the backend at construction time based on [`DecoderConfig`]. The GPU
//! path spawns a long-lived ffmpeg sidecar process and pipes raw H.264 NAL data
//! to it, reading back planar YUV420. The CPU path uses openh264 in-process,
//! which is lower latency on machines without CUDA but slower per frame.
//!
//! # Example
//!
//! ```no_run
//! use vidarax_core::webrtc::decode::{Decoder, DecoderConfig};
//!
//! let config = DecoderConfig { gpu_available: false };
//! let mut decoder = Decoder::new(&config);
//! // Feed a raw H.264 NAL unit (Annex B or AVCC bytes):
//! // let frame = decoder.decode(&nal_bytes).unwrap();
//! ```

use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use openh264::formats::YUVSource;

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
    /// When `true`, the NVDEC GPU path is used; otherwise openh264 CPU.
    pub gpu_available: bool,
}

impl DecoderConfig {
    /// Auto-detect GPU availability by probing `nvidia-smi`.
    ///
    /// Falls back to CPU (`gpu_available: false`) if the binary is not found
    /// or returns a non-zero exit status.
    pub fn auto_detect() -> Self {
        let gpu_available = Command::new("nvidia-smi")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        Self { gpu_available }
    }
}

/// H.264 decoder that wraps either an NVDEC ffmpeg sidecar or an openh264 instance.
pub enum Decoder {
    /// GPU: long-lived ffmpeg sidecar using NVDEC / h264_cuvid (~0.5 ms/frame).
    ///
    /// Requires a CUDA-capable GPU and `ffmpeg` with `h264_cuvid` compiled in.
    NvDec {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        width: u32,
        height: u32,
    },
    /// CPU: openh264 in-process decoder (~2–5 ms/frame on ARM).
    Software {
        decoder: openh264::decoder::Decoder,
    },
}

impl Decoder {
    /// Create a decoder. Selects NVDEC when `config.gpu_available` is `true`,
    /// otherwise falls back to openh264 software decode.
    ///
    /// # Panics
    ///
    /// Panics if the selected backend cannot be initialised (ffmpeg not found
    /// for NVDEC, or openh264 library init failure for software).
    pub fn new(config: &DecoderConfig) -> Self {
        if config.gpu_available {
            Self::new_nvdec(1920, 1080)
        } else {
            Self::new_software()
        }
    }

    fn new_software() -> Self {
        let decoder =
            openh264::decoder::Decoder::new().expect("openh264 decoder initialisation failed");
        Decoder::Software { decoder }
    }

    fn new_nvdec(width: u32, height: u32) -> Self {
        let mut child = Command::new("ffmpeg")
            .args([
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "nv12",
                "-c:v",
                "h264_cuvid",
                "-f",
                "h264",
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
            .expect("ffmpeg NVDEC spawn failed — is ffmpeg with h264_cuvid available?");

        let stdin = child.stdin.take().expect("ffmpeg stdin missing");
        let stdout = BufReader::new(child.stdout.take().expect("ffmpeg stdout missing"));

        Decoder::NvDec {
            child,
            stdin,
            stdout,
            width,
            height,
        }
    }

    /// Decode raw H.264 NAL unit bytes into a packed YUV420 frame.
    ///
    /// For NVDEC: writes `nals` to the ffmpeg stdin pipe, then reads back one
    /// full YUV frame from stdout. The caller must ensure `nals` contains
    /// exactly one complete access unit.
    ///
    /// For software: passes `nals` directly to openh264. Returns `Err` if
    /// openh264 reports no frame for the given NAL (e.g. SPS/PPS only).
    pub fn decode(&mut self, nals: &[u8]) -> Result<YuvFrame, String> {
        match self {
            Decoder::NvDec {
                stdin,
                stdout,
                width,
                height,
                ..
            } => Self::decode_nvdec(stdin, stdout, *width, *height, nals),
            Decoder::Software { decoder } => Self::decode_software(decoder, nals),
        }
    }

    fn decode_nvdec(
        stdin: &mut ChildStdin,
        stdout: &mut BufReader<ChildStdout>,
        width: u32,
        height: u32,
    nals: &[u8],
    ) -> Result<YuvFrame, String> {
        let w = width as usize;
        let h = height as usize;

        stdin
            .write_all(nals)
            .map_err(|e| format!("ffmpeg write: {e}"))?;
        stdin.flush().map_err(|e| format!("ffmpeg flush: {e}"))?;

        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        let mut y = vec![0u8; y_size];
        let mut u = vec![0u8; uv_size];
        let mut v = vec![0u8; uv_size];

        stdout
            .read_exact(&mut y)
            .map_err(|e| format!("ffmpeg read Y plane: {e}"))?;
        stdout
            .read_exact(&mut u)
            .map_err(|e| format!("ffmpeg read U plane: {e}"))?;
        stdout
            .read_exact(&mut v)
            .map_err(|e| format!("ffmpeg read V plane: {e}"))?;

        Ok(YuvFrame {
            y,
            u,
            v,
            width,
            height,
        })
    }

    fn decode_software(
        decoder: &mut openh264::decoder::Decoder,
        nals: &[u8],
    ) -> Result<YuvFrame, String> {
        let maybe_yuv = decoder
            .decode(nals)
            .map_err(|e| format!("openh264 decode error: {e}"))?;

        let yuv = maybe_yuv.ok_or_else(|| {
            "openh264: no frame output (NAL may be SPS/PPS or buffered)".to_string()
        })?;

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
        if let Decoder::NvDec { child, .. } = self {
            // Best-effort kill; ignore errors during shutdown.
            let _ = child.kill();
        }
    }
}

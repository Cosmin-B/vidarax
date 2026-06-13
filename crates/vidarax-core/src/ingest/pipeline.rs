//! Pluggable video-decode backends behind a single trait.
//!
//! Decoding has two phases: cheap frame-signal extraction for the gate engine,
//! then selective JPEG extraction for the frames the gate keeps. Each backend
//! implements both phases, so callers swap decode implementations without
//! touching handler code.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::{Arc, OnceLock};

use crate::ingest::{
    decode_mp4_to_frame_signals, decode_selective_jpeg_frames, DecodedJpegFrame, DecodedMp4Batch,
    InputSource, Mp4DecodeConfig,
};

static DETECTED_BACKEND: OnceLock<PipelineBackend> = OnceLock::new();
type DecodeFactory = fn() -> Arc<dyn DecodePipeline>;
static DECODE_REGISTRY: OnceLock<RwLock<HashMap<&'static str, DecodeFactory>>> = OnceLock::new();

/// Hardware backend used for video decode and JPEG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// ffmpeg subprocess, CPU decode, CPU JPEG encode. Works everywhere.
    CpuFfmpeg,
    /// ffmpeg `-hwaccel nvdec` for decode, CUDA for JPEG encode. Requires an NVIDIA GPU.
    NvdecCuda,
    /// MLX framework for Apple Silicon. Requires macOS on an M-series chip.
    Mlx,
}

impl PipelineBackend {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" | "ffmpeg" | "cpu-ffmpeg" => Ok(Self::CpuFfmpeg),
            "nvdec" | "cuda" | "nvdec-cuda" | "gpu" => Ok(Self::NvdecCuda),
            "mlx" | "apple" | "metal" => Ok(Self::Mlx),
            other => Err(format!(
                "unknown decode backend '{other}', expected one of: cpu, nvdec, mlx"
            )),
        }
    }

    /// Detect the best available backend. The result is cached after the first call.
    pub fn auto_detect() -> Self {
        *DETECTED_BACKEND.get_or_init(|| {
            let detected = if std::process::Command::new(super::nvidia_smi_path())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
            {
                Self::NvdecCuda
            } else {
                #[cfg(target_os = "macos")]
                {
                    if std::process::Command::new("python3")
                        .args(["-c", "import mlx.core"])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false)
                    {
                        Self::Mlx
                    } else {
                        Self::CpuFfmpeg
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Self::CpuFfmpeg
                }
            };
            if decode_registry()
                .read()
                .expect("decode registry poisoned")
                .contains_key(detected.label())
            {
                detected
            } else {
                Self::CpuFfmpeg
            }
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::CpuFfmpeg => "cpu-ffmpeg",
            Self::NvdecCuda => "nvdec-cuda",
            Self::Mlx => "mlx",
        }
    }
}

/// What a backend can actually do today.
///
/// Stub backends report `hardware_decode = false` and fall back to the CPU path,
/// so a caller that selects an unimplemented backend gets a warning instead of
/// silent degradation.
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    pub hardware_decode: bool,
    pub notes: &'static str,
}

/// A decode backend. Implementations run the two-phase flow: frame signals for
/// the gate engine, then selective JPEG extraction for the frames it keeps.
pub trait DecodePipeline: Send + Sync {
    /// Phase 1: per-frame signals (hashes, luma) for the gate engine.
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String>;

    /// Phase 2: JPEG frames at `frame_indices` (computed from the gate output).
    /// An empty `frame_indices` returns no frames without invoking ffmpeg.
    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
    ) -> Result<Vec<DecodedJpegFrame>, String>;

    fn backend(&self) -> PipelineBackend;

    fn capabilities(&self) -> BackendCapabilities;
}

// ---------------------------------------------------------------------------
// CPU / ffmpeg backend
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct CpuFfmpegPipeline;

impl CpuFfmpegPipeline {
    pub fn new() -> Self {
        Self
    }
}

impl DecodePipeline for CpuFfmpegPipeline {
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String> {
        decode_mp4_to_frame_signals(source, config)
    }

    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }
        decode_selective_jpeg_frames(source, sample_fps, frame_indices, max_frames)
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::CpuFfmpeg
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            hardware_decode: false,
            notes: "CPU ffmpeg decode and JPEG encode",
        }
    }
}

// ---------------------------------------------------------------------------
// NVDEC + CUDA backend
// ---------------------------------------------------------------------------

pub struct NvdecCudaPipeline;

impl DecodePipeline for NvdecCudaPipeline {
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String> {
        // framemd5 is text output with no GPU benefit, so phase 1 stays on CPU.
        decode_mp4_to_frame_signals(source, config)
    }

    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }
        decode_selective_jpeg_frames_nvdec(source, sample_fps, frame_indices, max_frames)
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::NvdecCuda
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            hardware_decode: true,
            notes: "NVDEC GPU decode for JPEG phase, CPU framemd5 for signals",
        }
    }
}

fn decode_selective_jpeg_frames_nvdec(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
) -> Result<Vec<DecodedJpegFrame>, String> {
    use crate::ingest::{build_select_expr, FFMPEG_PROTOCOL_WHITELIST};
    use std::process::Command;

    let select_expr = build_select_expr(frame_indices);
    let vf_chain =
        format!("fps={sample_fps:.3},hwdownload,format=nv12,{select_expr},format=yuv420p");
    let frames_cap = frame_indices.len().min(max_frames).to_string();

    let output = Command::new(super::ffmpeg_path())
        .args([
            "-hwaccel",
            "nvdec",
            "-hwaccel_output_format",
            "cuda",
            "-v",
            "error",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            source.as_ffmpeg_input(),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &vf_chain,
            "-vsync",
            "vfr",
            "-frames:v",
            &frames_cap,
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ])
        .output()
        .map_err(|_| "failed to run ffmpeg with NVDEC".to_string())?;

    if !output.status.success() {
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            "ffmpeg nvdec decode failed"
        );
        return Err("GPU video decode failed".to_string());
    }

    let mut parsed =
        crate::ingest::parse_jpeg_stream_to_frames(&output.stdout, frame_indices.len())?;
    let usable = parsed.len().min(frame_indices.len());
    parsed.truncate(usable);
    for (frame, &idx) in parsed.iter_mut().zip(frame_indices.iter()) {
        frame.frame_index = idx;
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// MLX backend (Apple Silicon)
// ---------------------------------------------------------------------------

/// MLX is not yet implemented. Both phases fall back to the CPU ffmpeg path, and
/// [`BackendCapabilities`] reports the fallback so callers are not misled.
pub struct MlxPipeline;

impl DecodePipeline for MlxPipeline {
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String> {
        CpuFfmpegPipeline.decode_signals(source, config)
    }

    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        CpuFfmpegPipeline.decode_jpegs(source, sample_fps, frame_indices, max_frames)
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::Mlx
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            hardware_decode: false,
            notes: "MLX backend not yet implemented; falls back to CPU ffmpeg",
        }
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

fn cpu_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(CpuFfmpegPipeline::new())
}

fn nvdec_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(NvdecCudaPipeline)
}

fn mlx_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(MlxPipeline)
}

fn decode_registry() -> &'static RwLock<HashMap<&'static str, DecodeFactory>> {
    DECODE_REGISTRY.get_or_init(|| {
        let mut registry = HashMap::new();
        for name in ["cpu", "ffmpeg", "cpu-ffmpeg"] {
            registry.insert(name, cpu_pipeline as DecodeFactory);
        }
        for name in ["nvdec", "cuda", "nvdec-cuda", "gpu"] {
            registry.insert(name, nvdec_pipeline as DecodeFactory);
        }
        for name in ["mlx", "apple", "metal"] {
            registry.insert(name, mlx_pipeline as DecodeFactory);
        }
        RwLock::new(registry)
    })
}

pub fn register_decode_backend(name: &'static str, factory: DecodeFactory) {
    decode_registry()
        .write()
        .expect("decode registry poisoned")
        .insert(name, factory);
}

pub fn build_decode_pipeline(name: &str) -> Result<Arc<dyn DecodePipeline>, String> {
    let factory = {
        let registry = decode_registry()
            .read()
            .map_err(|_| "decode registry poisoned".to_string())?;
        registry.get(name).copied()
    };
    factory
        .map(|build| build())
        .ok_or_else(|| format!("unknown decode backend '{name}'"))
}

/// Build the decode pipeline for `backend`. A backend that selects no hardware
/// acceleration (an unimplemented stub) logs a warning so the fallback is visible.
pub fn create_pipeline(backend: PipelineBackend) -> Arc<dyn DecodePipeline> {
    let pipeline = build_decode_pipeline(backend.label()).expect("built-in decode backend missing");

    let caps = pipeline.capabilities();
    if backend != PipelineBackend::CpuFfmpeg && !caps.hardware_decode {
        tracing::warn!(
            backend = backend.label(),
            notes = caps.notes,
            "decode backend has no hardware acceleration; using CPU fallback"
        );
    }
    pipeline
}

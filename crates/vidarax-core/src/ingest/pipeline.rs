//! Abstract decode pipeline trait with pluggable backends.
//!
//! Backends:
//! - `CpuFfmpeg`: ffmpeg subprocess (works everywhere)
//! - `NvdecCuda`: ffmpeg -hwaccel nvdec + CUDA JPEG encoder (Hetzner GPU)
//! - `Mlx`: MLX Framework for Apple Silicon (Mac)

use std::sync::OnceLock;

use crate::ingest::{
    decode_mp4_to_frame_signals, decode_selective_jpeg_frames, DecodedJpegFrame,
    DecodedMp4Batch, InputSource, Mp4DecodeConfig,
};

static DETECTED_BACKEND: OnceLock<PipelineBackend> = OnceLock::new();

/// Which hardware backend to use for video decode + JPEG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// ffmpeg subprocess, CPU decode, CPU JPEG encode. Works everywhere.
    CpuFfmpeg,
    /// ffmpeg -hwaccel nvdec for decode, CUDA for JPEG encode. Requires NVIDIA GPU.
    NvdecCuda,
    /// MLX Framework for Apple Silicon. Requires macOS + M-series chip.
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

    /// Auto-detect best available backend (result cached after first call).
    pub fn auto_detect() -> Self {
        *DETECTED_BACKEND.get_or_init(|| {
            // Check for NVIDIA GPU
            if std::process::Command::new(super::nvidia_smi_path())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
            {
                return Self::NvdecCuda;
            }
            // Check for Apple Silicon MLX
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
                    return Self::Mlx;
                }
            }
            Self::CpuFfmpeg
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

/// Configuration for the decode pipeline.
#[derive(Debug, Clone)]
pub struct DecodePipelineConfig {
    pub backend: PipelineBackend,
    pub sample_fps: f32,
    pub max_frames: usize,
}

impl Default for DecodePipelineConfig {
    fn default() -> Self {
        Self {
            backend: PipelineBackend::CpuFfmpeg,
            sample_fps: 2.0,
            max_frames: 512,
        }
    }
}

/// Result of decoding a video source through the pipeline.
pub struct DecodeResult {
    /// Frame signals for the gate engine (from framemd5 or equivalent).
    pub batch: DecodedMp4Batch,
    /// Only the JPEG frames selected for VLM inference (not all frames).
    pub selected_jpegs: Vec<DecodedJpegFrame>,
}

/// Abstract decode pipeline. Implementations handle the full flow:
/// 1. Extract frame signals (hashes, luma, etc.) for the gate engine
/// 2. Selectively extract JPEG frames for VLM inference
///
/// The trait allows swapping backends (CPU/NVDEC/MLX) without changing
/// handler code.
pub trait DecodePipeline: Send + Sync {
    /// Decode the source in a single logical operation.
    /// `semantic_frame_indices` are pre-computed by `compute_semantic_frame_indices`.
    fn decode(
        &self,
        source: &InputSource,
        semantic_frame_indices: &[u64],
    ) -> Result<DecodeResult, String>;

    /// Which backend this pipeline uses.
    fn backend(&self) -> PipelineBackend;
}

// ---------------------------------------------------------------------------
// CPU/ffmpeg backend
// ---------------------------------------------------------------------------

pub struct CpuFfmpegPipeline {
    config: DecodePipelineConfig,
}

impl CpuFfmpegPipeline {
    pub fn new(config: DecodePipelineConfig) -> Self {
        Self { config }
    }
}

impl DecodePipeline for CpuFfmpegPipeline {
    fn decode(
        &self,
        source: &InputSource,
        semantic_frame_indices: &[u64],
    ) -> Result<DecodeResult, String> {
        let mp4_config = Mp4DecodeConfig {
            sample_fps: self.config.sample_fps,
            max_frames: self.config.max_frames,
        };

        // Pass 1: framemd5 for gate engine signals (cheap, no encoding)
        let batch = decode_mp4_to_frame_signals(source, mp4_config)?;

        // Pass 2: selective JPEG — only the frames needed for VLM
        let selected_jpegs = if semantic_frame_indices.is_empty() {
            Vec::new()
        } else {
            decode_selective_jpeg_frames(
                source,
                self.config.sample_fps,
                semantic_frame_indices,
                self.config.max_frames,
            )?
        };

        Ok(DecodeResult { batch, selected_jpegs })
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::CpuFfmpeg
    }
}

// ---------------------------------------------------------------------------
// NVDEC + CUDA backend (stub — full implementation in a follow-up task)
// ---------------------------------------------------------------------------

pub struct NvdecCudaPipeline {
    config: DecodePipelineConfig,
}

impl NvdecCudaPipeline {
    pub fn new(config: DecodePipelineConfig) -> Self {
        Self { config }
    }
}

impl DecodePipeline for NvdecCudaPipeline {
    fn decode(
        &self,
        source: &InputSource,
        semantic_frame_indices: &[u64],
    ) -> Result<DecodeResult, String> {
        let mp4_config = Mp4DecodeConfig {
            sample_fps: self.config.sample_fps,
            max_frames: self.config.max_frames,
        };

        // Pass 1: framemd5 (same as CPU — framemd5 is text output, no GPU benefit)
        let batch = decode_mp4_to_frame_signals(source, mp4_config)?;

        // Pass 2: NVDEC selective JPEG — GPU decode, CPU-side select, JPEG encode
        // Uses: ffmpeg -hwaccel nvdec -hwaccel_output_format cuda -i src
        //       -vf "fps=N,hwdownload,format=nv12,select='...',format=yuv420p"
        //       -vsync vfr -f image2pipe -vcodec mjpeg -
        let selected_jpegs = if semantic_frame_indices.is_empty() {
            Vec::new()
        } else {
            decode_selective_jpeg_frames_nvdec(
                source,
                self.config.sample_fps,
                semantic_frame_indices,
                self.config.max_frames,
            )?
        };

        Ok(DecodeResult { batch, selected_jpegs })
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::NvdecCuda
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

    if frame_indices.is_empty() {
        return Ok(Vec::new());
    }

    let select_expr = build_select_expr(frame_indices);
    let vf_chain = format!(
        "fps={sample_fps:.3},hwdownload,format=nv12,{select_expr},format=yuv420p"
    );
    let frames_cap = frame_indices.len().min(max_frames).to_string();

    let output = Command::new(super::ffmpeg_path())
        .args([
            "-hwaccel", "nvdec",
            "-hwaccel_output_format", "cuda",
            "-v", "error",
            "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST,
            "-i", source.as_ffmpeg_input(),
            "-an", "-sn", "-dn",
            "-vf", &vf_chain,
            "-vsync", "vfr",
            "-frames:v", &frames_cap,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
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

    let mut parsed = crate::ingest::parse_jpeg_stream_to_frames(&output.stdout, frame_indices.len())?;
    let usable = parsed.len().min(frame_indices.len());
    parsed.truncate(usable);
    for (frame, &idx) in parsed.iter_mut().zip(frame_indices.iter()) {
        frame.frame_index = idx;
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// MLX backend (stub — Apple Silicon, future implementation)
// ---------------------------------------------------------------------------

pub struct MlxPipeline {
    config: DecodePipelineConfig,
}

impl MlxPipeline {
    pub fn new(config: DecodePipelineConfig) -> Self {
        Self { config }
    }
}

impl DecodePipeline for MlxPipeline {
    fn decode(
        &self,
        source: &InputSource,
        semantic_frame_indices: &[u64],
    ) -> Result<DecodeResult, String> {
        // MLX path: use VideoToolbox for hardware decode on Apple Silicon,
        // then MLX for any GPU-side processing.
        // For now, fall back to CPU ffmpeg pipeline.
        // TODO: implement native VideoToolbox decode + MLX JPEG encode
        let fallback = CpuFfmpegPipeline::new(self.config.clone());
        fallback.decode(source, semantic_frame_indices)
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::Mlx
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create the appropriate decode pipeline based on config.
pub fn create_pipeline(config: DecodePipelineConfig) -> Box<dyn DecodePipeline> {
    match config.backend {
        PipelineBackend::CpuFfmpeg => Box::new(CpuFfmpegPipeline::new(config)),
        PipelineBackend::NvdecCuda => Box::new(NvdecCudaPipeline::new(config)),
        PipelineBackend::Mlx => Box::new(MlxPipeline::new(config)),
    }
}

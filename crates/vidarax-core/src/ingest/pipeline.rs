//! Pluggable video-decode backends behind a single trait.
//!
//! Decoding has two phases: cheap frame-signal extraction for the gate engine,
//! then selective JPEG extraction for the frames the gate keeps. Each backend
//! implements both phases, so callers swap decode implementations without
//! touching handler code.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::{Arc, OnceLock};

use crate::crop::CropRegion;
use crate::ingest::{
    decode_mp4_to_frame_signals, decode_selective_jpeg_frames, extract_video_clip,
    DecodedJpegFrame, DecodedMp4Batch, InputSource, Mp4DecodeConfig,
};

static DETECTED_BACKEND: OnceLock<PipelineBackend> = OnceLock::new();
type DecodeFactory = fn() -> Arc<dyn DecodePipeline>;
static DECODE_REGISTRY: OnceLock<RwLock<HashMap<&'static str, DecodeFactory>>> = OnceLock::new();

/// Hardware backend used for video decode and JPEG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// ffmpeg subprocess, CPU decode, CPU JPEG encode. Works everywhere.
    CpuFfmpeg,
    /// ffmpeg `-hwaccel nvdec` decodes on the GPU; frames are then downloaded and JPEG-encoded on
    /// the CPU. Requires an NVIDIA GPU.
    NvdecCuda,
    /// ffmpeg `-hwaccel videotoolbox` decodes on Apple Silicon's media engine for the JPEG
    /// phase; frames are JPEG-encoded on the CPU. Requires an ffmpeg build with VideoToolbox
    /// support.
    VideoToolbox,
}

impl PipelineBackend {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" | "ffmpeg" | "cpu-ffmpeg" => Ok(Self::CpuFfmpeg),
            "nvdec" | "cuda" | "nvdec-cuda" | "gpu" => Ok(Self::NvdecCuda),
            "mlx" | "apple" | "metal" | "videotoolbox" => Ok(Self::VideoToolbox),
            other => Err(format!(
                "unknown decode backend '{other}', expected one of: cpu, nvdec, videotoolbox"
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
                Self::CpuFfmpeg
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
            Self::VideoToolbox => "videotoolbox",
        }
    }
}

/// Backend capability flags used by fallback warnings.
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    pub hardware_decode: bool,
    pub notes: &'static str,
}

/// A decode backend. Implementations run the two-phase flow: frame signals for
/// the gate engine, then selective JPEG extraction for the frames it keeps.
pub trait DecodePipeline: Send + Sync {
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String>;

    /// An empty `frame_indices` returns no frames without invoking ffmpeg.
    ///
    /// `max_edge` optionally caps the longest edge of each emitted frame (see
    /// [`Mp4DecodeConfig::max_edge`]) — the "fewer pixels" lever for VLM token cost.
    /// `crop` optionally restricts each frame to a region of interest (see
    /// [`Mp4DecodeConfig::crop`]); pass the same crop used for the signals pass so
    /// the gate and the VLM agree on what was analyzed.
    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
        max_edge: Option<u32>,
        crop: Option<CropRegion>,
    ) -> Result<Vec<DecodedJpegFrame>, String>;

    /// Extract a short self-contained MP4 clip beginning at `start_s` and
    /// running for `duration_s` seconds, optionally restricted to `crop`. Clip-
    /// mode realtime analysis hands the VLM this moving window instead of stills,
    /// so it rides the same swappable backend as the two decode phases, and the
    /// crop keeps the clip pinned to the same region the gate saw.
    fn extract_clip(
        &self,
        source: &InputSource,
        start_s: f32,
        duration_s: f32,
        crop: Option<CropRegion>,
    ) -> Result<Vec<u8>, String>;

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
        max_edge: Option<u32>,
        crop: Option<CropRegion>,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }
        decode_selective_jpeg_frames(
            source,
            sample_fps,
            frame_indices,
            max_frames,
            max_edge,
            crop,
        )
    }

    fn extract_clip(
        &self,
        source: &InputSource,
        start_s: f32,
        duration_s: f32,
        crop: Option<CropRegion>,
    ) -> Result<Vec<u8>, String> {
        extract_video_clip(source, start_s, duration_s, crop)
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
        // framemd5 is text output with no GPU benefit.
        decode_mp4_to_frame_signals(source, config)
    }

    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
        max_edge: Option<u32>,
        crop: Option<CropRegion>,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }
        decode_selective_jpeg_frames_nvdec(
            source,
            sample_fps,
            frame_indices,
            max_frames,
            max_edge,
            crop,
        )
    }

    fn extract_clip(
        &self,
        source: &InputSource,
        start_s: f32,
        duration_s: f32,
        crop: Option<CropRegion>,
    ) -> Result<Vec<u8>, String> {
        // Local clips are a stream copy and remote ones a short segment re-encode,
        // so NVDEC buys no decode win here. Reuse the CPU extractor.
        extract_video_clip(source, start_s, duration_s, crop)
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
    max_edge: Option<u32>,
    crop: Option<CropRegion>,
) -> Result<Vec<DecodedJpegFrame>, String> {
    super::fetch::with_prefetched_downloadable_source(source, |source| {
        decode_selective_jpeg_frames_nvdec_inner(
            source,
            sample_fps,
            frame_indices,
            max_frames,
            max_edge,
            crop,
        )
    })?
}

fn decode_selective_jpeg_frames_nvdec_inner(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
    max_edge: Option<u32>,
    crop: Option<CropRegion>,
) -> Result<Vec<DecodedJpegFrame>, String> {
    use crate::ingest::{
        build_select_expr, ffmpeg_input_options_for_source, ffmpeg_protocol_whitelist_for_source,
    };
    use std::process::Command;

    let select_expr = build_select_expr(frame_indices);
    // The crop runs on the CPU side after hwdownload, so it must sit right after
    // `format=nv12` and before select/scale. `iw`/`ih` in the crop expression
    // resolve to the downloaded frame size, which is the source resolution.
    let crop = crop
        .filter(|c| !c.is_full_frame())
        .map(|c| format!("{},", c.ffmpeg_crop_filter()))
        .unwrap_or_default();
    let vf_chain = match super::ffmpeg::longest_edge_scale_filter(max_edge) {
        Some(scale) => format!(
            "fps={sample_fps:.3},hwdownload,format=nv12,{crop}{select_expr},format=yuv420p,{scale}"
        ),
        None => {
            format!("fps={sample_fps:.3},hwdownload,format=nv12,{crop}{select_expr},format=yuv420p")
        }
    };
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
            ffmpeg_protocol_whitelist_for_source(source),
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
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
// VideoToolbox (Apple Silicon) backend
// ---------------------------------------------------------------------------

pub struct MlxVideoToolboxPipeline;

impl DecodePipeline for MlxVideoToolboxPipeline {
    fn decode_signals(
        &self,
        source: &InputSource,
        config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String> {
        // framemd5 is text output with no hardware-decode benefit, same reasoning
        // as NvdecCudaPipeline above.
        decode_mp4_to_frame_signals(source, config)
    }

    fn decode_jpegs(
        &self,
        source: &InputSource,
        sample_fps: f32,
        frame_indices: &[u64],
        max_frames: usize,
        max_edge: Option<u32>,
        crop: Option<CropRegion>,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }
        decode_selective_jpeg_frames_videotoolbox(
            source,
            sample_fps,
            frame_indices,
            max_frames,
            max_edge,
            crop,
        )
    }

    fn extract_clip(
        &self,
        source: &InputSource,
        start_s: f32,
        duration_s: f32,
        crop: Option<CropRegion>,
    ) -> Result<Vec<u8>, String> {
        // Local clips are a stream copy and remote ones a short segment re-encode,
        // so VideoToolbox buys no decode win here either. Reuse the CPU extractor.
        extract_video_clip(source, start_s, duration_s, crop)
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::VideoToolbox
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            hardware_decode: true,
            notes: "VideoToolbox hardware decode (Apple Silicon); signals phase runs on CPU",
        }
    }
}

fn decode_selective_jpeg_frames_videotoolbox(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
    max_edge: Option<u32>,
    crop: Option<CropRegion>,
) -> Result<Vec<DecodedJpegFrame>, String> {
    super::fetch::with_prefetched_downloadable_source(source, |source| {
        decode_selective_jpeg_frames_videotoolbox_inner(
            source,
            sample_fps,
            frame_indices,
            max_frames,
            max_edge,
            crop,
        )
    })?
}

fn decode_selective_jpeg_frames_videotoolbox_inner(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
    max_edge: Option<u32>,
    crop: Option<CropRegion>,
) -> Result<Vec<DecodedJpegFrame>, String> {
    use crate::ingest::{
        build_select_expr, ffmpeg_input_options_for_source, ffmpeg_protocol_whitelist_for_source,
    };
    use std::process::Command;

    let select_expr = build_select_expr(frame_indices);
    // Unlike the nvdec path, this leaves -hwaccel_output_format unset, so
    // VideoToolbox hands decoded frames back in a normal software pixel format
    // and the filter chain here is the same crop+fps+select+scale chain the CPU
    // path uses. No hwdownload/format step is needed before the mjpeg encoder.
    let crop = crop
        .filter(|c| !c.is_full_frame())
        .map(|c| format!("{},", c.ffmpeg_crop_filter()))
        .unwrap_or_default();
    let vf_chain = match super::ffmpeg::longest_edge_scale_filter(max_edge) {
        Some(scale) => format!("{crop}fps={sample_fps:.3},{select_expr},{scale}"),
        None => format!("{crop}fps={sample_fps:.3},{select_expr}"),
    };
    let frames_cap = frame_indices.len().min(max_frames).to_string();

    let output = Command::new(super::ffmpeg_path())
        .args([
            "-hwaccel",
            "videotoolbox",
            "-v",
            "error",
            "-protocol_whitelist",
            ffmpeg_protocol_whitelist_for_source(source),
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
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
        .map_err(|_| "failed to run ffmpeg with VideoToolbox".to_string())?;

    if !output.status.success() {
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            "ffmpeg videotoolbox decode failed"
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
// Factory
// ---------------------------------------------------------------------------

fn cpu_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(CpuFfmpegPipeline::new())
}

fn nvdec_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(NvdecCudaPipeline)
}

fn videotoolbox_pipeline() -> Arc<dyn DecodePipeline> {
    Arc::new(MlxVideoToolboxPipeline)
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
        for name in ["mlx", "apple", "metal", "videotoolbox"] {
            registry.insert(name, videotoolbox_pipeline as DecodeFactory);
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

/// Build the decode pipeline for `backend`.
pub fn create_pipeline(backend: PipelineBackend) -> Arc<dyn DecodePipeline> {
    build_decode_pipeline(backend.label()).expect("built-in decode backend missing")
}

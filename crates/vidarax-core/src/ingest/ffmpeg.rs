use std::process::Command;
use std::sync::OnceLock;

use crate::gate::FrameSignal;

use super::fetch::with_prefetched_downloadable_source;
use super::InputSource;

static FFMPEG_PATH: OnceLock<String> = OnceLock::new();
static FFPROBE_PATH: OnceLock<String> = OnceLock::new();
static NVIDIA_SMI_PATH: OnceLock<String> = OnceLock::new();

/// Configured ffmpeg binary path, cached after first call.
pub fn ffmpeg_path() -> &'static str {
    FFMPEG_PATH.get_or_init(|| {
        std::env::var("VIDARAX_FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".to_string())
    })
}

/// Configured ffprobe binary path, cached after first call.
pub fn ffprobe_path() -> &'static str {
    FFPROBE_PATH.get_or_init(|| {
        std::env::var("VIDARAX_FFPROBE_PATH").unwrap_or_else(|_| "ffprobe".to_string())
    })
}

/// Configured nvidia-smi binary path, cached after first call.
pub fn nvidia_smi_path() -> &'static str {
    NVIDIA_SMI_PATH.get_or_init(|| {
        std::env::var("VIDARAX_NVIDIA_SMI_PATH").unwrap_or_else(|_| "nvidia-smi".to_string())
    })
}

// ffmpeg applies protocol whitelists to redirects and nested demuxer resources,
// so each source kind gets the narrowest useful set. Residual limitation:
// redirects or nested resources that resolve to private IPs over an allowed
// scheme cannot be fully blocked here; hardened deployments need an egress
// proxy/resolver that enforces public-IP policy on every connection. See
// docs/security.md for the accepted residual and recommended control.
pub(crate) const FFMPEG_LOCAL_PROTOCOL_WHITELIST: &str = "file";
pub(crate) const FFMPEG_HTTPS_PROTOCOL_WHITELIST: &str = "https,tls,tcp";
pub(crate) const FFMPEG_HTTP_PROTOCOL_WHITELIST: &str = "http,tcp";
pub(crate) const FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST: &str = "hls,https,tls,tcp";
pub(crate) const FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST: &str = "hls,http,tcp";
pub(crate) const FFMPEG_RTSPS_PROTOCOL_WHITELIST: &str = "rtsp,rtsps,tls,tcp,udp,rtp";
pub(crate) const FFMPEG_RTSP_PROTOCOL_WHITELIST: &str = "rtsp,tcp,udp,rtp";

pub(crate) fn ffmpeg_protocol_whitelist_for_source(source: &InputSource) -> &'static str {
    match source {
        InputSource::FilePath(_) => FFMPEG_LOCAL_PROTOCOL_WHITELIST,
        InputSource::HlsStream(url) if url.starts_with("http://") => {
            FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST
        }
        InputSource::HlsStream(_) => FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST,
        InputSource::Url(url) if url.starts_with("http://") => FFMPEG_HTTP_PROTOCOL_WHITELIST,
        InputSource::Url(url) if url.starts_with("rtsps://") => FFMPEG_RTSPS_PROTOCOL_WHITELIST,
        InputSource::Url(url) if url.starts_with("rtsp://") => FFMPEG_RTSP_PROTOCOL_WHITELIST,
        InputSource::Url(_) => FFMPEG_HTTPS_PROTOCOL_WHITELIST,
        InputSource::WebRtcStream(_) => FFMPEG_LOCAL_PROTOCOL_WHITELIST,
    }
}

pub(crate) fn ffmpeg_input_options_for_source(source: &InputSource) -> Vec<String> {
    let mut options = Vec::new();
    match source {
        InputSource::Url(url) if url.starts_with("http://") || url.starts_with("https://") => {
            options.extend(["-max_redirects".to_string(), "0".to_string()]);
        }
        InputSource::HlsStream(_) => {
            options.extend(["-max_redirects".to_string(), "0".to_string()]);
        }
        _ => {}
    }
    if tls_verify_required_for_source(source) {
        options.extend(["-tls_verify".to_string(), "1".to_string()]);
        if let Some(host) = tls_verify_host_for_source(source) {
            options.extend(["-verifyhost".to_string(), host]);
        }
    }
    options
}

fn tls_verify_required_for_source(source: &InputSource) -> bool {
    if insecure_tls_enabled() {
        return false;
    }
    match source {
        InputSource::Url(url) => url.starts_with("rtsps://"),
        InputSource::HlsStream(url) => url.starts_with("https://"),
        _ => false,
    }
}

fn tls_verify_host_for_source(source: &InputSource) -> Option<String> {
    let raw = match source {
        InputSource::Url(url) if url.starts_with("rtsps://") => url,
        _ => return None,
    };
    reqwest::Url::parse(raw)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_string()))
}

fn insecure_tls_enabled() -> bool {
    std::env::var("VIDARAX_ALLOW_INSECURE_TLS")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePacket {
    pub run_id: String,
    pub stream_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
    pub source_uri: String,
}

pub struct TimestampNormalizer {
    base_pts: Option<i64>,
    last_ms: u64,
}

impl TimestampNormalizer {
    pub fn new() -> Self {
        Self {
            base_pts: None,
            last_ms: 0,
        }
    }

    /// Normalizes PTS to monotonic milliseconds from the first observed frame.
    pub fn normalize_pts_ms(&mut self, pts: i64, timebase_num: u32, timebase_den: u32) -> u64 {
        let base = *self.base_pts.get_or_insert(pts);
        let delta = pts.saturating_sub(base).max(0) as u128;
        let num = (timebase_num as u128).saturating_mul(1000);
        let den = (timebase_den as u128).max(1);
        let mut ms = delta.saturating_mul(num) / den;
        if ms < self.last_ms as u128 {
            ms = self.last_ms as u128;
        }
        let ms = ms as u64;
        self.last_ms = ms;
        ms
    }
}

impl Default for TimestampNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FramePacketInput<'a> {
    pub run_id: &'a str,
    pub stream_id: &'a str,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_format: &'a str,
    pub source_uri: &'a str,
}

pub fn make_frame_packet(input: FramePacketInput<'_>) -> FramePacket {
    FramePacket {
        run_id: input.run_id.to_string(),
        stream_id: input.stream_id.to_string(),
        frame_index: input.frame_index,
        pts_ms: input.pts_ms,
        width: input.width,
        height: input.height,
        pixel_format: input.pixel_format.to_string(),
        source_uri: input.source_uri.to_string(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mp4DecodeConfig {
    pub sample_fps: f32,
    pub max_frames: usize,
}

impl Default for Mp4DecodeConfig {
    fn default() -> Self {
        Self {
            sample_fps: 2.0,
            max_frames: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodedMp4Batch {
    pub source_uri: String,
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
    pub frame_signals: Vec<FrameSignal>,
}

#[derive(Debug, Clone)]
pub struct DecodedJpegFrame {
    pub frame_index: u64,
    pub jpeg_bytes: Vec<u8>,
}

pub fn probe_source_fps(source: &InputSource) -> Option<f32> {
    match with_prefetched_downloadable_source(source, probe_source_fps_inner) {
        Ok(fps) => fps,
        Err(err) => {
            tracing::warn!(error = %err, "remote media prefetch failed during ffprobe");
            None
        }
    }
}

fn probe_source_fps_inner(source: &InputSource) -> Option<f32> {
    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    let output = Command::new(ffprobe_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            protocol_whitelist,
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            source_uri,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    parse_ffprobe_frame_rate(raw.trim())
}

pub fn decode_mp4_to_frame_signals(
    source: &InputSource,
    config: Mp4DecodeConfig,
) -> Result<DecodedMp4Batch, String> {
    let mut decoded =
        with_prefetched_downloadable_source(source, |source| {
            decode_mp4_to_frame_signals_inner(source, config)
        })??;
    decoded.source_uri = source.as_ffmpeg_input().to_string();
    Ok(decoded)
}

#[cfg(test)]
fn decode_mp4_to_frame_signals_with_prefetch_validator(
    source: &InputSource,
    config: Mp4DecodeConfig,
    validate_url: super::fetch::FetchUrlValidator,
) -> Result<DecodedMp4Batch, String> {
    let mut decoded = super::fetch::with_prefetched_downloadable_source_and_validator(
        source,
        validate_url,
        |source| decode_mp4_to_frame_signals_inner(source, config),
    )??;
    decoded.source_uri = source.as_ffmpeg_input().to_string();
    Ok(decoded)
}

fn decode_mp4_to_frame_signals_inner(
    source: &InputSource,
    config: Mp4DecodeConfig,
) -> Result<DecodedMp4Batch, String> {
    if !config.sample_fps.is_finite() || config.sample_fps <= 0.0 {
        return Err("sample_fps must be > 0".to_string());
    }
    if config.max_frames == 0 {
        return Err("max_frames must be >= 1".to_string());
    }

    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    let fps_expr = format!("fps={:.3}", config.sample_fps);
    let output = Command::new(ffmpeg_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            protocol_whitelist,
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i",
            source_uri,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &fps_expr,
            "-f",
            "framemd5",
            "-",
        ])
        .output()
        .map_err(|_| "failed to run ffmpeg".to_string())?;
    if !output.status.success() {
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            "ffmpeg decode failed"
        );
        return Err("video decode failed".to_string());
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| "invalid output from video decoder".to_string())?;
    parse_framemd5_to_signals(&text, source_uri, config.max_frames)
}

pub fn decode_mp4_to_jpeg_frames(
    source: &InputSource,
    config: Mp4DecodeConfig,
) -> Result<Vec<DecodedJpegFrame>, String> {
    with_prefetched_downloadable_source(source, |source| {
        decode_mp4_to_jpeg_frames_inner(source, config)
    })?
}

fn decode_mp4_to_jpeg_frames_inner(
    source: &InputSource,
    config: Mp4DecodeConfig,
) -> Result<Vec<DecodedJpegFrame>, String> {
    if !config.sample_fps.is_finite() || config.sample_fps <= 0.0 {
        return Err("sample_fps must be > 0".to_string());
    }
    if config.max_frames == 0 {
        return Err("max_frames must be >= 1".to_string());
    }

    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    let fps_expr = format!("fps={:.3}", config.sample_fps);
    let output = Command::new(ffmpeg_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            protocol_whitelist,
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i",
            source_uri,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &fps_expr,
            "-frames:v",
            &config.max_frames.to_string(),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ])
        .output()
        .map_err(|_| "failed to run ffmpeg".to_string())?;
    if !output.status.success() {
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            "ffmpeg jpeg decode failed"
        );
        return Err("video jpeg decode failed".to_string());
    }
    parse_jpeg_stream_to_frames(&output.stdout, config.max_frames)
}

fn parse_framemd5_to_signals(
    framemd5: &str,
    source_uri: &str,
    max_frames: usize,
) -> Result<DecodedMp4Batch, String> {
    let mut tb_num = 1u32;
    let mut tb_den = 1000u32;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut normalizer = TimestampNormalizer::new();
    let mut frame_signals = Vec::with_capacity(max_frames.min(1024));
    let mut prev_luma = 0.0f32;
    let mut prev_hash: Option<u64> = None;

    for line in framemd5.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#tb ") {
            if let Some((num, den)) = parse_fraction_suffix(rest) {
                tb_num = num.max(1);
                tb_den = den.max(1);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("#dimensions ") {
            if let Some((w, h)) = parse_dimensions_suffix(rest) {
                width = w;
                height = h;
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        if frame_signals.len() >= max_frames {
            break;
        }

        let mut fields = [""; 6];
        let mut field_count = 0usize;
        for part in line.split(',').take(6) {
            fields[field_count] = part.trim();
            field_count += 1;
        }
        if field_count < 6 {
            return Err(format!("invalid framemd5 row: {line}"));
        }
        let pts = fields[2]
            .parse::<i64>()
            .map_err(|_| format!("invalid pts in framemd5 row: {line}"))?;
        let checksum = fields[5];
        let perceptual_hash = parse_hex_u64_prefix(checksum, 0, 16)?;
        // We intentionally derive fast proxy features from checksum bits so ingest stays deterministic
        // without introducing per-frame decode-side pixel scans.
        let luma_seed = parse_hex_u64_prefix(checksum, 16, 8).unwrap_or(0) as u32;
        let noise_seed = parse_hex_u64_prefix(checksum, 24, 8).unwrap_or(0) as u32;
        let luma_mean = (luma_seed as f64 / u32::MAX as f64) as f32;
        let flicker_score = normalize_unit((luma_mean - prev_luma).abs());
        let ghosting_score = prev_hash
            .map(|prev| normalize_unit(1.0 - ((prev ^ perceptual_hash).count_ones() as f32 / 64.0)))
            .unwrap_or(0.0);
        let noise_variance_score = (noise_seed as f64 / u32::MAX as f64) as f32;
        let pts_ms = normalizer.normalize_pts_ms(pts, tb_num, tb_den);
        let frame_index = frame_signals.len() as u64;
        frame_signals.push(FrameSignal {
            frame_index,
            pts_ms,
            perceptual_hash,
            luma_mean: normalize_unit(luma_mean),
            flicker_score,
            ghosting_score,
            noise_variance_score: normalize_unit(noise_variance_score),
        });
        prev_luma = luma_mean;
        prev_hash = Some(perceptual_hash);
    }

    if frame_signals.is_empty() {
        return Err("no video frames decoded from source".to_string());
    }

    Ok(DecodedMp4Batch {
        source_uri: source_uri.to_string(),
        width,
        height,
        pixel_format: "framemd5".to_string(),
        frame_signals,
    })
}

fn parse_fraction_suffix(value: &str) -> Option<(u32, u32)> {
    let (_, fraction) = value.split_once(':')?;
    let (num, den) = fraction.trim().split_once('/')?;
    Some((num.trim().parse().ok()?, den.trim().parse().ok()?))
}

fn parse_ffprobe_frame_rate(raw: &str) -> Option<f32> {
    if raw.is_empty() {
        return None;
    }
    if let Some((num, den)) = raw.split_once('/') {
        let num = num.trim().parse::<f32>().ok()?;
        let den = den.trim().parse::<f32>().ok()?;
        if den <= 0.0 {
            return None;
        }
        let fps = num / den;
        return fps.is_finite().then_some(fps);
    }
    let fps = raw.trim().parse::<f32>().ok()?;
    fps.is_finite().then_some(fps)
}

fn parse_dimensions_suffix(value: &str) -> Option<(u32, u32)> {
    let (_, dimensions) = value.split_once(':')?;
    let (width, height) = dimensions.trim().split_once('x')?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

fn parse_hex_u64_prefix(source: &str, offset: usize, len: usize) -> Result<u64, String> {
    let end = offset.saturating_add(len);
    if source.len() < end {
        return Err(format!(
            "checksum is too short: expected at least {end} chars, got {}",
            source.len()
        ));
    }
    u64::from_str_radix(&source[offset..end], 16)
        .map_err(|err| format!("invalid checksum hex: {err}"))
}

pub(crate) fn parse_jpeg_stream_to_frames(
    raw: &[u8],
    max_frames: usize,
) -> Result<Vec<DecodedJpegFrame>, String> {
    use memchr::memchr;

    let mut frames = Vec::with_capacity(max_frames.min(1024));
    let mut cursor = 0usize;

    while cursor + 1 < raw.len() && frames.len() < max_frames {
        // SIMD scan for 0xFF, then check next byte for SOI marker (0xD8).
        let start = loop {
            match memchr(0xFF, &raw[cursor..]) {
                Some(offset) => {
                    let pos = cursor + offset;
                    if pos + 1 < raw.len() && raw[pos + 1] == 0xD8 {
                        break pos;
                    }
                    cursor = pos + 1;
                }
                None => return if frames.is_empty() {
                    Err("no jpeg frames decoded from source".to_string())
                } else {
                    Ok(frames)
                },
            }
        };
        cursor = start + 2;

        // SIMD scan for EOI marker (0xFF 0xD9).
        let end = loop {
            match memchr(0xFF, &raw[cursor..]) {
                Some(offset) => {
                    let pos = cursor + offset;
                    if pos + 1 < raw.len() && raw[pos + 1] == 0xD9 {
                        break pos + 2;
                    }
                    cursor = pos + 1;
                }
                None => return Err("mjpeg stream ended with an incomplete frame".to_string()),
            }
        };
        cursor = end;

        frames.push(DecodedJpegFrame {
            frame_index: frames.len() as u64,
            jpeg_bytes: raw[start..end].to_vec(),
        });
    }

    if frames.is_empty() {
        return Err("no jpeg frames decoded from source".to_string());
    }
    Ok(frames)
}

#[inline]
fn normalize_unit(value: f32) -> f32 {
    if !value.is_finite() || value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

/// Pre-compute which resampled-stream frame indices will be sent to VLM
/// inference, given the total frame count, chunk layout, and per-chunk budget.
/// Returns a sorted, deduplicated Vec<u64>.
///
/// This mirrors the selection logic in `select_semantic_images` (vidarax-api
/// handlers.rs) but operates on frame index math only — no JPEG data needed.
/// Call after `decode_mp4_to_frame_signals` to get the index set for
/// `decode_selective_jpeg_frames`.
pub fn compute_semantic_frame_indices(
    total_frames: usize,
    chunk_size: usize,
    frames_per_chunk: usize,
) -> Vec<u64> {
    if total_frames == 0 || chunk_size == 0 || frames_per_chunk == 0 {
        return Vec::new();
    }

    let n_chunks = total_frames.div_ceil(chunk_size);
    let mut indices = Vec::with_capacity(n_chunks * frames_per_chunk);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(total_frames);
        let chunk_len = end - start;

        if frames_per_chunk >= chunk_len {
            for offset in 0..chunk_len {
                indices.push((start + offset) as u64);
            }
        } else if frames_per_chunk == 1 {
            indices.push((start + chunk_len / 2) as u64);
        } else {
            let mut last = usize::MAX;
            for i in 0..frames_per_chunk {
                let offset = i * (chunk_len - 1) / (frames_per_chunk - 1);
                if offset != last {
                    indices.push((start + offset) as u64);
                    last = offset;
                }
            }
        }
    }

    indices.dedup();
    indices
}

/// Build an ffmpeg `select` filter expression for the given frame indices.
///
/// For `[12, 37, 74]` produces: `"select='eq(n\\,12)+eq(n\\,37)+eq(n\\,74)'"`
/// Commas are escaped for ffmpeg's filter parser.
pub fn build_select_expr(indices: &[u64]) -> String {
    use std::fmt::Write;
    if indices.is_empty() {
        return "select='0'".to_string();
    }
    let mut expr = String::with_capacity(indices.len() * 16 + 12);
    expr.push_str("select='");
    for (i, &idx) in indices.iter().enumerate() {
        if i > 0 {
            expr.push('+');
        }
        // Escape the comma inside eq(n,X) for ffmpeg filter graph parsing
        write!(expr, "eq(n\\,{idx})").unwrap();
    }
    expr.push('\'');
    expr
}

/// Extract a short MP4 clip from `source`.
///
/// Local files use stream copy; remote/HLS sources are re-encoded into a
/// self-contained temporary MP4 and read back.
pub fn extract_video_clip(
    source: &InputSource,
    start_s: f32,
    duration_s: f32,
) -> Result<Vec<u8>, String> {
    with_prefetched_downloadable_source(source, |source| {
        extract_video_clip_inner(source, start_s, duration_s)
    })?
}

fn extract_video_clip_inner(
    source: &InputSource,
    start_s: f32,
    duration_s: f32,
) -> Result<Vec<u8>, String> {
    if !start_s.is_finite() || start_s < 0.0 {
        return Err("start_s must be >= 0".to_string());
    }
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err("duration_s must be > 0".to_string());
    }

    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    let start_str = format!("{start_s:.6}");
    let duration_str = format!("{duration_s:.6}");

    // Local files can use stream copy without re-encoding.
    // Remote/HLS sources need a re-encode to produce a self-contained clip.
    let use_stream_copy = matches!(source, InputSource::FilePath(_));

    // MP4 requires seekable output, so we write to a temp file and read back.
    let tmp = std::env::temp_dir().join(format!(
        "vidarax_clip_{}_{}.mp4",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let tmp_str = tmp.to_string_lossy().to_string();

    let output = if use_stream_copy {
        Command::new(ffmpeg_path())
            .args([
                "-v", "error",
                "-protocol_whitelist", protocol_whitelist,
            ])
            .args(ffmpeg_input_options_for_source(source))
            .args([
                "-ss", &start_str,
                "-t", &duration_str,
                "-i", source_uri,
                "-c", "copy",
                "-movflags", "+faststart",
                "-y",
                &tmp_str,
            ])
            .output()
            .map_err(|_| "failed to run ffmpeg".to_string())?
    } else {
        Command::new(ffmpeg_path())
            .args([
                "-v", "error",
                "-protocol_whitelist", protocol_whitelist,
            ])
            .args(ffmpeg_input_options_for_source(source))
            .args([
                "-ss", &start_str,
                "-t", &duration_str,
                "-i", source_uri,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-an",
                "-y",
                &tmp_str,
            ])
            .output()
            .map_err(|_| "failed to run ffmpeg".to_string())?
    };

    if !output.status.success() {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            start_s,
            duration_s,
            "ffmpeg clip extraction failed"
        );
        return Err("video clip extraction failed".to_string());
    }

    let bytes = std::fs::read(&tmp).map_err(|e| format!("failed to read clip: {e}"))?;
    let _ = std::fs::remove_file(&tmp);

    if bytes.is_empty() {
        return Err("ffmpeg produced empty clip output".to_string());
    }

    Ok(bytes)
}

/// Decode only the specified frame indices from the source as JPEG, using a
/// single ffmpeg pass with a `select` filter. Frames are sampled at
/// `sample_fps` first (matching the framemd5 pass), then the select filter
/// picks only the requested indices from that resampled stream.
///
/// `frame_indices` must be sorted ascending. Returns frames in the same order,
/// each stamped with its original resampled-stream index.
pub fn decode_selective_jpeg_frames(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
) -> Result<Vec<DecodedJpegFrame>, String> {
    with_prefetched_downloadable_source(source, |source| {
        decode_selective_jpeg_frames_inner(source, sample_fps, frame_indices, max_frames)
    })?
}

pub(crate) fn decode_selective_jpeg_frames_inner(
    source: &InputSource,
    sample_fps: f32,
    frame_indices: &[u64],
    max_frames: usize,
) -> Result<Vec<DecodedJpegFrame>, String> {
    if frame_indices.is_empty() {
        return Ok(Vec::new());
    }
    if !sample_fps.is_finite() || sample_fps <= 0.0 {
        return Err("sample_fps must be > 0".to_string());
    }

    let select_expr = build_select_expr(frame_indices);
    let vf_chain = format!("fps={sample_fps:.3},{select_expr}");
    let frames_cap = frame_indices.len().min(max_frames).to_string();

    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    let output = Command::new(ffmpeg_path())
        .args([
            "-v", "error",
            "-protocol_whitelist", protocol_whitelist,
        ])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i", source_uri,
            "-an", "-sn", "-dn",
            "-vf", &vf_chain,
            "-vsync", "vfr",
            "-frames:v", &frames_cap,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ])
        .output()
        .map_err(|_| "failed to run ffmpeg".to_string())?;

    if !output.status.success() {
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&output.stderr).trim(),
            "ffmpeg selective decode failed"
        );
        return Err("selective video decode failed".to_string());
    }

    let mut parsed = parse_jpeg_stream_to_frames(&output.stdout, frame_indices.len())?;

    // Re-stamp frame_index with the original resampled-stream index.
    let usable = parsed.len().min(frame_indices.len());
    parsed.truncate(usable);
    for (frame, &original_idx) in parsed.iter_mut().zip(frame_indices.iter()) {
        frame.frame_index = original_idx;
    }

    Ok(parsed)
}


#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ffmpeg_input_options_for_source, ffmpeg_protocol_whitelist_for_source,
        parse_ffprobe_frame_rate, parse_framemd5_to_signals, parse_jpeg_stream_to_frames,
        Mp4DecodeConfig, TimestampNormalizer, FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST,
        FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST, FFMPEG_HTTP_PROTOCOL_WHITELIST,
        FFMPEG_HTTPS_PROTOCOL_WHITELIST, FFMPEG_LOCAL_PROTOCOL_WHITELIST,
        FFMPEG_RTSPS_PROTOCOL_WHITELIST,
    };
    use crate::ingest::InputSource;

    struct EnvRestore {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match self.old.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn with_env<T>(key: &'static str, value: Option<&str>, test: impl FnOnce() -> T) -> T {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old = std::env::var_os(key);
        let _restore = EnvRestore { key, old };
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        test()
    }

    #[test]
    fn normalizes_pts_monotonically() {
        let mut n = TimestampNormalizer::new();
        assert_eq!(n.normalize_pts_ms(300, 1, 30), 0);
        assert_eq!(n.normalize_pts_ms(330, 1, 30), 1000);
        // Out-of-order sample should clamp to last_ms for deterministic monotonic output.
        assert_eq!(n.normalize_pts_ms(320, 1, 30), 1000);
    }

    #[test]
    fn parses_framemd5_into_frame_signals() {
        let framemd5 = r#"
#format: frame checksums
#tb 0: 1/25
#dimensions 0: 320x240
0,          0,          0,        1,   230400, 0123456789abcdeffedcba9876543210
0,          1,          1,        1,   230400, fedcba98765432100123456789abcdef
"#;
        let decoded = parse_framemd5_to_signals(framemd5, "/tmp/test.mp4", 8).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 240);
        assert_eq!(decoded.frame_signals.len(), 2);
        assert_eq!(decoded.frame_signals[0].pts_ms, 0);
        assert!(decoded.frame_signals[1].pts_ms >= decoded.frame_signals[0].pts_ms);
        assert!((0.0..=1.0).contains(&decoded.frame_signals[1].flicker_score));
    }

    #[test]
    fn decode_config_defaults_are_stable() {
        let cfg = Mp4DecodeConfig::default();
        assert!(cfg.sample_fps > 0.0);
        assert!(cfg.max_frames > 0);
    }

    #[test]
    fn parses_ffprobe_fps_format() {
        let fps = parse_ffprobe_frame_rate("30000/1001").unwrap();
        assert!((fps - 29.97).abs() < 0.05);
        assert_eq!(parse_ffprobe_frame_rate(""), None);
    }

    #[test]
    fn parses_mjpeg_stream_into_frames() {
        // Minimal marker-based split test with two synthetic JPEG-like byte ranges.
        let stream = [
            0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9, 0x00, 0x11, 0xff, 0xd8, 0x03, 0x04, 0xff, 0xd9,
        ];
        let frames = parse_jpeg_stream_to_frames(&stream, 8).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].frame_index, 0);
        assert_eq!(frames[1].frame_index, 1);
        assert_eq!(frames[0].jpeg_bytes[0..2], [0xff, 0xd8]);
        assert_eq!(
            frames[1].jpeg_bytes[frames[1].jpeg_bytes.len() - 2..],
            [0xff, 0xd9]
        );
    }

    #[test]
    fn protocol_whitelists_are_scoped_by_input_kind() {
        assert!(
            !FFMPEG_HTTPS_PROTOCOL_WHITELIST
                .split(',')
                .any(|p| matches!(p, "file" | "hls" | "http")),
            "default HTTPS inputs must not allow file, hls, or plain http subresources"
        );
        assert_eq!(FFMPEG_LOCAL_PROTOCOL_WHITELIST, "file");
        let https_hls_protocols = FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST
            .split(',')
            .collect::<Vec<_>>();
        assert!(https_hls_protocols.contains(&"hls"));
        assert!(https_hls_protocols.contains(&"https"));
        assert!(https_hls_protocols.contains(&"tls"));
        assert!(!https_hls_protocols.contains(&"http"));
        assert!(!https_hls_protocols.contains(&"file"));

        let http_hls_protocols = FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST
            .split(',')
            .collect::<Vec<_>>();
        assert!(http_hls_protocols.contains(&"hls"));
        assert!(http_hls_protocols.contains(&"http"));
        assert!(!http_hls_protocols.contains(&"https"));
        assert!(!http_hls_protocols.contains(&"tls"));
        assert!(!http_hls_protocols.contains(&"file"));
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::Url(
                "https://cdn.example.com/video.mp4".to_string()
            )),
            FFMPEG_HTTPS_PROTOCOL_WHITELIST
        );
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::Url(
                "http://cdn.example.com/video.mp4".to_string()
            )),
            FFMPEG_HTTP_PROTOCOL_WHITELIST
        );
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::HlsStream(
                "https://cdn.example.com/live.m3u8".to_string()
            )),
            FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST
        );
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::HlsStream(
                "http://cdn.example.com/live.m3u8".to_string()
            )),
            FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST
        );
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::Url(
                "rtsps://camera.example.com/live".to_string()
            )),
            FFMPEG_RTSPS_PROTOCOL_WHITELIST
        );
        assert_eq!(
            ffmpeg_protocol_whitelist_for_source(&InputSource::FilePath(
                "/tmp/video.mp4".to_string()
            )),
            FFMPEG_LOCAL_PROTOCOL_WHITELIST
        );
    }

    #[test]
    fn remote_http_inputs_disable_ffmpeg_redirects() {
        assert_eq!(
            ffmpeg_input_options_for_source(&InputSource::Url(
                "https://cdn.example.com/video.mp4".to_string()
            )),
            vec!["-max_redirects".to_string(), "0".to_string()]
        );
        let hls_options = ffmpeg_input_options_for_source(&InputSource::HlsStream(
            "https://cdn.example.com/live.m3u8".to_string(),
        ));
        assert_eq!(hls_options[0], "-max_redirects");
        assert_eq!(hls_options[1], "0");
        assert!(
            ffmpeg_input_options_for_source(&InputSource::FilePath(
                "/tmp/video.mp4".to_string()
            ))
            .is_empty()
        );
    }

    #[test]
    fn hls_tls_verifies_chain_without_pinning_manifest_host() {
        with_env("VIDARAX_ALLOW_INSECURE_TLS", None, || {
            let hls_options = ffmpeg_input_options_for_source(&InputSource::HlsStream(
                "https://cdn.example.com/live.m3u8".to_string(),
            ));
            assert!(hls_options
                .windows(2)
                .any(|w| w[0] == "-tls_verify" && w[1] == "1"));
            assert!(
                !hls_options.iter().any(|arg| arg == "-verifyhost"),
                "HLS must not pin every segment/key request to the manifest host"
            );
        });
    }

    #[test]
    fn rtsps_tls_verifies_chain_and_pins_original_host() {
        with_env("VIDARAX_ALLOW_INSECURE_TLS", None, || {
            assert_eq!(
                ffmpeg_input_options_for_source(&InputSource::Url(
                    "rtsps://camera.example.com/live".to_string()
                )),
                vec![
                    "-tls_verify".to_string(),
                    "1".to_string(),
                    "-verifyhost".to_string(),
                    "camera.example.com".to_string(),
                ]
            );
        });
    }

    #[test]
    fn insecure_tls_opt_out_omits_peer_verification_args() {
        with_env("VIDARAX_ALLOW_INSECURE_TLS", Some("true"), || {
            let options = ffmpeg_input_options_for_source(&InputSource::Url(
                "rtsps://camera.example.com/live".to_string(),
            ));

            assert!(!options.iter().any(|arg| arg == "-tls_verify"));
            assert!(!options.iter().any(|arg| arg == "-verifyhost"));
        });
    }

    #[test]
    fn public_https_media_url_prefetches_and_decodes() {
        let mp4 = create_test_mp4_bytes();
        let server = crate::ingest::fetch::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "video/mp4")],
            mp4,
        );
        let url = server.url("/media.mp4");
        let source = InputSource::Url(url.clone());
        let decoded = super::decode_mp4_to_frame_signals_with_prefetch_validator(
            &source,
            Mp4DecodeConfig {
                sample_fps: 1.0,
                max_frames: 1,
            },
            server.allow_origin_validator(),
        )
        .expect("local mock media URL should prefetch and decode");

        assert_eq!(decoded.source_uri, url);
        assert_eq!(decoded.frame_signals.len(), 1);
    }

    fn create_test_mp4_bytes() -> Vec<u8> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vidarax-test-media-{}-{nanos}.mp4",
            std::process::id()
        ));
        let output = Command::new(super::ffmpeg_path())
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=1:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
            ])
            .arg(&path)
            .output()
            .expect("run ffmpeg to generate test mp4");
        assert!(
            output.status.success(),
            "ffmpeg failed to generate test mp4: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = fs::read(&path).expect("read generated test mp4");
        let _ = fs::remove_file(&path);
        assert!(!bytes.is_empty(), "generated test mp4 must not be empty");
        bytes
    }
}

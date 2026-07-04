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
        .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
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
    let mut decoded = with_prefetched_downloadable_source(source, |source| {
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
        .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i", source_uri, "-an", "-sn", "-dn", "-vf", &fps_expr, "-f", "framemd5", "-",
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

    // A real perceptual hash needs real pixels, which the framemd5 pass doesn't
    // expose: MD5 avalanches, so the Hamming distance between two *different*
    // frames is ~50% noise and the hash only reliably flags byte-identical
    // frames. Run one cheap extra pass that emits an 8×8 grayscale grid per
    // frame and average-hash it. Both passes use the same `fps` sampler, so
    // hash[i] lines up with framemd5 frame i. Degrade to the (weak) checksum
    // hash per-frame if the pass fails, so ingest never regresses to an error.
    let ahashes = compute_ahashes_from_source(source, config.sample_fps, config.max_frames)
        .unwrap_or_else(|err| {
            tracing::warn!(%err, "perceptual-hash pass failed; falling back to checksum hash");
            Vec::new()
        });
    parse_framemd5_to_signals(&text, source_uri, config.max_frames, &ahashes)
}

/// Compute a real 64-bit average-hash per sampled frame by decoding the source a
/// second time into an 8×8 grayscale grid (ffmpeg's `area` scaler is a box
/// average, the same idea as [`crate::webrtc::signals`]'s luma downscale) and
/// hashing each grid. Frames are produced by the same `fps={sample_fps}` sampler
/// as the framemd5 pass, so the returned hashes align index-for-index.
fn compute_ahashes_from_source(
    source: &InputSource,
    sample_fps: f32,
    max_frames: usize,
) -> Result<Vec<u64>, String> {
    let source_uri = source.as_ffmpeg_input();
    let protocol_whitelist = ffmpeg_protocol_whitelist_for_source(source);
    // fps first (same frame set as framemd5), then squash to 8×8 luma. `area`
    // box-averages on downscale; `format=gray` makes it one byte per cell.
    let vf_expr = format!("fps={sample_fps:.3},scale=8:8:flags=area,format=gray");
    let output = Command::new(ffmpeg_path())
        .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i", source_uri, "-an", "-sn", "-dn", "-vf", &vf_expr, "-frames:v",
            &max_frames.to_string(), "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ])
        .output()
        .map_err(|_| "failed to run ffmpeg".to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg perceptual-hash pass failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(ahashes_from_gray_grid(&output.stdout))
}

/// Pack a stream of 8×8 (= 64-byte) grayscale grids into average-hashes, one per
/// complete grid. A trailing partial grid (should not happen for `-pix_fmt
/// gray` output) is ignored.
fn ahashes_from_gray_grid(raw: &[u8]) -> Vec<u64> {
    raw.chunks_exact(64).map(ahash_cell_grid).collect()
}

/// Average-hash one 8×8 grayscale grid (64 bytes, row-major): bit `i` is set iff
/// cell `i` is strictly brighter than the 64-cell mean. This is the same bit
/// convention as [`crate::webrtc::signals`]'s `perceptual_hash_y`, so hashes
/// from the file-ingest and live-stream paths are directly Hamming-comparable.
fn ahash_cell_grid(cells: &[u8]) -> u64 {
    debug_assert_eq!(cells.len(), 64, "average-hash grid must be 8x8");
    let mean = cells.iter().map(|&c| c as u32).sum::<u32>() / 64;
    let mut hash = 0u64;
    for (i, &c) in cells.iter().enumerate() {
        if c as u32 > mean {
            hash |= 1u64 << i;
        }
    }
    hash
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
        .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
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
    ahashes: &[u64],
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
        // Prefer the real average-hash from the pixel pass (aligned by frame
        // index); fall back to the checksum slice only when that pass produced
        // fewer frames — e.g. it failed and returned an empty vec. The checksum
        // hash is a weak signal (MD5 avalanches) but keeps ingest working.
        let perceptual_hash = ahashes
            .get(frame_signals.len())
            .copied()
            .map_or_else(|| parse_hex_u64_prefix(checksum, 0, 16), Ok)?;
        // luma/noise stay checksum-derived proxies so ingest stays deterministic
        // without a per-frame pixel scan; only the perceptual hash (the n_phash /
        // ghosting signal) is made real above.
        let luma_seed = parse_hex_u64_prefix(checksum, 16, 8).unwrap_or(0) as u32;
        let noise_seed = parse_hex_u64_prefix(checksum, 24, 8).unwrap_or(0) as u32;
        let luma_mean = (luma_seed as f64 / u32::MAX as f64) as f32;
        let flicker_score = normalize_unit((luma_mean - prev_luma).abs());
        let ghosting_score = prev_hash
            .map(|prev| normalize_unit((prev ^ perceptual_hash).count_ones() as f32 / 64.0))
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
        // Normalised Hamming *distance* (hd/64), same polarity as the live
        // webrtc path (signals.rs): a bigger hash delta = higher score. The old
        // `1.0 - hd/64` (similarity) was inverted — a static screen scored 1.0
        // and tripped the gate's ghosting threshold on every frame.
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
                None => {
                    return if frames.is_empty() {
                        Err("no jpeg frames decoded from source".to_string())
                    } else {
                        Ok(frames)
                    }
                }
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
/// Callers normally pass sorted, de-duplicated indices (the semantic path does);
/// anything else is normalized to sorted-unique first, so the selected frame set
/// never depends on input order — it always equals the set of distinct indices.
///
/// A naive `eq(n,a)+eq(n,b)+...` with one term per frame blows up ffmpeg's
/// filter-graph parser on dense cadences: a few hundred `eq()` terms exhaust
/// the expression evaluator's allocation ("Error initializing filters /
/// Cannot allocate memory") and the whole selective decode fails. Since the
/// semantic selection is structured — either evenly-strided single frames
/// (1 frame/chunk) or equal-length clusters (N frames/chunk) — we can pick the
/// exact same frames with a constant-size expression.
///
/// We coalesce indices into consecutive runs, then greedily collapse any
/// maximal arithmetic *block* (equal-length runs at a constant stride) of at
/// least three runs into ONE `mod`-based term. Shorter or irregular groups
/// stay as literal per-run `eq`/`between` terms. Crucially this survives a
/// partial final chunk: a clip whose frame count isn't a multiple of
/// `chunk_size` puts its last midpoint off-stride, and only that stray frame
/// falls out of the collapse — the dense bulk stays O(1).
///
/// - `[42]`               → `select='eq(n\,42)'`
/// - `[12, 37, 74]`       → `select='eq(n\,12)+eq(n\,37)+eq(n\,74)'` (sparse: one term each)
/// - `[2, 7, .., 832]`    → `select='between(n\,2\,832)*eq(mod(n-2\,5)\,0)'` (strided: O(1))
/// - `[2, 7, .., 827, 831]`→ `select='between(n\,2\,827)*eq(mod(n-2\,5)\,0)+eq(n\,831)'` (partial tail)
/// - `[0,1, 5,6, 10,11]`  → `select='between(n\,0\,11)*lt(mod(n\,5)\,2)'` (clusters: O(1))
///
/// Commas are escaped (`\,`) for ffmpeg's filter parser, matching the
/// established quoting convention.
pub fn build_select_expr(indices: &[u64]) -> String {
    if indices.is_empty() {
        return "select='0'".to_string();
    }

    // Block-peeling below assumes strictly-ascending, unique indices — the sole
    // production caller (compute_semantic_frame_indices) always passes them that
    // way. Honor the "defensive" promise without taxing that hot path: normalize
    // (sort + dedup) ONLY when the input isn't already strictly ascending, so the
    // common case stays zero-alloc. Normalizing also removes the one u64 overflow
    // risk — the `*hi + 1` extend check can only run when a strictly-larger index
    // follows, so hi < u64::MAX there and the add can't wrap.
    let normalized: Vec<u64>;
    let indices: &[u64] = if indices.windows(2).all(|w| w[0] < w[1]) {
        indices
    } else {
        let mut v = indices.to_vec();
        v.sort_unstable();
        v.dedup();
        normalized = v;
        &normalized
    };

    // Coalesce into maximal consecutive runs [lo, hi].
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &idx in indices {
        match runs.last_mut() {
            Some((_, hi)) if idx == *hi + 1 => *hi = idx,
            _ => runs.push((idx, idx)),
        }
    }

    // Greedily emit terms. A maximal arithmetic *block* — runs of equal length
    // L whose starts are separated by a constant stride S — is exactly
    //   { n in [first, last] : (n - first) mod S < L }
    // and collapses to ONE O(1) term (separate runs imply S > L, so they never
    // overlap). We only collapse blocks of >= COLLAPSE runs: that bounds the
    // output no matter the frame count, while short/irregular groups stay as
    // literal per-run terms (preserving historical output for sparse sets).
    //
    // Peeling blocks greedily — instead of demanding the whole set be regular —
    // is what survives a partial final chunk: the dense stride-S bulk collapses
    // and only the stray off-stride tail frame(s) spill into cheap `eq` terms.
    const COLLAPSE: usize = 3;
    let mut terms: Vec<String> = Vec::new();
    let mut i = 0;
    while i < runs.len() {
        let (lo, hi) = runs[i];
        let l = hi - lo + 1;
        // Extend a regular block [i..=j] sharing the stride of its first gap.
        let mut j = i;
        if i + 1 < runs.len() {
            let s = runs[i + 1].0 - runs[i].0;
            while j + 1 < runs.len()
                && runs[j + 1].0 - runs[j].0 == s
                && runs[j + 1].1 - runs[j + 1].0 + 1 == l
            {
                j += 1;
            }
            if j - i + 1 >= COLLAPSE {
                let first = runs[i].0;
                let last = runs[j].1;
                let modarg = if first == 0 {
                    "n".to_string()
                } else {
                    format!("n-{first}")
                };
                terms.push(if l == 1 {
                    format!("between(n\\,{first}\\,{last})*eq(mod({modarg}\\,{s})\\,0)")
                } else {
                    format!("between(n\\,{first}\\,{last})*lt(mod({modarg}\\,{s})\\,{l})")
                });
                i = j + 1;
                continue;
            }
        }
        // Not a collapsible block: emit this single run literally and advance
        // by one so the next run gets its own chance to start a block.
        terms.push(if lo == hi {
            format!("eq(n\\,{lo})")
        } else {
            format!("between(n\\,{lo}\\,{hi})")
        });
        i += 1;
    }

    let cap = terms.iter().map(|t| t.len() + 1).sum::<usize>() + 10;
    let mut expr = String::with_capacity(cap);
    expr.push_str("select='");
    for (k, t) in terms.iter().enumerate() {
        if k > 0 {
            expr.push('+');
        }
        expr.push_str(t);
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
            .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
            .args(ffmpeg_input_options_for_source(source))
            .args([
                "-ss",
                &start_str,
                "-t",
                &duration_str,
                "-i",
                source_uri,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                &tmp_str,
            ])
            .output()
            .map_err(|_| "failed to run ffmpeg".to_string())?
    } else {
        Command::new(ffmpeg_path())
            .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
            .args(ffmpeg_input_options_for_source(source))
            .args([
                "-ss",
                &start_str,
                "-t",
                &duration_str,
                "-i",
                source_uri,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
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
        .args(["-v", "error", "-protocol_whitelist", protocol_whitelist])
        .args(ffmpeg_input_options_for_source(source))
        .args([
            "-i",
            source_uri,
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
        ahash_cell_grid, ahashes_from_gray_grid, ffmpeg_input_options_for_source,
        ffmpeg_protocol_whitelist_for_source, parse_ffprobe_frame_rate, parse_framemd5_to_signals,
        parse_jpeg_stream_to_frames, Mp4DecodeConfig, TimestampNormalizer,
        FFMPEG_HLS_HTTPS_PROTOCOL_WHITELIST,
        FFMPEG_HLS_HTTP_PROTOCOL_WHITELIST, FFMPEG_HTTPS_PROTOCOL_WHITELIST,
        FFMPEG_HTTP_PROTOCOL_WHITELIST, FFMPEG_LOCAL_PROTOCOL_WHITELIST,
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
        // Empty ahashes → per-frame fallback to the checksum-derived hash, which
        // is exactly the legacy behavior this test pins.
        let decoded = parse_framemd5_to_signals(framemd5, "/tmp/test.mp4", 8, &[]).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 240);
        assert_eq!(decoded.frame_signals.len(), 2);
        assert_eq!(decoded.frame_signals[0].pts_ms, 0);
        assert!(decoded.frame_signals[1].pts_ms >= decoded.frame_signals[0].pts_ms);
        assert!((0.0..=1.0).contains(&decoded.frame_signals[1].flicker_score));
    }

    #[test]
    fn real_ahashes_override_checksum_hash() {
        // When the pixel pass supplies hashes, they must land in perceptual_hash
        // verbatim (aligned by frame index), not the checksum slice.
        let framemd5 = r#"
#format: frame checksums
#tb 0: 1/25
#dimensions 0: 320x240
0,          0,          0,        1,   230400, 0123456789abcdeffedcba9876543210
0,          1,          1,        1,   230400, fedcba98765432100123456789abcdef
"#;
        let real = [0xDEAD_BEEF_0000_0001u64, 0x0000_0000_0000_00F0];
        let decoded = parse_framemd5_to_signals(framemd5, "/tmp/test.mp4", 8, &real).unwrap();
        assert_eq!(decoded.frame_signals[0].perceptual_hash, real[0]);
        assert_eq!(decoded.frame_signals[1].perceptual_hash, real[1]);
        // Ghosting is the normalised Hamming *distance* hd/64 — same polarity as
        // the live webrtc path (signals.rs), where a bigger hash delta = higher
        // score. Static frames score ~0 (not 1), matching the gate's artifact model.
        let hd = (real[0] ^ real[1]).count_ones() as f32;
        let expect = (hd / 64.0).clamp(0.0, 1.0);
        assert!((decoded.frame_signals[1].ghosting_score - expect).abs() < 1e-6);
    }

    #[test]
    fn ahash_cell_grid_sets_bits_above_mean() {
        // Mean of 0..64 (bytes 0,1,..,63) is 31 (integer div). Cells strictly
        // above 31 are indices 32..=63 → the high 32 bits set, low 32 clear.
        let cells: Vec<u8> = (0..64u8).collect();
        let h = ahash_cell_grid(&cells);
        assert_eq!(h, 0xFFFF_FFFF_0000_0000);
        // A flat grid has no cell above the mean → all-zero hash.
        assert_eq!(ahash_cell_grid(&[128u8; 64]), 0);
    }

    #[test]
    fn ahashes_from_gray_grid_splits_per_frame() {
        // Two 64-byte grids back to back → two hashes; a trailing partial grid
        // (not 64 bytes) is dropped by chunks_exact.
        let mut buf = vec![200u8; 64]; // frame 0: flat → hash 0
        buf.extend((0..64u8).collect::<Vec<_>>()); // frame 1: ramp
        buf.extend([1u8, 2, 3]); // stray partial grid, ignored
        let hashes = ahashes_from_gray_grid(&buf);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], 0);
        assert_eq!(hashes[1], 0xFFFF_FFFF_0000_0000);
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
        assert!(ffmpeg_input_options_for_source(&InputSource::FilePath(
            "/tmp/video.mp4".to_string()
        ))
        .is_empty());
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

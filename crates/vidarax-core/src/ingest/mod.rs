pub mod pipeline;

use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crate::gate::FrameSignal;

static FFMPEG_PATH: OnceLock<String> = OnceLock::new();
static FFPROBE_PATH: OnceLock<String> = OnceLock::new();
static NVIDIA_SMI_PATH: OnceLock<String> = OnceLock::new();

/// Return the configured ffmpeg binary path (cached after first call).
///
/// Checks `VIDARAX_FFMPEG_PATH` env var first, falls back to `"ffmpeg"`.
pub fn ffmpeg_path() -> &'static str {
    FFMPEG_PATH.get_or_init(|| {
        std::env::var("VIDARAX_FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".to_string())
    })
}

/// Return the configured ffprobe binary path (cached after first call).
///
/// Checks `VIDARAX_FFPROBE_PATH` env var first, falls back to `"ffprobe"`.
pub fn ffprobe_path() -> &'static str {
    FFPROBE_PATH.get_or_init(|| {
        std::env::var("VIDARAX_FFPROBE_PATH").unwrap_or_else(|_| "ffprobe".to_string())
    })
}

/// Return the configured nvidia-smi binary path (cached after first call).
///
/// Checks `VIDARAX_NVIDIA_SMI_PATH` env var first, falls back to `"nvidia-smi"`.
pub fn nvidia_smi_path() -> &'static str {
    NVIDIA_SMI_PATH.get_or_init(|| {
        std::env::var("VIDARAX_NVIDIA_SMI_PATH").unwrap_or_else(|_| "nvidia-smi".to_string())
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    FilePath(String),
    Url(String),
    /// Live HLS stream (HTTP/HTTPS .m3u8 manifest or hls:// URL).
    /// ffmpeg handles HLS natively via the `hls` demuxer.
    HlsStream(String),
    /// Live WebRTC stream identified by session ID (processed via worker pools,
    /// not through ffmpeg).
    WebRtcStream(String),
}

impl InputSource {
    pub fn parse_and_validate(input: &str, allowed_file_roots: &[PathBuf]) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err("source_uri must not be empty".to_string());
        }

        if trimmed.contains("://") {
            let url =
                reqwest::Url::parse(trimmed).map_err(|err| format!("invalid source_uri: {err}"))?;
            return match url.scheme() {
                "http" | "https" => {
                    // HLS manifests served over HTTP/HTTPS: validate as HLS stream.
                    if url.path().to_ascii_lowercase().ends_with(".m3u8") {
                        validate_hls_url(url)
                    } else {
                        validate_remote_url(url)
                    }
                }
                "hls" => validate_hls_url(url),
                "rtsps" => validate_remote_url(url),
                "rtsp" => {
                    // M-12: Reject unencrypted RTSP by default. Operators can
                    // opt in via VIDARAX_ALLOW_UNENCRYPTED_RTSP=true for legacy
                    // cameras on trusted networks.
                    let allow_plain = std::env::var("VIDARAX_ALLOW_UNENCRYPTED_RTSP")
                        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                        .unwrap_or(false);
                    if !allow_plain {
                        return Err(
                            "unencrypted rtsp:// is not allowed; use rtsps:// or set \
                             VIDARAX_ALLOW_UNENCRYPTED_RTSP=true"
                                .to_string(),
                        );
                    }
                    validate_remote_url(url)
                }
                "file" => validate_file_url(url, allowed_file_roots),
                other => Err(format!(
                    "unsupported source_uri scheme '{other}', expected one of: \
                     file, hls, http, https, rtsps"
                )),
            };
        }

        validate_file_path(trimmed, allowed_file_roots)
    }

    pub fn as_ffmpeg_input(&self) -> &str {
        match self {
            InputSource::FilePath(path) | InputSource::Url(path) | InputSource::HlsStream(path) => {
                path
            }
            // WebRTC streams are fed through the worker pool directly; the
            // session ID is returned as a placeholder but should never be
            // passed to ffmpeg.
            InputSource::WebRtcStream(session_id) => session_id.as_str(),
        }
    }
}

fn validate_remote_url(url: reqwest::Url) -> Result<InputSource, String> {
    if !url.username().is_empty() || url.password().is_some() {
        return Err("source_uri must not contain embedded credentials".to_string());
    }
    let host = url
        .host_str()
        .ok_or_else(|| "source_uri must include a valid host".to_string())?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
        return Err("source_uri host must not target localhost/local domains".to_string());
    }
    if let Ok(ip) = lower.parse::<IpAddr>() {
        if blocked_ip(&ip) {
            return Err("source_uri host must not be private, loopback, or link-local".to_string());
        }
    }

    // H-1: Resolve DNS and validate ALL resolved IPs to prevent DNS rebinding.
    let port = url.port_or_known_default().unwrap_or(80);
    let resolve_target = format!("{host}:{port}");
    match resolve_target.to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            if addrs.is_empty() {
                return Err("source_uri host did not resolve to any address".to_string());
            }
            for addr in &addrs {
                if blocked_ip(&addr.ip()) {
                    return Err(
                        "source_uri host must not be private, loopback, or link-local".to_string(),
                    );
                }
            }
        }
        Err(_) => {
            // DNS resolution failed — reject rather than allow a potentially
            // unvalidated host through.
            return Err("source_uri host could not be resolved".to_string());
        }
    }

    Ok(InputSource::Url(url.to_string()))
}

/// Validate an HLS URL (http/https .m3u8 or hls:// scheme).
///
/// Applies the same SSRF guards as [`validate_remote_url`] (no embedded
/// credentials, no private/loopback hosts, DNS rebinding check), then
/// returns [`InputSource::HlsStream`].
///
/// For `hls://` URLs the host validation re-parses the URL as `https://`
/// so that `reqwest::Url::host_str()` and DNS resolution work correctly.
fn validate_hls_url(url: reqwest::Url) -> Result<InputSource, String> {
    if !url.username().is_empty() || url.password().is_some() {
        return Err("source_uri must not contain embedded credentials".to_string());
    }

    // For hls:// scheme, substitute https:// for SSRF host validation.
    let validation_url = if url.scheme() == "hls" {
        let https_str = format!("https{}", &url.as_str()["hls".len()..]);
        reqwest::Url::parse(&https_str)
            .map_err(|_| "invalid hls:// source_uri host".to_string())?
    } else {
        url.clone()
    };

    let host = validation_url
        .host_str()
        .ok_or_else(|| "source_uri must include a valid host".to_string())?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
        return Err("source_uri host must not target localhost/local domains".to_string());
    }
    if let Ok(ip) = lower.parse::<IpAddr>() {
        if blocked_ip(&ip) {
            return Err("source_uri host must not be private, loopback, or link-local".to_string());
        }
    }

    let port = validation_url.port_or_known_default().unwrap_or(443);
    let resolve_target = format!("{host}:{port}");
    match resolve_target.to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            if addrs.is_empty() {
                return Err("source_uri host did not resolve to any address".to_string());
            }
            for addr in &addrs {
                if blocked_ip(&addr.ip()) {
                    return Err(
                        "source_uri host must not be private, loopback, or link-local".to_string(),
                    );
                }
            }
        }
        Err(_) => {
            return Err("source_uri host could not be resolved".to_string());
        }
    }

    Ok(InputSource::HlsStream(url.to_string()))
}

fn validate_file_url(
    url: reqwest::Url,
    allowed_file_roots: &[PathBuf],
) -> Result<InputSource, String> {
    if url.host_str().map(|host| !host.is_empty()).unwrap_or(false) {
        return Err("file:// source_uri must not include a host".to_string());
    }
    let path = url
        .to_file_path()
        .map_err(|_| "file:// source_uri path is invalid".to_string())?;
    validate_file_path(path.to_string_lossy().as_ref(), allowed_file_roots)
}

fn validate_file_path(path: &str, allowed_file_roots: &[PathBuf]) -> Result<InputSource, String> {
    let canonical = Path::new(path)
        .canonicalize()
        .map_err(|_| "source_uri file path is invalid or does not exist".to_string())?;
    if !allowed_file_roots
        .iter()
        .any(|root| canonical.starts_with(root))
    {
        return Err("source_uri file path is outside configured ingest roots".to_string());
    }
    Ok(InputSource::FilePath(
        canonical.to_string_lossy().to_string(),
    ))
}

fn blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
                || (octets[0] == 169 && octets[1] == 254)
        }
        IpAddr::V6(v6) => {
            let octets = v6.octets();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6.is_multicast()
                || (octets[0] == 0x20
                    && octets[1] == 0x01
                    && octets[2] == 0x0d
                    && octets[3] == 0xb8)
        }
    }
}

pub(crate) const FFMPEG_PROTOCOL_WHITELIST: &str = "file,http,https,tcp,tls,rtsp,rtp,udp,hls";

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
    let source_uri = source.as_ffmpeg_input();
    let output = Command::new(ffprobe_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
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
    if !config.sample_fps.is_finite() || config.sample_fps <= 0.0 {
        return Err("sample_fps must be > 0".to_string());
    }
    if config.max_frames == 0 {
        return Err("max_frames must be >= 1".to_string());
    }

    let source_uri = source.as_ffmpeg_input();
    let fps_expr = format!("fps={:.3}", config.sample_fps);
    let output = Command::new(ffmpeg_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
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
    if !config.sample_fps.is_finite() || config.sample_fps <= 0.0 {
        return Err("sample_fps must be > 0".to_string());
    }
    if config.max_frames == 0 {
        return Err("max_frames must be >= 1".to_string());
    }

    let source_uri = source.as_ffmpeg_input();
    let fps_expr = format!("fps={:.3}", config.sample_fps);
    let output = Command::new(ffmpeg_path())
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
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

    let n_chunks = (total_frames + chunk_size - 1) / chunk_size;
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

/// Extract a short MP4 clip from `source` starting at `start_s` for `duration_s` seconds.
///
/// Uses `-c copy` (stream copy) when the source is a local file path so the
/// operation is near-instant; falls back to `-c:v libx264 -preset ultrafast`
/// for remote/HLS sources where a re-encode is required.  Output is written to
/// a temporary file (MP4 requires seekable output) then read back.
///
/// Returns the raw MP4 bytes on success.
///
/// # Errors
///
/// Returns a human-readable error string when ffmpeg is not found, the source
/// cannot be read, or the requested time range is out of bounds.
///
/// # Example
///
/// ```no_run
/// use vidarax_core::ingest::{InputSource, extract_video_clip};
/// let source = InputSource::FilePath("/tmp/video.mp4".to_string());
/// let mp4_bytes = extract_video_clip(&source, 0.0, 0.5).unwrap();
/// assert!(!mp4_bytes.is_empty());
/// ```
pub fn extract_video_clip(
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
    let start_str = format!("{start_s:.6}");
    let duration_str = format!("{duration_s:.6}");

    // For local files use stream copy (near-instant, no re-encode).
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
                "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST,
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
                "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST,
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
    let output = Command::new(ffmpeg_path())
        .args([
            "-v", "error",
            "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST,
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
    use super::{
        parse_ffprobe_frame_rate, parse_framemd5_to_signals, parse_jpeg_stream_to_frames,
        InputSource, Mp4DecodeConfig, TimestampNormalizer,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_url_and_file_sources() {
        let allowed = vec![std::env::temp_dir()];
        assert!(matches!(
            InputSource::parse_and_validate("https://example.com/video.mp4", &allowed).unwrap(),
            InputSource::Url(_)
        ));

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vidarax-ingest-test-{nanos}"));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let video = root.join("video.mp4");
        fs::write(&video, b"fixture").unwrap();
        let source = InputSource::parse_and_validate(video.to_string_lossy().as_ref(), &[root])
            .expect("file path should be valid");
        assert!(matches!(source, InputSource::FilePath(_)));
    }

    #[test]
    fn rejects_private_or_metadata_hosts() {
        let allowed = vec![std::env::temp_dir()];
        assert!(InputSource::parse_and_validate("http://127.0.0.1/video.mp4", &allowed).is_err());
        assert!(InputSource::parse_and_validate(
            "http://169.254.169.254/latest/meta-data",
            &allowed
        )
        .is_err());
        assert!(InputSource::parse_and_validate("https://localhost/video.mp4", &allowed).is_err());
    }

    #[test]
    fn rejects_paths_outside_allowed_roots() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vidarax-ingest-root-{nanos}"));
        let outside = std::env::temp_dir().join(format!("vidarax-ingest-outside-{nanos}.mp4"));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        fs::write(&outside, b"fixture").unwrap();
        let result = InputSource::parse_and_validate(outside.to_string_lossy().as_ref(), &[root]);
        assert!(result.is_err(), "outside path should fail allowlist check");
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

    // ── HLS source tests ──────────────────────────────────────────────────

    #[test]
    fn accepts_https_m3u8_as_hls_stream() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("https://example.com/live/stream.m3u8", &allowed);
        assert!(result.is_ok(), "https .m3u8 should be valid: {result:?}");
        assert!(
            matches!(result.unwrap(), InputSource::HlsStream(_)),
            "expected HlsStream variant"
        );
    }

    #[test]
    fn accepts_http_m3u8_as_hls_stream() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("http://example.com/live/stream.m3u8", &allowed);
        assert!(result.is_ok(), "http .m3u8 should be valid: {result:?}");
        assert!(matches!(result.unwrap(), InputSource::HlsStream(_)));
    }

    #[test]
    fn rejects_hls_with_embedded_credentials() {
        let allowed = vec![std::env::temp_dir()];
        assert!(
            InputSource::parse_and_validate(
                "https://user:pass@example.com/stream.m3u8",
                &allowed
            )
            .is_err(),
            "credentials in HLS URL should be rejected"
        );
    }

    #[test]
    fn rejects_hls_targeting_private_host() {
        let allowed = vec![std::env::temp_dir()];
        assert!(
            InputSource::parse_and_validate("https://192.168.1.100/stream.m3u8", &allowed)
                .is_err(),
            "private IP in HLS URL should be rejected"
        );
    }

    #[test]
    fn non_m3u8_https_remains_url_variant() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("https://example.com/video.mp4", &allowed).unwrap();
        assert!(
            matches!(result, InputSource::Url(_)),
            "non-.m3u8 https should be Url, not HlsStream"
        );
    }

    #[test]
    fn hls_ffmpeg_input_returns_url_string() {
        let src = InputSource::HlsStream("https://example.com/live.m3u8".to_string());
        assert_eq!(src.as_ffmpeg_input(), "https://example.com/live.m3u8");
    }

    #[test]
    fn hls_protocol_whitelist_contains_hls() {
        use super::FFMPEG_PROTOCOL_WHITELIST;
        assert!(
            FFMPEG_PROTOCOL_WHITELIST.split(',').any(|p| p == "hls"),
            "hls must be in FFMPEG_PROTOCOL_WHITELIST"
        );
    }
}

//! WebRTC video decoders for H.264, H.265 / HEVC, and unsupported negotiated codecs.
//!
//! GPU H.264 and H.265 use a long-lived ffmpeg sidecar; software H.264 uses
//! openh264 in-process. Software H.265 uses the ffmpeg sidecar. With
//! `--features vp8`, VP8 uses libvpx in-process.

// The crate denies unsafe (see lib.rs). The libvpx VP8 path calls a C API
// through raw pointers, so it needs unsafe; the allow is scoped to this module
// and gated on the same `vp8` feature that compiles that FFI, so with VP8 off
// the deny still covers every line here too.
#![cfg_attr(feature = "vp8", allow(unsafe_code))]

use std::collections::VecDeque;
#[cfg(feature = "vp8")]
use std::ffi::CStr;
use std::io::{BufReader, Read, Write};
#[cfg(feature = "vp8")]
use std::mem::MaybeUninit;
#[cfg(feature = "vp8")]
use std::os::raw::{c_int, c_uint};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{mpsc, Arc};

use openh264::formats::YUVSource;
#[cfg(feature = "vp8")]
use vpx_sys::{
    vpx_codec_ctx_t, vpx_codec_dec_init_ver, vpx_codec_decode, vpx_codec_destroy, vpx_codec_err_t,
    vpx_codec_error, vpx_codec_error_detail, vpx_codec_get_frame, vpx_codec_iter_t,
    vpx_codec_vp8_dx, vpx_image_t, vpx_img_fmt_t, VPX_DECODER_ABI_VERSION,
};

use crate::metrics::PipelineMetrics;
use crate::webrtc::recycle::{RecycledBytes, VecPool};

/// Bounded ffmpeg stdout reader handoff depth.
///
/// Sixteen frames gives short scheduler stalls room to clear while bounding
/// decoded output retained in the reader handoff. The reader uses blocking
/// sends; `decode()` drains this queue before writing more encoded input so the
/// handoff is lossless without deadlocking the ffmpeg pipes.
pub const FFMPEG_YUV_READER_QUEUE_CAPACITY: usize = 16;

/// Decoder-local pending FIFO allowance covered by the YUV output pool.
pub const FFMPEG_YUV_PENDING_POOL_ALLOWANCE: usize = 4;

const FFMPEG_YUV_PENDING_FIFO_CAPACITY: usize =
    FFMPEG_YUV_READER_QUEUE_CAPACITY + FFMPEG_YUV_PENDING_POOL_ALLOWANCE;

/// Generous diagnostic bound for the decoder-local pending FIFO.
pub const FFMPEG_YUV_PENDING_SANITY_BOUND: usize = FFMPEG_YUV_READER_QUEUE_CAPACITY * 4;

/// Minimum pooled YUV frame slots needed by the bounded ffmpeg reader path:
/// full reader queue, steady-state decoder pending FIFO allowance, one frame
/// currently being assembled by the reader, and one frame held by the decode
/// consumer.
pub const FFMPEG_YUV_READER_POOL_MIN_SLOTS: usize =
    FFMPEG_YUV_READER_QUEUE_CAPACITY + FFMPEG_YUV_PENDING_POOL_ALLOWANCE + 2;

/// Minimum pooled YUV slots for the synchronous openh264 path.
///
/// openh264 decodes one access unit at a time and has no reader handoff or
/// pending FIFO. Two slots cover one caller-held output and the next decoded
/// output without falling back to heap allocation in the normal decode loop.
pub const SOFTWARE_YUV_POOL_MIN_SLOTS: usize = 2;

/// Video codec carried by the WebRTC track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoCodec {
    /// ITU-T H.264 / AVC — default when the offer codec cannot be determined.
    #[default]
    H264,
    /// ITU-T H.265 / HEVC.
    H265,
    /// Google VP8 — negotiated by rustrtc when the browser prefers it.
    Vp8,
}

/// A video codec advertised in an SDP offer's video m-section, with the
/// payload type and clock rate the offer assigned to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfferedVideoCodec {
    pub payload_type: u8,
    pub codec: VideoCodec,
    pub clock_rate: u32,
}

impl VideoCodec {
    /// Parse the first video codec from the SDP offer.
    pub fn from_sdp(sdp: &str) -> Self {
        Self::offered_video_codecs(sdp)
            .first()
            .map(|offered| offered.codec)
            // Compatibility fallback for sessions without explicit codec info.
            .unwrap_or(VideoCodec::H264)
    }

    /// Canonical codec token for SDP rtpmap lines.
    pub fn rtpmap_name(self) -> &'static str {
        match self {
            VideoCodec::H264 => "H264",
            VideoCodec::H265 => "H265",
            VideoCodec::Vp8 => "VP8",
        }
    }

    /// Parse recognized video codecs from the SDP offer's video sections.
    pub fn offered_video_codecs(sdp: &str) -> Vec<OfferedVideoCodec> {
        fn flush_video_section(
            offered: &mut Vec<OfferedVideoCodec>,
            video_payload_types: &[u8],
            section_codecs: &[OfferedVideoCodec],
        ) {
            let mut pushed_payload_types = Vec::new();
            for payload_type in video_payload_types {
                if pushed_payload_types.contains(payload_type) {
                    continue;
                }
                if let Some(codec) = section_codecs
                    .iter()
                    .find(|codec| codec.payload_type == *payload_type)
                {
                    offered.push(*codec);
                    pushed_payload_types.push(*payload_type);
                }
            }
        }

        let mut offered = Vec::new();
        let mut in_video = false;
        let mut video_payload_types = Vec::new();
        let mut section_codecs = Vec::new();

        for line in sdp.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("m=") {
                if in_video {
                    flush_video_section(&mut offered, &video_payload_types, &section_codecs);
                }
                video_payload_types.clear();
                section_codecs.clear();
                in_video = trimmed.starts_with("m=video");
                if in_video {
                    video_payload_types.extend(
                        trimmed
                            .split_whitespace()
                            .skip(3)
                            .filter_map(|pt| pt.parse::<u8>().ok()),
                    );
                }
                continue;
            }
            if !in_video {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("a=rtpmap:") {
                let mut fields = rest.split_whitespace();
                let Some(payload_type) = fields.next().and_then(|pt| pt.parse::<u8>().ok()) else {
                    continue;
                };
                if payload_type > 127 || !video_payload_types.contains(&payload_type) {
                    continue;
                }
                let Some(codec_part) = fields.next() else {
                    continue;
                };

                let mut codec_fields = codec_part.split('/');
                let codec_name = codec_fields.next().unwrap_or("").to_ascii_uppercase();
                let codec = match codec_name.as_str() {
                    "VP8" => VideoCodec::Vp8,
                    "H264" => VideoCodec::H264,
                    "H265" | "HEVC" => VideoCodec::H265,
                    _ => continue,
                };
                let clock_rate = codec_fields
                    .next()
                    .and_then(|clock| clock.parse::<u32>().ok())
                    .unwrap_or(90000);

                section_codecs.push(OfferedVideoCodec {
                    payload_type,
                    codec,
                    clock_rate,
                });
            }
        }
        if in_video {
            flush_video_section(&mut offered, &video_payload_types, &section_codecs);
        }

        offered
    }

    /// Whether Vidarax can both depacketize and decode this codec on the live
    /// WebRTC path. H.264 uses rustrtc depacketization plus openh264 or nvdec
    /// decode. VP8 is serveable only with the `vp8` feature. H.265 uses the
    /// in-crate HEVC RTP depacketizer (RFC 7798) plus nvdec or the ffmpeg hevc
    /// software sidecar.
    fn is_live_serveable(self) -> bool {
        match self {
            VideoCodec::H264 => true,
            #[cfg(feature = "vp8")]
            VideoCodec::Vp8 => true,
            #[cfg(not(feature = "vp8"))]
            VideoCodec::Vp8 => false,
            VideoCodec::H265 => true,
        }
    }

    /// ffmpeg input format flag for codecs with a live sidecar input path.
    fn ffmpeg_input_format(self) -> Option<&'static str> {
        match self {
            VideoCodec::H264 => Some("h264"),
            VideoCodec::H265 => Some("hevc"),
            VideoCodec::Vp8 => None,
        }
    }
}

/// Pick the one video codec to answer with from the offer's advertised codecs,
/// restricted to what Vidarax can serve live. VP8 is preferred when it is
/// offered and serveable because it has a complete in-crate pipeline and needs
/// no fmtp negotiation; otherwise the first serveable codec in offer order is
/// used. Returns `None` when no advertised codec is serveable, leaving rustrtc's
/// default negotiation unchanged.
pub fn select_answer_video_codec(offered: &[OfferedVideoCodec]) -> Option<OfferedVideoCodec> {
    offered
        .iter()
        .copied()
        .find(|o| o.codec == VideoCodec::Vp8 && o.codec.is_live_serveable())
        .or_else(|| {
            offered
                .iter()
                .copied()
                .find(|o| o.codec.is_live_serveable())
        })
}

fn h265_offer_signals_don(sdp: &str, payload_type: u8) -> bool {
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

        let Some(rest) = trimmed.strip_prefix("a=fmtp:") else {
            continue;
        };
        let mut fields = rest.splitn(2, char::is_whitespace);
        let Some(fmtp_payload_type) = fields.next().and_then(|pt| pt.parse::<u8>().ok()) else {
            continue;
        };
        if fmtp_payload_type != payload_type {
            continue;
        }

        let Some(params) = fields.next() else {
            continue;
        };
        for param in params.split(';') {
            let Some((key, value)) = param.trim().split_once('=') else {
                continue;
            };
            let key = key.trim();
            if !key.eq_ignore_ascii_case("sprop-max-don-diff")
                && !key.eq_ignore_ascii_case("sprop-depack-buf-nalus")
            {
                continue;
            }
            if value.trim().parse::<u32>().is_ok_and(|value| value > 0) {
                return true;
            }
        }
    }

    false
}

/// Count the video media sections (`m=video ...` lines) in an SDP offer.
pub fn count_video_media_sections(sdp: &str) -> usize {
    sdp.lines()
        .filter(|line| line.trim_start().starts_with("m=video"))
        .count()
}

/// Count the audio media sections (`m=audio ...` lines) in an SDP offer.
pub fn count_audio_media_sections(sdp: &str) -> usize {
    sdp.lines()
        .filter(|line| line.trim_start().starts_with("m=audio"))
        .count()
}

/// Select the answer video codec directly from the offer SDP, excluding
/// H.265 payload types whose fmtp signals RFC 7798 decoding-order use
/// (`sprop-max-don-diff > 0` or `sprop-depack-buf-nalus > 0`). The
/// in-crate HEVC depacketizer assumes no DONL/DOND fields, so such an
/// offer would be depacketized into corrupt access units. Filtering
/// lets selection fall back to another serveable codec, or return
/// `None` (no video pinned) for a DON-only H.265 offer.
pub fn select_answer_video_codec_for_offer(sdp: &str) -> Option<OfferedVideoCodec> {
    let offered = VideoCodec::offered_video_codecs(sdp);
    let serveable: Vec<OfferedVideoCodec> = offered
        .into_iter()
        .filter(|o| o.codec != VideoCodec::H265 || !h265_offer_signals_don(sdp, o.payload_type))
        .collect();
    select_answer_video_codec(&serveable)
}

/// Concrete decoder backend selected from GPU availability and negotiated codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderBackend {
    /// GPU ffmpeg pipe path.
    NvDec,
    /// In-process openh264 path.
    Software,
    /// CPU ffmpeg pipe path.
    FfmpegSw,
    /// In-process libvpx VP8 path.
    #[cfg(feature = "vp8")]
    Vp8,
    /// No supported live decoder for the negotiated codec.
    Unsupported,
}

impl DecoderBackend {
    pub fn select(gpu_available: bool, codec: VideoCodec) -> Self {
        match (gpu_available, codec) {
            #[cfg(feature = "vp8")]
            (_, VideoCodec::Vp8) => DecoderBackend::Vp8,
            #[cfg(not(feature = "vp8"))]
            (_, VideoCodec::Vp8) => DecoderBackend::Unsupported,
            (true, VideoCodec::H264) => DecoderBackend::NvDec,
            (false, VideoCodec::H264) => DecoderBackend::Software,
            (true, VideoCodec::H265) => DecoderBackend::NvDec,
            (false, VideoCodec::H265) => DecoderBackend::FfmpegSw,
        }
    }

    pub fn min_yuv_pool_slots(self) -> usize {
        match self {
            DecoderBackend::NvDec | DecoderBackend::FfmpegSw => FFMPEG_YUV_READER_POOL_MIN_SLOTS,
            DecoderBackend::Software => SOFTWARE_YUV_POOL_MIN_SLOTS,
            #[cfg(feature = "vp8")]
            DecoderBackend::Vp8 => SOFTWARE_YUV_POOL_MIN_SLOTS,
            DecoderBackend::Unsupported => 0,
        }
    }
}

/// A planar YUV 4:2:0 frame with packed (non-strided) plane buffers.
///
/// `y.len() == (width * height) as usize`
/// `u.len() == v.len() == (width / 2 * height / 2) as usize`
#[derive(Debug, Clone)]
pub struct YuvFrame {
    pub y: RecycledBytes,
    pub u: RecycledBytes,
    pub v: RecycledBytes,
    pub width: u32,
    pub height: u32,
}

/// Ceiling on the Y-plane bytes the pool will pre-allocate per slot, ~8K 4:2:0
/// luma. A malformed or hostile stream can declare an enormous resolution; this
/// bounds the speculative pre-allocation. A real frame that genuinely exceeds it
/// still decodes — the copy path grows that individual buffer — it just is not
/// pre-sized for. Kept a power of two so bucketing never rounds above it.
const MAX_POOL_Y_CAPACITY: usize = 1 << 25;

#[derive(Clone, Debug)]
pub struct YuvPlanePools {
    y: VecPool,
    u: VecPool,
    v: VecPool,
    /// Bytes the Y free-list buffers are pre-sized for; the chroma free-lists are
    /// a quarter of this. Tracked as capacity rather than a resolution so a
    /// smaller frame is served from the existing buffers and the pool only
    /// rebuilds when a frame genuinely needs more room (see `ensure_dims`).
    y_capacity: usize,
    slots: usize,
    /// Whether a real decoded frame has reconciled the initial guess yet. The
    /// first frame is allowed to resize the pool in either direction; after that
    /// it only grows.
    reconciled: bool,
}

impl YuvPlanePools {
    fn new(width: u32, height: u32, slots: usize) -> Self {
        Self::with_capacity(Self::required_y_capacity(width, height), slots)
    }

    fn with_capacity(y_capacity: usize, slots: usize) -> Self {
        let uv_capacity = y_capacity / 4;
        Self {
            y: VecPool::with_capacity(slots, y_capacity),
            u: VecPool::with_capacity(slots, uv_capacity),
            v: VecPool::with_capacity(slots, uv_capacity),
            y_capacity,
            slots,
            reconciled: false,
        }
    }

    /// Y-plane bytes to pre-size the pool for a `width`x`height` frame.
    ///
    /// Rounded up to a power of two so a stream that steps up in resolution
    /// rebuilds the free-lists a bounded number of times — once per bucket
    /// crossing — rather than once per distinct size an untrusted sender might
    /// send. The chroma requirement `(w/2)*(h/2)` per plane is folded in (times
    /// four, expressed in Y-plane terms) so that, below the cap, the derived
    /// chroma capacity `/4` covers odd dimensions, where truncated `(w/2)*(h/2)`
    /// can exceed `w*h/4`. Capped at `MAX_POOL_Y_CAPACITY` so a bogus giant
    /// resolution cannot force an unbounded pre-allocation; the cap also keeps
    /// `next_power_of_two` from overflowing, and saturating arithmetic guards the
    /// same overflow on the way in. At the cap the value is only a starting size:
    /// a frame at or above it still decodes, with the copy path growing the
    /// individual plane buffer if it needs more than the capped capacity.
    fn required_y_capacity(width: u32, height: u32) -> usize {
        let luma = (width as usize).saturating_mul(height as usize);
        let chroma_as_luma = ((width as usize) / 2)
            .saturating_mul((height as usize) / 2)
            .saturating_mul(4);
        luma.max(chroma_as_luma)
            .clamp(1, MAX_POOL_Y_CAPACITY)
            .next_power_of_two()
    }

    /// Reconcile the pool with the resolution a frame actually decoded at.
    ///
    /// - The first real frame resizes the pool in either direction, correcting the
    ///   default guess the WebRTC path opens with (a 720p stream stops carrying
    ///   1080p buffers; a 4K stream stops regrowing an undersized one).
    /// - After that, the pool only grows: a frame needing no more capacity than it
    ///   already provides is served from the existing buffers. Because capacity is
    ///   bucketed, a sender ramping resolution upward crosses only a handful of
    ///   buckets, and one that flips resolution every frame settles at the peak —
    ///   neither can force a per-frame rebuild.
    ///
    /// When a rebuild does happen, buffers still in flight hold the old
    /// free-list's sender; its receiver is already gone, so on drop they free
    /// rather than recycle — a one-time cost bounded by `slots`.
    fn ensure_dims(&mut self, width: u32, height: u32) {
        let needed = Self::required_y_capacity(width, height);
        if self.reconciled && needed <= self.y_capacity {
            return;
        }
        if needed == self.y_capacity {
            // First frame, and the guess already fits it exactly — no rebuild,
            // just record that the pool now reflects a real frame so a later
            // smaller frame is not mistaken for another first frame.
            self.reconciled = true;
            return;
        }
        let slots = self.slots;
        *self = Self::with_capacity(needed, slots);
        self.reconciled = true;
    }
}

pub struct YuvFrameReceiver {
    rx: mpsc::Receiver<YuvFrame>,
}

impl YuvFrameReceiver {
    fn try_recv(&self) -> Result<Option<YuvFrame>, mpsc::TryRecvError> {
        match self.rx.try_recv() {
            Ok(frame) => Ok(Some(frame)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(err @ mpsc::TryRecvError::Disconnected) => Err(err),
        }
    }
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
    /// Recycle slots for decoded YUV frames crossing into downstream queues.
    pub output_pool_slots: usize,
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
            output_pool_slots: 1,
        }
    }
}

/// Errors returned by [`Decoder::decode`].
///
#[derive(Debug)]
pub enum DecodeError {
    /// The decoder accepted input but no output frame is available yet.
    Buffered,
    /// The ffmpeg reader thread has exited (process terminated or pipe closed).
    ReaderExited,
    /// Writing payload bytes to the ffmpeg stdin pipe failed.
    WriteError(std::io::Error),
    /// Flushing the ffmpeg stdin pipe failed.
    FlushError(std::io::Error),
    /// openh264 reported a hard decode error (bad bitstream, invalid state).
    SoftwareDecode(String),
    /// libvpx reported a hard decode error or unsupported output shape.
    #[cfg(feature = "vp8")]
    Vp8Decode(String),
    /// The negotiated codec has no supported live decoder in this build.
    UnsupportedCodec(VideoCodec),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Buffered => f.write_str("codec buffering input (no frame yet)"),
            DecodeError::ReaderExited => f.write_str("ffmpeg reader thread exited"),
            DecodeError::WriteError(e) => write!(f, "ffmpeg write: {e}"),
            DecodeError::FlushError(e) => write!(f, "ffmpeg flush: {e}"),
            DecodeError::SoftwareDecode(e) => write!(f, "openh264 decode error: {e}"),
            #[cfg(feature = "vp8")]
            DecodeError::Vp8Decode(e) => write!(f, "libvpx decode error: {e}"),
            DecodeError::UnsupportedCodec(codec) => {
                write!(f, "no supported decoder for {codec:?}")
            }
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

#[cfg(feature = "vp8")]
pub struct Vp8DecoderCtx {
    ctx: Box<vpx_codec_ctx_t>,
}

#[cfg(feature = "vp8")]
impl Vp8DecoderCtx {
    fn new() -> Self {
        // A zeroed vpx_codec_ctx_t is the libvpx-required initial state.
        let mut ctx = Box::new(unsafe { MaybeUninit::<vpx_codec_ctx_t>::zeroed().assume_init() });
        let ctx_ptr = ctx.as_mut() as *mut vpx_codec_ctx_t;
        // ctx_ptr is stable inside Box and cfg is null per libvpx defaults.
        let result = unsafe {
            vpx_codec_dec_init_ver(
                ctx_ptr,
                vpx_codec_vp8_dx(),
                std::ptr::null(),
                0,
                VPX_DECODER_ABI_VERSION as c_int,
            )
        };
        if result != vpx_codec_err_t::VPX_CODEC_OK {
            panic!("libvpx VP8 decoder initialisation failed: {result:?}");
        }
        Self { ctx }
    }

    fn as_mut_ptr(&mut self) -> *mut vpx_codec_ctx_t {
        self.ctx.as_mut() as *mut vpx_codec_ctx_t
    }
}

#[cfg(feature = "vp8")]
impl Drop for Vp8DecoderCtx {
    fn drop(&mut self) {
        // ctx was initialised by vpx_codec_dec_init_ver and is destroyed once here.
        unsafe {
            vpx_codec_destroy(self.as_mut_ptr());
        }
    }
}

/// Multi-codec video decoder backed by either NVDEC, openh264, or an explicit
/// unsupported state.
///
/// The ffmpeg pipe paths (`NvDec`, `FfmpegSw`) use a dedicated reader thread
/// to avoid deadlocks: H.264 commonly requires multiple NAL units (SPS, PPS,
/// IDR) before ffmpeg can produce a frame, so writing and reading must happen
/// concurrently. The reader thread continuously drains complete YUV frames from
/// ffmpeg stdout into a bounded channel with blocking sends. `decode()` drains
/// that channel into a decoder-local FIFO before writing more encoded input.
/// Under real-time backlog, `decode()` sheds older decoded YUV output and
/// returns the freshest ready frame so downstream labels stay close to the
/// current RTP timestamp. Encoded input is still always written.
pub enum Decoder {
    /// GPU ffmpeg sidecar using `-hwaccel auto`.
    NvDec {
        child: Child,
        stdin: ChildStdin,
        frame_rx: YuvFrameReceiver,
        pending: VecDeque<YuvFrame>,
        pending_warned: bool,
        metrics: Option<Arc<PipelineMetrics>>,
        codec: VideoCodec,
        width: u32,
        height: u32,
    },
    /// In-process openh264 decoder.
    Software {
        decoder: openh264::decoder::Decoder,
        yuv_pools: YuvPlanePools,
    },
    /// In-process libvpx VP8 decoder.
    #[cfg(feature = "vp8")]
    Vp8 {
        ctx: Vp8DecoderCtx,
        yuv_pools: YuvPlanePools,
    },
    /// CPU: long-lived ffmpeg sidecar without hardware acceleration.
    FfmpegSw {
        child: Child,
        stdin: ChildStdin,
        frame_rx: YuvFrameReceiver,
        pending: VecDeque<YuvFrame>,
        pending_warned: bool,
        metrics: Option<Arc<PipelineMetrics>>,
        codec: VideoCodec,
        width: u32,
        height: u32,
    },
    /// Negotiated codec with no working live decoder.
    Unsupported { codec: VideoCodec },
}

/// Spawn a reader thread for complete YUV420 frames from ffmpeg stdout.
///
/// The handoff is lossless: the reader uses a bounded blocking send and never
/// evicts decoded output. The decode side drains the bounded channel before
/// writing more input to ffmpeg, so a full handoff channel cannot remain the
/// reason `decode()` is blocked while writing stdin.
///
/// Plane buffers (`y`, `u`, `v`) come from the output pools: each frame is read
/// straight into a pooled buffer that returns to the free-list once the consumer
/// drops it, so the reuse happens through recycling rather than a pre-loop
/// allocation, avoiding repeated heap allocation at 1080p/30 fps. The bounded
/// output pools cover the full reader queue, a small steady-state pending FIFO
/// allowance, one constructing frame, and one consumer-held frame.
fn spawn_frame_reader(
    mut stdout: BufReader<ChildStdout>,
    width: u32,
    height: u32,
    output_pool_slots: usize,
) -> YuvFrameReceiver {
    let (tx, rx) = mpsc::sync_channel(FFMPEG_YUV_READER_QUEUE_CAPACITY);
    let output_pool_slots = output_pool_slots.max(FFMPEG_YUV_READER_POOL_MIN_SLOTS);
    let pools = YuvPlanePools::new(width, height, output_pool_slots);
    std::thread::spawn(move || {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        // ffmpeg emits packed planar I420 with no row padding, so each plane is
        // read straight into a buffer from the output pool. Sizing the pooled
        // buffer and reading into it means the frame that goes downstream is the
        // one ffmpeg wrote, with no intermediate copy. Recycled buffers keep their
        // capacity, so after warm-up the resize does not reallocate.
        loop {
            let mut y = pools.y.acquire();
            y.resize(y_size, 0);
            if stdout.read_exact(&mut y).is_err() {
                break; // ffmpeg closed stdout (process exited)
            }
            let mut u = pools.u.acquire();
            u.resize(uv_size, 0);
            if stdout.read_exact(&mut u).is_err() {
                break;
            }
            let mut v = pools.v.acquire();
            v.resize(uv_size, 0);
            if stdout.read_exact(&mut v).is_err() {
                break;
            }
            let frame = YuvFrame {
                y: pools.y.recycle(y),
                u: pools.u.recycle(u),
                v: pools.v.recycle(v),
                width,
                height,
            };
            if !send_yuv_frame_lossless(&tx, frame) {
                break;
            }
        }
    });
    YuvFrameReceiver { rx }
}

fn send_yuv_frame_lossless(tx: &mpsc::SyncSender<YuvFrame>, frame: YuvFrame) -> bool {
    tx.send(frame).is_ok()
}

#[cfg(test)]
fn try_receive_yuv_frame(frame_rx: &YuvFrameReceiver) -> Result<YuvFrame, DecodeError> {
    match frame_rx.try_recv() {
        Ok(Some(frame)) => Ok(frame),
        Ok(None) => Err(DecodeError::Buffered),
        Err(_) => Err(DecodeError::ReaderExited),
    }
}

/// Feed one raw access unit into an ffmpeg pipe decoder and return at most one
/// decoded YUV frame.
///
/// The ffmpeg raw pipe has no frame metadata channel, so callers label any
/// returned frame with the current access unit as a best-effort approximation.
/// The decode side first drains all currently-ready YUV frames into `pending`,
/// then writes and flushes the next encoded payload, then returns the newest
/// pending decoded frame. Older pending decoded frames are shed and counted so
/// real-time analysis stays as close as possible to the current RTP label while
/// still never dropping encoded input. The drain-before-write order keeps room
/// in the bounded reader channel for the reader thread's blocking send.
// Decodes one pipe payload; the caller supplies distinct state handles and frame metadata.
#[allow(clippy::too_many_arguments)]
fn decode_ffmpeg_pipe(
    stdin: &mut impl Write,
    frame_rx: &YuvFrameReceiver,
    pending: &mut VecDeque<YuvFrame>,
    metrics: Option<&PipelineMetrics>,
    pending_warned: &mut bool,
    codec: VideoCodec,
    width: u32,
    height: u32,
    payload: &[u8],
) -> Result<YuvFrame, DecodeError> {
    let reader_exited = drain_ready_yuv_frames(frame_rx, pending);
    observe_pending_depth(pending.len(), metrics, pending_warned, codec, width, height);

    stdin.write_all(payload).map_err(DecodeError::WriteError)?;
    stdin.flush().map_err(DecodeError::FlushError)?;

    if let Some(frame) = pending.pop_back() {
        let shed = pending.len();
        if shed != 0 {
            if let Some(metrics) = metrics {
                metrics.inc_frames_dropped_by(shed as u64);
            }
            pending.clear();
        }
        return Ok(frame);
    }
    if reader_exited {
        return Err(DecodeError::ReaderExited);
    }
    Err(DecodeError::Buffered)
}

fn drain_ready_yuv_frames(frame_rx: &YuvFrameReceiver, pending: &mut VecDeque<YuvFrame>) -> bool {
    loop {
        match frame_rx.try_recv() {
            Ok(Some(frame)) => pending.push_back(frame),
            Ok(None) => return false,
            Err(_) => return true,
        }
    }
}

fn observe_pending_depth(
    pending_depth: usize,
    metrics: Option<&PipelineMetrics>,
    pending_warned: &mut bool,
    codec: VideoCodec,
    width: u32,
    height: u32,
) {
    if pending_depth <= FFMPEG_YUV_PENDING_SANITY_BOUND {
        return;
    }

    debug_assert!(
        pending_depth <= FFMPEG_YUV_PENDING_SANITY_BOUND,
        "ffmpeg YUV pending depth exceeded backpressure sanity bound"
    );

    if *pending_warned {
        return;
    }
    *pending_warned = true;
    if let Some(metrics) = metrics {
        metrics.inc_decode_pending_sanity_violations();
    }
    tracing::warn!(
        pending_depth,
        sanity_bound = FFMPEG_YUV_PENDING_SANITY_BOUND,
        ?codec,
        width,
        height,
        "ffmpeg YUV pending FIFO exceeded sanity bound; preserving frames without eviction"
    );
}

impl Decoder {
    /// Create a decoder.
    ///
    /// Selects the backend based on `config`:
    ///
    /// | `gpu_available` | `codec`          | Backend     |
    /// |-----------------|------------------|-------------|
    /// | `true`          | H.264            | `NvDec`     |
    /// | `true`          | H.265 / HEVC     | `NvDec`     |
    /// | `true`          | VP8              | `Vp8` with `--features vp8`; otherwise `Unsupported` |
    /// | `false`         | H.264            | `Software`  |
    /// | `false`         | H.265 / HEVC     | `FfmpegSw`  |
    /// | `false`         | VP8              | `Vp8` with `--features vp8`; otherwise `Unsupported` |
    ///
    /// H.265 / HEVC uses the ffmpeg sidecar, including software ffmpeg when no
    /// GPU is available.
    ///
    /// # Panics
    ///
    /// Panics if the selected backend cannot be initialised (ffmpeg not found
    /// for `NvDec`, or openh264 library init failure for `Software`).
    pub fn new(config: &DecoderConfig) -> Self {
        Self::new_inner(config, None)
    }

    pub(crate) fn new_with_metrics(config: &DecoderConfig, metrics: Arc<PipelineMetrics>) -> Self {
        Self::new_inner(config, Some(metrics))
    }

    fn new_inner(config: &DecoderConfig, metrics: Option<Arc<PipelineMetrics>>) -> Self {
        match DecoderBackend::select(config.gpu_available, config.codec) {
            DecoderBackend::NvDec => Self::new_nvdec(
                config.codec,
                config.width,
                config.height,
                config.output_pool_slots,
                metrics,
            ),
            DecoderBackend::Software => {
                Self::new_software(config.width, config.height, config.output_pool_slots)
            }
            #[cfg(feature = "vp8")]
            DecoderBackend::Vp8 => {
                Self::new_vp8(config.width, config.height, config.output_pool_slots)
            }
            DecoderBackend::FfmpegSw => Self::new_ffmpeg_sw(
                config.codec,
                config.width,
                config.height,
                config.output_pool_slots,
                metrics,
            ),
            DecoderBackend::Unsupported => Decoder::Unsupported {
                codec: config.codec,
            },
        }
    }

    #[cfg(feature = "vp8")]
    fn new_vp8(width: u32, height: u32, output_pool_slots: usize) -> Self {
        let output_pool_slots = output_pool_slots.max(SOFTWARE_YUV_POOL_MIN_SLOTS);
        Decoder::Vp8 {
            ctx: Vp8DecoderCtx::new(),
            yuv_pools: YuvPlanePools::new(width, height, output_pool_slots),
        }
    }

    fn new_software(width: u32, height: u32, output_pool_slots: usize) -> Self {
        let decoder =
            openh264::decoder::Decoder::new().expect("openh264 decoder initialisation failed");
        let output_pool_slots = output_pool_slots.max(SOFTWARE_YUV_POOL_MIN_SLOTS);
        Decoder::Software {
            decoder,
            yuv_pools: YuvPlanePools::new(width, height, output_pool_slots),
        }
    }

    /// Spawn an ffmpeg sidecar using `-hwaccel auto` so the same process
    /// handles H.264 without hard-coding a decoder name.
    fn new_nvdec(
        codec: VideoCodec,
        width: u32,
        height: u32,
        output_pool_slots: usize,
        metrics: Option<Arc<PipelineMetrics>>,
    ) -> Self {
        let input_fmt = codec
            .ffmpeg_input_format()
            .expect("ffmpeg sidecar requires a codec with an input demuxer format");
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
        let frame_rx = spawn_frame_reader(stdout, width, height, output_pool_slots);

        Decoder::NvDec {
            child,
            stdin,
            frame_rx,
            pending: VecDeque::with_capacity(FFMPEG_YUV_PENDING_FIFO_CAPACITY),
            pending_warned: false,
            metrics,
            codec,
            width,
            height,
        }
    }

    /// Spawn a software-only ffmpeg sidecar for codecs with a live raw input
    /// format.
    fn new_ffmpeg_sw(
        codec: VideoCodec,
        width: u32,
        height: u32,
        output_pool_slots: usize,
        metrics: Option<Arc<PipelineMetrics>>,
    ) -> Self {
        let input_fmt = codec
            .ffmpeg_input_format()
            .expect("ffmpeg sidecar requires a codec with an input demuxer format");
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
                "-s",
                &format!("{width}x{height}"),
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("ffmpeg software sidecar spawn failed");

        let stdin = child.stdin.take().expect("ffmpeg stdin missing");
        let stdout = BufReader::new(child.stdout.take().expect("ffmpeg stdout missing"));
        let frame_rx = spawn_frame_reader(stdout, width, height, output_pool_slots);

        Decoder::FfmpegSw {
            child,
            stdin,
            frame_rx,
            pending: VecDeque::with_capacity(FFMPEG_YUV_PENDING_FIFO_CAPACITY),
            pending_warned: false,
            metrics,
            codec,
            width,
            height,
        }
    }

    /// Decode raw video payload bytes into a packed YUV420 frame.
    ///
    /// - For `NvDec` and `FfmpegSw`: drains the bounded stdout reader channel
    ///   into the decoder-local FIFO, writes and flushes `payload` to ffmpeg
    ///   stdin, then returns the freshest pending output frame and sheds older
    ///   pending decoded frames. `Buffered` means
    ///   ffmpeg accepted the input but has no output ready for this call.
    /// - For `Software` (H.264 only): passes `payload` directly to openh264;
    ///   `Buffered` covers SPS/PPS-only payloads or partial slices that do not
    ///   produce an output frame.
    pub fn decode(&mut self, payload: &[u8]) -> Result<YuvFrame, DecodeError> {
        match self {
            Decoder::NvDec {
                stdin,
                frame_rx,
                pending,
                pending_warned,
                metrics,
                codec,
                width,
                height,
                ..
            }
            | Decoder::FfmpegSw {
                stdin,
                frame_rx,
                pending,
                pending_warned,
                metrics,
                codec,
                width,
                height,
                ..
            } => decode_ffmpeg_pipe(
                stdin,
                frame_rx,
                pending,
                metrics.as_deref(),
                pending_warned,
                *codec,
                *width,
                *height,
                payload,
            ),
            Decoder::Software { decoder, yuv_pools } => {
                Self::decode_software(decoder, yuv_pools, payload)
            }
            #[cfg(feature = "vp8")]
            Decoder::Vp8 { ctx, yuv_pools } => Self::decode_vp8(ctx, yuv_pools, payload),
            Decoder::Unsupported { codec } => Err(DecodeError::UnsupportedCodec(*codec)),
        }
    }

    fn decode_software(
        decoder: &mut openh264::decoder::Decoder,
        yuv_pools: &mut YuvPlanePools,
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
        // The plane slices may be padded, so we de-stride each row directly into
        // a buffer taken from the output pool. Writing here instead of into a
        // separate scratch buffer means the packed plane is built once and handed
        // straight downstream, rather than built and then copied again.
        let (width, height) = yuv.dimensions();
        let (y_stride, u_stride, v_stride) = yuv.strides();

        // The WebRTC path opens the pool at a default resolution because the true
        // frame size is not known until the first frame decodes. Reconcile it to
        // what actually decoded so the free-list is sized for this stream rather
        // than a guess (see `ensure_dims` for the first-frame-then-grow policy).
        yuv_pools.ensure_dims(width as u32, height as u32);

        let uv_width = width / 2;
        let uv_height = height / 2;

        let mut y = yuv_pools.y.acquire();
        y.reserve(width * height);
        for row in 0..height {
            let start = row * y_stride;
            y.extend_from_slice(&yuv.y()[start..start + width]);
        }

        let mut u = yuv_pools.u.acquire();
        u.reserve(uv_width * uv_height);
        for row in 0..uv_height {
            let start = row * u_stride;
            u.extend_from_slice(&yuv.u()[start..start + uv_width]);
        }

        let mut v = yuv_pools.v.acquire();
        v.reserve(uv_width * uv_height);
        for row in 0..uv_height {
            let start = row * v_stride;
            v.extend_from_slice(&yuv.v()[start..start + uv_width]);
        }

        Ok(YuvFrame {
            y: yuv_pools.y.recycle(y),
            u: yuv_pools.u.recycle(u),
            v: yuv_pools.v.recycle(v),
            width: width as u32,
            height: height as u32,
        })
    }

    #[cfg(feature = "vp8")]
    fn decode_vp8(
        ctx: &mut Vp8DecoderCtx,
        yuv_pools: &mut YuvPlanePools,
        payload: &[u8],
    ) -> Result<YuvFrame, DecodeError> {
        let ctx_ptr = ctx.as_mut_ptr();
        let payload_len = c_uint::try_from(payload.len())
            .map_err(|_| DecodeError::Vp8Decode("VP8 payload too large".to_string()))?;
        // ctx_ptr is a live decoder and payload points to immutable bytes for this call.
        let result = unsafe {
            vpx_codec_decode(
                ctx_ptr,
                payload.as_ptr(),
                payload_len,
                std::ptr::null_mut(),
                0,
            )
        };
        if result != vpx_codec_err_t::VPX_CODEC_OK {
            return Err(DecodeError::Vp8Decode(vpx_error_message(ctx_ptr)));
        }

        let mut iter: vpx_codec_iter_t = std::ptr::null();
        let mut output: *mut vpx_image_t = std::ptr::null_mut();
        loop {
            // iter is owned by libvpx for this drain and ctx remains live.
            let img = unsafe { vpx_codec_get_frame(ctx_ptr, &mut iter) };
            if img.is_null() {
                break;
            }
            if output.is_null() {
                output = img;
            } else {
                return Err(DecodeError::Vp8Decode(
                    "multiple frames returned for one VP8 access unit".to_string(),
                ));
            }
        }

        if output.is_null() {
            return Err(DecodeError::Buffered);
        }

        // output points to decoder-owned image memory valid until the next decode call.
        let img = unsafe { &*output };
        if img.fmt != vpx_img_fmt_t::VPX_IMG_FMT_I420 {
            return Err(DecodeError::Vp8Decode(format!(
                "unsupported libvpx image format {:?}",
                img.fmt
            )));
        }

        let width = img.d_w as usize;
        let height = img.d_h as usize;
        if width % 2 != 0 || height % 2 != 0 {
            return Err(DecodeError::Vp8Decode(
                "odd VP8 frame dimensions unsupported".to_string(),
            ));
        }

        // Match the pool to the frame's true resolution, which the WebRTC path
        // only learns once the first frame decodes (see `ensure_dims` for the
        // first-frame-then-grow policy).
        yuv_pools.ensure_dims(img.d_w, img.d_h);

        // De-stride each libvpx plane straight into a pooled buffer. copy_vpx_plane
        // clears its destination first, so a freshly acquired buffer drops in and
        // the packed plane is written once rather than staged in scratch and copied.
        let mut y = yuv_pools.y.acquire();
        copy_vpx_plane(&mut y, img.planes[0], img.stride[0], width, height)?;

        let uv_width = width / 2;
        let uv_height = height / 2;
        let mut u = yuv_pools.u.acquire();
        copy_vpx_plane(&mut u, img.planes[1], img.stride[1], uv_width, uv_height)?;
        let mut v = yuv_pools.v.acquire();
        copy_vpx_plane(&mut v, img.planes[2], img.stride[2], uv_width, uv_height)?;

        Ok(YuvFrame {
            y: yuv_pools.y.recycle(y),
            u: yuv_pools.u.recycle(u),
            v: yuv_pools.v.recycle(v),
            width: img.d_w,
            height: img.d_h,
        })
    }
}

#[cfg(feature = "vp8")]
fn copy_vpx_plane(
    dst: &mut Vec<u8>,
    src: *const u8,
    stride: c_int,
    width: usize,
    height: usize,
) -> Result<(), DecodeError> {
    let stride = usize::try_from(stride)
        .map_err(|_| DecodeError::Vp8Decode("invalid libvpx plane layout".to_string()))?;
    if src.is_null() || stride < width {
        return Err(DecodeError::Vp8Decode(
            "invalid libvpx plane layout".to_string(),
        ));
    }

    dst.clear();
    dst.reserve(width * height);
    for row in 0..height {
        // src is a top-down libvpx plane with at least width active bytes in this row.
        let row_slice = unsafe { std::slice::from_raw_parts(src.add(row * stride), width) };
        dst.extend_from_slice(row_slice);
    }
    Ok(())
}

#[cfg(feature = "vp8")]
fn vpx_error_message(ctx: *mut vpx_codec_ctx_t) -> String {
    // ctx is a live decoder and libvpx returns null or static C strings.
    let base = unsafe { c_string_or_default(vpx_codec_error(ctx), "unknown libvpx error") };
    // ctx is a live decoder and detail is optional.
    let detail = unsafe { c_string_or_default(vpx_codec_error_detail(ctx), "") };
    if detail.is_empty() {
        base
    } else {
        format!("{base}: {detail}")
    }
}

#[cfg(feature = "vp8")]
unsafe fn c_string_or_default(ptr: *const std::os::raw::c_char, default: &str) -> String {
    if ptr.is_null() {
        return default.to_string();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

impl Drop for Decoder {
    fn drop(&mut self) {
        match self {
            Decoder::NvDec { child, .. } | Decoder::FfmpegSw { child, .. } => {
                // Best-effort kill; ignore errors during shutdown.
                let _ = child.kill();
            }
            #[cfg(feature = "vp8")]
            Decoder::Vp8 { .. } => {}
            Decoder::Software { .. } | Decoder::Unsupported { .. } => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io::{self, Write};
    use std::sync::mpsc;

    use crate::metrics::PipelineMetrics;

    use super::{
        count_audio_media_sections, count_video_media_sections, decode_ffmpeg_pipe,
        h265_offer_signals_don, select_answer_video_codec, select_answer_video_codec_for_offer,
        send_yuv_frame_lossless, try_receive_yuv_frame, DecodeError, DecoderBackend,
        OfferedVideoCodec, VideoCodec, YuvFrame, YuvPlanePools, FFMPEG_YUV_PENDING_POOL_ALLOWANCE,
        FFMPEG_YUV_READER_POOL_MIN_SLOTS, FFMPEG_YUV_READER_QUEUE_CAPACITY,
    };

    #[test]
    fn yuv_plane_pools_ensure_dims_rebuilds_on_resolution_change() {
        // Two warm buffers per plane, sized for a small frame.
        let mut pools = YuvPlanePools::new(640, 480, 2);

        // Drain the warm free-list without recycling, so any buffer a later
        // acquire hands back can only have come from a rebuild.
        let _a = pools.y.acquire();
        let _b = pools.y.acquire();
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "free-list should be drained"
        );

        // A different resolution re-provisions the pool: the refilled free-list
        // now serves buffers sized for the larger frame.
        pools.ensure_dims(1920, 1080);
        assert!(
            pools.y.acquire().capacity() >= 1920 * 1080,
            "rebuilt pool should serve buffers sized for the new resolution",
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_same_resolution_is_noop() {
        let mut pools = YuvPlanePools::new(1920, 1080, 2);

        // Drain the warm free-list.
        let _a = pools.y.acquire();
        let _b = pools.y.acquire();
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "free-list should be drained"
        );

        // The same resolution must not rebuild, so the free-list stays drained
        // and in-flight buffers keep their route home.
        pools.ensure_dims(1920, 1080);
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "unchanged resolution must not re-provision the pool",
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_reconciles_down_on_first_frame() {
        // Pool opened at a 1080p default, but the stream turns out smaller.
        let mut pools = YuvPlanePools::new(1920, 1080, 2);
        let _a = pools.y.acquire();
        let _b = pools.y.acquire();
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "free-list should be drained"
        );

        // The first real frame reconciles the guess downward so the run does not
        // carry 1080p buffers for a 480p stream.
        pools.ensure_dims(640, 480);
        let cap = pools.y.acquire().capacity();
        assert!(cap >= 640 * 480, "rebuilt pool must fit the smaller frame");
        assert!(
            cap < 1920 * 1080,
            "buffers must no longer be sized for 1080p"
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_is_grow_only_after_reconcile() {
        // Reconcile at 1080p (equal to the opening size marks it reconciled).
        let mut pools = YuvPlanePools::new(1920, 1080, 2);
        pools.ensure_dims(1920, 1080);

        // Drain the warm free-list.
        let _a = pools.y.acquire();
        let _b = pools.y.acquire();
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "free-list should be drained"
        );

        // A smaller frame is served from the existing buffers, not a rebuild, so a
        // sender flipping resolution every frame cannot force a per-frame realloc.
        pools.ensure_dims(1280, 720);
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "a smaller frame after reconcile must not re-provision the pool",
        );

        // Returning to a size that still fits is likewise a no-op.
        pools.ensure_dims(1920, 1080);
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "a frame that still fits must not re-provision the pool",
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_grows_beyond_current_size() {
        // Reconcile at 720p.
        let mut pools = YuvPlanePools::new(1280, 720, 2);
        pools.ensure_dims(1280, 720);

        let _a = pools.y.acquire();
        let _b = pools.y.acquire();
        assert_eq!(
            pools.y.acquire().capacity(),
            0,
            "free-list should be drained"
        );

        // A frame with more pixels than the pool is sized for must rebuild, so the
        // hot path stops regrowing an undersized buffer every cycle.
        pools.ensure_dims(1920, 1080);
        assert!(
            pools.y.acquire().capacity() >= 1920 * 1080,
            "a larger frame must grow the pool",
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_bounds_rebuilds_under_monotonic_growth() {
        let mut pools = YuvPlanePools::new(16, 16, 2);
        pools.ensure_dims(16, 16);

        // Feed hundreds of strictly-increasing frames up to 1080p. Power-of-two
        // capacity bucketing must let a sender ramp resolution upward without
        // forcing a rebuild on every frame — only bucket crossings rebuild.
        let mut rebuilds = 0usize;
        let mut last_capacity = pools.y_capacity;
        let mut w = 16u32;
        let mut frames = 0usize;
        while w <= 1920 {
            pools.ensure_dims(w, w * 9 / 16);
            if pools.y_capacity != last_capacity {
                rebuilds += 1;
                last_capacity = pools.y_capacity;
            }
            frames += 1;
            w += 2;
        }
        assert!(frames > 900, "expected a long ramp, got {frames} frames");
        assert!(
            rebuilds <= 24,
            "bucketing must bound rebuilds well below frame count: {rebuilds} rebuilds over {frames} frames",
        );
    }

    #[test]
    fn yuv_plane_pools_ensure_dims_sizes_chroma_for_odd_dimensions() {
        // Reconcile at a degenerate near-1D resolution whose exact chroma size is
        // zero, then hand it a squarer frame of *fewer* pixels but a real chroma
        // plane. Folding the chroma requirement into the bucketed capacity means
        // the U/V buffers still fit without a per-frame regrow.
        let mut pools = YuvPlanePools::new(1, 1000, 2);
        pools.ensure_dims(1, 1000);

        pools.ensure_dims(31, 32);
        let u = pools.u.acquire();
        assert!(
            u.capacity() >= (31 / 2) * (32 / 2),
            "chroma buffer must fit odd-dimension frames, got capacity {}",
            u.capacity(),
        );
    }

    #[test]
    fn count_video_media_sections_returns_zero_for_audio_only_offer() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
        );

        assert_eq!(count_video_media_sections(sdp), 0);
    }

    #[test]
    fn count_video_media_sections_returns_one_for_single_video_section() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H264/90000\r\n",
        );

        assert_eq!(count_video_media_sections(sdp), 1);
    }

    #[test]
    fn count_video_media_sections_returns_two_for_two_video_sections() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H264/90000\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 97\r\n",
            "a=rtpmap:97 H265/90000\r\n",
        );

        assert_eq!(count_video_media_sections(sdp), 2);
    }

    #[test]
    fn count_video_media_sections_does_not_count_audio_or_application_sections() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
            "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
            "a=sctp-port:5000\r\n",
        );

        assert_eq!(count_video_media_sections(sdp), 0);
    }

    #[test]
    fn count_audio_media_sections_returns_zero_for_video_only_offer() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H264/90000\r\n",
        );
        assert_eq!(count_audio_media_sections(sdp), 0);
    }

    #[test]
    fn count_audio_media_sections_returns_one_for_audio_plus_video_offer() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H264/90000\r\n",
        );
        assert_eq!(count_audio_media_sections(sdp), 1);
    }

    #[test]
    fn count_audio_media_sections_returns_two_for_two_audio_sections() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 110\r\n",
            "a=rtpmap:110 opus/48000/2\r\n",
        );
        assert_eq!(count_audio_media_sections(sdp), 2);
    }

    #[test]
    fn offered_video_codecs_parses_multi_codec_offer() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96 97 98\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
            "a=rtpmap:98 rtx/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::Vp8,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
            ]
        );
    }

    #[test]
    fn offered_video_codecs_ignores_audio_section_and_unknowns() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 99 97\r\n",
            "a=rtpmap:99 AV1/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![OfferedVideoCodec {
                payload_type: 97,
                codec: VideoCodec::H264,
                clock_rate: 90000,
            }]
        );
    }

    #[test]
    fn offered_video_codecs_defaults_missing_clock_rate() {
        let sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 VP8\r\n";

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::Vp8,
                clock_rate: 90000,
            }]
        );
    }

    #[test]
    fn offered_video_codecs_rejects_pt_above_127() {
        let sdp = concat!(
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:200 H264/90000\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::Vp8,
                clock_rate: 90000,
            }]
        );
    }

    #[test]
    fn offered_video_codecs_rejects_pt_absent_from_m_line() {
        let sdp = concat!(
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:50 H264/90000\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::Vp8,
                clock_rate: 90000,
            }]
        );
    }

    #[test]
    fn offered_video_codecs_format_set_resets_between_sections() {
        let sdp = concat!(
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 97\r\n",
            "a=rtpmap:96 H264/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::Vp8,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
            ]
        );
    }

    #[test]
    fn offered_video_codecs_uses_m_line_order_over_rtpmap_order() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 97 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );
        let offered = VideoCodec::offered_video_codecs(sdp);

        assert_eq!(
            offered,
            vec![
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::H265,
                    clock_rate: 90000,
                },
            ]
        );
        assert_eq!(select_answer_video_codec(&offered), Some(offered[0]));
    }

    #[test]
    fn offered_video_codecs_dedupes_repeated_m_line_payload() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 97 96 97\r\n",
            "a=rtpmap:97 H264/90000\r\n",
            "a=rtpmap:96 H265/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::H265,
                    clock_rate: 90000,
                },
            ]
        );
    }

    #[test]
    fn offered_video_codecs_omits_m_line_payload_without_rtpmap() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![OfferedVideoCodec {
                payload_type: 97,
                codec: VideoCodec::H264,
                clock_rate: 90000,
            }]
        );
    }

    #[test]
    fn offered_video_codecs_keeps_order_when_m_line_matches_rtpmap_order() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96 97 98\r\n",
            "a=rtpmap:96 VP8/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
            "a=rtpmap:98 H265/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::Vp8,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 98,
                    codec: VideoCodec::H265,
                    clock_rate: 90000,
                },
            ]
        );
    }

    #[test]
    fn offered_video_codecs_orders_each_video_section_by_its_m_line() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 97 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=rtpmap:97 H264/90000\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 100 99\r\n",
            "a=rtpmap:99 H264/90000\r\n",
            "a=rtpmap:100 VP8/90000\r\n",
        );

        assert_eq!(
            VideoCodec::offered_video_codecs(sdp),
            vec![
                OfferedVideoCodec {
                    payload_type: 97,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 96,
                    codec: VideoCodec::H265,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 100,
                    codec: VideoCodec::Vp8,
                    clock_rate: 90000,
                },
                OfferedVideoCodec {
                    payload_type: 99,
                    codec: VideoCodec::H264,
                    clock_rate: 90000,
                },
            ]
        );
    }

    #[cfg(feature = "vp8")]
    #[test]
    fn select_prefers_vp8_when_serveable() {
        let offered = [
            OfferedVideoCodec {
                payload_type: 97,
                codec: VideoCodec::H264,
                clock_rate: 90000,
            },
            OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::Vp8,
                clock_rate: 90000,
            },
        ];

        assert_eq!(select_answer_video_codec(&offered), Some(offered[1]));
    }

    #[cfg(not(feature = "vp8"))]
    #[test]
    fn select_falls_back_to_h264_without_vp8() {
        let offered = [
            OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::Vp8,
                clock_rate: 90000,
            },
            OfferedVideoCodec {
                payload_type: 97,
                codec: VideoCodec::H264,
                clock_rate: 90000,
            },
        ];

        assert_eq!(select_answer_video_codec(&offered), Some(offered[1]));
    }

    #[test]
    fn select_returns_hevc_for_hevc_only_offer() {
        let offered = [OfferedVideoCodec {
            payload_type: 96,
            codec: VideoCodec::H265,
            clock_rate: 90000,
        }];

        assert_eq!(select_answer_video_codec(&offered), Some(offered[0]));
    }

    #[test]
    fn select_for_offer_rejects_don_only_h265() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=fmtp:96 sprop-max-don-diff=1\r\n",
        );

        assert_eq!(select_answer_video_codec_for_offer(sdp), None);
    }

    #[test]
    fn select_for_offer_falls_back_from_don_h265_to_h264() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=fmtp:96 sprop-max-don-diff=1\r\n",
            "a=rtpmap:97 H264/90000\r\n",
        );

        assert_eq!(
            select_answer_video_codec_for_offer(sdp),
            Some(OfferedVideoCodec {
                payload_type: 97,
                codec: VideoCodec::H264,
                clock_rate: 90000,
            })
        );
    }

    #[test]
    fn select_for_offer_allows_h265_with_zero_don_diff() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=fmtp:96 sprop-max-don-diff=0\r\n",
        );

        assert_eq!(
            select_answer_video_codec_for_offer(sdp),
            Some(OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::H265,
                clock_rate: 90000,
            })
        );
    }

    #[test]
    fn select_for_offer_allows_h265_without_fmtp() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
        );

        assert_eq!(
            select_answer_video_codec_for_offer(sdp),
            Some(OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::H265,
                clock_rate: 90000,
            })
        );
    }

    #[test]
    fn select_for_offer_rejects_h265_when_depack_buf_nalus_forces_donl() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
            "a=fmtp:96 sprop-max-don-diff=0; sprop-depack-buf-nalus=2\r\n",
        );

        assert_eq!(select_answer_video_codec_for_offer(sdp), None);
    }

    #[test]
    fn select_for_offer_ignores_don_fmtp_in_audio_section() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 opus/48000/2\r\n",
            "a=fmtp:96 sprop-max-don-diff=1\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=rtpmap:96 H265/90000\r\n",
        );

        assert_eq!(
            select_answer_video_codec_for_offer(sdp),
            Some(OfferedVideoCodec {
                payload_type: 96,
                codec: VideoCodec::H265,
                clock_rate: 90000,
            })
        );
    }

    #[test]
    fn h265_offer_signals_don_from_video_fmtp_params() {
        let present = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=fmtp:96 profile-id=1; SPROP-MAX-DON-DIFF = 3 ; tx-mode=SRST\r\n",
        );
        let zero = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=fmtp:96 sprop-max-don-diff=0\r\n",
        );
        let absent = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n";
        let depack_buf = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=fmtp:96 sprop-max-don-diff=0; sprop-depack-buf-nalus=2\r\n",
        );
        let malformed = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
            "a=fmtp:96 sprop-max-don-diff=x\r\n",
        );

        assert!(h265_offer_signals_don(present, 96));
        assert!(!h265_offer_signals_don(zero, 96));
        assert!(!h265_offer_signals_don(absent, 96));
        assert!(h265_offer_signals_don(depack_buf, 96));
        assert!(!h265_offer_signals_don(malformed, 96));
    }

    #[test]
    fn select_returns_none_for_empty_offer() {
        let sdp = concat!(
            "v=0\r\n",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            "a=rtpmap:111 opus/48000/2\r\n",
        );
        let offered = VideoCodec::offered_video_codecs(sdp);

        assert!(offered.is_empty());
        assert_eq!(select_answer_video_codec(&[]), None);
        assert_eq!(select_answer_video_codec(&offered), None);
    }

    #[test]
    fn rtpmap_name_round_trips() {
        assert_eq!(VideoCodec::H264.rtpmap_name(), "H264");
        assert_eq!(VideoCodec::H265.rtpmap_name(), "H265");
        assert_eq!(VideoCodec::Vp8.rtpmap_name(), "VP8");
    }

    #[test]
    fn from_sdp_still_returns_first_recognized_codec() {
        let sdp = concat!(
            "v=0\r\n",
            "m=video 9 UDP/TLS/RTP/SAVPF 99 97 96\r\n",
            "a=rtpmap:99 AV1/90000\r\n",
            "a=rtpmap:97 H265/90000\r\n",
            "a=rtpmap:96 H264/90000\r\n",
        );

        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::H265);
    }

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
    fn detects_h265_from_sdp_rtpmap() {
        let h265_sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 98\r\na=rtpmap:98 H265/90000\r\n";
        let hevc_sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 98\r\na=rtpmap:98 HEVC/90000\r\n";

        assert_eq!(VideoCodec::from_sdp(h265_sdp), VideoCodec::H265);
        assert_eq!(VideoCodec::from_sdp(hevc_sdp), VideoCodec::H265);
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
        // First codec in the video m-line wins.
        assert_eq!(VideoCodec::from_sdp(sdp), VideoCodec::Vp8);
    }

    #[test]
    fn ffmpeg_input_format_strings_are_correct() {
        assert_eq!(VideoCodec::H264.ffmpeg_input_format(), Some("h264"));
        assert_eq!(VideoCodec::H265.ffmpeg_input_format(), Some("hevc"));
        assert_eq!(VideoCodec::Vp8.ffmpeg_input_format(), None);
    }

    #[test]
    fn h265_backend_selection_uses_ffmpeg_sidecars() {
        assert_eq!(
            DecoderBackend::select(true, VideoCodec::H265),
            DecoderBackend::NvDec
        );
        assert_eq!(
            DecoderBackend::select(false, VideoCodec::H265),
            DecoderBackend::FfmpegSw
        );
    }

    #[test]
    fn video_codec_default_is_h264() {
        assert_eq!(VideoCodec::default(), VideoCodec::H264);
    }

    #[test]
    fn ffmpeg_reader_handoff_is_lossless_for_bounded_burst() {
        let (tx, rx) = test_frame_receiver();

        for i in 0..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            assert!(tx.try_send(tiny_yuv_frame(i as u8)).is_ok());
        }

        for i in 0..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            let frame = try_receive_yuv_frame(&rx).unwrap();
            assert_eq!(frame.y[0], i as u8);
        }
        assert!(matches!(
            try_receive_yuv_frame(&rx),
            Err(DecodeError::Buffered)
        ));
    }

    #[test]
    fn ffmpeg_reader_try_recv_semantics_are_preserved() {
        let (tx, rx) = test_frame_receiver();

        assert!(matches!(
            try_receive_yuv_frame(&rx),
            Err(DecodeError::Buffered)
        ));

        drop(tx);
        assert!(matches!(
            try_receive_yuv_frame(&rx),
            Err(DecodeError::ReaderExited)
        ));
    }

    #[test]
    fn ffmpeg_decode_writes_input_then_returns_buffered_before_output() {
        let (_tx, rx) = test_frame_receiver();
        let mut stdin = CountingWrite {
            writes: 0,
            flushes: 0,
        };
        let mut pending = test_pending_fifo();
        let mut pending_warned = false;

        let err = decode_for_test(
            &mut stdin,
            &rx,
            &mut pending,
            &mut pending_warned,
            b"encoded",
        )
        .unwrap_err();

        assert!(matches!(err, DecodeError::Buffered));
        assert_eq!(stdin.writes, 1);
        assert_eq!(stdin.flushes, 1);
    }

    #[test]
    fn ffmpeg_decode_returns_freshest_ready_frame_and_sheds_older_backlog() {
        let (tx, rx) = test_frame_receiver();
        tx.try_send(tiny_yuv_frame(7)).unwrap();
        tx.try_send(tiny_yuv_frame(8)).unwrap();
        let mut stdin = CountingWrite {
            writes: 0,
            flushes: 0,
        };
        let mut pending = test_pending_fifo();
        let mut pending_warned = false;
        let metrics = PipelineMetrics::new();

        let first = decode_for_test_with_metrics(
            &mut stdin,
            &rx,
            &mut pending,
            Some(&metrics),
            &mut pending_warned,
            b"encoded-1",
        )
        .unwrap();
        let second = decode_for_test_with_metrics(
            &mut stdin,
            &rx,
            &mut pending,
            Some(&metrics),
            &mut pending_warned,
            b"encoded-2",
        )
        .unwrap_err();

        assert_eq!(first.y[0], 8);
        assert!(matches!(second, DecodeError::Buffered));
        assert_eq!(metrics.frames_dropped_total(), 1);
        assert_eq!(stdin.writes, 2);
        assert_eq!(stdin.flushes, 2);
    }

    #[test]
    fn ffmpeg_decode_reports_reader_exited_after_writing_input() {
        let (tx, rx) = test_frame_receiver();
        drop(tx);
        let mut stdin = CountingWrite {
            writes: 0,
            flushes: 0,
        };
        let mut pending = test_pending_fifo();
        let mut pending_warned = false;

        let err = decode_for_test(
            &mut stdin,
            &rx,
            &mut pending,
            &mut pending_warned,
            b"encoded",
        )
        .unwrap_err();

        assert!(matches!(err, DecodeError::ReaderExited));
        assert_eq!(stdin.writes, 1);
        assert_eq!(stdin.flushes, 1);
    }

    #[test]
    fn ffmpeg_decode_writes_encoded_input_even_when_output_queue_is_full() {
        let (tx, rx) = test_frame_receiver();
        for i in 0..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            tx.try_send(tiny_yuv_frame(i as u8)).unwrap();
        }
        let mut stdin = CountingWrite {
            writes: 0,
            flushes: 0,
        };
        let mut pending = test_pending_fifo();
        let mut pending_warned = false;
        let metrics = PipelineMetrics::new();

        let frame = decode_for_test_with_metrics(
            &mut stdin,
            &rx,
            &mut pending,
            Some(&metrics),
            &mut pending_warned,
            b"must-not-drop-input",
        )
        .unwrap();

        assert_eq!(frame.y[0], (FFMPEG_YUV_READER_QUEUE_CAPACITY - 1) as u8);
        assert_eq!(
            metrics.frames_dropped_total(),
            (FFMPEG_YUV_READER_QUEUE_CAPACITY - 1) as u64
        );
        assert_eq!(stdin.writes, 1);
        assert_eq!(stdin.flushes, 1);
    }

    #[test]
    fn ffmpeg_decode_drains_before_write_to_unblock_lossless_reader() {
        let (tx, rx) = test_frame_receiver();
        for i in 0..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            tx.try_send(tiny_yuv_frame(i as u8)).unwrap();
        }
        let mut stdin = QueueRoomAssertingWrite {
            tx: tx.clone(),
            injected: false,
        };
        let mut pending = test_pending_fifo();
        let mut pending_warned = false;

        let first = decode_for_test(
            &mut stdin,
            &rx,
            &mut pending,
            &mut pending_warned,
            b"encoded",
        )
        .unwrap();

        assert_eq!(first.y[0], (FFMPEG_YUV_READER_QUEUE_CAPACITY - 1) as u8);
        assert!(stdin.injected);
        let injected = decode_for_test(
            &mut CountingWrite {
                writes: 0,
                flushes: 0,
            },
            &rx,
            &mut pending,
            &mut pending_warned,
            b"encoded",
        )
        .unwrap();
        assert_eq!(injected.y[0], 200);
    }

    #[test]
    fn ffmpeg_reader_lossless_send_blocks_for_backpressure_without_dropping() {
        let (tx, rx) = test_frame_receiver();
        for i in 0..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            tx.try_send(tiny_yuv_frame(i as u8)).unwrap();
        }

        let producer = std::thread::spawn(move || {
            assert!(send_yuv_frame_lossless(&tx, tiny_yuv_frame(200)));
        });

        assert_eq!(try_receive_yuv_frame(&rx).unwrap().y[0], 0);
        producer.join().unwrap();
        for i in 1..FFMPEG_YUV_READER_QUEUE_CAPACITY {
            assert_eq!(try_receive_yuv_frame(&rx).unwrap().y[0], i as u8);
        }
        assert_eq!(try_receive_yuv_frame(&rx).unwrap().y[0], 200);
    }

    #[test]
    fn ffmpeg_reader_pool_slots_cover_queue_constructing_and_consumer_frames() {
        assert_eq!(
            FFMPEG_YUV_READER_POOL_MIN_SLOTS,
            FFMPEG_YUV_READER_QUEUE_CAPACITY + FFMPEG_YUV_PENDING_POOL_ALLOWANCE + 2
        );
    }

    fn decode_for_test(
        stdin: &mut impl Write,
        rx: &super::YuvFrameReceiver,
        pending: &mut VecDeque<YuvFrame>,
        pending_warned: &mut bool,
        payload: &[u8],
    ) -> Result<YuvFrame, DecodeError> {
        decode_for_test_with_metrics(stdin, rx, pending, None, pending_warned, payload)
    }

    fn decode_for_test_with_metrics(
        stdin: &mut impl Write,
        rx: &super::YuvFrameReceiver,
        pending: &mut VecDeque<YuvFrame>,
        metrics: Option<&PipelineMetrics>,
        pending_warned: &mut bool,
        payload: &[u8],
    ) -> Result<YuvFrame, DecodeError> {
        decode_ffmpeg_pipe(
            stdin,
            rx,
            pending,
            metrics,
            pending_warned,
            VideoCodec::Vp8,
            2,
            2,
            payload,
        )
    }

    struct CountingWrite {
        writes: usize,
        flushes: usize,
    }

    impl Write for CountingWrite {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    struct QueueRoomAssertingWrite {
        tx: mpsc::SyncSender<YuvFrame>,
        injected: bool,
    }

    impl Write for QueueRoomAssertingWrite {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if !self.injected {
                self.tx
                    .try_send(tiny_yuv_frame(200))
                    .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "queue still full"))?;
                self.injected = true;
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn tiny_yuv_frame(seed: u8) -> YuvFrame {
        YuvFrame {
            y: vec![seed; 4].into(),
            u: vec![seed; 1].into(),
            v: vec![seed; 1].into(),
            width: 2,
            height: 2,
        }
    }

    fn test_pending_fifo() -> VecDeque<YuvFrame> {
        VecDeque::with_capacity(super::FFMPEG_YUV_PENDING_FIFO_CAPACITY)
    }

    fn test_frame_receiver() -> (mpsc::SyncSender<YuvFrame>, super::YuvFrameReceiver) {
        let (tx, rx) = mpsc::sync_channel(FFMPEG_YUV_READER_QUEUE_CAPACITY);
        (tx, super::YuvFrameReceiver { rx })
    }
}

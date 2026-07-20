//! Process admission units for one live media-pipeline generation.

use crate::webrtc::decode::{DecoderBackend, VideoCodec};
use crate::webrtc::session::{MAX_RTP_ACCESS_UNIT_BYTES, RTP_FRAME_QUEUE_CAPACITY};
use crate::webrtc::signals::MAX_JPEG_BYTES_PER_FRAME;
use crate::webrtc::workers::{
    decode_output_pool_slots, jpeg_pool_slots, per_stream_analysis_workers,
    per_stream_decode_workers, per_stream_vlm_workers, WorkerPoolConfig,
};

/// Capacity-plan allowance for ffmpeg demux/codec/filter state outside the
/// explicit parent-owned queues and pools.
pub const SIDECAR_PROCESS_MEMORY_RESERVATION_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSessionResources {
    pub worker_threads: usize,
    pub reserved_bytes: u64,
    pub rtp_queue_bytes: u64,
    pub decoded_frame_bytes: u64,
    pub jpeg_payload_bytes: u64,
    pub scratch_bytes: u64,
    pub sidecar_bytes: u64,
}

impl MediaSessionResources {
    /// Build a conservative capacity envelope for one generation. Payload
    /// terms use hard limits enforced at queue ingress; decoded pools use the
    /// negotiated codec/backend topology. Arithmetic saturates so an invalidly
    /// enormous configured resolution is rejected by admission, not wrapped.
    pub fn for_pipeline(cfg: &WorkerPoolConfig, codec: VideoCodec, clip_mode: bool) -> Self {
        let decode_workers = per_stream_decode_workers(cfg.decode_workers);
        let analysis_workers = if clip_mode {
            per_stream_analysis_workers(cfg.analysis_workers)
        } else {
            0
        };
        let vlm_workers = per_stream_vlm_workers(cfg.vlm_workers);
        let event_writer_workers = usize::from(!clip_mode);
        let clip_accumulator_workers = usize::from(clip_mode);
        let sidecar_reader_workers = usize::from(matches!(
            DecoderBackend::select(cfg.gpu_available, codec),
            DecoderBackend::NvDec | DecoderBackend::FfmpegSw
        ));
        let worker_threads = decode_workers
            .saturating_add(analysis_workers)
            .saturating_add(vlm_workers)
            .saturating_add(event_writer_workers)
            .saturating_add(clip_accumulator_workers)
            .saturating_add(sidecar_reader_workers);

        let pixels = u64::from(cfg.decode_width).saturating_mul(u64::from(cfg.decode_height));
        let yuv_frame_bytes = pixels.saturating_mul(3).saturating_div(2);
        let decoded_frame_bytes = yuv_frame_bytes
            .saturating_mul(decode_output_pool_slots(cfg.gpu_available, codec) as u64);
        let jpeg_payload_bytes = (MAX_JPEG_BYTES_PER_FRAME as u64)
            .saturating_mul(jpeg_pool_slots(cfg.analysis_workers, cfg.vlm_workers) as u64);
        let rtp_queue_bytes = (MAX_RTP_ACCESS_UNIT_BYTES as u64)
            .saturating_mul((RTP_FRAME_QUEUE_CAPACITY + decode_workers + 1) as u64);
        // One YCbCr interleave scratch per decoder and one base64 request
        // buffer per VLM worker. Provider protocols may require base64 on the
        // wire; it is bounded here but never used for durable image storage.
        let decode_scratch = pixels
            .saturating_mul(3)
            .saturating_mul(decode_workers as u64);
        let provider_request_scratch = (MAX_JPEG_BYTES_PER_FRAME as u64)
            .saturating_mul(4)
            .saturating_div(3)
            .saturating_mul(vlm_workers as u64);
        let scratch_bytes = decode_scratch.saturating_add(provider_request_scratch);
        let sidecar_bytes = if sidecar_reader_workers == 0 {
            0
        } else {
            SIDECAR_PROCESS_MEMORY_RESERVATION_BYTES
        };
        let reserved_bytes = rtp_queue_bytes
            .saturating_add(decoded_frame_bytes)
            .saturating_add(jpeg_payload_bytes)
            .saturating_add(scratch_bytes)
            .saturating_add(sidecar_bytes);

        Self {
            worker_threads,
            reserved_bytes,
            rtp_queue_bytes,
            decoded_frame_bytes,
            jpeg_payload_bytes,
            scratch_bytes,
            sidecar_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MediaSessionResources;
    use crate::webrtc::decode::VideoCodec;
    use crate::webrtc::session::WebRtcConfig;
    use crate::webrtc::workers::WorkerPoolConfig;

    #[test]
    fn default_session_plan_is_finite_and_clip_has_one_more_worker() {
        let cfg = WorkerPoolConfig::from(&WebRtcConfig::default());
        let keyframe = MediaSessionResources::for_pipeline(&cfg, VideoCodec::H264, false);
        let clip = MediaSessionResources::for_pipeline(&cfg, VideoCodec::H264, true);
        assert_eq!(keyframe.worker_threads, 4);
        assert_eq!(clip.worker_threads, 5);
        assert!(keyframe.reserved_bytes > keyframe.rtp_queue_bytes);
        assert!(keyframe.reserved_bytes < u64::MAX);
    }
}

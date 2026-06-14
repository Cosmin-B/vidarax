pub mod pipeline;

mod ffmpeg;
mod validate;

pub use ffmpeg::{
    build_select_expr, compute_semantic_frame_indices, decode_mp4_to_frame_signals,
    decode_mp4_to_jpeg_frames, decode_selective_jpeg_frames, extract_video_clip, ffmpeg_path,
    ffprobe_path, nvidia_smi_path, probe_source_fps, DecodedJpegFrame, DecodedMp4Batch,
    FramePacket, FramePacketInput, Mp4DecodeConfig, TimestampNormalizer, make_frame_packet,
};
pub(crate) use ffmpeg::{parse_jpeg_stream_to_frames, FFMPEG_PROTOCOL_WHITELIST};
pub use validate::InputSource;

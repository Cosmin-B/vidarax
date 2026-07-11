pub mod pipeline;

mod fetch;
mod ffmpeg;
mod validate;

pub use fetch::{prepare_source_for_reuse, PreparedSource};
pub use ffmpeg::{
    build_select_expr, compute_semantic_frame_indices, decode_mp4_to_frame_signals,
    decode_mp4_to_jpeg_frames, decode_selective_jpeg_frames, extract_video_clip, ffmpeg_path,
    ffprobe_path, make_frame_packet, nvidia_smi_path, probe_source_fps, DecodedJpegFrame,
    DecodedMp4Batch, FramePacket, FramePacketInput, Mp4DecodeConfig, Timebase, TimestampNormalizer,
};
pub(crate) use ffmpeg::{
    ffmpeg_input_options_for_source, ffmpeg_protocol_whitelist_for_source,
    parse_jpeg_stream_to_frames,
};
pub use validate::InputSource;

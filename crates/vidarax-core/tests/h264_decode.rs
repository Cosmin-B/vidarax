use vidarax_core::webrtc::decode::{Decoder, DecoderConfig, VideoCodec, YuvFrame};

#[test]
fn software_decoder_creates_without_panic() {
    let config = DecoderConfig {
        gpu_available: false,
        codec: VideoCodec::H264,
        width: 1280,
        height: 720,
        output_pool_slots: 1,
    };
    let decoder = Decoder::new(&config);
    assert!(matches!(decoder, Decoder::Software { .. }));
}

#[test]
fn ffmpeg_sw_decoder_selects_for_vp8_no_gpu() {
    let config = DecoderConfig {
        gpu_available: false,
        codec: VideoCodec::Vp8,
        width: 1280,
        height: 720,
        output_pool_slots: 1,
    };
    let decoder = Decoder::new(&config);
    assert!(matches!(decoder, Decoder::FfmpegSw { .. }));
}

#[test]
fn yuv_frame_dimensions() {
    let frame = YuvFrame {
        y: vec![128; 1920 * 1080].into(),
        u: vec![128; 960 * 540].into(),
        v: vec![128; 960 * 540].into(),
        width: 1920,
        height: 1080,
    };
    assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
    assert_eq!(frame.u.len(), (frame.width / 2 * frame.height / 2) as usize);
    assert_eq!(frame.v.len(), (frame.width / 2 * frame.height / 2) as usize);
}

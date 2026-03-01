use vidarax_core::webrtc::decode::{Decoder, DecoderConfig, YuvFrame};

#[test]
fn software_decoder_creates_without_panic() {
    let config = DecoderConfig { gpu_available: false };
    let decoder = Decoder::new(&config);
    assert!(matches!(decoder, Decoder::Software { .. }));
}

#[test]
fn yuv_frame_dimensions() {
    let frame = YuvFrame {
        y: vec![128; 1920 * 1080],
        u: vec![128; 960 * 540],
        v: vec![128; 960 * 540],
        width: 1920,
        height: 1080,
    };
    assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
    assert_eq!(frame.u.len(), (frame.width / 2 * frame.height / 2) as usize);
    assert_eq!(frame.v.len(), (frame.width / 2 * frame.height / 2) as usize);
}

use vidarax_core::webrtc::decode::YuvFrame;
use vidarax_core::webrtc::recycle::VecPool;
use vidarax_core::webrtc::signals::{yuv_to_frame_signal, yuv_to_jpeg};

fn make_gray_frame(luma: u8, width: u32, height: u32) -> YuvFrame {
    YuvFrame {
        y: vec![luma; (width * height) as usize].into(),
        u: vec![128; (width / 2 * height / 2) as usize].into(),
        v: vec![128; (width / 2 * height / 2) as usize].into(),
        width,
        height,
    }
}

#[test]
fn luma_mean_matches_uniform_frame() {
    let frame = make_gray_frame(200, 64, 64);
    let signal = yuv_to_frame_signal(&frame, 0, 0, None);
    let expected = 200.0_f32 / 255.0;
    assert!(
        (signal.luma_mean - expected).abs() < 0.01,
        "luma_mean={}, expected={}",
        signal.luma_mean,
        expected
    );
}

#[test]
fn identical_frames_have_zero_flicker() {
    let frame = make_gray_frame(128, 64, 64);
    let first = yuv_to_frame_signal(&frame, 0, 0, None);
    let second = yuv_to_frame_signal(&frame, 1, 33, Some(&first));
    assert!(
        second.flicker_score < 0.01,
        "flicker={} (expected < 0.01)",
        second.flicker_score
    );
    assert!(
        second.ghosting_score < 0.01,
        "ghosting={} (expected < 0.01)",
        second.ghosting_score
    );
}

#[test]
fn different_frames_have_nonzero_flicker() {
    let dark = make_gray_frame(50, 64, 64);
    let bright = make_gray_frame(200, 64, 64);
    let first = yuv_to_frame_signal(&dark, 0, 0, None);
    let second = yuv_to_frame_signal(&bright, 1, 33, Some(&first));
    assert!(
        second.flicker_score > 0.3,
        "flicker={} (expected > 0.3)",
        second.flicker_score
    );
}

#[test]
fn jpeg_encoding_produces_valid_jpeg() {
    let frame = make_gray_frame(128, 64, 64);
    let mut scratch = Vec::new();
    let jpeg_pool = VecPool::with_slots(1);
    let jpeg = yuv_to_jpeg(&frame, 80, &mut scratch, &jpeg_pool);
    assert!(jpeg.len() > 100, "JPEG too small: {} bytes", jpeg.len());
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "not a JPEG header");
}

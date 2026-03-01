//! YUV frame to FrameSignal conversion and JPEG encoding.
//!
//! This module bridges decoded YUV frames from the [`super::decode`] module
//! to the gate engine's [`crate::gate::FrameSignal`] type, and provides JPEG
//! thumbnail encoding for downstream consumers.

use crate::gate::FrameSignal;
use crate::webrtc::decode::YuvFrame;

/// Compute a 64-bit perceptual hash from the Y (luma) plane.
///
/// Downscales the luma plane to an 8×8 grid by block-averaging, computes the
/// mean of all 64 cells, then sets each bit based on whether the cell value is
/// above the mean. This yields a hash that is robust to small brightness
/// changes while detecting structural differences.
fn perceptual_hash_y(y: &[u8], width: u32, height: u32) -> u64 {
    let w = width as usize;
    let h = height as usize;
    let block_w = w / 8;
    let block_h = h / 8;
    if block_w == 0 || block_h == 0 {
        return 0;
    }

    let mut grid = [0u32; 64];
    for by in 0..8usize {
        for bx in 0..8usize {
            let mut sum = 0u64;
            let mut count = 0u32;
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let px = bx * block_w + dx;
                    let py = by * block_h + dy;
                    if px < w && py < h {
                        sum += y[py * w + px] as u64;
                        count += 1;
                    }
                }
            }
            grid[by * 8 + bx] = if count > 0 {
                (sum / count as u64) as u32
            } else {
                0
            };
        }
    }

    let mean: u32 = grid.iter().sum::<u32>() / 64;
    let mut hash = 0u64;
    for (i, &val) in grid.iter().enumerate() {
        if val > mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Compute a [`FrameSignal`] from a decoded YUV 4:2:0 frame.
///
/// When `prev` is `None` (first frame), `flicker_score` and `ghosting_score`
/// are set to `0.0`. Subsequent calls should pass the previous `FrameSignal`
/// so that temporal deltas can be computed.
///
/// # Arguments
///
/// * `yuv` - The decoded YUV frame to analyse.
/// * `frame_index` - Monotonically increasing frame counter from the stream.
/// * `pts_ms` - Presentation timestamp in milliseconds.
/// * `prev` - Optional previous frame signal for delta computation.
pub fn yuv_to_frame_signal(
    yuv: &YuvFrame,
    frame_index: u64,
    pts_ms: u64,
    prev: Option<&FrameSignal>,
) -> FrameSignal {
    let y = &yuv.y;
    let w = yuv.width;
    let h = yuv.height;
    let pixel_count = y.len() as f64;

    // Luma mean normalised to [0, 1].
    let luma_sum: u64 = y.iter().map(|&b| b as u64).sum();
    let luma_mean = (luma_sum as f64 / pixel_count / 255.0) as f32;

    // Luma variance normalised. Max u8 variance is bounded by (127.5)^2 ≈ 16256.
    // We use 4096 as a practical normalisation cap to keep the score in [0, 1]
    // for typical noisy frames without clipping signal content.
    let mean_f64 = luma_sum as f64 / pixel_count;
    let variance: f64 = y
        .iter()
        .map(|&b| {
            let d = b as f64 - mean_f64;
            d * d
        })
        .sum::<f64>()
        / pixel_count;
    let noise_variance_score = (variance / 4096.0).min(1.0) as f32;

    // Perceptual hash from luma plane.
    let perceptual_hash = perceptual_hash_y(y, w, h);

    // Temporal delta scores — require a previous frame.
    let (flicker_score, ghosting_score) = match prev {
        Some(p) => {
            // Flicker: absolute luma mean delta.
            let flicker = (luma_mean - p.luma_mean).abs();
            // Ghosting: normalised Hamming distance between perceptual hashes.
            let hamming = (perceptual_hash ^ p.perceptual_hash).count_ones();
            let ghosting = hamming as f32 / 64.0;
            (flicker, ghosting)
        }
        None => (0.0f32, 0.0f32),
    };

    FrameSignal {
        frame_index,
        pts_ms,
        perceptual_hash,
        luma_mean,
        flicker_score,
        ghosting_score,
        noise_variance_score,
    }
}

/// Encode a YUV 4:2:0 frame as a JPEG byte buffer.
///
/// Converts YUV to RGB using BT.601 coefficients, then encodes with
/// `jpeg-encoder` at the requested quality level (1–100, higher = better).
///
/// # Panics
///
/// Panics only if the frame dimensions are inconsistent with the plane buffers
/// (i.e. a bug in the caller, not a user error).
pub fn yuv_to_jpeg(yuv: &YuvFrame, quality: u8) -> Vec<u8> {
    let w = yuv.width as usize;
    let h = yuv.height as usize;

    // Convert YUV420 to packed RGB.
    let mut rgb = Vec::with_capacity(w * h * 3);
    for py in 0..h {
        for px in 0..w {
            let y_val = yuv.y[py * w + px] as f32;
            let u_val = yuv.u[(py / 2) * (w / 2) + (px / 2)] as f32 - 128.0;
            let v_val = yuv.v[(py / 2) * (w / 2) + (px / 2)] as f32 - 128.0;

            // BT.601 YCbCr → RGB
            let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344_136 * u_val - 0.714_136 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, quality);
    encoder
        .encode(&rgb, yuv.width as u16, yuv.height as u16, jpeg_encoder::ColorType::Rgb)
        .expect("JPEG encoding failed");
    buf
}

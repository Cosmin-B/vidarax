//! YUV frame to FrameSignal conversion and JPEG encoding.
//!
//! This module bridges decoded YUV frames from the [`super::decode`] module
//! to the gate engine's [`crate::gate::FrameSignal`] type, and provides JPEG
//! thumbnail encoding for downstream consumers.

use crate::crop::PixelCrop;
use crate::gate::FrameSignal;
use crate::webrtc::decode::YuvFrame;
use crate::webrtc::recycle::{RecycledBytes, VecPool};

/// Capacity to reserve up front on a freshly-pooled JPEG buffer.
///
/// Reserving once, lazily, inside [`yuv_to_jpeg`] lets a buffer skip the run of
/// small doubling reallocations it would otherwise walk on its first encode. We
/// do it per in-flight buffer instead of pre-sizing every pool slot, so idle
/// slots stay small and the memory a worker holds tracks how many frames are
/// actually moving through it. A frame that needs more still grows from here —
/// this is a floor, not a cap.
const JPEG_TYPICAL_CAPACITY: usize = 256 * 1024;
/// Hard bound for a JPEG payload that can enter downstream queues. The encoder
/// writes into owned memory first; an oversized result is recycled immediately
/// and never becomes queued work.
pub const MAX_JPEG_BYTES_PER_FRAME: usize = 2 * 1024 * 1024;

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

    // Stride by 4 in both dx/dy — reduces iterations ~16× while preserving
    // hash quality (block averages converge quickly for perceptual hashing).
    let stride = 4;
    let mut grid = [0u32; 64];
    for by in 0..8usize {
        for bx in 0..8usize {
            let mut sum = 0u64;
            let mut count = 0u32;
            let base_x = bx * block_w;
            let base_y = by * block_h;
            // By construction: base_x + dx < 8*(w/8) <= w, same for y.
            // Bounds check is always true, so we skip it.
            let mut dy = 0;
            while dy < block_h {
                let row = (base_y + dy) * w;
                let mut dx = 0;
                while dx < block_w {
                    sum += y[row + base_x + dx] as u64;
                    count += 1;
                    dx += stride;
                }
                dy += stride;
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

/// A decoded frame whose planes are too small for the dimensions it claims.
///
/// A working 4:2:0 decoder never emits one. This is the guard against a corrupt
/// or truncated decode: every pixel routine below indexes the planes straight
/// from `width`/`height`, so a short plane would read past its end. We treat
/// that as recoverable state and drop the one frame rather than fault the
/// capture — which, under `panic = "abort"`, an out-of-bounds read would do.
#[derive(Debug, Clone)]
pub struct MalformedFrame {
    pub width: u32,
    pub height: u32,
    detail: String,
}

impl std::fmt::Display for MalformedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "malformed {}x{} frame: {}",
            self.width, self.height, self.detail
        )
    }
}

impl std::error::Error for MalformedFrame {}

/// Confirm a decoded frame's planes hold enough samples to read it safely.
///
/// The signal and JPEG paths walk the luma plane across `width * height` and
/// each chroma plane across `(width/2) * (height/2)`. A 4:2:0 frame has even
/// dimensions by construction, so anything odd or zero-sized is already wrong.
/// Once a frame clears this check, every downstream index stays in bounds.
pub fn check_frame(yuv: &YuvFrame) -> Result<(), MalformedFrame> {
    let reject = |detail: String| {
        Err(MalformedFrame {
            width: yuv.width,
            height: yuv.height,
            detail,
        })
    };

    let (w, h) = (yuv.width as usize, yuv.height as usize);
    if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
        return reject("dimensions must be even and non-zero for 4:2:0".to_string());
    }

    // checked_mul so a 32-bit usize can't wrap w*h down to a small value and
    // wave a truncated frame through the length check; on such a target an
    // overflowing frame is itself malformed. chroma is at most luma/4, so once
    // luma has not overflowed its plain product cannot either.
    let Some(luma) = w.checked_mul(h) else {
        return reject(format!("dimensions {w}x{h} overflow a usize"));
    };
    let chroma = (w / 2) * (h / 2);
    if yuv.y.len() < luma || yuv.u.len() < chroma || yuv.v.len() < chroma {
        return reject(format!(
            "short planes: y={} u={} v={}, need y>={luma} and u,v>={chroma}",
            yuv.y.len(),
            yuv.u.len(),
            yuv.v.len(),
        ));
    }

    Ok(())
}

/// Reusable Y/U/V plane buffers for the live-path crop.
///
/// The crop runs on every decoded frame before the gate decides to keep it, so
/// allocating three fresh plane buffers per frame would put per-frame heap
/// traffic on the hot path. This pools them the same way the JPEG encoder pools
/// its output: [`crop_yuv`] draws cleared buffers here and the returned frame's
/// planes recycle back on drop.
#[derive(Clone, Debug)]
pub struct CropPool {
    y: VecPool,
    u: VecPool,
    v: VecPool,
}

impl CropPool {
    pub fn new(slots: usize) -> Self {
        let slots = slots.max(1);
        Self {
            y: VecPool::with_slots(slots),
            u: VecPool::with_slots(slots),
            v: VecPool::with_slots(slots),
        }
    }
}

/// Copy the sub-rectangle `rect` out of `yuv` into a packed I420 frame drawn
/// from `pool`.
///
/// I420 planes are stored without stride, so a sub-rectangle is not a contiguous
/// slice; this re-packs row by row into pooled plane buffers. Cropping here,
/// right after decode, means the perceptual-hash/luma signals and the JPEG the
/// VLM sees are all restricted to the same region of the screen.
///
/// Returns `None` (leave the frame untouched) when the source planes are too
/// short to read `rect` safely or `rect` does not fit inside the frame, so a
/// malformed frame is never indexed out of bounds — the caller's existing
/// [`check_frame`] guard still sees the original and drops it. `rect` is expected
/// even-aligned and in-bounds, as produced by
/// [`crate::crop::CropRegion::resolve`].
pub fn crop_yuv(yuv: &YuvFrame, rect: PixelCrop, pool: &CropPool) -> Option<YuvFrame> {
    let src_w = yuv.width as usize;
    let src_h = yuv.height as usize;
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let cw = rect.width as usize;
    let ch = rect.height as usize;

    // The rectangle must be even (4:2:0 chroma grid) and fit inside the frame.
    if cw == 0 || ch == 0 || cw % 2 != 0 || ch % 2 != 0 || x0 % 2 != 0 || y0 % 2 != 0 {
        return None;
    }
    if x0 + cw > src_w || y0 + ch > src_h {
        return None;
    }
    // Confirm the source planes actually hold the frame they claim before we
    // index them; a malformed frame falls through to the caller unchanged.
    if check_frame(yuv).is_err() {
        return None;
    }

    let mut y = pool.y.acquire();
    y.reserve(cw * ch);
    for row in 0..ch {
        let start = (y0 + row) * src_w + x0;
        y.extend_from_slice(&yuv.y[start..start + cw]);
    }

    // Chroma is half resolution in each axis; offsets and extents halve cleanly
    // because the luma rectangle is even on all sides.
    let src_cw = src_w / 2;
    let ccw = cw / 2;
    let cch = ch / 2;
    let cx0 = x0 / 2;
    let cy0 = y0 / 2;
    let mut u = pool.u.acquire();
    let mut v = pool.v.acquire();
    u.reserve(ccw * cch);
    v.reserve(ccw * cch);
    for row in 0..cch {
        let start = (cy0 + row) * src_cw + cx0;
        u.extend_from_slice(&yuv.u[start..start + ccw]);
        v.extend_from_slice(&yuv.v[start..start + ccw]);
    }

    Some(YuvFrame {
        y: pool.y.recycle(y),
        u: pool.u.recycle(u),
        v: pool.v.recycle(v),
        width: rect.width,
        height: rect.height,
    })
}

/// Compute a [`FrameSignal`] from a decoded YUV 4:2:0 frame.
///
/// Validates the frame with [`check_frame`] first, so a `YuvFrame` whose planes
/// are shorter than its claimed dimensions returns [`MalformedFrame`] instead of
/// panicking out of bounds. Callers already on the validated hot path use
/// `yuv_to_frame_signal_unchecked` (crate-internal) to skip the re-check.
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
///
/// # Errors
///
/// Returns [`MalformedFrame`] when the frame fails [`check_frame`].
pub fn yuv_to_frame_signal(
    yuv: &YuvFrame,
    frame_index: u64,
    pts_ms: u64,
    prev: Option<&FrameSignal>,
) -> Result<FrameSignal, MalformedFrame> {
    check_frame(yuv)?;
    Ok(yuv_to_frame_signal_unchecked(
        yuv,
        frame_index,
        pts_ms,
        prev,
    ))
}

/// Compute a [`FrameSignal`] from a frame that has *already* cleared
/// [`check_frame`].
///
/// This reads both planes by dimension and does no bounds guarding of its own,
/// so passing a frame with planes shorter than `width * height` panics. The
/// worker path validates once per frame and then calls this directly; external
/// callers should prefer the checked [`yuv_to_frame_signal`].
pub(crate) fn yuv_to_frame_signal_unchecked(
    yuv: &YuvFrame,
    frame_index: u64,
    pts_ms: u64,
    prev: Option<&FrameSignal>,
) -> FrameSignal {
    let y = &yuv.y;
    let w = yuv.width;
    let h = yuv.height;

    // Single-pass luma mean + variance via E[X²] - E[X]² with 4× subsampling.
    // Subsampling every 4th pixel cuts work 4× while converging to the same
    // statistics for typical video frames.
    //
    // Walk only the active w*h samples, not the whole buffer: check_frame
    // guarantees y holds at least that many, and a plane the decoder happens to
    // pad past its dimensions shouldn't drag its trailing bytes into the stats.
    let active_luma = w as usize * h as usize;
    let stride = 4;
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut count = 0u64;
    let mut i = 0;
    while i < active_luma {
        let v = y[i] as u64;
        sum += v;
        sum_sq += v * v;
        count += 1;
        i += stride;
    }

    let mean_raw = sum as f64 / count as f64;
    let luma_mean = (mean_raw / 255.0) as f32;

    // Variance via E[X²] - E[X]². Normalised with 4096 cap to keep [0, 1].
    let variance = (sum_sq as f64 / count as f64) - (mean_raw * mean_raw);
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

/// A frame that could not be turned into a JPEG.
///
/// Every reason is a per-frame, recoverable condition: a frame that fails
/// [`check_frame`] before the encoder ever sees it (from the public
/// [`yuv_to_jpeg`]), a frame whose dimensions exceed the encoder's `u16` size
/// fields, or a frame the encoder itself refuses. Either way the caller drops
/// that one frame's thumbnail and the stream keeps running.
#[derive(Debug, Clone)]
pub struct JpegEncodeError {
    pub width: u32,
    pub height: u32,
    detail: String,
}

impl std::fmt::Display for JpegEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not encode {}x{} frame as jpeg: {}",
            self.width, self.height, self.detail
        )
    }
}

impl std::error::Error for JpegEncodeError {}

impl From<MalformedFrame> for JpegEncodeError {
    fn from(err: MalformedFrame) -> Self {
        JpegEncodeError {
            width: err.width,
            height: err.height,
            detail: err.detail,
        }
    }
}

/// Encode a decoded YUV 4:2:0 frame into a pooled JPEG buffer.
///
/// Validates the frame with [`check_frame`] first, so a `YuvFrame` with planes
/// shorter than its claimed dimensions returns [`JpegEncodeError`] instead of
/// panicking on the interleave. Callers already on the validated hot path use
/// `yuv_to_jpeg_unchecked` (crate-internal) to skip the re-check.
///
/// See `yuv_to_jpeg_unchecked` (crate-internal) for the `scratch`/`output_pool`
/// contract.
///
/// # Errors
///
/// Returns [`JpegEncodeError`] when the frame fails [`check_frame`], exceeds the
/// dimensions the encoder's `u16` size fields can hold, or is rejected by the
/// encoder; in every case the caller drops the thumbnail and the stream keeps
/// running.
pub fn yuv_to_jpeg(
    yuv: &YuvFrame,
    quality: u8,
    scratch: &mut Vec<u8>,
    output_pool: &VecPool,
) -> Result<RecycledBytes, JpegEncodeError> {
    check_frame(yuv)?;
    yuv_to_jpeg_unchecked(yuv, quality, scratch, output_pool)
}

/// Encode a frame that has *already* cleared [`check_frame`] into a pooled JPEG.
///
/// The planar YUV is interleaved straight into YCbCr, which the encoder accepts
/// natively — so neither side pays for a BT.601 float conversion, and the chroma
/// upsample is a nearest-neighbour lookup rather than per-pixel float math.
///
/// `scratch` is the caller's reused interleave buffer, held across frames so the
/// large YCbCr staging area isn't reallocated every call. The finished JPEG
/// lands in a buffer drawn from `output_pool` and returns to it on drop.
///
/// The interleave below indexes the planes by dimension with no bounds guarding,
/// so a frame that has not cleared [`check_frame`] panics. The worker path
/// validates once per frame and then calls this directly; external callers
/// should prefer the checked [`yuv_to_jpeg`].
///
/// # Errors
///
/// Returns [`JpegEncodeError`] when the dimensions exceed what the encoder's
/// `u16` size fields can hold, or if the encoder itself rejects an in-range
/// frame. On a validated, in-range frame the latter is effectively unreachable,
/// but it stays typed so the caller can drop the thumbnail and keep the stream
/// running rather than take the encoder's word on faith.
pub(crate) fn yuv_to_jpeg_unchecked(
    yuv: &YuvFrame,
    quality: u8,
    scratch: &mut Vec<u8>,
    output_pool: &VecPool,
) -> Result<RecycledBytes, JpegEncodeError> {
    // The encoder takes width and height as u16. A larger frame would be
    // truncated in the encode call and produce a valid-looking jpeg at the
    // wrong size, so reject it here before interleaving anything.
    if yuv.width > u16::MAX as u32 || yuv.height > u16::MAX as u32 {
        return Err(JpegEncodeError {
            width: yuv.width,
            height: yuv.height,
            detail: format!(
                "dimensions exceed the {}px the jpeg encoder accepts",
                u16::MAX
            ),
        });
    }

    let w = yuv.width as usize;
    let h = yuv.height as usize;
    let half_w = w / 2;

    // Reuse scratch buffer for YCbCr interleave; avoids per-frame heap allocation.
    scratch.clear();
    scratch.reserve(w * h * 3);
    for py in 0..h {
        let y_row = py * w;
        let c_row = (py / 2) * half_w;
        for px in 0..w {
            scratch.push(yuv.y[y_row + px]);
            scratch.push(yuv.u[c_row + px / 2]);
            scratch.push(yuv.v[c_row + px / 2]);
        }
    }

    let mut buf = output_pool.acquire();
    // Grow a fresh buffer to the reserve size in one shot so the encoder's first
    // write doesn't climb through a chain of small reallocations. Once a buffer
    // has cycled through the pool this is a no-op.
    buf.reserve(JPEG_TYPICAL_CAPACITY);
    let encoder = jpeg_encoder::Encoder::new(&mut buf, quality);
    let outcome = encoder.encode(
        scratch,
        yuv.width as u16,
        yuv.height as u16,
        jpeg_encoder::ColorType::Ycbcr,
    );

    // Return the backing buffer to the pool whichever way the encode went — a
    // rejected frame must not bleed a slot out of the free-list. On the error
    // path this handle drops at the end of the match and the slot goes back.
    let bytes = output_pool.recycle(buf);
    match outcome {
        Ok(()) if bytes.len() <= MAX_JPEG_BYTES_PER_FRAME => Ok(bytes),
        Ok(()) => Err(JpegEncodeError {
            width: yuv.width,
            height: yuv.height,
            detail: format!(
                "encoded payload exceeds the {} byte pipeline limit",
                MAX_JPEG_BYTES_PER_FRAME
            ),
        }),
        Err(err) => Err(JpegEncodeError {
            width: yuv.width,
            height: yuv.height,
            detail: err.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32) -> YuvFrame {
        let (wu, hu) = (w as usize, h as usize);
        YuvFrame {
            y: vec![128u8; wu * hu].into(),
            u: vec![128u8; (wu / 2) * (hu / 2)].into(),
            v: vec![128u8; (wu / 2) * (hu / 2)].into(),
            width: w,
            height: h,
        }
    }

    #[test]
    fn check_frame_accepts_well_formed_and_rejects_broken_frames() {
        assert!(check_frame(&solid_frame(16, 16)).is_ok());

        // A luma plane one sample short of width*height would read out of bounds
        // in the interleave and hash loops.
        let mut short = solid_frame(16, 16);
        short.y = vec![128u8; 16 * 16 - 1].into();
        assert!(check_frame(&short).is_err());

        // Zero and odd dimensions can't describe a 4:2:0 frame.
        assert!(check_frame(&solid_frame(0, 16)).is_err());
        let mut odd = solid_frame(16, 16);
        odd.width = 15;
        assert!(check_frame(&odd).is_err());
    }

    #[test]
    fn crop_yuv_repacks_the_requested_rectangle() {
        // 4x4 luma with distinct values 0..16, 2x2 chroma 0..4 / 100..104.
        let frame = YuvFrame {
            y: (0u8..16).collect::<Vec<_>>().into(),
            u: vec![0u8, 1, 2, 3].into(),
            v: vec![100u8, 101, 102, 103].into(),
            width: 4,
            height: 4,
        };
        // Bottom-right 2x2 luma quadrant.
        let rect = PixelCrop {
            x: 2,
            y: 2,
            width: 2,
            height: 2,
        };
        let pool = CropPool::new(2);
        let cropped = crop_yuv(&frame, rect, &pool).expect("in-bounds crop");
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        // Rows (2,3) x cols (2,3): y = [10,11, 14,15].
        assert_eq!(&*cropped.y, &[10, 11, 14, 15]);
        // Chroma is 1x1 at (1,1): u = [3], v = [103].
        assert_eq!(&*cropped.u, &[3]);
        assert_eq!(&*cropped.v, &[103]);
        // The re-packed frame is itself well formed.
        assert!(check_frame(&cropped).is_ok());
    }

    #[test]
    fn crop_yuv_rejects_out_of_bounds_and_malformed() {
        let frame = solid_frame(16, 16);
        let pool = CropPool::new(2);
        // Runs past the right edge.
        assert!(crop_yuv(
            &frame,
            PixelCrop {
                x: 8,
                y: 0,
                width: 12,
                height: 4
            },
            &pool
        )
        .is_none());
        // Odd rectangle can't map onto the 4:2:0 chroma grid.
        assert!(crop_yuv(
            &frame,
            PixelCrop {
                x: 0,
                y: 0,
                width: 3,
                height: 4
            },
            &pool
        )
        .is_none());
        // A short-plane source is left for check_frame to reject, not indexed.
        let mut short = solid_frame(16, 16);
        short.y = vec![128u8; 16 * 16 - 1].into();
        assert!(crop_yuv(
            &short,
            PixelCrop {
                x: 0,
                y: 0,
                width: 8,
                height: 8
            },
            &pool
        )
        .is_none());
    }

    #[test]
    fn yuv_to_jpeg_emits_valid_jpeg_and_keeps_pool_buffer_reserved() {
        let frame = solid_frame(16, 16);
        let mut scratch = Vec::new();
        let pool = VecPool::with_slots(1);

        let jpeg = yuv_to_jpeg(&frame, 75, &mut scratch, &pool).expect("a solid frame encodes");
        // A well-formed JPEG is bracketed by SOI (FFD8) and EOI (FFD9) markers.
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing JPEG SOI marker");
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "missing JPEG EOI marker"
        );

        // Returning the buffer to the pool must retain the lazy reservation, so
        // the next frame reuses this allocation instead of doubling up from zero.
        drop(jpeg);
        let recycled = pool.acquire();
        assert!(
            recycled.capacity() >= JPEG_TYPICAL_CAPACITY,
            "pooled buffer lost its reservation: {} < {JPEG_TYPICAL_CAPACITY}",
            recycled.capacity(),
        );
    }

    #[test]
    fn yuv_to_frame_signal_rejects_a_malformed_frame_instead_of_panicking() {
        // The stride-4 luma stats loop reads index 252 of a 16x16 frame, so a Y
        // plane of 252 bytes (16*16 - 4) is genuinely short enough to index out
        // of bounds in the unchecked body. The checked wrapper must reject it
        // instead: if the check regressed, this would panic on that read.
        let mut short = solid_frame(16, 16);
        short.y = vec![128u8; 16 * 16 - 4].into();
        assert!(
            yuv_to_frame_signal(&short, 0, 0, None).is_err(),
            "the checked helper must reject a short frame, not panic on it"
        );
        // A well-formed frame still yields a signal.
        assert!(yuv_to_frame_signal(&solid_frame(16, 16), 7, 33, None).is_ok());
    }

    #[test]
    fn yuv_to_jpeg_rejects_a_malformed_frame_instead_of_panicking() {
        // A chroma plane one sample short would index past its end in the
        // interleave loop, before the encoder ever runs.
        let mut short = solid_frame(16, 16);
        short.u = vec![128u8; (16 / 2) * (16 / 2) - 1].into();
        let mut scratch = Vec::new();
        let pool = VecPool::with_slots(1);
        let err = yuv_to_jpeg(&short, 75, &mut scratch, &pool)
            .expect_err("the checked helper must reject a short frame, not panic on it");
        // The MalformedFrame reason is folded into the JpegEncodeError, keeping
        // the frame's dimensions so the drop is attributable.
        assert_eq!(err.width, 16);
        assert_eq!(err.height, 16);
    }

    #[test]
    fn yuv_to_jpeg_rejects_dimensions_larger_than_the_encoder_accepts() {
        // The encoder's width/height are u16. 65538 would truncate to 2 in that
        // cast and encode a valid-looking jpeg at the wrong size, so an in-range
        // planar frame at those dims must be rejected rather than silently
        // mis-encoded. The planes are sized for the claimed dims so check_frame
        // passes and the u16 guard is what catches it.
        let frame = solid_frame(65538, 2);
        let mut scratch = Vec::new();
        let pool = VecPool::with_slots(1);
        let err = yuv_to_jpeg(&frame, 75, &mut scratch, &pool)
            .expect_err("dimensions past u16 must be rejected, not truncated");
        assert_eq!(err.width, 65538);
    }
}

//! Region-of-interest crop shared by the file/URI decode path and the live WHIP
//! path.
//!
//! A crop is normalized: `x`, `y`, `width`, `height` are fractions of the frame
//! in `[0, 1]`, with `(0, 0)` at the top-left. Fractions keep one crop valid
//! whatever the decoded resolution turns out to be, which matters here for two
//! reasons: the file path never learns the frame size in Rust (ffmpeg does the
//! decode in a subprocess), and a live screen can renegotiate its resolution
//! mid-session. So the same `{x, y, width, height}` a caller sends works against
//! a 720p clip and a 4K screen without translation.

use serde::{Deserialize, Serialize};

/// A normalized rectangle selecting the part of each frame to analyze.
///
/// All four values are fractions of the frame. `x`/`y` are the top-left corner,
/// `width`/`height` the extent, so a centered half-size crop is
/// `{ x: 0.25, y: 0.25, width: 0.5, height: 0.5 }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CropRegion {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Why a [`CropRegion`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropError {
    /// A value was NaN or infinite.
    NonFinite,
    /// The origin fell outside `[0, 1]`.
    OriginOutOfRange,
    /// Width or height was zero or negative.
    NonPositiveExtent,
    /// The rectangle ran past the right or bottom edge (`x + width > 1`, etc.).
    ExceedsFrame,
}

impl std::fmt::Display for CropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            CropError::NonFinite => "crop values must be finite",
            CropError::OriginOutOfRange => "crop x and y must be in [0, 1]",
            CropError::NonPositiveExtent => "crop width and height must be > 0",
            CropError::ExceedsFrame => {
                "crop must stay within the frame (x + width <= 1, y + height <= 1)"
            }
        };
        f.write_str(msg)
    }
}

impl std::error::Error for CropError {}

/// A crop resolved to whole even pixels within a concrete frame size.
///
/// Even offsets and extents keep the rectangle aligned to the 4:2:0 chroma grid,
/// which both the JPEG encoder and `check_frame` on the live path require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelCrop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A crop resolved against a concrete frame size.
///
/// The three cases are deliberately distinct so a caller never confuses "analyze
/// the whole frame" with "the requested region is too small to represent." The
/// difference matters: falling back to the full frame for a too-small crop would
/// analyze *more* of the screen than the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedCrop {
    /// The crop covers the whole frame; use the frame unchanged.
    Full,
    /// Crop to this rectangle.
    Rect(PixelCrop),
    /// The region is smaller than one 2x2 chroma block at this frame size, or
    /// its origin falls outside the frame. The caller should drop the frame, not
    /// widen it back to full.
    TooSmall,
}

#[inline]
fn floor_to_even(v: u32) -> u32 {
    v & !1
}

impl CropRegion {
    /// Reject a crop that is not a usable sub-rectangle of the frame. A small
    /// epsilon absorbs float rounding right at the `1.0` edge so a caller-sent
    /// `{x: 0.5, width: 0.5}` is not spuriously rejected.
    pub fn validate(&self) -> Result<(), CropError> {
        for v in [self.x, self.y, self.width, self.height] {
            if !v.is_finite() {
                return Err(CropError::NonFinite);
            }
        }
        if self.width <= 0.0 || self.height <= 0.0 {
            return Err(CropError::NonPositiveExtent);
        }
        if self.x < 0.0 || self.y < 0.0 || self.x > 1.0 || self.y > 1.0 {
            return Err(CropError::OriginOutOfRange);
        }
        const EPS: f32 = 1e-4;
        if self.x + self.width > 1.0 + EPS || self.y + self.height > 1.0 + EPS {
            return Err(CropError::ExceedsFrame);
        }
        Ok(())
    }

    /// True when the crop selects the whole frame, so applying it is a no-op.
    pub fn is_full_frame(&self) -> bool {
        self.x <= 0.0 && self.y <= 0.0 && self.width >= 1.0 && self.height >= 1.0
    }

    /// ffmpeg `crop` filter for this region.
    ///
    /// Uses `iw`/`ih` so ffmpeg resolves the rectangle against the real input
    /// size in its own process. Width, height, and offsets are floored to even
    /// (`floor(.../2)*2`) so the cropped frame stays valid for the yuv420p /
    /// mjpeg consumers downstream. Emit this as the first filter in a chain so
    /// any later `fps`, `select`, or `scale` operates on already-cropped pixels.
    pub fn ffmpeg_crop_filter(&self) -> String {
        format!(
            "crop=floor(iw*{w:.6}/2)*2:floor(ih*{h:.6}/2)*2:floor(iw*{x:.6}/2)*2:floor(ih*{y:.6}/2)*2",
            w = self.width,
            h = self.height,
            x = self.x,
            y = self.y,
        )
    }

    /// Resolve this crop against a concrete `frame_w x frame_h` frame.
    ///
    /// Fractions are clamped to `[0, 1]` defensively so an unvalidated crop can
    /// never produce an out-of-frame rectangle. A crop that covers the frame
    /// resolves to [`ResolvedCrop::Full`]; one too small to represent (under 2x2
    /// after even-flooring, or with an origin off the frame) to
    /// [`ResolvedCrop::TooSmall`] so the caller drops the frame rather than
    /// analyzing the whole screen.
    pub fn resolve(&self, frame_w: u32, frame_h: u32) -> ResolvedCrop {
        if self.is_full_frame() {
            return ResolvedCrop::Full;
        }
        if frame_w < 2 || frame_h < 2 {
            return ResolvedCrop::TooSmall;
        }
        let clampf = |v: f32| v.clamp(0.0, 1.0);
        let fw = frame_w as f32;
        let fh = frame_h as f32;

        let x = floor_to_even((clampf(self.x) * fw) as u32);
        let y = floor_to_even((clampf(self.y) * fh) as u32);
        if x >= frame_w || y >= frame_h {
            return ResolvedCrop::TooSmall;
        }

        // Never run past the frame; keep the clamped extent even too.
        let w = floor_to_even((clampf(self.width) * fw) as u32).min(floor_to_even(frame_w - x));
        let h = floor_to_even((clampf(self.height) * fh) as u32).min(floor_to_even(frame_h - y));
        if w < 2 || h < 2 {
            return ResolvedCrop::TooSmall;
        }
        if x == 0 && y == 0 && w == frame_w && h == frame_h {
            return ResolvedCrop::Full;
        }
        ResolvedCrop::Rect(PixelCrop {
            x,
            y,
            width: w,
            height: h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_normal_and_full_frame() {
        assert!(CropRegion {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.5
        }
        .validate()
        .is_ok());
        assert!(CropRegion {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0
        }
        .validate()
        .is_ok());
        // Flush against the right/bottom edge is fine.
        assert!(CropRegion {
            x: 0.5,
            y: 0.5,
            width: 0.5,
            height: 0.5
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn validate_rejects_bad_crops() {
        assert_eq!(
            CropRegion {
                x: f32::NAN,
                y: 0.0,
                width: 0.5,
                height: 0.5
            }
            .validate(),
            Err(CropError::NonFinite)
        );
        assert_eq!(
            CropRegion {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.5
            }
            .validate(),
            Err(CropError::NonPositiveExtent)
        );
        assert_eq!(
            CropRegion {
                x: -0.1,
                y: 0.0,
                width: 0.5,
                height: 0.5
            }
            .validate(),
            Err(CropError::OriginOutOfRange)
        );
        assert_eq!(
            CropRegion {
                x: 0.7,
                y: 0.0,
                width: 0.5,
                height: 0.5
            }
            .validate(),
            Err(CropError::ExceedsFrame)
        );
    }

    #[test]
    fn ffmpeg_filter_is_even_floored_and_comma_free() {
        let f = CropRegion {
            x: 0.25,
            y: 0.1,
            width: 0.5,
            height: 0.5,
        }
        .ffmpeg_crop_filter();
        assert!(f.starts_with("crop="));
        // Commas would split the filtergraph; the expression must use only `:`.
        assert!(!f.contains(','), "crop expr must not contain commas: {f}");
        assert!(f.contains("floor(iw*0.500000/2)*2"));
    }

    #[test]
    fn resolve_centers_and_aligns() {
        let r = CropRegion {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.5,
        }
        .resolve(1920, 1080);
        assert_eq!(
            r,
            ResolvedCrop::Rect(PixelCrop {
                x: 480,
                y: 270,
                width: 960,
                height: 540
            })
        );
    }

    #[test]
    fn resolve_full_frame_is_full() {
        assert_eq!(
            CropRegion {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0
            }
            .resolve(1280, 720),
            ResolvedCrop::Full
        );
    }

    #[test]
    fn resolve_near_full_crop_is_a_rect_not_full() {
        // A sub-1.0 width even-floors below the frame width, so a near-full crop
        // stays a real Rect rather than collapsing to Full.
        let ResolvedCrop::Rect(r) = (CropRegion {
            x: 0.0,
            y: 0.0,
            width: 0.99,
            height: 0.99,
        })
        .resolve(1920, 1080) else {
            panic!("expected a Rect for a near-full crop");
        };
        assert!(r.width < 1920 && r.width > 1800, "width was {}", r.width);
        assert!(
            r.height < 1080 && r.height > 1000,
            "height was {}",
            r.height
        );
    }

    #[test]
    fn resolve_clamps_overshoot_and_stays_even() {
        // Odd frame dims plus a crop that would run off the right edge: the
        // result must be even and stay inside the frame.
        let ResolvedCrop::Rect(r) = (CropRegion {
            x: 0.5,
            y: 0.5,
            width: 0.9,
            height: 0.9,
        })
        .resolve(101, 101) else {
            panic!("expected a usable rect");
        };
        assert_eq!(r.x % 2, 0);
        assert_eq!(r.y % 2, 0);
        assert_eq!(r.width % 2, 0);
        assert_eq!(r.height % 2, 0);
        assert!(r.x + r.width <= 101);
        assert!(r.y + r.height <= 101);
    }

    #[test]
    fn resolve_tiny_or_offscreen_crop_is_too_small() {
        // Sub-2px region: must NOT fall back to the whole frame.
        assert_eq!(
            CropRegion {
                x: 0.0,
                y: 0.0,
                width: 0.001,
                height: 0.001
            }
            .resolve(640, 480),
            ResolvedCrop::TooSmall
        );
        // Origin flush against the right edge (validates under the epsilon slack)
        // has nowhere to extend.
        assert_eq!(
            CropRegion {
                x: 1.0,
                y: 0.0,
                width: 0.00005,
                height: 0.5
            }
            .resolve(640, 480),
            ResolvedCrop::TooSmall
        );
    }
}

//! Explicit image-coordinate provenance for decoded frames and durable events.
//!
//! `vidarax.image.v1` uses source-image pixels with `(0, 0)` at the top-left,
//! `x` increasing right, and `y` increasing down. Normalized regions use the
//! same origin and axes in the closed interval `[0, 1]`.

use serde::{Deserialize, Serialize};

use crate::crop::{CropRegion, PixelCrop, ResolvedCrop};

/// Stable identifier written beside serialized [`FrameCoordinates`].
pub const IMAGE_COORDINATE_SCHEMA: &str = "vidarax.image.v1";

/// Width and height in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelExtent {
    pub width: u32,
    pub height: u32,
}

/// Rectangle in source-image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Rectangle in normalized source-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// The transform from a source video frame to the pixels inspected by the
/// deterministic filter.
///
/// The value is plain frame metadata: it owns no buffers and performs no heap
/// allocation when copied between pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameCoordinates {
    /// Dimensions before crop or resize.
    pub source_extent: PixelExtent,
    /// Caller-requested region in normalized source coordinates.
    pub requested_region: NormalizedRect,
    /// Exact even-aligned source pixels selected after resolving the request.
    pub resolved_region: PixelRect,
    /// Dimensions after crop, before any optional model-transport resize.
    pub analysis_extent: PixelExtent,
}

impl FrameCoordinates {
    /// Coordinate provenance for an uncropped frame.
    pub fn full_frame(width: u32, height: u32) -> Self {
        let extent = PixelExtent { width, height };
        Self {
            source_extent: extent,
            requested_region: NormalizedRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            resolved_region: PixelRect {
                x: 0,
                y: 0,
                width,
                height,
            },
            analysis_extent: extent,
        }
    }

    /// Resolve an optional normalized crop against a concrete source frame.
    ///
    /// Returns `None` when the requested crop is too small to represent. This
    /// matches the live crop path: the caller must drop that frame rather than
    /// silently widen the request to the full image.
    pub fn resolve(width: u32, height: u32, crop: Option<CropRegion>) -> Option<Self> {
        let Some(crop) = crop else {
            return Some(Self::full_frame(width, height));
        };
        let requested_region = NormalizedRect {
            x: crop.x,
            y: crop.y,
            width: crop.width,
            height: crop.height,
        };
        let rect = match crop.resolve(width, height) {
            ResolvedCrop::Full => PixelCrop {
                x: 0,
                y: 0,
                width,
                height,
            },
            ResolvedCrop::Rect(rect) => rect,
            ResolvedCrop::TooSmall => return None,
        };
        Some(Self {
            source_extent: PixelExtent { width, height },
            requested_region,
            resolved_region: PixelRect {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            },
            analysis_extent: PixelExtent {
                width: rect.width,
                height: rect.height,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_frame_contract_is_explicit() {
        let coordinates = FrameCoordinates::full_frame(1920, 1080);
        assert_eq!(coordinates.source_extent.width, 1920);
        assert_eq!(coordinates.resolved_region.x, 0);
        assert_eq!(coordinates.resolved_region.height, 1080);
        assert_eq!(coordinates.analysis_extent, coordinates.source_extent);
    }

    #[test]
    fn crop_preserves_request_and_exact_even_pixel_resolution() {
        let coordinates = FrameCoordinates::resolve(
            1920,
            1080,
            Some(CropRegion {
                x: 0.25,
                y: 0.1,
                width: 0.5,
                height: 0.5,
            }),
        )
        .expect("representable crop");

        assert_eq!(coordinates.requested_region.x, 0.25);
        assert_eq!(coordinates.resolved_region.x, 480);
        assert_eq!(coordinates.resolved_region.y, 108);
        assert_eq!(coordinates.analysis_extent.width, 960);
        assert_eq!(coordinates.analysis_extent.height, 540);
    }

    #[test]
    fn unrepresentable_crop_is_not_relabelled_as_full_frame() {
        let coordinates = FrameCoordinates::resolve(
            16,
            16,
            Some(CropRegion {
                x: 0.5,
                y: 0.5,
                width: 0.01,
                height: 0.01,
            }),
        );
        assert!(coordinates.is_none());
    }

    #[test]
    fn coordinate_value_has_no_owned_heap_fields() {
        assert_eq!(
            std::mem::size_of::<FrameCoordinates>(),
            std::mem::size_of::<PixelExtent>() * 2
                + std::mem::size_of::<PixelRect>()
                + std::mem::size_of::<NormalizedRect>()
        );
        assert!(!std::mem::needs_drop::<FrameCoordinates>());
    }
}

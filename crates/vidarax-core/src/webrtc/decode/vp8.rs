//! libvpx context ownership and decoder-owned plane copies.
//! No image pointer escapes a decode call or survives the next libvpx call.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::os::raw::{c_int, c_uint};
use vpx_sys::{
    vpx_codec_ctx_t, vpx_codec_dec_init_ver, vpx_codec_decode, vpx_codec_destroy, vpx_codec_err_t,
    vpx_codec_error, vpx_codec_error_detail, vpx_codec_get_frame, vpx_codec_iter_t,
    vpx_codec_vp8_dx, vpx_image_t, vpx_img_fmt_t, VPX_DECODER_ABI_VERSION,
};

use super::{DecodeError, YuvFrame, YuvPlanePools};

pub struct Vp8DecoderCtx {
    ctx: Box<vpx_codec_ctx_t>,
}

impl Vp8DecoderCtx {
    pub(super) fn new() -> Result<Self, DecodeError> {
        // A zeroed vpx_codec_ctx_t is the libvpx-required initial state.
        let mut ctx = Box::new(unsafe { MaybeUninit::<vpx_codec_ctx_t>::zeroed().assume_init() });
        let ctx_ptr = ctx.as_mut() as *mut vpx_codec_ctx_t;
        // ctx_ptr is stable inside Box and cfg is null per libvpx defaults.
        let result = unsafe {
            vpx_codec_dec_init_ver(
                ctx_ptr,
                vpx_codec_vp8_dx(),
                std::ptr::null(),
                0,
                VPX_DECODER_ABI_VERSION as c_int,
            )
        };
        if result != vpx_codec_err_t::VPX_CODEC_OK {
            return Err(DecodeError::Vp8Decode(format!(
                "libvpx VP8 decoder initialisation failed: {result:?}"
            )));
        }
        Ok(Self { ctx })
    }

    fn as_mut_ptr(&mut self) -> *mut vpx_codec_ctx_t {
        self.ctx.as_mut() as *mut vpx_codec_ctx_t
    }
}

impl Drop for Vp8DecoderCtx {
    fn drop(&mut self) {
        // ctx was initialised by vpx_codec_dec_init_ver and is destroyed once here.
        unsafe {
            vpx_codec_destroy(self.as_mut_ptr());
        }
    }
}

pub(super) fn decode(
    ctx: &mut Vp8DecoderCtx,
    yuv_pools: &mut YuvPlanePools,
    payload: &[u8],
) -> Result<YuvFrame, DecodeError> {
    let ctx_ptr = ctx.as_mut_ptr();
    let payload_len = c_uint::try_from(payload.len())
        .map_err(|_| DecodeError::Vp8Decode("VP8 payload too large".to_string()))?;
    // ctx_ptr is a live decoder and payload points to immutable bytes for this call.
    let result = unsafe {
        vpx_codec_decode(
            ctx_ptr,
            payload.as_ptr(),
            payload_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != vpx_codec_err_t::VPX_CODEC_OK {
        return Err(DecodeError::Vp8Decode(vpx_error_message(ctx_ptr)));
    }

    let mut iter: vpx_codec_iter_t = std::ptr::null();
    let mut output: *mut vpx_image_t = std::ptr::null_mut();
    loop {
        // iter is owned by libvpx for this drain and ctx remains live.
        let img = unsafe { vpx_codec_get_frame(ctx_ptr, &mut iter) };
        if img.is_null() {
            break;
        }
        if output.is_null() {
            output = img;
        } else {
            return Err(DecodeError::Vp8Decode(
                "multiple frames returned for one VP8 access unit".to_string(),
            ));
        }
    }

    if output.is_null() {
        return Err(DecodeError::Buffered);
    }

    // output points to decoder-owned image memory valid until the next decode call.
    let img = unsafe { &*output };
    if img.fmt != vpx_img_fmt_t::VPX_IMG_FMT_I420 {
        return Err(DecodeError::Vp8Decode(format!(
            "unsupported libvpx image format {:?}",
            img.fmt
        )));
    }

    let width = img.d_w as usize;
    let height = img.d_h as usize;
    if width % 2 != 0 || height % 2 != 0 {
        return Err(DecodeError::Vp8Decode(
            "odd VP8 frame dimensions unsupported".to_string(),
        ));
    }

    // Match the pool to the frame's true resolution, which the WebRTC path
    // only learns once the first frame decodes (see `ensure_dims` for the
    // first-frame-then-grow policy).
    yuv_pools.ensure_dims(img.d_w, img.d_h);

    // De-stride each libvpx plane straight into a pooled buffer. copy_vpx_plane
    // clears its destination first, so a freshly acquired buffer drops in and
    // the packed plane is written once rather than staged in scratch and copied.
    let mut y = yuv_pools.y.acquire();
    copy_vpx_plane(&mut y, img.planes[0], img.stride[0], width, height)?;

    let uv_width = width / 2;
    let uv_height = height / 2;
    let mut u = yuv_pools.u.acquire();
    copy_vpx_plane(&mut u, img.planes[1], img.stride[1], uv_width, uv_height)?;
    let mut v = yuv_pools.v.acquire();
    copy_vpx_plane(&mut v, img.planes[2], img.stride[2], uv_width, uv_height)?;

    Ok(YuvFrame {
        y: yuv_pools.y.recycle(y),
        u: yuv_pools.u.recycle(u),
        v: yuv_pools.v.recycle(v),
        width: img.d_w,
        height: img.d_h,
    })
}

fn copy_vpx_plane(
    dst: &mut Vec<u8>,
    src: *const u8,
    stride: c_int,
    width: usize,
    height: usize,
) -> Result<(), DecodeError> {
    let stride = usize::try_from(stride)
        .map_err(|_| DecodeError::Vp8Decode("invalid libvpx plane layout".to_string()))?;
    if src.is_null() || stride < width {
        return Err(DecodeError::Vp8Decode(
            "invalid libvpx plane layout".to_string(),
        ));
    }

    dst.clear();
    dst.reserve(width * height);
    for row in 0..height {
        // src is a top-down libvpx plane with at least width active bytes in this row.
        let row_slice = unsafe { std::slice::from_raw_parts(src.add(row * stride), width) };
        dst.extend_from_slice(row_slice);
    }
    Ok(())
}

fn vpx_error_message(ctx: *mut vpx_codec_ctx_t) -> String {
    // ctx is a live decoder and libvpx returns null or static C strings.
    let base = unsafe { c_string_or_default(vpx_codec_error(ctx), "unknown libvpx error") };
    // ctx is a live decoder and detail is optional.
    let detail = unsafe { c_string_or_default(vpx_codec_error_detail(ctx), "") };
    if detail.is_empty() {
        base
    } else {
        format!("{base}: {detail}")
    }
}

unsafe fn c_string_or_default(ptr: *const std::os::raw::c_char, default: &str) -> String {
    if ptr.is_null() {
        return default.to_string();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

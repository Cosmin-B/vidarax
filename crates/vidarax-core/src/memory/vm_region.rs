use std::fmt::{Display, Formatter};
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmError {
    InvalidSize,
    PageSizeUnavailable,
    ReserveFailed(i32),
    ReleaseFailed(i32),
}

impl Display for VmError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::InvalidSize => write!(f, "invalid virtual memory reservation size"),
            VmError::PageSizeUnavailable => write!(f, "unable to determine system page size"),
            VmError::ReserveFailed(code) => write!(f, "mmap failed with errno={code}"),
            VmError::ReleaseFailed(code) => write!(f, "munmap failed with errno={code}"),
        }
    }
}

impl std::error::Error for VmError {}

pub struct VmRegion {
    ptr: NonNull<u8>,
    len: usize,
    page_size: usize,
}

// Safety: VmRegion is an owned mmap region. It does not enforce aliasing rules
// by itself, so higher-level allocators must guarantee safe concurrent access.
unsafe impl Send for VmRegion {}
// Safety: sharing VmRegion only shares region metadata and base pointer.
// Mutable access must be synchronized by callers.
unsafe impl Sync for VmRegion {}

impl VmRegion {
    pub fn reserve_and_commit(bytes: usize) -> Result<Self, VmError> {
        let page_size = page_size()?;
        if bytes == 0 {
            return Err(VmError::InvalidSize);
        }
        let len = align_up(bytes, page_size).ok_or(VmError::InvalidSize)?;

        #[cfg(unix)]
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(VmError::ReserveFailed(errno()));
            }
            Ok(Self {
                ptr: NonNull::new_unchecked(ptr.cast::<u8>()),
                len,
                page_size,
            })
        }

        #[cfg(not(unix))]
        {
            let _ = len;
            let _ = page_size;
            unimplemented!("VmRegion currently supports unix targets only");
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // Safe: region pointer/length are valid for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safe: mutable borrow guarantees exclusive access to region bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for VmRegion {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            if libc::munmap(self.ptr.as_ptr().cast(), self.len) != 0 {
                let _ = VmError::ReleaseFailed(errno());
            }
        }
    }
}

#[inline]
fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

#[cfg(unix)]
#[inline]
fn errno() -> i32 {
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        *libc::__errno_location()
    }
}

fn page_size() -> Result<usize, VmError> {
    #[cfg(unix)]
    unsafe {
        let v = libc::sysconf(libc::_SC_PAGESIZE);
        if v <= 0 {
            return Err(VmError::PageSizeUnavailable);
        }
        Ok(v as usize)
    }

    #[cfg(not(unix))]
    {
        Err(VmError::PageSizeUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::VmRegion;

    #[test]
    fn reserve_and_write() {
        let mut region = VmRegion::reserve_and_commit(4096).expect("region");
        assert!(region.len() >= 4096);
        region.as_mut_slice()[0] = 7;
        assert_eq!(region.as_slice()[0], 7);
    }
}

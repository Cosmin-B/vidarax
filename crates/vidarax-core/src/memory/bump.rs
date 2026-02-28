use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::memory::vm_region::{VmError, VmRegion};

pub struct VmBumpArena {
    region: VmRegion,
    head: AtomicUsize,
}

impl VmBumpArena {
    pub fn new(bytes: usize) -> Result<Self, VmError> {
        Ok(Self {
            region: VmRegion::reserve_and_commit(bytes)?,
            head: AtomicUsize::new(0),
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.region.len()
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.head.load(Ordering::Acquire)
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.used())
    }

    /// Resets allocation offset. Existing pointers become invalid immediately.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that all pointers/references previously allocated
    /// from this arena are no longer used after reset, and that any required
    /// destructor logic has already run.
    pub unsafe fn reset(&mut self) {
        *self.head.get_mut() = 0;
    }

    pub fn alloc_bytes(&self, bytes: usize, align: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(bytes, align).ok()?;
        self.alloc_layout(layout)
    }

    pub fn alloc_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        let capacity = self.capacity();
        loop {
            let current = self.head.load(Ordering::Relaxed);
            let aligned = align_up(current, layout.align())?;
            let end = aligned.checked_add(layout.size())?;
            if end > capacity {
                return None;
            }

            if self
                .head
                .compare_exchange_weak(current, end, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let ptr = unsafe { self.region.as_ptr().add(aligned) };
                return NonNull::new(ptr);
            }
        }
    }

    pub fn alloc_value<T>(&self, value: T) -> Option<NonNull<T>> {
        let ptr = self.alloc_layout(Layout::new::<T>())?;
        let typed = ptr.cast::<T>();
        // Safe: memory region is committed and aligned for T.
        unsafe {
            typed.as_ptr().write(value);
        }
        Some(typed)
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

#[cfg(test)]
mod tests {
    use super::VmBumpArena;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn allocates_and_tracks_usage() {
        let arena = VmBumpArena::new(4096).expect("arena");
        let a = arena.alloc_bytes(64, 16).expect("a");
        let b = arena.alloc_bytes(128, 32).expect("b");
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert!(arena.used() >= 192);
    }

    #[test]
    fn supports_lock_free_parallel_allocations() {
        let arena = Arc::new(VmBumpArena::new(1 << 20).expect("arena"));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let a = Arc::clone(&arena);
            handles.push(thread::spawn(move || {
                let mut ok = 0usize;
                for _ in 0..1024 {
                    if a.alloc_bytes(64, 16).is_some() {
                        ok += 1;
                    }
                }
                ok
            }));
        }

        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 4096);
    }
}

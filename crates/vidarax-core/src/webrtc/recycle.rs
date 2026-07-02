use std::ops::Deref;

/// Small fixed free-list for byte buffers that cross worker boundaries.
///
/// The free-list uses non-blocking `kanal` operations only. When the list is
/// empty, callers get a transient `Vec`; when it is full on return, the buffer
/// is dropped. This gives bounded reuse without a hand-rolled lock-free stack.
#[derive(Debug, Clone)]
pub struct VecPool {
    free_tx: kanal::Sender<Vec<u8>>,
    free_rx: kanal::Receiver<Vec<u8>>,
}

impl VecPool {
    pub fn with_slots(slots: usize) -> Self {
        Self::with_capacity(slots, 0)
    }

    pub fn with_capacity(slots: usize, vec_capacity: usize) -> Self {
        let slots = slots.max(1);
        let (free_tx, free_rx) = kanal::bounded(slots);
        for _ in 0..slots {
            let _ = free_tx.try_send(Vec::with_capacity(vec_capacity));
        }
        Self { free_tx, free_rx }
    }

    pub fn acquire(&self) -> Vec<u8> {
        let mut bytes = self.free_rx.try_recv().ok().flatten().unwrap_or_default();
        bytes.clear();
        bytes
    }

    pub fn recycle(&self, bytes: Vec<u8>) -> RecycledBytes {
        RecycledBytes {
            bytes,
            free_tx: Some(self.free_tx.clone()),
        }
    }

    pub fn copy_from_slice(&self, data: &[u8]) -> RecycledBytes {
        let mut bytes = self.acquire();
        bytes.clear();
        bytes.reserve(data.len());
        bytes.extend_from_slice(data);
        self.recycle(bytes)
    }
}

pub struct RecycledBytes {
    bytes: Vec<u8>,
    free_tx: Option<kanal::Sender<Vec<u8>>>,
}

impl RecycledBytes {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Clone for RecycledBytes {
    fn clone(&self) -> Self {
        self.bytes.clone().into()
    }
}

impl Default for RecycledBytes {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl From<Vec<u8>> for RecycledBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            free_tx: None,
        }
    }
}

impl<const N: usize> From<[u8; N]> for RecycledBytes {
    fn from(bytes: [u8; N]) -> Self {
        Vec::from(bytes).into()
    }
}

impl AsRef<[u8]> for RecycledBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Deref for RecycledBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl std::fmt::Debug for RecycledBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecycledBytes")
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for RecycledBytes {
    fn drop(&mut self) {
        let Some(free_tx) = &self.free_tx else {
            return;
        };
        let mut bytes = std::mem::take(&mut self.bytes);
        bytes.clear();
        let _ = free_tx.try_send(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::VecPool;

    #[test]
    fn reusable_vec_pool_round_trips_backing_allocation() {
        let pool = VecPool::with_capacity(1, 64);

        let mut first = pool.acquire();
        first.extend_from_slice(b"frame-a");
        let first_addr = first.as_ptr().addr();
        let first_capacity = first.capacity();
        let bytes = pool.recycle(first);
        assert_eq!(&bytes[..], b"frame-a");

        drop(bytes);

        let second = pool.acquire();
        assert_eq!(second.capacity(), first_capacity);
        assert_eq!(second.as_ptr().addr(), first_addr);
    }
}

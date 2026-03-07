use std::array;
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Slot<T> {
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// Slots are concurrently accessed by exactly one producer and one consumer.
unsafe impl<T: Send> Sync for Slot<T> {}

/// Cache-line-separated ring buffer control block.
///
/// `head` (read by consumer, written by producer) and `tail` (written by
/// producer, read by consumer) each occupy their own 64-byte cache line so
/// that the two threads never invalidate each other's L1 cache entries
/// (false sharing / cache-line ping-pong).
#[repr(C)]
struct Inner<T, const N: usize> {
    head: AtomicUsize,
    _pad0: [u8; 56], // pad head to its own 64-byte cache line
    tail: AtomicUsize,
    _pad1: [u8; 56], // pad tail to its own 64-byte cache line
    slots: [Slot<T>; N],
}

impl<T, const N: usize> Inner<T, N> {
    fn new() -> Self {
        assert!(N > 0, "SPSC channel capacity must be > 0");
        assert!(N.is_power_of_two(), "SPSC channel capacity must be a power of two");
        Self {
            head: AtomicUsize::new(0),
            _pad0: [0u8; 56],
            tail: AtomicUsize::new(0),
            _pad1: [0u8; 56],
            slots: array::from_fn(|_| Slot::new()),
        }
    }

    #[inline]
    fn index(pos: usize) -> usize {
        pos & (N - 1)
    }
}

impl<T, const N: usize> Drop for Inner<T, N> {
    fn drop(&mut self) {
        let mut head = *self.head.get_mut();
        let tail = *self.tail.get_mut();
        while head != tail {
            let idx = Self::index(head);
            // Safe: only the drop path can access these remaining slots.
            unsafe {
                (*self.slots[idx].value.get()).assume_init_drop();
            }
            head = head.wrapping_add(1);
        }
    }
}

pub struct Producer<T, const N: usize> {
    inner: Arc<Inner<T, N>>,
    _not_sync: PhantomData<Cell<()>>,
}

pub struct Consumer<T, const N: usize> {
    inner: Arc<Inner<T, N>>,
    _not_sync: PhantomData<Cell<()>>,
}

pub fn spsc_channel<T, const N: usize>() -> (Producer<T, N>, Consumer<T, N>) {
    let inner = Arc::new(Inner::<T, N>::new());
    (
        Producer {
            inner: Arc::clone(&inner),
            _not_sync: PhantomData,
        },
        Consumer {
            inner,
            _not_sync: PhantomData,
        },
    )
}

impl<T, const N: usize> Producer<T, N> {
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.inner.head.load(Ordering::Acquire);
        let tail = self.inner.tail.load(Ordering::Relaxed);
        tail.wrapping_sub(head)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == N
    }

    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.inner.head.load(Ordering::Acquire);
        let tail = self.inner.tail.load(Ordering::Relaxed);
        if tail.wrapping_sub(head) == N {
            return Err(value);
        }

        let idx = Inner::<T, N>::index(tail);
        // Safe: producer is the only writer to tail slot.
        unsafe {
            (*self.inner.slots[idx].value.get()).write(value);
        }
        self.inner
            .tail
            .store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl<T, const N: usize> Consumer<T, N> {
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pop(&self) -> Option<T> {
        let tail = self.inner.tail.load(Ordering::Acquire);
        let head = self.inner.head.load(Ordering::Relaxed);
        if tail == head {
            return None;
        }

        let idx = Inner::<T, N>::index(head);
        // Safe: consumer is the only reader from head slot after visibility.
        let value = unsafe { (*self.inner.slots[idx].value.get()).assume_init_read() };
        self.inner
            .head
            .store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::spsc_channel;
    use std::thread;

    #[test]
    fn push_pop_roundtrip() {
        let (producer, consumer) = spsc_channel::<u64, 4>();
        assert_eq!(consumer.pop(), None);
        assert!(producer.push(10).is_ok());
        assert!(producer.push(11).is_ok());
        assert_eq!(consumer.pop(), Some(10));
        assert_eq!(consumer.pop(), Some(11));
        assert_eq!(consumer.pop(), None);
    }

    #[test]
    fn full_queue_rejects() {
        let (producer, consumer) = spsc_channel::<u64, 2>();
        assert!(producer.push(1).is_ok());
        assert!(producer.push(2).is_ok());
        assert_eq!(producer.push(3), Err(3));
        assert_eq!(consumer.pop(), Some(1));
        assert!(producer.push(3).is_ok());
    }

    #[test]
    fn cross_thread_spsc() {
        let (producer, consumer) = spsc_channel::<u64, 64>();
        let prod_handle = thread::spawn(move || {
            for n in 0..10_000_u64 {
                loop {
                    if producer.push(n).is_ok() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        let cons_handle = thread::spawn(move || {
            let mut expected = 0_u64;
            while expected < 10_000 {
                match consumer.pop() {
                    Some(n) => {
                        assert_eq!(n, expected);
                        expected += 1;
                    }
                    None => std::hint::spin_loop(),
                }
            }
        });

        prod_handle.join().unwrap();
        cons_handle.join().unwrap();
    }
}

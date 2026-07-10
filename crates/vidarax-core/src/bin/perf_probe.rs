use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use vidarax_core::gate::{FrameSignal, GateConfig, GateEngine, GateEventType};

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() {
    let gate_stats = bench_gate_path(120_000);
    let alloc_stats = allocation_probe(60_000);

    println!(
        "{{\"gate_process\":{{\"p50_ns\":{},\"p95_ns\":{}}},\"allocations\":{{\"total\":{},\"per_frame\":{:.6}}}}}",
        gate_stats.p50_ns,
        gate_stats.p95_ns,
        alloc_stats.total,
        alloc_stats.per_frame
    );
}

struct GateStats {
    p50_ns: u64,
    p95_ns: u64,
}

struct AllocStats {
    total: u64,
    per_frame: f64,
}

fn bench_gate_path(samples: usize) -> GateStats {
    let mut gate = GateEngine::new(GateConfig::default());
    let mut durations = Vec::with_capacity(samples);

    for i in 0..samples {
        let signal = FrameSignal {
            frame_index: i as u64,
            pts_ms: (i as u64) * 33,
            perceptual_hash: ((i as u64) << 7) ^ 0xA5A5_A5A5_A5A5_A5A5,
            luma_mean: ((i % 100) as f32) / 100.0,
            flicker_score: if i % 120 == 0 { 0.7 } else { 0.0 },
            ghosting_score: if i % 160 == 0 { 0.8 } else { 0.0 },
            noise_variance_score: if i % 220 == 0 { 0.9 } else { 0.0 },
        };
        let start = Instant::now();
        let event = gate.process(signal);
        let elapsed = start.elapsed().as_nanos() as u64;
        // Commit kept keyframes so the reference frame advances like a real
        // stream. process() now only classifies; without this the gate would
        // never gain a reference and every sample would hit the initial-frame
        // branch instead of the steady-state comparison path we mean to time.
        if event.event_type == GateEventType::KeepKeyframe {
            gate.commit_keyframe(signal);
        }
        durations.push(elapsed);
    }

    durations.sort_unstable();
    GateStats {
        p50_ns: percentile(&durations, 50),
        p95_ns: percentile(&durations, 95),
    }
}

fn allocation_probe(frames: usize) -> AllocStats {
    let mut gate = GateEngine::new(GateConfig::default());
    let before = ALLOCATIONS.load(Ordering::Relaxed);

    for i in 0..frames {
        let signal = FrameSignal {
            frame_index: i as u64,
            pts_ms: (i as u64) * 33,
            perceptual_hash: i as u64,
            luma_mean: 0.5,
            flicker_score: 0.0,
            ghosting_score: 0.0,
            noise_variance_score: 0.0,
        };
        let event = gate.process(signal);
        if event.event_type == GateEventType::KeepKeyframe {
            gate.commit_keyframe(signal);
        }
    }

    let total = ALLOCATIONS.load(Ordering::Relaxed).saturating_sub(before);
    AllocStats {
        total,
        per_frame: total as f64 / frames as f64,
    }
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    let n = sorted.len();
    let idx = ((n.saturating_sub(1)) * p) / 100;
    sorted[idx]
}

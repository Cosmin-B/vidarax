---
title: Allocation discipline
description: The counting allocator in the perf probe, the pointer-identity tests that pin pool reuse, and the release-gate scripts as mechanism.
---

The hot paths in vidarax are written to a stated policy: no per-frame heap allocation in the core ingest and gate loops, pre-allocated pools everywhere buffers cross threads (see [Development](/docs/development/#contributing-basics) for the policy text). The repository enforces the policy three ways: a global counting allocator in a probe binary that measures allocation events per frame, unit tests that assert pool reuse by pointer identity rather than by effect, and release-gate scripts that turn the allocation probe into a pass-or-fail check. The pointer-identity tests run in the ordinary workspace test suite; the release-gate scripts do not run them. This page describes each mechanism.

## The counting allocator

`crates/vidarax-core/src/bin/perf_probe.rs` (built only with the `perf-probe` feature) installs a `#[global_allocator]` that counts allocation events in a process-wide atomic:

```rust
struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    // alloc_zeroed and realloc increment the same counter;
    // dealloc does not.
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;
```

Every `alloc`, `alloc_zeroed`, and `realloc` anywhere in the process increments the counter and delegates to the system allocator; `dealloc` is passed through uncounted, so the number is a count of allocation events, not a live-bytes gauge. The probe's `main` runs two workloads against a default-config `GateEngine`: a timing pass over `gate.process` calls, and an allocation pass that reads the counter before and after the loop and divides the delta by the frame count. It prints one JSON object carrying the timing summary plus `allocations.total` and `allocations.per_frame`, which is the interface the gate scripts consume with `jq`.

The design constraint worth knowing before modifying it: because the allocator is global, anything else the probe process does (logging, JSON formatting) also counts, so the measured region is kept to the bare gate loop with the counter snapshot taken immediately around it, and the workload constructs each `FrameSignal` on the stack.

## Pointer-identity tests

Asserting "the pool was reused" by observing behavior is weak; the tests instead assert identity of the backing memory. The canonical one is in `crates/vidarax-core/src/webrtc/recycle.rs`:

```rust
let mut first = pool.acquire();
first.extend_from_slice(b"frame-a");
let first_addr = first.as_ptr().addr();
let first_capacity = first.capacity();
let bytes = pool.recycle(first);
drop(bytes);

let second = pool.acquire();
assert_eq!(second.capacity(), first_capacity);
assert_eq!(second.as_ptr().addr(), first_addr);
```

`reusable_vec_pool_round_trips_backing_allocation` proves the round trip through `VecPool::acquire`, `recycle`, and `RecycledBytes::drop` hands back the same heap allocation, address and capacity intact. Any regression in the free-list plumbing (a hidden clone, a clear that frees, a drop path that stops returning buffers) breaks the address equality even if behavior still looks correct.

The same technique pins reuse at other layers:

- `yuv_to_jpeg_emits_valid_jpeg_and_keeps_pool_buffer_reserved` in `webrtc/signals.rs` asserts that a JPEG buffer returned to the pool keeps its lazy capacity reservation, so the next frame's encode does not climb through doubling reallocations from zero.
- `yuv_plane_pools_ensure_dims_rebuilds_on_resolution_change` in `webrtc/decode.rs` drains a plane pool's free-list and proves that a later acquire can only be pool-served after a deliberate rebuild, pinning the first-frame-then-grow policy.
- In `crates/vidarax-api/src/state.rs`, `registry_keeps_same_map_snapshot_for_existing_run_event` asserts with `Arc::ptr_eq` that a non-structural run event updates per-run atomics in place without replacing the registry entry, and `publishing_append_reuses_unchanged_run_tail_arc` asserts that publishing a timeline snapshot shares untouched runs' tails by `Arc` instead of cloning them.

The complementary sizing tests, such as the one deriving the 484-slot JPEG pool bound, are described in [Media plane](/docs/internals/media-plane/#pool-sizing-as-a-sum-over-in-flight-positions).

## The release-gate scripts

Three scripts under `scripts/` turn the probe and the replay tests into gates. Each check compares an observed value against a ceiling supplied by an environment variable, so thresholds are tightened or relaxed without editing the scripts, and any breach exits non-zero. The pointer-identity tests above are not part of these scripts; they run with `cargo test --workspace`.

| Script | What it checks |
|---|---|
| `validate_replay_and_schema.sh` | Runs the `replay_schema` integration test: gate decisions replay deterministically to a pinned fingerprint, and the published JSON Schemas accept their reference fixtures and reject invalid ones. See [WAL and events](/docs/internals/wal-and-events/#validation-replay-and-schema-gates). |
| `bench_regression.sh` | Builds and runs `perf_probe` in release mode and compares `allocations.per_frame` against `VIDARAX_MAX_ALLOC_PER_FRAME`. This is the executable form of the no-per-frame-allocation policy: a change that adds a heap allocation inside `gate.process` fails this script. |
| `release_gates.sh` | Runs both of the above, then builds release binaries for `vidarax-cli` and `vidarax-api` and compares their file sizes against `VIDARAX_MAX_CLI_SIZE_BYTES` and `VIDARAX_MAX_API_SIZE_BYTES`, then runs the probe's timing regression gate against its configured ceiling. |

Conceptually the ceilings guard four different regressions: semantic drift in the gate (replay fingerprint), allocation creep on the per-frame path (allocations per frame), dependency and code-size creep in the shipped artifacts (binary sizes), and timing regressions in the gate decision itself. The operational procedure around running them is in [Operations](/docs/operations/#release-gates).

## Edge cases and limits

- The allocator counts events, not bytes or lifetimes; a change that replaces many small allocations with one huge one improves the metric while possibly worsening memory behavior. The pool sizing tests and the byte-capacity caps cover that axis separately.
- The probe exercises the gate loop only. Allocation discipline in the decode, JPEG, and channel paths is enforced by the pointer-identity and pool-sizing tests, not by the probe.
- `Vec::as_ptr().addr()` equality can in principle hold accidentally if an allocator reuses a freed address; the recycle test avoids that by keeping the pool slot occupied through the round trip, so the buffer is never returned to the system allocator at all.
- The binary-size gates measure the uncompressed on-disk release binaries; they exist to catch accidental dependency growth, not to bound deployment artifacts, which may be stripped or compressed differently.
- `perf_probe` must be built in release mode for either gate to mean anything; the scripts do this themselves (`cargo run --release --features perf-probe`), so run them through the scripts rather than invoking the binary ad hoc.

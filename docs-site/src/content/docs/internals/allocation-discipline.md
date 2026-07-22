---
title: Allocation discipline
description: The counting allocator in the perf probe, pointer-identity tests for pool reuse, and executable release checks.
---

The hot paths in Vidarax follow a stated policy: no selected-global-allocator calls per frame after warmup in the core ingest and filter loops, with reusable pools wherever buffers cross threads (see [Development](/docs/development/#contributing-basics)). The repository checks the policy three ways: a counting allocator in a probe binary, unit tests that assert pool reuse by pointer identity, and release scripts that turn the probe into a pass-or-fail check. Pointer-identity tests run in the ordinary workspace test suite. The release scripts do not run them.

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

Every `alloc`, `alloc_zeroed`, and `realloc` routed through this Rust global allocator increments the counter and delegates to the system allocator. `dealloc` passes through uncounted. The value is an allocation-event count, not a live-bytes gauge. It does not observe stack storage, direct OS mappings, allocations inside foreign libraries that bypass Rust's selected allocator, or storage managed by a separate arena. The probe runs a timing pass and an allocation pass around `GateEngine::process`, then prints `allocations.total` and `allocations.per_frame`. The release script checks the integer total, so one allocation cannot disappear through decimal formatting.

The design constraint worth knowing before modifying it: because the allocator is global, anything else the probe process does (logging, JSON formatting) also counts, so the measured region is kept to the bare gate loop with the counter snapshot taken immediately around it, and the workload constructs each `FrameSignal` on the stack.

## Pointer-identity tests

Asserting "the pool was reused" by observing behavior is weak. The tests instead assert identity of the backing memory. The canonical one is in `crates/vidarax-core/src/webrtc/recycle.rs`:

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
- In `crates/vidarax-api/src/state.rs`, `registry_keeps_same_map_snapshot_for_existing_run_event` asserts with `Arc::ptr_eq` that a non-structural run event updates per-run atomics in place without replacing the registry entry. `publishing_append_reuses_unchanged_run_tail_arc` asserts that timeline publication shares untouched runs' tails through `Arc` and never clones them.

The complementary sizing tests, such as the one deriving the 484-slot JPEG pool bound, are described in [Media plane](/docs/internals/media-plane/#pool-sizing-as-a-sum-over-in-flight-positions).

## The release-check scripts

Three scripts under `scripts/` turn the probe and replay tests into release checks. Each check compares an observed value against a ceiling supplied by an environment variable, so thresholds can change without editing the scripts, and any breach exits non-zero. The pointer-identity tests above are not part of these scripts. They run with `cargo test --workspace`.

| Script | What it checks |
|---|---|
| `validate_replay_and_schema.sh` | Runs the `replay_schema` integration test: gate decisions replay deterministically to a pinned fingerprint, and the published JSON Schemas accept their reference fixtures and reject invalid ones. See [WAL and events](/docs/internals/wal-and-events/#validation-replay-and-schema-gates). |
| `bench_regression.sh` | Builds and runs `perf_probe` in release mode and compares integer `allocations.total` against `VIDARAX_MAX_ALLOC_TOTAL` (default 0). A first counted allocation inside the measured loop fails the script. |
| `release_gates.sh` | Runs both of the above, then builds release binaries for `vidarax-cli` and `vidarax-api` and compares their file sizes against `VIDARAX_MAX_CLI_SIZE_BYTES` and `VIDARAX_MAX_API_SIZE_BYTES`, then runs the probe's timing regression gate against its configured ceiling. |

The ceilings guard four different regressions: semantic drift in the filter (replay fingerprint), counted allocation creep on the per-frame path, dependency and code-size creep in shipped artifacts, and timing regressions in the filter decision. The operational procedure is in [Operations](/docs/operations/#release-checks).

## Edge cases and limits

- The allocator counts selected-allocator events, not bytes or lifetimes. One huge allocation scores better than many small ones even if memory behavior worsens. Pool sizing tests and byte-capacity caps cover that axis separately.
- The probe exercises the gate loop only. Allocation discipline in the decode, JPEG, and channel paths is enforced by the pointer-identity and pool-sizing tests, not by the probe.
- `Vec::as_ptr().addr()` equality can in principle hold accidentally if an allocator reuses a freed address. The recycle test avoids that by keeping the pool slot occupied through the round trip, so the buffer is never returned to the system allocator at all.
- The binary-size gates measure the uncompressed on-disk release binaries. They exist to catch accidental dependency growth, not to bound deployment artifacts, which may be stripped or compressed differently.
- `perf_probe` must be built in release mode for either check to be meaningful. The scripts run `cargo run --release --features perf-probe` with the required configuration.

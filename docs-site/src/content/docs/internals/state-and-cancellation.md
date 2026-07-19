---
title: State and cancellation
description: AppState layout, RAII slot reservation, the single-winner delete protocol, and how request cancellation is kept away from session ownership.
---

`AppState` in `crates/vidarax-api/src/state.rs` is the shared state behind every handler: the run registry, the timeline writer, session maps, and the concurrency guards around them. Its design goal is that every invariant survives concurrent requests racing each other and futures being cancelled mid-flight. The mechanisms are RAII guards whose `Drop` runs on normal returns, cancellation, and unwinding panics; compare-and-swap claims that admit exactly one winner; and lock-free snapshots readers can load without blocking writers. Release builds use `panic = "abort"`, so an abort does not run destructors. This page covers the layout, the `StreamSlotGuard` reservation, the insert-to-spawn window in `whip.rs`, single-winner deletion, and the snapshot registry. Event persistence itself is in [WAL and events](/docs/internals/wal-and-events/).

## AppState layout

`AppState` is a `Clone` wrapper around `Arc<AppStateInner>` with a `Deref` impl, so handlers clone a pointer, never state. The fields of `AppStateInner` group as:

| Group | Fields | Concurrency mechanism |
|---|---|---|
| Counters | `run_seq`, `request_seq` | `AtomicU64`, `fetch_add` |
| Durable log | `wal_path`, `timeline_tx` | Single writer thread behind a bounded `mpsc::channel` (`TIMELINE_WRITER_QUEUE_CAP`, 1024) |
| Read acceleration | `timeline_snapshot: Arc<ArcSwap<RingSnapshot>>` | Copy-on-write snapshot, swapped whole by the writer |
| Run registry | `run_registry: Arc<RunRegistry>` | `DashMap` shards plus per-run atomics |
| Stream limits | `stream_reservations: DashMap<String, usize>`, `active_stream_limit`, `stream_ttl_secs` | Shard-entry critical sections |
| WebRTC sessions | `sessions: DashMap<String, SessionEntry>`, `session_slots: AtomicUsize`, `media_budget`, `media_reservations`, `reclaimed_sessions: Mutex<ReclaimedSessions>` | Exact session admission plus atomic byte/thread reservations; shard entries; short synchronous tombstone-map critical sections |
| Plumbing | `provider`, `decode_pipeline`, `security_policy`, metrics, `spacetime_client`, `webrtc_config`, `tenant_label_maps`, `ingest_file_roots`, `novelty_config` | Immutable after construction |

Run state is not a mutable row. `RunState` holds only atomics (`created: AtomicBool`, `delete_state: AtomicU8`, `state: AtomicU8` encoding `StreamState`, `last_activity_ms: AtomicU64`, plus an `ArcSwap` principal key and a `Notify`), so once a run exists, ordinary events nudge those atomics in place; a test pins that the registry entry's `Arc` pointer is unchanged across a non-structural event. `RunRegistry` shards runs and a `by_principal` index across two `DashMap`s, and `apply_structural_event` is careful to never hold a lock in one map while touching the other, taking them in the same order as the reader `count_active_runs_for_principal`, so the pair cannot deadlock.

## Stream slot reservation: StreamSlotGuard

Run creation must respect a per-principal active stream limit, but the check (count active runs) and the commit (append `run_created`) are separated by awaits. `try_reserve_stream_slot` closes the race by holding the principal's `DashMap` shard entry across the read and the increment:

```rust
let entry = self.stream_reservations.entry(principal_key.to_string());
let reserved = match &entry {
    Entry::Occupied(slot) => *slot.get(),
    Entry::Vacant(_) => 0,
};
let committed = self.count_active_runs_for_principal(principal_key, now_ms);
if committed.saturating_add(reserved) >= self.active_stream_limit {
    return None;
}
```

Two creators for the same principal cannot both pass the same snapshot, because the second blocks on the shard entry until the first has incremented. The committed count lives in the run registry, a different map, so reading it under this entry is safe. On success the caller gets a `#[must_use]` `StreamSlotGuard` whose `Drop` releases the slot under the same shard-entry discipline:

```rust
impl Drop for StreamSlotGuard {
    fn drop(&mut self) {
        if let Entry::Occupied(mut slot) = self.reservations.entry(self.principal_key.clone()) {
            if *slot.get() <= 1 {
                slot.remove();
            } else {
                *slot.get_mut() -= 1;
            }
        }
    }
}
```

The entry is held across the check and the remove, so a concurrent reservation for the same principal cannot race the decrement, and a count that reaches zero deletes its key, keeping the map bounded to principals that actually hold reservations. Because release is `Drop`, it runs on early returns, `?` propagation, future cancellation, and unwinding panics. It does not run after a process abort, including a panic in the release profile. The guard may briefly overlap with its own committed run (the reservation is released after `run_created` is durable and visible); that overlap can only reject a racing creator early, never admit one past the limit.

## The insert-to-spawn window in whip.rs

WHIP session creation must never leave a live session that nothing watches. `start_whip_session_transaction` orders its steps so every failure path is covered, and `start_whip_session` runs the whole transaction in a detached `tokio::spawn` so an HTTP client that disconnects mid-request cannot cancel it halfway:

1. Reserve the stream slot (`StreamSlotGuard`); on refusal, close the peer connection and return 409.
2. Build a `MediaSessionResources` envelope from the negotiated codec, configured resolution, queue payload limits, pool topology, mode, and fixed worker count. Atomically reserve both process-wide bytes and worker threads; on refusal, close and return 503 without appending `run_created`.
3. Append `run_created` durably; on failure, close and return 500. Dropping the media permit releases both capacity dimensions.
4. Reserve one `session_slots` count with an atomic compare-and-swap, then insert the session and its media permit. The reservation makes the `MAX_WEBRTC_SESSIONS` (100) cap exact under concurrent starts; an ID collision releases it immediately. On failure, the just-created run is tombstoned (`run_deleted`) with bounded inline retries and a detached retry task as backstop, then 503.
5. `spawn_session_reclaimer`, which watches the rustrtc peer state and reclaims the session on `Disconnected`, `Failed`, or `Closed`. The winning removal drops the media permit exactly once.

There is no suspension point between session insertion and reclaimer spawn. The reclaimed-session tombstone update uses a short `std::sync::Mutex` critical section, so cancellation cannot expose a visible session without its watcher. The whole startup transaction also runs in a detached `tokio::spawn`, which lets durable cleanup and response-independent work finish if an HTTP client disconnects. Everything after the reclaimer spawn (channel construction and worker startup) is safe to await because by then the session has an owner that will clean it up.

Worker startup itself is generation-owned. `spawn_pipeline` returns a
`PipelineRuntime` containing stage-tagged join handles. Partial startup raises
the shared stop signal and bounded-joins the already-created prefix. After a
successful start, the supervisor treats the first unexpected exit as a fault,
closes the peer, stops siblings, and waits out a join deadline derived from
the configured VLM pass timeouts, the backend fallback count, the admission
wait, and the novelty embedding timeout (`supervise_join_deadline_from`),
long enough for
an in-flight inference call to drain. A worker that misses that deadline is
left detached and reported as a forced shutdown, and the session's media
reservation is kept because the detached thread still holds its memory. The
peer-state reclaimer still owns the durable tombstone, so fault and DELETE
races converge on the same single-winner removal.

Reclaim itself is race-safe by construction: `remove_session_for_run` uses `DashMap::remove_if` to check run ownership and remove under one shard lock, so exactly one caller (DELETE handler or peer-state watcher) wins cleanup, and the winner writes a `ReclaimedSessions` record so a later DELETE of the same session stays idempotent instead of returning 404. That tombstone map is bounded by both a TTL (`RECLAIMED_SESSION_TTL_MS`, ten minutes) and a cap (`RECLAIMED_SESSION_MAX_ENTRIES`, 1024).

## Single-winner deletion

`DELETE /v1/runs/{id}` is soft deletion: it appends one `run_deleted` event. The protocol guarantees that concurrent deletes append that event exactly once. Per run, `delete_state: AtomicU8` moves through three states, `RUN_DELETE_LIVE` (0), `RUN_DELETE_APPEND_IN_FLIGHT` (1), `RUN_DELETE_DELETED` (2), and the claim is a compare-and-swap:

```rust
fn begin_delete_append(&self) -> RunDeleteState {
    match self.delete_state.compare_exchange(
        RUN_DELETE_LIVE,
        RUN_DELETE_APPEND_IN_FLIGHT,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => RunDeleteState::Claimed,
        Err(RUN_DELETE_DELETED) => RunDeleteState::AlreadyDeleted,
        Err(RUN_DELETE_APPEND_IN_FLIGHT) => RunDeleteState::InFlight,
        Err(_) => RunDeleteState::InFlight,
    }
}
```

`append_run_deleted_for_stream_idempotent_async` loops over the claim result:

| Claim result | Action |
|---|---|
| `Claimed` | Wrap the run in a `RunDeleteAppendGuard` and append `run_deleted`; the guard commits to `DELETED` when the append lands, or its `Drop` rolls the state back to `LIVE` if the append failed, so a failed WAL write leaves the run deletable again |
| `AlreadyDeleted` | Append nothing; synthesize a `run_deleted` event for the response so repeat DELETEs stay idempotent with no new WAL write |
| `InFlight` | Wait on the run's `Notify` until the winner commits or rolls back, then re-loop |
| `Missing` | Run unknown to the registry (long-forgotten or never created here); append `run_deleted` unconditionally |

Both commit and rollback call `notify_waiters`, so blocked concurrent deleters always make progress. The registry keeps deleted runs visible for idempotency in a FIFO capped at `DELETED_RUN_RETENTION_CAP` (4096); beyond that, the oldest deleted entries are forgotten and a re-DELETE of one takes the `Missing` path, which appends a fresh `run_deleted`, an accepted cost for long-forgotten runs. The nonblocking append API refuses `run_deleted` outright (`"run_deleted requires confirmed append"`): deletion must be confirmed, never fire-and-forget.

## The swap-published snapshot registry

Readers of recent events never take a lock. The timeline writer thread owns a map of per-run event tails (`VecDeque<TimelineEvent>` capped at `RUN_EVENT_TAIL_CAP`, 256 events) and, after each append, publishes an immutable `RingSnapshot` of all tails through `ArcSwap::store`. Publication is copy-on-write at run granularity: only the appended run's tail is rebuilt; every other run's tail is shared by `Arc` with the previous snapshot (a test pins the pointer equality). Subscribers, meaning `GET /v1/runs/{id}/events` pollers via `read_run_events_from`, do `timeline_snapshot.load()` and serve any cursor the tail still covers; a cursor older than the tail's front falls back to a WAL scan with identical ordering, so a warm hit and a cold miss are indistinguishable to the client. The tails map itself is bounded by `WARM_RUN_TAIL_CAP` (1024) runs with least-recently-appended eviction; eviction only removes the read accelerator, never durable data.

Sequence counters differ by level. `run_seq` and `request_seq` on `AppState` are atomics (`fetch_add` with `AcqRel`), and per-session frame sequence numbers in the media path are one shared `Arc<AtomicU64>` across a session's track tasks (see [WebRTC ingest](/docs/internals/webrtc-ingest/)). The WAL `seq` is not an atomic: it is ordinary state owned by the single timeline-writer thread, and its safety comes from serialization through the writer's bounded channel. That channel serialization is also what makes the WAL sink safe to call from worker OS threads without an async runtime or a lock.

## Edge cases and limits

- The active-session map and its atomic admission count are updated in a fixed order. A reservation may briefly exist before its `DashMap` entry, but it can only reject another start early; it cannot admit session 101.
- `apply_expiry` derives `Expired` at read time from `last_activity_ms` and the TTL; nothing writes an expiry event, so expiry costs no WAL traffic and reverses if a keepalive lands (state is re-derived on the next read).
- `count_active_runs_for_principal` re-checks each run's live principal against the index bucket, so a bucket whose membership drifted from the run's `ArcSwap`-swapped principal cannot inflate another principal's count.
- `synthetic_run_deleted_event` stamps the snapshot's current `max_seq` rather than allocating a new sequence number; it exists only to shape an HTTP response, never to be persisted.
- The blocking flavor of the delete wait (`wait_delete_append_blocking`) spins with `yield_now`; it exists for non-async callers such as worker threads and is bounded by the winner's append latency.

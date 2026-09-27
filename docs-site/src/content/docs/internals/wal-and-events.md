---
title: WAL and events
description: The write-ahead log format, event families and append ownership, the event sink, and replay checks.
---

The local WAL is the authoritative event store. Live-session worker events and
handler lifecycle events follow the same append path, even when SpacetimeDB is
configured. Selected JPEGs, retained MP4 windows, and generated WAV files live
in content-addressed binary directories while event JSON carries their references. The format and
writer live in `crates/vidarax-core/src/timeline.rs`, the append pipeline in
`crates/vidarax-api/src/state.rs`, and the worker bridge and blob writers in
`crates/vidarax-api/src/wal_sink.rs`.

## File format

The WAL is at `${VIDARAX_DATA_DIR}/timeline.wal` (data directory defaults to `.vidarax-data`). New events use one checksummed V2 record per line. The record contains a byte length and CRC32 over the original six-field event body:

```
V2 \t body_byte_len \t crc32_hex \t seq \t run_id \t stream_id \t pts_ms \t kind \t payload
```

The encoder escapes `\`, tab, and newline in string fields (`\\`, `\t`, `\n`) in one pass, after checking the exact escaped size. Existing six-field lines remain readable, so a legacy file can have older unchecksummed records followed by V2 records. On Unix the WAL is opened with mode `0o600`. Keyframe events record the JPEG hash, size, and file reference. JPEG files are stored under `${VIDARAX_DATA_DIR}/keyframes/blobs/`. Retained MP4 windows and generated WAV files are stored under `${VIDARAX_DATA_DIR}/media/blobs/`. Their events also record the hash, media type, byte count, and file reference.

`WalWriter::append` writes a complete V2 line and waits for the filesystem to sync it before returning. The API writer groups commands already waiting in its queue, up to 32 commands or until the batch reaches a 256 KiB target, then synchronizes once before making the events readable and replying. A large record can take the batch over the byte target and share the sync with earlier records. The writer does not wait for more commands, so a single confirmed append syncs immediately. A `run_deleted` event ends its own batch so later commands see that the run has been deleted. The `timeline.wal.lock` file prevents two writers from using the same WAL name. The writer also locks the WAL file itself, so opening a hard-link alias cannot bypass the lock.

On macOS, file and directory syncs use `F_FULLFSYNC`. Other platforms use Rust's `sync_all`. [Apple's `fsync` documentation](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html) explains why the stronger request matters for sudden power loss. These operations depend on the filesystem and device completing the requested flush. An archive or replica on another machine is needed if the local disk is destroyed.

## Format and recovery contract

The exact rules a maintainer or operator can rely on, as implemented today:

- Versioning: new lines start with `V2`. Earlier six-field lines remain accepted but have no integrity checksum. This release does not rewrite old history.
- Line size: V2 event bodies have a 64 MiB safety limit, with a small framing allowance. A larger event is rejected before any bytes are written.
- Corruption detection: a complete V2 line with a wrong length, CRC32, or invalid event body fails replay with its byte offset. Legacy lines have only structural validation, so a well-formed edit to a legacy line can go undetected.
- Incomplete tail: any unterminated final line is ignored on reads and truncated on exclusive writer open, including a short V2 marker or a plausible-looking legacy record. A complete valid V2 record can survive even if its reply is lost. Retrying after a lost reply can therefore append it again.
- Interior corruption and invalid UTF-8: replay fails instead of skipping a record. Inspect and preserve the original file before manual repair.
- Write failure: after any attempted V2 write fails, the writer is poisoned and rejects further appends until reopened and recovered. A pre-write validation rejection does not poison it.
- Local ownership: the sidecar lock file must reside with the WAL, and the WAL inode is locked too. These locks and sync operations require a local filesystem. Their behavior on network filesystems has not been tested.
- Sequence continuity: API startup requires global sequence numbers to begin at 1 and increase by exactly 1, including legacy records. Duplicate, missing, or reordered numbers stop startup before it derives run state.

## Who appends each event family

One thread, `vidarax-timeline-writer`, writes all events. It assigns each event's sequence number and wall-clock timestamp, writes the batch, and syncs the WAL. It then updates the readable byte limit before updating the run registry and cached events, so reads from disk can see the records behind the new run state. Async handlers call `AppState::append_run_event_async` and receive their reply after the sync. A request rejected before writing can reuse its sequence number. If a write or sync fails, further appends fail until recovery.

The writer does not wait for queue space: a confirmed append returns an error if the command queue or byte budget is full. Once queued, the writer finishes the command even if the caller cancels while awaiting the acknowledgement. Retried state transitions therefore need an idempotency rule. `run_deleted` has one, while ordinary telemetry events are append-only observations.

After committing the batch, the writer sends a best-effort notification through
a bounded broadcast ring. SSE and webhook consumers use it to wake up, then
read events by sequence from the WAL. A slow subscriber never blocks the writer.

`read_events_after` reads at most one fixed-size batch from disk for SSE
reconnect and webhook recovery, up to the last committed byte limit. Neither
consumer loads the whole event history
into memory. Webhook registrations and deletions are timeline events. Delivery
attempts, checkpoints, and dead-letter records use the separate
`webhook-delivery.wal`, so they cannot trigger another webhook delivery.

Handler-appended kinds, by string literal in `handlers.rs` (all through `append_run_event_async`):

| Kind | Appended when |
|---|---|
| `run_created` | `POST /v1/runs`, and WHIP session start in `whip.rs` |
| `ingest_received` | An ingest request is accepted (file, URL, or realtime attach) |
| `frames_decoded` | A decode pass finishes. Payload carries the per-frame signals |
| `marker_emitted` | The gate produces a marker (one event per marker) |
| `analysis_generated` | A deterministic analysis pass completes |
| `semantic_chunk_inferred` | A chunk finishes tiered VLM inference |
| `multimodal_moment` | Native A/V reasoning returns a timestamped audible, visual, or combined moment |
| `semantic_chunk_generated` | A chunk's semantic result is recorded |
| `semantic_fallback_activated` | The semantic path falls back (for example, no provider) |
| `inference_completed` | `POST /v1/infer` completes |
| `run_completed` | Analysis completes or a WHIP resource terminates gracefully |
| `run_failed` | A live peer or its state channel ends unexpectedly |
| `stop_requested` | `POST /v1/runs/{id}/stop` |
| `keepalive_refreshed` | `POST /v1/runs/{id}/keepalive` |
| `run_deleted` | `DELETE /v1/runs/{id}` or creation-failure tombstoning |
| `operator_feedback_submitted` | Operator feedback commits to the run timeline |

Delivery state is also durable:

| Kind | Appended when |
|---|---|
| `webhook_registered` | A signed webhook registration commits |
| `webhook_deleted` | A webhook registration is removed |

Policy lifecycle handlers append these kinds:

| Kind | Appended when |
|---|---|
| `policy_revision_created` | An immutable policy revision is created |
| `policy_deployment_requested` | Shadow, canary, or active promotion starts |
| `policy_deployment_acknowledged` | The requested promotion is accepted |
| `policy_deployment_rejected` | Generation ownership or configuration rejects promotion |
| `policy_rollback_requested` | Rollback starts |
| `policy_rollback_acknowledged` | Rollback is accepted |
| `policy_rollback_rejected` | Rollback is rejected |
| `policy_replay_evaluated` | Candidate replay completes for a revision |

Concurrent semantic workers publish `semantic_chunk_inferred` as each chunk finishes, so WAL sequence captures completion order. Consumers that reconstruct source order must sort by `chunk_index`.

Worker-emitted kinds arrive through the `EventSink` trait. The sink writes the worker's `event_type` string straight through as the WAL `kind`:

| Kind | Emitted by |
|---|---|
| `vlm` / `vlm_tiered` | Keyframe VLM worker. Tiered suffix when the second pass answered |
| `clip_vlm` / `clip_vlm_tiered` | Clip VLM worker |
| `state_transition` | VLM worker, when consecutive descriptions diverge past the word-overlap threshold |
| `loop_detected` | Frame filter or analysis worker, once per loop entry |
| `keyframe_stored` | The sink's `store_keyframe_sync`, recording keyframe metadata |
| `restricted_zone_activity_entered` | The live restricted-zone state machine enters its active state |
| `trigger.<event_type>` | A trigger program emits an assertion. The suffix is the declared event type. |

`transition_state` in `state.rs` is the authoritative map from kinds to run status: `run_created` yields `Pending`. `ingest_received`, `analysis_generated`, `inference_completed`, and `keepalive_refreshed` yield `Processing`. `run_completed`, `run_failed`, and `stop_requested` yield `Completed`, `Failed`, and `Cancelled`. Every other kind leaves the status untouched and only advances `last_activity_ms`. This is why `GET /v1/runs/{id}/state` needs no status column: status is a fold over the run's events.

## The WAL event sink

`WalEventSink` is the live-session `EventSink` in every configuration. It receives the run ID on each sink call and holds the optional SpacetimeDB mirror:

```rust
pub struct WalEventSink {
    state: AppState,
    keyframe_blob_root: PathBuf,
    spacetime_event_mirror: Option<SpacetimeClient>,
}
```

`emit_event_sync` wraps the worker fields (`session_id`, `frame_index`, `pts_ms`, `coordinate_schema`, `coordinates`, `confidence`, `description`) in JSON and calls the confirmed local append. After that succeeds, it attempts the SpacetimeDB mirror. Mirror failure is logged and does not undo local durability. A failed confirmed event or keyframe write exits its worker, so the pipeline supervisor marks the session as failed and stops its other workers. `emit_event_nonblocking` uses the detached local append and never mirrors because a network call would violate its nonblocking contract.

The writer accepts at most 1,024 queued commands and 64 MiB of request strings held while preparing, queuing, or writing commands. It counts the serialized JSON before allocating it and holds the byte reservation until the request is rejected or processed. This limit excludes the caller's JSON, the buffer used to encode WAL records, cached committed events, and replies. A detached event that exceeds the available command or byte capacity is dropped with a warning and increments the drop counter.

`store_keyframe_sync` hashes the raw JPEG, writes and synchronizes a `0o600` content-addressed blob, installs its final name without replacing a concurrent writer's blob, synchronizes the directory, and then appends `keyframe_stored` with `image_ref`, media type, byte count, SHA-256, and `vidarax.image.v1` coordinate provenance. Reuse verifies the existing bytes and hash before the event append. A crash after blob installation but before the WAL append can leave an unreferenced blob. Automatic startup reconciliation or retention-based garbage collection is not implemented yet. This path requires a local filesystem that supports hard links and directory synchronization.

The append methods handle saturation as follows:

| Path | Caller | Blocking | Full command queue or byte budget | May append `run_deleted` |
|---|---|---|---|---|
| `append_run_event_async` | tokio handlers | awaits ack after admission | returns a retryable error before admission | yes, via the idempotent claim |
| `append_run_event` | worker threads | blocks on ack after admission | returns a retryable error before admission | yes, via the idempotent claim |
| `append_run_event_nonblocking` | hot paths | no | drops event | refused with an error |

`run_deleted` is special-cased on every path: it routes through the single-winner claim described in [State and cancellation](/docs/internals/state-and-cancellation/#single-winner-deletion), so the deletion event is appended exactly once per run while the deletion claim is retained, and only through a confirmed append. The retention is bounded: deleted-run records live in a FIFO capped at 4,096 entries, and once a record is evicted, a later DELETE of the same run takes the unknown-run path and appends another `run_deleted`.

## Replay and reads

On startup, `AppState::from_wal` reads the whole file and validates global sequence continuity before rebuilding the run registry and warm per-run tails. It seeds the writer's sequence counter from the final validated event, so numbering continues where it left off. An event for an unknown run registers that run on first sight. Both warm and cold reads use the same validated event history.

`read_run_events_from` uses the in-memory snapshot when it still contains the requested events. Otherwise it scans the WAL and filters by run, using `spawn_blocking` for async calls. The scan stops at the last published byte limit even if the file contains newer bytes that have not been acknowledged. Each scan reads all events up to that limit. An index of file offsets per run would reduce this work if these scans become frequent.

For backups, run `vidarax-archive` while the API is stopped. Opening the WAL runs recovery and removes an incomplete final record. The command checks the complete manifest and its 16 MiB size limit before uploading any object. It then uploads WAL chunks and referenced JPEGs under names derived from their hashes and writes the manifest last. Restore checks every object before moving the restored data directory into place. Retained MP4 and WAV files and webhook delivery state are stored separately and are not included in this snapshot. See [Offline archive and restore](/docs/deployment/#offline-archive-and-restore).

## Validation: replay and schema gates

`scripts/validate_replay_and_schema.sh` is one command:

```bash
cargo test -p vidarax-core --test replay_schema
```

The `replay_schema` integration test (`crates/vidarax-core/tests/replay_schema.rs`) enforces three properties:

- Deterministic replay. It feeds `fixtures/replay/frame-signals.json` through the frame gate twice and requires identical event streams, then hashes event types, reason codes, and frame indexes with FNV and compares against a pinned fingerprint constant. A deliberate change to frame-gate semantics also requires updating the fixture and fingerprint.
- Schema acceptance. `schemas/processing-config.schema.json` and `schemas/frame-metadata.schema.json` must accept their reference fixtures.
- Schema rejection. A frame-metadata instance missing required fields must fail validation, proving the schema actually constrains.

The same script is the first step of `scripts/release_gates.sh`, so no release ships with drifted frame-gate behavior or schemas; see [Allocation discipline](/docs/internals/allocation-discipline/#the-release-check-scripts) for the rest of that pipeline.

## Edge cases and limits

- `pts_ms` on WAL events written by the timeline writer is epoch milliseconds at append time, while worker payloads carry the media-relative `pts_ms` inside the JSON payload. Consumers that need media time must read the payload field.
- The `payload` column is stored as a serialized JSON string. The writer never parses it except for `run_created`, where `principal_key` is extracted for the registry.
- Detached appends acknowledge no durable write. Saturation drops the event with a warning, and later writer failures are logged. Use a confirmed append when the caller needs a durable result.
- Complete malformed lines cause a replay error. Preserve the WAL before manual repair.
- A deleted run's tail is removed from the snapshot immediately, so its reads always take the WAL scan path, where the `run_deleted` event is visible to the deletion checks.

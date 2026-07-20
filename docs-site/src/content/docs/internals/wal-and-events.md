---
title: WAL and events
description: The write-ahead log format, every event kind and who appends it, the WAL event sink, and the replay and schema gates.
---

The local WAL is the authoritative event store. Live-session worker events and handler lifecycle events follow the same append path, even when SpacetimeDB is configured; the optional client mirrors blocking description events only after the WAL commit. Selected JPEGs are durable too, but they live in a content-addressed blob directory rather than inside event JSON. The format and writer live in `crates/vidarax-core/src/timeline.rs`, the append pipeline in `crates/vidarax-api/src/state.rs`, and the worker bridge and blob writer in `crates/vidarax-api/src/wal_sink.rs`.

## File format

The WAL is plain text at `${VIDARAX_DATA_DIR}/timeline.wal` (data directory defaults to `.vidarax-data`), one event per line, six tab-separated fields:

```
seq \t run_id \t stream_id \t pts_ms \t kind \t payload
```

`TimelineEvent::encode_line` escapes `\`, tab, and newline in the string fields (`\\`, `\t`, `\n`) in a single pass; `decode_line` reverses it and returns `None` for malformed lines, which `read_all_events` silently skips. That handles the common torn-final-line case, with two limits spelled out in the [format and recovery contract](#format-and-recovery-contract) below: a line torn at a byte boundary that is not valid UTF-8 fails the whole read, and a torn line that still decodes as a well-formed record replays undetected, because there is no checksum. On Unix the file is opened with mode `0o600` (owner read and write only). Keyframe events carry blob metadata; raw JPEG bytes are written under `${VIDARAX_DATA_DIR}/keyframes/blobs/`.

`WalWriter::append` writes one line and flushes:

```rust
pub fn append(&mut self, event: &TimelineEvent) -> Result<(), TimelineError> {
    writeln!(self.file, "{}", event.encode_line())?;
    self.file.flush()?;
    Ok(())
}
```

There is no `sync_all` call, so durability is write-through to the OS, not to the platter: a process crash loses nothing acknowledged, while an OS or power failure can lose recently appended lines. `timeline.rs` also defines a `DualWriter` that appends the WAL first, then a secondary `EventIndex`, with `reconcile_missing` to repair the index from the WAL; it encodes the "WAL is the source of truth" contract but is not wired into the API server path.

## Format and recovery contract

The exact rules a maintainer or operator can rely on, as implemented today:

- Versioning: none. A line carries no format-version marker; compatibility is positional on the six tab-separated fields. Changing the field set is a breaking change with no migration hook.
- Line size: unbounded. Neither the writer nor the reader enforces a maximum line length; the payload column is as large as the serialized JSON.
- Corruption detection: none beyond decodability. There is no checksum or length framing. A damaged line that still decodes into six well-formed fields replays as if it were real.
- Undecodable lines: `decode_line` returns `None` and `read_all_events` skips the line silently. This is what recovers the common crash case, a torn final line that is valid UTF-8 but incomplete.
- Invalid UTF-8: fatal to the read, not the line. `read_all_events` iterates `BufRead::lines`, and a line containing invalid UTF-8 makes the whole call return an error. At startup this fails `AppState::from_wal`, so the server does not start; on the read path it surfaces as an internal error on the affected endpoints.
- Repair: manual. Nothing truncates or rewrites the file automatically. The procedure for a file that fails the UTF-8 read is to copy it aside, remove or truncate the damaged tail, and restart; the writer reseeds its sequence counter from the highest surviving `seq`.
- The file is append-only by contract. Hand edits are not detected; an edited line that no longer decodes disappears silently from replay.

## Who appends, and every kind

All appends funnel through one thread, `vidarax-timeline-writer`, which owns the `WalWriter`, assigns `seq`, stamps `pts_ms` with wall-clock milliseconds, applies the event to the run registry, updates the in-memory tails, and publishes a fresh snapshot. Async handlers reach it through `AppState::append_run_event_async`; the append is acknowledged over a oneshot channel only after the WAL write succeeded, and a failed write rolls the writer's sequence counter back so the numbering stays dense. Cancellation has one linearization point: before the bounded-channel send completes, the caller still owns the command and no append occurs; after it completes, the writer owns the command and finishes it even if the request disappears while waiting for the acknowledgement. Retried state transitions therefore need an idempotency rule; `run_deleted` has one, while ordinary telemetry events are append-only observations.

Handler-appended kinds, by string literal in `handlers.rs` (all through `append_run_event_async`):

| Kind | Appended when |
|---|---|
| `run_created` | `POST /v1/runs`, and WHIP session start in `whip.rs` |
| `ingest_received` | An ingest request is accepted (file, URL, or realtime attach) |
| `frames_decoded` | A decode pass finishes; payload carries the per-frame signals |
| `marker_emitted` | The gate produces a marker (one event per marker) |
| `analysis_generated` | A deterministic analysis pass completes |
| `semantic_chunk_inferred` | A chunk finishes tiered VLM inference |
| `semantic_chunk_generated` | A chunk's semantic result is recorded |
| `semantic_fallback_activated` | The semantic path falls back (for example, no provider) |
| `inference_completed` | `POST /v1/infer` completes |
| `run_completed` | A run reaches its terminal success state |
| `stop_requested` | `POST /v1/runs/{id}/stop` |
| `keepalive_refreshed` | `POST /v1/runs/{id}/keepalive` |
| `run_deleted` | `DELETE /v1/runs/{id}`, WHIP reclaim, or creation-failure tombstoning |

Concurrent semantic workers publish `semantic_chunk_inferred` as each chunk finishes, so WAL sequence captures completion order rather than `chunk_index` order. Consumers that reconstruct source order must sort by `chunk_index`.

Worker-emitted kinds arrive through the `EventSink` trait rather than a handler. The sink writes the worker's `event_type` string straight through as the WAL `kind`:

| Kind | Emitted by |
|---|---|
| `vlm` / `vlm_tiered` | Keyframe VLM worker; tiered suffix when the second pass answered |
| `clip_vlm` / `clip_vlm_tiered` | Clip VLM worker |
| `state_transition` | VLM worker, when consecutive descriptions diverge past the word-overlap threshold |
| `loop_detected` | Gate or analysis worker, once per loop entry |
| `keyframe_stored` | The sink's `store_keyframe_sync`, recording keyframe metadata |

`transition_state` in `state.rs` is the authoritative map from kinds to run status: `run_created` yields `Pending`; `ingest_received`, `analysis_generated`, `inference_completed`, and `keepalive_refreshed` yield `Processing`; `run_completed`, `run_failed`, and `stop_requested` yield `Completed`, `Failed`, and `Cancelled`. Every other kind leaves the status untouched and only advances `last_activity_ms`. This is why `GET /v1/runs/{id}/state` needs no status column: status is a fold over the run's events.

## The WAL event sink

`WalEventSink` is the live-session `EventSink` in every configuration. It receives the run ID on each sink call and holds the optional SpacetimeDB mirror:

```rust
pub struct WalEventSink {
    state: AppState,
    keyframe_blob_root: PathBuf,
    spacetime_event_mirror: Option<SpacetimeClient>,
}
```

`emit_event_sync` wraps the worker fields (`session_id`, `frame_index`, `pts_ms`, `coordinate_schema`, `coordinates`, `confidence`, `description`) in JSON and calls the confirmed local append. After that succeeds, it attempts the SpacetimeDB mirror; mirror failure is logged and does not undo local durability. `emit_event_nonblocking` uses the detached local append and never mirrors because a network call would violate its nonblocking contract. When the writer queue (capacity `TIMELINE_WRITER_QUEUE_CAP`, 1024) is full, that detached event is dropped with a warning.

`store_keyframe_sync` hashes the raw JPEG, atomically writes a `0o600` content-addressed blob if the hash is new, and then appends `keyframe_stored` with `image_ref`, media type, byte count, SHA-256, and `vidarax.image.v1` coordinate provenance. The blob write is flushed but not fsynced. If the blob write fails, no metadata event is appended; duplicate content reuses the existing file. A crash after the blob rename but before the WAL append can leave an unreferenced blob. Automatic startup reconciliation or retention-based garbage collection is not implemented yet.

Three append flavors, one contract table:

| Path | Caller | Blocking | Full queue | May append `run_deleted` |
|---|---|---|---|---|
| `append_run_event_async` | tokio handlers | awaits ack | awaits capacity | yes, via the idempotent claim |
| `append_run_event` | worker threads | blocks on ack | yield-and-retry | yes, via the idempotent claim |
| `append_run_event_nonblocking` | hot paths | no | drops event | refused with an error |

`run_deleted` is special-cased on every path: it routes through the single-winner claim described in [State and cancellation](/docs/internals/state-and-cancellation/#single-winner-deletion), so the deletion event is appended exactly once per run while the deletion claim is retained, and only through a confirmed append. The retention is bounded: deleted-run records live in a FIFO capped at 4,096 entries, and once a record is evicted, a later DELETE of the same run takes the unknown-run path and appends another `run_deleted`.

## Replay and reads

On startup, `AppState::from_wal` reads the whole file, rebuilds the run registry with `apply_structural_event` per event, rebuilds the warm per-run tails, and seeds the writer's sequence counter from the observed maximum, so numbering continues where it left off. Replay is order-tolerant: an event for an unknown run registers the run on first sight, and `insert_event_by_seq` places late arrivals by sequence number and drops exact duplicates.

Reads have two tiers. `read_run_events_from` serves an advancing cursor from the swap-published snapshot when the run's in-memory tail still covers it; otherwise it falls back to `read_all_events`, a full-file scan filtered by run, executed under `spawn_blocking` on the async path. The scan is linear in total events. A per-run offset index is the clear next step when cold-read volume makes that cost material.

## Validation: replay and schema gates

`scripts/validate_replay_and_schema.sh` is one command:

```bash
cargo test -p vidarax-core --test replay_schema
```

The `replay_schema` integration test (`crates/vidarax-core/tests/replay_schema.rs`) enforces three properties:

- Deterministic replay. It feeds `fixtures/replay/frame-signals.json` through the gate twice and requires identical event streams, then hashes event types, reason codes, and frame indexes with FNV and compares against a pinned fingerprint constant. Any change to gate semantics fails the gate until the fixture and fingerprint are updated deliberately.
- Schema acceptance. `schemas/processing-config.schema.json` and `schemas/frame-metadata.schema.json` must accept their reference fixtures.
- Schema rejection. A frame-metadata instance missing required fields must fail validation, proving the schema actually constrains.

The same script is the first step of `scripts/release_gates.sh`, so no release ships with drifted gate behavior or schemas; see [Allocation discipline](/docs/internals/allocation-discipline/#the-release-gate-scripts) for the rest of that pipeline.

## Edge cases and limits

- `pts_ms` on WAL events written by the timeline writer is epoch milliseconds at append time, while worker payloads carry the media-relative `pts_ms` inside the JSON payload; consumers that need media time must read the payload field.
- The `payload` column is stored as a serialized JSON string; the writer never parses it except for `run_created`, where `principal_key` is extracted for the registry.
- Detached appends provide no failure signal to the caller beyond a server-side warning log; anything a client must be able to observe should use a confirmed append.
- `read_all_events` skipping undecodable lines means manual edits to the WAL fail silently; treat the file as append-only.
- A deleted run's tail is removed from the snapshot immediately, so its reads always take the WAL scan path, where the `run_deleted` event is visible to the deletion checks.

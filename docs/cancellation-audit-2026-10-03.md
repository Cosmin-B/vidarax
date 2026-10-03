# Vidarax cancellation audit

Inspected the working tree at base `3363a4e`, including uncommitted Gemini/dense-review changes. Initial source line references below precede the regression test subsequently added to `handlers.rs`; references before its test module are unaffected. The audit followed the installed Native Systems skill and Vidarax profile. It did not invoke the separate installed `codex-audit` CLI workflow, make provider calls, read credentials, publish, or deploy.

The written [Rain talk](https://sunshowers.io/posts/cancelling-async-rust/), [talk notes](https://github.com/sunshowers/cancelling-async-rust), [RFD 397](https://rfd.shared.oxide.computer/rfd/0397), and [RFD 400](https://rfd.shared.oxide.computer/rfd/0400) informed the distinction between a locally cancel-unsafe operation and a reachable cancellation that violates a global invariant. No claim is made to have watched the video.

## Ranked findings

### 1. High: deleting a webhook can commit its tombstone while delivery remains active

Locations: `crates/vidarax-api/src/delivery.rs:657-671`, `:817-824`, `:1184-1226`.

`delete_webhook` queues `webhook_deleted` and awaits the WAL acknowledgement before sending the coordinator's Remove command. Reset the HTTP/2 stream while that acknowledgement is pending: the handler is dropped, the independently owned timeline command still commits, and Remove is never sent. The coordinator treats every timeline notification only as a wake-up; it does not apply registration/deletion events to its live map. The hook therefore remains active and can continue sending subsequent events until an explicit retry or process restart.

The violated invariant is that a durable `webhook_deleted` removes the registration. This is explicit in `docs-site/src/content/docs/internals/wal-and-events.md:80-85` and in the restart reconstruction at `delivery.rs:1218-1220`; current process behavior disagrees with replay. This is pre-existing relative to the reviewed base.

Actual cancellation source was checked in pinned Hyper 1.10.1: `src/proto/h2/server.rs:451-460` polls a pending service future, observes RST_STREAM, returns a terminal error, and drops the handler. This applies equally to the create path below.

Minimum complete fix: make the durable deletion plus coordinator removal one independently owned transaction, and retain/join that ownership during service shutdown. Preserve the worker's current rule of finishing/retrying an already-started delivery; do not introduce cancellation of partially acknowledged outbound requests without an explicit at-least-once policy.

Suggested bounded offline test: set up an active hook locally without making a delivery request; pause the WAL writer, run DELETE until its append is queued, abort the request, resume the writer, and assert both the durable tombstone and removal from the live coordinator. The current code fails the latter assertion.

### 2. High, fixed during review: journal failure leaves the completion sender permanently blocked

Locations: `crates/vidarax-api/src/handlers.rs:2401`, `:2418-2448`; `crates/vidarax-api/src/semantic_infer.rs:676-677`, `:709-710`.

The journal async block borrows the receiver declared in the outer handler. A blob/WAL append failure returns from that block through `?`, but the receiver remains alive in the handler. `tokio::join!` continues waiting for semantic dispatch. After its bounded completion queue fills, dispatch waits forever in `send`; no consumer remains. This is a progress bug uncovered in the cancellation audit, rather than itself requiring cancellation. Cancellation is currently the only way for the request to escape.

This bug was present at base `3363a4e`; dense-review changes add ordering and deduplication inside the affected journal. It must not be described as newly introduced by those changes.

A regression was added, with authorization, only to the `handlers.rs` unit-test region: `semantic_journal_failure_returns_without_blocking_remaining_completions`. It creates a tiny local 20-frame MP4, submits four chunks with concurrency one, and uses a mocked provider to activate existing WAL failure injection on the first inference. It expects a 500 within two seconds. On the old implementation it failed with `Elapsed(())` in 2.12 seconds:

```text
cargo test --offline -p vidarax-api --lib semantic_journal_failure_returns_without_blocking_remaining_completions -- --exact handlers::tests::semantic_journal_failure_returns_without_blocking_remaining_completions --nocapture
FAILED: journal failure must close its receiver so completion sends can finish: Elapsed(())
```

The implementation applied the minimum deadlock fix: the journal now owns its completion receiver and drops it on all exits, allowing senders to observe channel closure. The same regression passed after the change in 0.23 seconds (1 passed; 263 filtered out). Its final assertion checks the sanitized HTTP 500/internal_error response and at least three mocked inference calls; it does not require private WAL error text to appear in the public response. A separate behavioral choice remains: send errors are presently ignored, so closing the receiver still permits all remaining chunks to be inferred. If failures should stop further paid work, stop admitting new chunks when the journal closes while explicitly handling any already-started blocking operations.

### 3. Medium: cancelling webhook creation leaks reservations or strands durable registrations

Locations: `crates/vidarax-api/src/delivery.rs:215-229`, `:567-595`, `:750-773`.

Reserve inserts a pending entry into the coordinator before its acknowledgement is delivered. Cancelling the request at that acknowledgement leaves a live reservation without a WAL registration; the failed reply send does not roll it back. Cancelling during the subsequent WAL acknowledgement can instead leave a durable registration without ever activating a worker. Neither has an expiry/reconciler. Repetition can consume the process-wide 64-hook capacity, and the registered-but-pending variant does not deliver until explicitly repaired or restarted.

This is pre-existing. Minimum fix: own reserve, persist, activate, and failure rollback within one transaction that survives request cancellation. Reservation cancellation must not depend on another async operation owned by the already-cancelled HTTP future. A coordinator-owned reservation guard/transaction is also viable; checking reply closure once is insufficient to cover cancellation during the later WAL phase.

### 4. Medium, H3 feature only: shutdown returns before H1/H2 requests drain

Locations: `crates/vidarax-api/src/server.rs:104`, `:127-128`, `:148-156`; `crates/vidarax-api/src/lib.rs:150-157`; `crates/vidarax-api/src/main.rs:3-6`.

The experimental H3 server spawns the H1/H2 graceful server and discards its handle. On SIGTERM/SIGINT the H3 accept loop breaks and returns immediately. `run` then returns and `#[tokio::main]` shuts down its runtime, cancelling the H1/H2 drain task and detached H3 request tasks. The comment promising that H1/H2 drains does not match this lifetime. In-flight HTTP requests are cut off rather than drained under this configuration.

This is pre-existing and does not apply to default H1/H2-only serving. Minimum fix: retain and await the H1/H2 server task and explicitly own/drain H3 connection/request tasks before returning from the outer server. Existing H1/H2 drain tests do not cover this orchestration.

## Context-dependent risks, not promoted to confirmed cancellation findings

- A reason request owns both `JoinSet` and the completion journal. Cancelling it drops pending/out-of-order results, aborts async child tasks, and leaves already-started `spawn_blocking` providers/extractions running. The dispatcher owns its provider admission permit inside the blocking closure, which prevents prematurely returning that capacity. The docs explicitly allow recorded extraction to finish after the caller leaves. The new bounded test `semantic_infer::tests::cancelled_dispatch_holds_permit_until_blocking_provider_exits` passed (1 passed, 263 filtered out, 0.00 seconds): after aborting the dispatcher, the mock provider is still running, the dispatch permit remains held, and the completion channel closes without a result; releasing the provider returns the permit and still emits no completion. There is no equally explicit contract guaranteeing paid inference results after disconnect. If that durability is required, own inference completion and journaling together outside the HTTP request; a token alone cannot repair it.
- Source-ordered buffering at `handlers.rs:2420-2442` expands the loss window: completed later chunks can wait behind an unfinished earlier chunk. Current `docs/gemini-flash.md:77-78` intentionally specifies source order, and the docs-site now distinguishes this pending branch behavior from completion order in earlier versions. Neither ordering is intrinsically wrong; the ownership and persistence contract matters.
- `live_audio.rs:392-425` still uses blocking FFmpeg stdin `write_all` followed by `wait_with_output`, with no supervisor deadline or kill/reap guard. A slow/wedged child or full-duplex pipe deadlock can hold a started blocking job after its async waiter is cancelled, and Tokio runtime destruction waits for blocking jobs. This separate path was not repaired by the recorded-extraction timeout fix. A concrete FFmpeg deadlock fixture was not run in this audit, so it remains a targeted follow-up rather than a validated exploit/regression claim.
- Gemini File API cleanup is synchronous inside provider work: ordinary caller cancellation does not skip its final deletion because started blocking work continues. Timeout/HTTP errors where the remote server created a resource but no resource identifier was received remain an uncertain-outcome remote-operation problem; a missing cancellation token is not the cause.

## Protections and existing coverage checked in source

- `state.rs:1214-1235`: an admitted append transfers its payload, byte permit, and delete guard to the independent writer before awaiting the reply. The writer syncs and publishes before replying (`:2008-2044`); caller cancellation does not undo ownership.
- `state.rs:2314-2338`: delete guards restore a failed claim; successful guards commit after WAL completion. Existing tests include cancelled admitted append permit retention, cancelled deletion retry, concurrent idempotent deletion, queued post-tombstone rejection, and closed-channel drain.
- `handlers.rs:418-456` and `whip.rs:989-1065`: stop/delete and WHIP reclamation use detached transactions to keep durable state and media teardown together across HTTP cancellation. Existing tests exercise cancellation during reclaim and termination. These proofs are about request cancellation, not automatically about runtime shutdown.
- `webrtc/session.rs:797-850`, `:872-885`: track receive/send is raced against explicit teardown, with accepted in-flight frame discard only at teardown. Track tasks are signalled and joined on normal session exit.
- `webrtc/runtime.rs:223-239`: already-cancelled config acknowledgements are rejected before applying queued updates; the two-second timeout is not by itself evidence of a bad update path.
- `semantic_infer.rs:683-749`: bounded task admission and join-error mapping prevent unlimited async task creation and associate a child failure with its chunk.
- Recorded FFmpeg extraction now has the 120-second supervisor, kill/reap, and output limits. Existing tests cover timeout/reaping, stderr saturation/overflow, and failed WAV preparation without consuming the retained-evidence budget. This audit does not extend those claims to the separate live-audio FFmpeg path.
- No production `tokio::sync::Mutex`/RwLock held across an await, or production `try_join!` cascade, was found in the searched API/core paths. Synchronous state-owner critical sections and WAL I/O do not have async partial-write cancellation points. SSE/Webhook delivery replay uses durable cursors; duplicate outbound delivery after an uncertain response is consistent with at-least-once delivery.

This reviewer executed the journal regression before the fix (failed at its two-second timeout) and after the fix (passed), plus the new blocking-provider cancellation regression (passed). `cargo fmt --all` completed successfully. Existing tests listed here were inspected, not claimed to have been rerun by this reviewer. The main reviewer applied the journal ownership fix and reran the focused checks. No webhook or H3 fixes were made; those findings remain audit-only.


## Main-reviewer pattern matrix

These classifications distinguish execution evidence from source inspection.
The main reviewer independently traced webhook reserve/persist/activate,
tombstone/remove, coordinator notifications, experimental server shutdown,
and journal/dispatch ownership. Ranked webhook and H3 findings remain source
validated; no network webhook delivery or H3 shutdown fixture was executed.

| Talk pattern / concern | Vidarax boundary | Result and evidence |
| --- | --- | --- |
| Future drop propagates into children | Request-owned JoinSet and journal | Tested: abort closes completion channel; underlying blocking work continues; paid-result persistence remains a contract gap. |
| Cancellation correctness requires global invariant | Webhook durable deletion versus live hook map | Confirmed source finding 1; runtime replay and current state can disagree. |
| Async mutex exposes intermediate state | Searched API/core locks | No production Tokio mutex/RwLock spanning await found; synchronous state critical sections inspected. |
| Reserve then await leaves partial transaction | Webhook create | Confirmed source finding 3; pending reservation and durable-but-inactive hook have no reconciliation. |
| Async future accidentally never awaited | Searched detached task ownership and ignored handles | Intentional detached startup/teardown inspected; H3 ignored handle is finding 4. No blanket claim for every future in repository. |
| select loses an operation | WebRTC teardown and coordinator wakeups | Inspected: frame discard on teardown is accepted; webhook notifications only wake, do not reconcile. |
| try_join short-circuits siblings | Searched API/core production paths | No production try_join found. Current join avoids that cascade but required receiver ownership to make progress. |
| Channel send can be cancelled or wait forever | Semantic result queue | Tested failure before fix: timeout; after fix: HTTP500. Result delivery is still request-owned. |
| Reserve capacity before transferring payload | Timeline admission | Tested: cancelled admitted append retains byte permit until writer completion. |
| Timeouts drop futures, not blocking work | Provider spawn_blocking | Tested: provider semaphore remains held through real worker exit; no journal result survives dispatch abort. |
| Partial async writes / cursor loss | Timeline WAL writer | Inspected synchronous actor writes with owned commands; no async write_all cancellation point on this path. |
| Pin/resume and queue fairness | Awaited MPSC and bounded JoinSet | No independently verified fairness defect; not a proof of global fairness. |
| Actor owns transaction | Timeline writer; webhook coordinator | WAL transfer tested safe; webhook HTTP-owned multi-step transaction is still incomplete. |
| Cancellation token alone insufficient | Recorded provider/extraction | Token would not repair durable/live webhook divergence or guarantee result persistence; ownership must match contract. |
| Runtime shutdown cancels detached tasks | Default H1/H2 versus experimental H3 | Default drain source inspected; H3 source finding 4, no runtime fixture. |
| Panic differs from cancellation | Guards and provider join errors | Source inspected: unwinding guards exist; release panic=abort does not run cleanup. |
| Async drop / remote uncertain outcome | Gemini File API cleanup | Blocking cleanup inspected; remote accepted-but-unacknowledged upload remains unverified live gap. |

Main-reviewer checks after the final receiver fix: five cancellation tests passed
(state append/delete, WHIP reclaim/terminate, blocking provider); journal regression
passed; all three real-footage/mocked dense-review tests passed; strict workspace
Clippy passed with incremental caching disabled; formatting and diff checks passed.
No paid provider calls were made. The installed `/codex-audit` CLI workflow was
not executed; this is a direct review plus the user-requested Astra xhigh review.

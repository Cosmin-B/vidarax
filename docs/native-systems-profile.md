# Runtime contracts

Vidarax turns recorded media and live WebRTC streams into timestamped events.
This file owns the repository-specific scope for native-systems work. Source
code owns the capacities and state transitions linked below.

## Authority and targets

- Maintainer: Cosmin B. Scope: this repository, including the Rust services,
  Python sidecars, TypeScript SDK and Vue UI.
- Authoritative source: `Cosmin-B/vidarax`, `docs/native-systems-profile.md`.
  Initial source baseline: `442543456969326724f78dab132d50f5aea5331e`.
- Portable reasoning: `native-systems-c-cpp`, entrypoint SHA-256
  `5390ae3df1f7cdb00508053213332b7f1d42e8aab0da5d48c7958c945a8e98c0`.
  Apply ownership, lifetime,
  capacity and topology rules to each language. Keep workload rules here and
  implementation invariants beside their source.
- Rust: edition 2021, MSRV 1.89. Release builds use LTO, one codegen unit and
  `panic = "abort"`; a panic cannot provide release cleanup.
- CI covers Linux Rust builds, optional VP8, the SDK and UI. Python runtime
  checks cover Linux, macOS and Windows. Apple Silicon decode is an optional
  host-specific backend. No CPU affinity, cache-line size or NUMA contract is
  established.
- There are no generated or installed copies of this profile. No global
  skill or cross-project profile is changed by repository work.

## Target paths and permitted costs

The live decode, signal and gate loops are the allocation-sensitive paths.
Their steady-state pooled buffers must retain ownership through their last
consumer. Pool reuse checks do not measure every allocation in a pipeline.
The process-wide allocator probe measures the gate loop only.

HTTP parsing, configuration, archive tooling and model requests already use
owned strings, vectors, shared state and asynchronous tasks. Keep those costs
outside the small per-frame loops. A control-plane lock does not justify a new
lock-free abstraction. A blocking media tool must run outside the Tokio
executor and have an owner through exit and cleanup.

One live stream has one stateful decoder, gate, loop detector and temporal VLM
context. Parallelism is across sessions. The API admits each generation's
memory and worker-thread reservation before starting it. The reservation must
survive a detached worker until that worker exits.

See [media-plane ownership](../docs-site/src/content/docs/internals/media-plane.md),
[allocation costs](../docs-site/src/content/docs/internals/allocation-discipline.md)
and [request and generation lifetimes](../docs-site/src/content/docs/internals/state-and-cancellation.md).

## Ownership, representation and failure

| Resource | Owner and transfer | Capacity and release |
|---|---|---|
| RTP access unit | Async track task transfers owned bytes to decode | Bounded channel; wait while active, release on stop. Oversized units are rejected before copying. |
| Decoded YUV planes | Decoder transfers pooled bytes through its reader channel | Pool positions include the reader, queue, decoder pending frames and consumer. Decoder drop ends the child and joins its reader. |
| JPEG work | Decode transfers pooled bytes through bounded stage channels | Queue-specific wait or drop behavior; pools cover all retained positions. A pool may allocate when empty and free when full. |
| Native MP4 window | One admitted inference task extracts and owns one window | Per-window and process byte limits apply until the provider releases the bytes. |
| Live generation | Supervisor owns stage handles and stop state | Stop siblings on an unexpected exit; join them before releasing shared generation resources. |
| WAL event | Timeline writer owns durable append order | Commit before publishing durable state. Caller cancellation cannot undo an accepted transaction. |
| Sidecar request | Caller owns framing and request deadline; sidecar owns decode | Bound network lengths before reading or decoding payloads. Treat sidecar replies as external data. |
| SDK/UI request | Caller or mounted component owns the request and callbacks | Cancellation and replacement must release readers, timers and listeners. Late callbacks cannot replace current state. |

The source constants and sizing functions are authoritative:

- [Live queues and pool positions](../crates/vidarax-core/src/webrtc/workers.rs).
- [Decoder dimensions, reader queue and pending frames](../crates/vidarax-core/src/webrtc/decode.rs).
- [Generation admission](../crates/vidarax-api/src/state.rs).
- [Recorded inference admission](../crates/vidarax-api/src/semantic_infer.rs).
- [Archive format and restore](../crates/vidarax-archive/src/lib.rs).

Source timestamps and persisted identifiers retain their external contract.
Image and video conversion may change pixels but must preserve source-time
mapping. A different representation retains the original byte limits, bounds
and lifetime requirements.

## Trust and process boundaries

HTTP fields, uploaded files, remote media, model replies, archive manifests and
sidecar replies are external inputs. Check their lengths and structure where
they enter. Normalize stable configuration before repeated work. An internal
mutation must preserve the prepared invariant.

ffmpeg/ffprobe and Python model execution cross process boundaries. These
boundaries separate execution and failure; they do not provide an OS security
sandbox. Optional remote providers receive media only when the caller selects
and authorizes them. Model compatibility does not establish visual accuracy.

## Focused checks and measurement

Use the affected unit or integration test first, then the workspace checks in
[`ci.yml`](../.github/workflows/ci.yml). Reuse
[`release_gates.sh`](../scripts/release_gates.sh) for release size and gate
checks. Its defaults are 25 MB for the CLI, 45 MB for the API and 50,000 ns for
gate p95. These are regression ceilings, not end-to-end service budgets.

Report a performance comparison with matched input, output, compiler, build
mode and machine. Include process startup, conversion, transfer and cleanup
when they contribute to the caller-visible operation. Source inspection alone
does not establish a speedup.

The repository formatter governs touched code. Public comments describe
ownership, units, blocking and failure when they matter to the caller.
Implementation comments explain the invariant, ordering or resource cost.
Keep one noun for each object and remove line-by-line narration.

## Measurement limits

The release gate measures the gate operation and executable size. It does not
measure end-to-end request latency or service throughput. The local ingest
comparison includes process startup and cleanup for its stated inputs.

The live Gemini comparison used one synthetic, one-second clip with four known
defect frames. The dense request detected that defect; the sparse request did
not. This result does not establish recall on other media or the provider's
effective sampling rate. The reported cost uses response token counts and
published prices; it is an estimate, not an account invoice.

See [Resources through completion](runtime-resource-changes.md) for the
behavior changes, retained-capacity ledger and matched local ingest costs.
The [Core ownership map](core-ownership-map.md) covers each core module and
explains the execution, lifetime and protocol boundaries.

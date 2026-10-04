# Core ownership map

The core uses Rust ownership to keep frame state local and to release resources
at their consuming boundary. A module boundary separates a resource owner,
execution domain, protocol or independently used mechanism. File length alone
does not justify another layer.

Paths below are relative to `crates/vidarax-core/src`. The source defines the
exact capacities and failure behavior.

| Module | Owner and lifetime | Boundary and storage |
|---|---|---|
| [`admission`](../crates/vidarax-core/src/admission.rs) | One shared scheduler; each permit owns an active reservation. | Mutex-protected counts and a bounded waiter queue. Permit drop releases capacity. |
| [`audio_sidecar`](../crates/vidarax-core/src/audio_sidecar.rs) | One client owns its connection and reconnect state. | Audio framing, payload limits and reply checks stay at the binary protocol boundary. |
| [`backends`](../crates/vidarax-core/src/backends.rs) | Startup owns parsed configuration and the constructed provider graph. | Environment expansion and provider selection happen before repeated inference. Configuration strings remain owned. |
| [`coordinates`](../crates/vidarax-core/src/coordinates.rs) | Plain copied coordinate records. | Source dimensions and crop coordinates cross API and evidence boundaries without a service object. |
| [`crop`](../crates/vidarax-core/src/crop.rs) | A prepared region is copied into decode work. | Region checks and pixel conversion share one coordinate contract. |
| [`dedup`](../crates/vidarax-core/src/dedup.rs) | Each VLM worker owns its last emitted description. | Hash comparison has an exact-text guard. The retained string reuses its capacity. |
| [`embedding_sidecar`](../crates/vidarax-core/src/embedding_sidecar.rs) | Each client owns one connection and its backoff. | Fixed-width embeddings and bounded JPEG framing. Failed exchanges close the connection. |
| [`gate`](../crates/vidarax-core/src/gate.rs) | Each stream owns its reference frame. | Copied signals and enum decisions; classification and reference commit are separate operations. |
| [`gemini`](../crates/vidarax-core/src/gemini.rs) | The provider owns HTTP pools and learned retry state; a call owns uploaded-file cleanup. | Vendor payloads, sampling controls and the bounded thinking retry stay together. Serial attempts share a deadline. |
| [`ingest/fetch`](../crates/vidarax-core/src/ingest/fetch.rs) | Prepared media owns its temporary file until the last caller releases it. | Remote fetch checks each redirect, pins approved addresses and bounds retained media. |
| [`ingest/ffmpeg`](../crates/vidarax-core/src/ingest/ffmpeg.rs) | Each recorded operation owns its command, media result and temporary paths. | Media commands, timestamps and parsers stay together. Child execution uses the shared process owner. |
| [`ingest/pipeline`](../crates/vidarax-core/src/ingest/pipeline.rs) | Startup selects a reusable decode backend. | CPU, NVDEC and VideoToolbox implementations share the recorded decode contract. Hardware probes use bounded child execution. |
| [`ingest/validate`](../crates/vidarax-core/src/ingest/validate.rs) | The input boundary prepares a source and permitted roots. | File canonicalization, scheme checks and public-address checks happen before media work. |
| [`loop_detector`](../crates/vidarax-core/src/loop_detector.rs) | Each stream owns eight recent hashes. | An explicit populated length keeps unused slots out of repetition counts. Reset invalidates history. |
| [`media_process`](../crates/vidarax-core/src/media_process.rs) | One invocation owns the child and its pipe threads through cleanup. | Independent subprocess lifetime, output capacity and timeout policy. Fetch and media parsers receive a completed result. |
| [`metrics`](../crates/vidarax-core/src/metrics.rs) | Shared atomics retain counts; callers record observations. | Fixed histograms and relaxed observability counters. Text allocation happens when exporting metrics. |
| [`novelty`](../crates/vidarax-core/src/novelty.rs) | Each stream owns its signature ring and quantization scratch. | Storage is prepared at construction. Scoring borrows embeddings; commit advances the semantic anchor. |
| [`pipeline`](../crates/vidarax-core/src/pipeline.rs) | Each stream owns the gate, context window and reusable metadata output. | Gate and context phases finish per frame. No intermediate gate-event batch is retained. Returned metadata borrows the pipeline. |
| [`provider`](../crates/vidarax-core/src/provider.rs) | Reusable providers own transports; each request owns its media and deadline. | Routing and vendor-neutral inference contracts. Network payloads use owned strings; admission permits cover active calls. |
| [`sidecar_io`](../crates/vidarax-core/src/sidecar_io.rs) | An exchange borrows a client-owned socket. | One absolute deadline covers partial reads and writes for both sidecar protocols. |
| [`tiered_vlm`](../crates/vidarax-core/src/tiered_vlm.rs) | One logical call carries its request through both model passes. | Second-pass selection and token accounting stay together. The returned request lets callers recover reusable media buffers. |
| [`timeline`](../crates/vidarax-core/src/timeline.rs) | A writer owns the WAL inode lock, ownership lock and committed byte boundary. | Durable append and recovery share the record contract. A failed write poisons the owner until recovery. macOS synchronization is a narrow platform exception. |
| [`trigger`](../crates/vidarax-core/src/trigger.rs) | Each stream owns the prepared program and bounded VM state. | Fixed instruction and state limits; frame evaluation returns a copied assertion. |
| [`zone`](../crates/vidarax-core/src/zone.rs) | Each stream owns zone-entry state and its shared immutable policy. | Hysteresis counters and transitions stay together; evidence writing belongs to another stage. |
| [`webrtc/clip`](../crates/vidarax-core/src/webrtc/clip.rs) | The accumulator owns its sampled window; VLM work receives the retained frames. | PTS-based sampling and bounded clip frames. Worker handles belong to the generation supervisor. |
| [`webrtc/decode`](../crates/vidarax-core/src/webrtc/decode.rs) | One decoder owns its child, reader and newest pending output. | Codec selection, bounded reader handoff and pooled YUV transfer. Drop disconnects the handoff, reaps the child and joins the reader. |
| [`webrtc/decode/vp8`](../crates/vidarax-core/src/webrtc/decode/vp8.rs) | One context owns libvpx state; decode borrows its image pointers until plane copying ends. | Feature-gated C FFI is the only decoder module allowed to use unsafe operations. VP8 runs in process. |
| [`webrtc/depacketize`](../crates/vidarax-core/src/webrtc/depacketize.rs) | The track receiver owns codec-specific fragment state. | Enforce the existing compressed-payload limit before appending fragments. Overflow discards the incomplete unit and permits clean recovery. |
| [`webrtc/recycle`](../crates/vidarax-core/src/webrtc/recycle.rs) | A byte handle owns its buffer and returns it on drop. | Bounded free-list reuse uses existing channel primitives. An empty pool may allocate; a full pool frees excess returns. |
| [`webrtc/resources`](../crates/vidarax-core/src/webrtc/resources.rs) | Generation startup computes the declared video reservation. | Queue positions and rounded pool capacities come from their source owners. RTP reservations include stream framing and capped buffer growth. |
| [`webrtc/runtime`](../crates/vidarax-core/src/webrtc/runtime.rs) | One generation owns its stage handles and stop state. | Configuration acceptance shares a gate with cancellation. A worker drains at most one command-queue capacity per media iteration. Shutdown reports workers that exceed the join deadline. |
| [`webrtc/session`](../crates/vidarax-core/src/webrtc/session.rs) | The session future owns track tasks and their stop signal. | Negotiation and RTP ingress share peer lifetime. Teardown wakes blocked receives and sends, then joins track tasks. |
| [`webrtc/signals`](../crates/vidarax-core/src/webrtc/signals.rs) | A caller borrows decoded planes and owns encoding scratch. | Pixel bounds, frame signals and JPEG encoding. The writer enforces the JPEG limit during encoding. |
| [`webrtc/workers`](../crates/vidarax-core/src/webrtc/workers.rs) | Each stage owns its local gate or VLM state; generation startup owns rollback. | Explicit stage parameters and bounded channels connect decode, analysis, evidence and inference. These owners share pipeline wiring without a new manager layer. |
| [`bin/perf_probe`](../crates/vidarax-core/src/bin/perf_probe.rs) | A standalone process owns its measurement buffers and allocation counter. | Measures gate processing and gate-path allocations. Its results do not describe the entire pipeline. |
| [`lib`](../crates/vidarax-core/src/lib.rs), [`ingest/mod`](../crates/vidarax-core/src/ingest/mod.rs), [`webrtc/mod`](../crates/vidarax-core/src/webrtc/mod.rs) | No runtime storage. | Expose the public mechanisms and keep implementation helpers private. |

## Error text at the consuming boundary

Recorded semantic work in the API retains `SemanticFailure` variants, including
`Cancelled` and numeric HTTP status. The event writer formats the existing wire
text. External failure details retain their owned diagnostic string. A fixed
string builder would add no value for a literal error code that can remain an
enum until serialization. Provider labels borrow their static canonical text.

## Scope of the resource contracts

Fixed frame state does not make the entire service allocation-free. Provider
serialization, recorded batches, exported metrics and WAL records allocate
owned data. Provider timeout adjustments currently clone request media, and
remote HTTP response reads have no separate response-byte ceiling. These costs
are outside the gate and frame-analysis measurements.

The video reservation does not cover every allocator, transport, audio-track
or external codec allocation. ffmpeg separates decoder execution and failure;
the optional libvpx path runs within the host process. See
[Runtime contracts](native-systems-profile.md) and
[Resources through completion](runtime-resource-changes.md) for exact bounds,
measured changes and remaining limits.

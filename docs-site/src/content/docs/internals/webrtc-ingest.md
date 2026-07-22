---
title: WebRTC ingest
description: The WHIP signalling path, how RTP becomes Annex B decoder input, and the session lifecycle from offer to teardown.
---

The WHIP path in `crates/vidarax-api/src/whip.rs` and `vidarax-core/src/webrtc/session.rs` accepts a browser's SDP offer, negotiates one inbound video stream, and feeds its RTP into the per-session worker pipeline. The answer, depacketizer, and decoder agree on one codec. RTP ingress into the decoder stays lossless and ordered. Every visible session has a durable `run_created` event and a watcher that reclaims it. [Media plane](/docs/internals/media-plane/) describes the downstream workers. [State and cancellation](/docs/internals/state-and-cancellation/) covers ownership and cancellation.

## Endpoint contract

RFC 9725 signalling maps to three handlers:

| Endpoint | Handler | Success | Errors |
|---|---|---|---|
| `POST /v1/stream/whip` | `whip_offer` | 201, `application/sdp` answer, `Location: /v1/stream/whip/{sess_id}`, `x-vidarax-run-id` header | 400 empty or non-UTF-8 offer or bad attach header, 409 stream limit, 415 unserveable video, 500 negotiation or persistence failure, 503 session, media-capacity, or id-collision refusal |
| `PATCH /v1/stream/whip/{sess_id}` | `whip_ice` | 204 | 400 non-UTF-8 body, 403 principal mismatch, 404 unknown session |
| `DELETE /v1/stream/whip/{sess_id}` | `whip_terminate` | 200 | 403, 404, 500 cleanup incomplete (retryable) |
| `PATCH /v1/stream/whip/{sess_id}/prompt` | `whip_update_prompt` | 200 after worker acknowledgement | 403, 404, 409 closed generation, 503 acknowledgement timeout |

Session IDs are `sess-` plus 16 hex characters from the OS RNG (`new_session_id`), with a timestamp fallback if the RNG is unavailable. Per-session configuration can ride the offer in the `x-attach-config` header: base64url-encoded JSON (`AttachStreamRequest`) carrying an initial prompt, a token-rate cap, an optional clip-mode config, a normalized crop, a generation-static restricted-zone activity policy, or one compiled trigger program, capped at 8 KiB encoded (`ATTACH_CONFIG_HEADER_MAX_ENCODED_LEN`). Trigger programs and restricted-zone policy are mutually exclusive. The restricted-zone region becomes the decode crop so its motion score has an explicit image-space meaning. Conflicting crop or clip-mode combinations are rejected before workers start.

## Offer handling

`WebRtcSession::new` inspects the offer before rustrtc sees it, because the answer, the depacketizer, and the decode routing must be pinned to the same codec:

- `select_answer_video_codec_for_offer` parses the video m-sections into `OfferedVideoCodec` entries (payload type, codec, clock rate) and picks one live-serveable codec. VP8 is preferred when offered and built in (complete in-crate pipeline, no fmtp negotiation). Otherwise the first serveable codec in offer order wins.
- H.265 payload types whose fmtp signals RFC 7798 decoding-order use (`sprop-max-don-diff > 0` or `sprop-depack-buf-nalus > 0`) are excluded, because the in-crate HEVC depacketizer assumes no DONL/DOND fields and would assemble corrupt access units. Selection then falls back to another serveable codec or to none.
- Offers with more than one video m-section are rejected with 415: a single global answer capability cannot answer multiple video sections correctly, and WHIP ingest is single-stream by design.
- An offer that advertises video but no serveable codec is rejected with 415. This prevents rustrtc's default H.264 depacketizer from mis-parsing another codec.

The session then installs the selected codec as the sole video capability, installs `VidaraxDepacketizerFactory` (rustrtc's H.264 depacketizer for H.264. In-crate depacketizers for H.265 and VP8), and pre-creates a recvonly video transceiver so the offer's video section reuses a receiver built with that factory, not rustrtc's default. One recvonly audio transceiver is pre-created per offered audio section so each is answered `recvonly` (RFC 3264 forbids answering a `sendonly` offer with `sendonly`). Vidarax never consumes the audio. After `set_remote_description`, the handler waits up to 3 seconds for a first local ICE candidate so the answer carries usable host candidates, then creates and applies the answer. Trickle ICE continues over PATCH, where each body line is parsed as a candidate (an `a=` prefix is stripped). A candidate that fails to apply still returns 204, since the connection may complete on other candidates.

`whip_setup_error_response` splits failures: unserveable video is the client's 415, everything else in negotiation is a 500.

## Session start

`start_whip_session` generates the session id, resolves the caller's principal, allocates a run id, and runs `start_whip_session_transaction` in a detached task so client disconnects cannot cancel it mid-flight. The transaction's ordering (stream slot, byte/thread capacity reservation, durable `run_created`, session insert, reclaimer spawn) and the detached-task reasoning behind it are covered in [State and cancellation](/docs/internals/state-and-cancellation/#the-insert-to-spawn-window-in-whiprs). After that, the media wiring happens:

```rust
let (frame_tx, frame_rx) = kanal::bounded::<RtpFrame>(RTP_FRAME_QUEUE_CAPACITY);
let (stream_tx, stream_rx) = kanal::bounded::<StreamFrame>(STREAM_FRAME_QUEUE_CAPACITY);
let (vlm_tx, vlm_rx) = kanal::bounded::<KeyframeWork>(VLM_WORK_QUEUE_CAPACITY);

let run_future = session.run(frame_tx, Arc::clone(&metrics_arc));
tokio::spawn(run_future);
```

Every session uses `WalEventSink`, so worker events are visible through the local timeline. If SpacetimeDB is attached, the sink mirrors successful blocking description events after their WAL commit (see [WAL and events](/docs/internals/wal-and-events/#the-wal-event-sink)). The inference provider falls back to a `NullInferenceProvider` when none is configured, so the pipeline still decodes and filters frames. That fallback emits the explicit description "(no inference provider configured)". Restricted-zone and trigger pipelines each add one bounded eight-item binary queue and one supervised writer. A firing trigger stores the JPEG in the content-addressed sidecar, commits `trigger.<event_type>`, dispatches requested metadata-only actions, then forwards the same pooled buffer to optional semantic inference. Saturated inference can drop semantic work without erasing the already-durable local assertion. Worker threads are spawned inside one `tokio::task::spawn_blocking` call. `PipelineRuntime` owns their stage-tagged join handles and supervises the generation as one unit. The VLM worker owns prompt and schema values after startup. `PATCH /prompt` sends a bounded generation-tagged command and returns success only after that worker acknowledges the update. Restricted-zone policy and trigger bytecode stay fixed for the generation and never enter shared mutable frame-path state.

## How RTP becomes decoder input

`session.run` returns an owned future driving the rustrtc event loop. For each `PeerConnectionEvent::Track` whose kind is video, it spawns a tokio task that receives media samples and converts each into an `RtpFrame`:

- Depacketization has already happened inside rustrtc via the configured factory, so a video sample is a complete access unit for H.264 and H.265, or a raw VP8 payload.
- For H.264 and H.265, the 4-byte Annex B start code `00 00 00 01` is prepended (`ANNEX_B_START`), because rustrtc delivers NAL payloads without the framing the ffmpeg sidecar expects. VP8 passes through unframed.
- `pts_ms` is derived from the RTP timestamp on its 90 kHz clock (`rtp_timestamp / 90`).
- `seq` comes from one `Arc<AtomicU64>` shared by all track tasks of the session. It is a per-session monotone counter, not the RTP sequence number.
- Payload bytes are copied into buffers from a per-task `VecPool` sized by `rtp_nal_pool_slots`: `RTP_FRAME_QUEUE_CAPACITY` (128) queued frames plus one held by the decode worker plus one being constructed.
- An access unit above `MAX_RTP_ACCESS_UNIT_BYTES` (2 MiB) is dropped before that pipeline-owned copy, so the item-bounded queue also has a finite payload-byte envelope.

The enqueue is lossless while the session is active. `enqueue_rtp_frame_lossless` awaits capacity on the bounded channel. A gap in an ordered stream would corrupt the stateful decoder, so when decode falls behind, the tokio task yields and sustained overload reaches the WebRTC media layer. Jitter buffers, NACKs, and keyframe requests handle loss there. On teardown, the session sends a monotonic stop signal to every track task and joins them. An in-flight frame may be discarded at that explicit boundary so a blocked sender cannot hold the decoder open forever. [Decode sidecar](/docs/internals/decode-sidecar/#decoder-warm-up) covers the worker on the other side and its SPS/PPS/IDR warm-up.

Audio samples on a video track are ignored. A receive error ends the track task.

## Lifecycle from offer to teardown

A session ends through one of two doors, both funneling into the same reclaim transaction:

- Explicit WHIP `DELETE`. The handler verifies ownership against the live session entry or a bounded reclaimed-session record, marks the close as graceful, and runs `reclaim_whip_session` in a detached task. This terminates the WHIP resource and keeps the Vidarax run with its captured media references.
- The peer-state watcher. `spawn_session_reclaimer` holds a `tokio::sync::watch` receiver of `PeerConnectionState` and fires on `Disconnected`, `Failed`, or `Closed` (reasons `peer_disconnected`, `peer_failed`, `peer_closed`), or when the channel itself closes.

`reclaim_whip_session_transaction` claims the session exactly once, commits the outcome, and only then removes and closes it. A graceful WHIP termination emits `run_completed`. An unexpected peer loss emits `run_failed` with a fixed reason. `POST /v1/runs/{id}/stop` has already emitted `stop_requested`. Only `DELETE /v1/runs/{id}` emits `run_deleted`. Captured assertions and media references remain available for review. The watcher retries failed terminal WAL appends with exponential backoff between `WHIP_RECLAIM_INITIAL_BACKOFF` and `WHIP_RECLAIM_MAX_BACKOFF`. Failure to insert a session at creation time still tombstones the freshly created run, because that run was never usable.

Closing the peer connection marks the generation stopping before `session.run` drops `frame_tx`. Workers then treat channel closure as a clean stop. The supervisor joins every stage. The decoder's `Drop` kills and waits for its ffmpeg child and joins the stdout reader. An unexpected stage exit instead faults the generation, closes the same peer, stops siblings, and records the fixed stage/reason telemetry.

## Edge cases and limits

- `whip_offer` accepts any `Content-Type` and only logs it. The body is treated as raw SDP text either way.
- An empty PATCH body is the end-of-candidates signal and returns 204 without touching the session.
- DELETE idempotency has a horizon: reclaimed-session records expire after `RECLAIMED_SESSION_TTL_MS` (ten minutes) or past `RECLAIMED_SESSION_MAX_ENTRIES` (1024) entries, after which a repeat DELETE returns 404.
- The response headers degrade safely. An unrepresentable `Location` falls back to `/`, and an unrepresentable run id header becomes empty. The created session remains active.
- `x-attach-config` larger than the encoded cap is rejected with a pointer to `PATCH /prompt` for large prompts.
- Prompt and schema updates have a synchronization point. The bounded command carries the active generation and a one-shot acknowledgement. A request that times out drops its acknowledgement, and the worker discards that cancelled command before applying it.

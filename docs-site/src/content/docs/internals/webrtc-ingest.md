---
title: WebRTC ingest
description: The WHIP signalling path, how RTP becomes Annex B decoder input, and the session lifecycle from offer to teardown.
---

The WHIP path in `crates/vidarax-api/src/whip.rs` and `vidarax-core/src/webrtc/session.rs` accepts a browser's SDP offer, negotiates one inbound video stream, and feeds its RTP into the per-session worker pipeline. It guarantees that the answered codec, the depacketizer, and the decoder always agree; that RTP ingress into the decoder is lossless and ordered; and that every visible session has both a durable `run_created` event and a watcher that reclaims it. This page walks offer handling, the RTP-to-decoder conversion, and the lifecycle. The workers downstream are in [Media plane](/internals/media-plane/); the ownership and cancellation invariants in [State and cancellation](/internals/state-and-cancellation/).

## Endpoint contract

RFC 9725 signalling maps to three handlers:

| Endpoint | Handler | Success | Errors |
|---|---|---|---|
| `POST /v1/stream/whip` | `whip_offer` | 201, `application/sdp` answer, `Location: /v1/stream/whip/{sess_id}`, `x-vidarax-run-id` header | 400 empty or non-UTF-8 offer or bad attach header, 409 stream limit, 415 unserveable video, 500 negotiation or persistence failure, 503 session limit or id collision |
| `PATCH /v1/stream/whip/{sess_id}` | `whip_ice` | 204 | 400 non-UTF-8 body, 403 principal mismatch, 404 unknown session |
| `DELETE /v1/stream/whip/{sess_id}` | `whip_terminate` | 200 | 403, 404, 500 cleanup incomplete (retryable) |
| `PATCH /v1/stream/whip/{sess_id}/prompt` | `whip_update_prompt` | 200 JSON echo | 403, 404 |

Session IDs are `sess-` plus 16 hex characters from the OS RNG (`new_session_id`), with a timestamp fallback if the RNG is unavailable. Per-session configuration can ride the offer in the `x-attach-config` header: base64url-encoded JSON (`AttachStreamRequest`) carrying an initial prompt, a token-rate cap, and an optional clip-mode config, capped at 8 KiB encoded (`ATTACH_CONFIG_HEADER_MAX_ENCODED_LEN`).

## Offer handling

`WebRtcSession::new` inspects the offer before rustrtc sees it, because the answer, the depacketizer, and the decode routing must be pinned to the same codec:

- `select_answer_video_codec_for_offer` parses the video m-sections into `OfferedVideoCodec` entries (payload type, codec, clock rate) and picks one live-serveable codec. VP8 is preferred when offered and built in (complete in-crate pipeline, no fmtp negotiation); otherwise the first serveable codec in offer order wins.
- H.265 payload types whose fmtp signals RFC 7798 decoding-order use (`sprop-max-don-diff > 0` or `sprop-depack-buf-nalus > 0`) are excluded, because the in-crate HEVC depacketizer assumes no DONL/DOND fields and would assemble corrupt access units. Selection then falls back to another serveable codec or to none.
- Offers with more than one video m-section are rejected with 415: a single global answer capability cannot answer multiple video sections correctly, and WHIP ingest is single-stream by design.
- An offer that advertises video but no serveable codec is rejected with 415 rather than letting rustrtc's default H.264 depacketizer mis-parse whatever the peer sends.

The session then installs the selected codec as the sole video capability, installs `VidaraxDepacketizerFactory` (rustrtc's H.264 depacketizer for H.264; in-crate depacketizers for H.265 and VP8), and pre-creates a recvonly video transceiver so the offer's video section reuses a receiver built with that factory, not rustrtc's default. One recvonly audio transceiver is pre-created per offered audio section so each is answered `recvonly` (RFC 3264 forbids answering a `sendonly` offer with `sendonly`); vidarax never consumes the audio. After `set_remote_description`, the handler waits up to 3 seconds for a first local ICE candidate so the answer carries usable host candidates, then creates and applies the answer. Trickle ICE continues over PATCH, where each body line is parsed as a candidate (an `a=` prefix is stripped); a candidate that fails to apply still returns 204, since the connection may complete on other candidates.

`whip_setup_error_response` splits failures: unserveable video is the client's 415, everything else in negotiation is a 500.

## Session start

`start_whip_session` generates the session id, resolves the caller's principal, allocates a run id, and runs `start_whip_session_transaction` in a detached task so client disconnects cannot cancel it mid-flight. The transaction's ordering (slot reservation, durable `run_created`, session insert, reclaimer spawn) and the detached-task reasoning behind it are covered in [State and cancellation](/internals/state-and-cancellation/#the-insert-to-spawn-window-in-whiprs). After that, the media wiring happens:

```rust
let (frame_tx, frame_rx) = kanal::bounded::<RtpFrame>(RTP_FRAME_QUEUE_CAPACITY);
let (stream_tx, stream_rx) = kanal::bounded::<StreamFrame>(STREAM_FRAME_QUEUE_CAPACITY);
let (vlm_tx, vlm_rx) = kanal::bounded::<KeyframeWork>(VLM_WORK_QUEUE_CAPACITY);

let run_future = session.run(frame_tx, Arc::clone(&metrics_arc));
tokio::spawn(run_future);
```

Every session uses `WalEventSink`, so worker events are visible through the local timeline. If SpacetimeDB is attached, the sink mirrors successful blocking description events after their WAL commit (see [WAL and events](/internals/wal-and-events/#the-wal-event-sink)). The inference provider falls back to a `NullInferenceProvider` when none is configured, so the pipeline still decodes and gates. That fallback emits the explicit description "(no inference provider configured)". Worker threads are spawned inside one `tokio::task::spawn_blocking` call; their wiring receives the live novelty settings, while the token-rate cap comes from the session so an attach-config value set before worker start is honored. The session's prompt and guided-JSON handles (`ArcSwap`s) are shared with the workers, which lets `PATCH /prompt` affect the next keyframe without restarting threads.

## How RTP becomes decoder input

`session.run` returns an owned future driving the rustrtc event loop. For each `PeerConnectionEvent::Track` whose kind is video, it spawns a tokio task that receives media samples and converts each into an `RtpFrame`:

- Depacketization has already happened inside rustrtc via the configured factory, so a video sample is a complete access unit for H.264 and H.265, or a raw VP8 payload.
- For H.264 and H.265, the 4-byte Annex B start code `00 00 00 01` is prepended (`ANNEX_B_START`), because rustrtc delivers NAL payloads without start codes while openh264 and the ffmpeg sidecar require them. VP8 passes through unframed.
- `pts_ms` is derived from the RTP timestamp on its 90 kHz clock (`rtp_timestamp / 90`).
- `seq` comes from one `Arc<AtomicU64>` shared by all track tasks of the session; it is a per-session monotone counter, not the RTP sequence number.
- Payload bytes are copied into buffers from a per-task `VecPool` sized by `rtp_nal_pool_slots`: `RTP_FRAME_QUEUE_CAPACITY` (128) queued frames plus one held by the decode worker plus one being constructed.

The enqueue is lossless: `enqueue_rtp_frame_lossless` awaits capacity on the bounded channel instead of dropping. A gap in an ordered stream would corrupt the stateful decoder, so when decode falls behind, the tokio task yields and sustained overload backpressures the WebRTC media layer, where jitter buffers, NACKs, and keyframe requests are the right tools for real-time loss. The decode worker on the other side of this channel, and the decoder warm-up that consumes the first SPS/PPS/IDR units, are covered in [Decode sidecar](/internals/decode-sidecar/#decoder-warm-up).

Audio samples on a video track are ignored; a receive error ends the track task.

## Lifecycle from offer to teardown

A session ends through one of two doors, both funneling into the same reclaim transaction:

- Explicit `DELETE`. The handler verifies ownership against the live session entry or, if a watcher already reclaimed it, against the reclaimed-session record (accepted only once the run is actually marked deleted), then runs `reclaim_whip_session` in a detached task.
- The peer-state watcher. `spawn_session_reclaimer` holds a `tokio::sync::watch` receiver of `PeerConnectionState` and fires on `Disconnected`, `Failed`, or `Closed` (reasons `peer_disconnected`, `peer_failed`, `peer_closed`), or when the channel itself closes.

`reclaim_whip_session_transaction` appends the run's `run_deleted` exactly once through the idempotent claim, then removes the session under `remove_session_for_run` (ownership-checked, single winner), closes the peer connection, and bumps the removal metric. The watcher path wraps this in retries with exponential backoff between `WHIP_RECLAIM_INITIAL_BACKOFF` and `WHIP_RECLAIM_MAX_BACKOFF`, stopping early when it observes the work already done (session re-owned by a different run, or run already deleted). Failure to insert a session at creation time tombstones the freshly created run with `WHIP_CREATE_TOMBSTONE_INLINE_ATTEMPTS` inline attempts and a detached retry task as backstop, so a failed creation cannot leave an active run behind.

Closing the peer connection ends `session.run` on its next poll, which drops `frame_tx`; the decode worker's `recv` loop then ends, its channels close downstream, and the worker threads exit stage by stage, with the decoder's `Drop` killing any ffmpeg sidecar.

## Edge cases and limits

- `whip_offer` accepts any `Content-Type` and only logs it; the body is treated as raw SDP text either way.
- An empty PATCH body is the end-of-candidates signal and returns 204 without touching the session.
- DELETE idempotency has a horizon: reclaimed-session records expire after `RECLAIMED_SESSION_TTL_MS` (ten minutes) or past `RECLAIMED_SESSION_MAX_ENTRIES` (1024) entries, after which a repeat DELETE returns 404.
- The response headers degrade safely: an unrepresentable `Location` falls back to `/`, and an unrepresentable run id header to an empty value, rather than failing the created session.
- `x-attach-config` larger than the encoded cap is rejected with a pointer to `PATCH /prompt` for large prompts.
- Prompt and schema updates are read per keyframe by the workers; there is no synchronization point, so an update applies to the next keyframe after the swap, not retroactively.

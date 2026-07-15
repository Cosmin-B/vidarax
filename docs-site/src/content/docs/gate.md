---
title: The gate
description: The per-frame gate engine and tiered escalation to a second-pass model.
---

Running a vision language model on every frame of a video is the wrong shape for the problem: most frames say nothing new. Vidarax filters before it infers. Every frame passes the deterministic gate, which works from per-frame statistics without any model call; only frames the gate keeps become candidates for inference; and inference itself is tiered, with a second-pass model consulted only when the first pass reports low confidence.

## The per-frame gate

Every decoded frame is first reduced to a small fixed-size signal: a perceptual hash, a mean luma value, and scores for flicker, ghosting, and noise variance. The gate engine compares each signal against the last kept frame and emits exactly one decision per frame:

- Keep keyframe: the frame goes downstream for JPEG encoding and possible inference.
- Suspect artifact: the frame is marked as a visual anomaly.
- Skip: nothing new; the frame is dropped.

Each decision carries a reason code:

| Reason | Trigger |
|--------|---------|
| `initial_frame` | The first frame of a stream is always kept. |
| `scene_cut` | The perceptual-hash distance to the last kept frame crosses the configured threshold. |
| `periodic_keepalive` | A frame is kept at a configured frame interval even when nothing changed, so downstream consumers always have a recent keyframe. |
| `exposure_shift` | The mean luma moved more than the configured threshold. |
| `flicker_suspected` | The flicker score crossed its threshold. |
| `ghosting_suspected` | The ghosting score crossed its threshold. |
| `noise_variance_spike` | The noise variance score crossed its threshold. |
| `no_trigger` | Nothing fired; the frame is skipped. |

All thresholds live in a single gate configuration with defaults defined in code. The gate is deterministic: the same frames in the same order produce the same decisions, which is what makes replay testing of the pipeline possible. It runs on a single-threaded hot path and avoids per-frame heap allocation by design; in the live pipeline, JPEG encoding happens only for frames the gate keeps, not for every decoded frame.

Two small companions run beside the gate:

- Loop detection. A fixed-size ring of recent perceptual hashes detects when the stream is showing the same screen repeatedly; a loop fires when enough recent hashes sit within a hash-distance threshold of the current frame.
- Description dedup. When a loop causes the gate to fire on perceptually identical frames, a dedup filter compares each outgoing VLM description against the last emitted one and suppresses exact repeats, so the event stream does not fill with identical descriptions.

## The semantic novelty gate

The frame gate asks whether pixels changed enough to inspect. The semantic novelty gate asks whether a kept frame changed enough to justify another VLM call. Set `VIDARAX_NOVELTY_EMBEDDING_ADDR` to enable it on live WHIP capture; without that address the pipeline behaves as before.

The reusable library gate can fuse three distances between a candidate and a rolling window:

- OCR text change, compared with MinHash set similarity.
- Embedding change, compared with cosine similarity.
- Perceptual-hash change, compared with bit distance.

That general gate maps the fused score to three outcomes:

- Admit: clearly new; run the full model.
- Drop: clearly redundant; spend nothing.
- Escalate: ambiguous; the caller runs a bounded yes-or-no check to break the tie.

Live capture deliberately uses a narrower binary policy: a SigLIP2 embedding either reuses the last successful description or runs the VLM. It does not call a second model to settle ambiguity. Reuse is bounded by a capture-time TTL and cumulative embedding drift; shadow sampling sends a configurable fraction of reuse decisions through the VLM to measure disagreement. Sidecar timeouts and malformed embeddings fail open by running the VLM.

The gate anchor advances only after a successful, non-empty VLM description. Reused frames and failed inference do not change it. The embedding sidecar receives raw JPEG bytes over a length-prefixed TCP protocol and returns 768 little-endian `f32` values; no image is transformed into JSON or base64. See [Deployment](/operations/) for operations and the repository's `docs/deployment.md` for calibration and provider/hardware evidence commands.

## Tiered escalation to a second-pass model

Frames the gate keeps reach inference, and inference itself is two-tiered. A first-pass model handles the common case. When the first pass returns a confidence below the configured threshold, the same frame is re-run against a second-pass model, and the second answer is used.

The mechanism is three rules:

- Tiering is a configuration of two model IDs and a confidence threshold. When both IDs are the same, tiering is inactive and no second pass ever runs.
- Escalation triggers only when tiering is active and first-pass confidence is strictly below the threshold.
- The second pass has its own output token budget.

The first-pass model handles every frame; the second-pass model runs only on the frames the first pass was uncertain about. The `/v1/models` catalog reports each model's tier and runtime availability.

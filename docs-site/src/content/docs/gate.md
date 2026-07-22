---
title: The per-frame filter
description: The deterministic per-frame filter, semantic novelty reuse, and tiered model escalation.
---

Most adjacent video frames carry no new operational meaning. Vidarax first applies a deterministic filter built from image statistics. Selected frames become inference candidates, and a second model can handle cases where the first returns a low schema confidence score.

## The deterministic filter

Every decoded frame is first reduced to a small fixed-size signal: a perceptual hash, a mean luma value, and scores for flicker, ghosting, and noise variance. The filter compares each signal against the last kept frame and emits exactly one decision per frame:

- Keep keyframe: the frame goes downstream for JPEG encoding and possible inference.
- Suspect artifact: the frame is marked as a visual anomaly.
- Skip: nothing new. The frame is dropped.

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
| `no_trigger` | Nothing fired. The frame is skipped. |

All thresholds live in one filter configuration with defaults defined in code. The same frames in the same order produce the same decisions, which makes replay testing possible. The filter runs on a single-threaded hot path and avoids selected-global-allocator calls after warmup in the measured probe. In the live pipeline, JPEG encoding happens only for selected frames.

Two small companions run beside the filter:

- Loop detection. A fixed-size ring of recent perceptual hashes detects when the stream is showing the same screen repeatedly. A loop fires when enough recent hashes sit within a hash-distance threshold of the current frame.
- Description dedup. When a loop causes the filter to select perceptually identical frames, a dedup filter compares each outgoing VLM description against the last emitted one and suppresses exact repeats.

## The semantic novelty filter

The per-frame filter asks whether pixels changed enough to inspect. The semantic novelty filter asks whether a selected frame changed enough to justify another VLM call. Set `VIDARAX_NOVELTY_EMBEDDING_ADDR` to enable it on live WHIP capture. Without that address the pipeline runs the VLM for each selected frame.

The reusable library gate can fuse three distances between a candidate and a rolling window:

- OCR text change, compared with MinHash set similarity.
- Embedding change, compared with cosine similarity.
- Perceptual-hash change, compared with bit distance.

The reusable library maps the fused score to three outcomes:

- Admit: clearly new. Run the full model.
- Drop: clearly redundant. Spend nothing.
- Escalate: ambiguous. The caller runs a bounded yes-or-no check to break the tie.

Live capture deliberately uses a narrower binary policy: a SigLIP2 embedding either reuses the last successful description or runs the VLM. It does not call a second model to settle ambiguity. Reuse is bounded by a capture-time TTL and cumulative embedding drift. Shadow sampling sends a configurable fraction of reuse decisions through the VLM to measure disagreement. Sidecar timeouts and malformed embeddings fail open by running the VLM.

The shipped reuse threshold is deliberately conservative. It is a safe starting point, not a universal calibration: operators should select a threshold against labelled, representative streams and the latency of the provider they actually run.

The novelty anchor advances only after a successful, non-empty VLM description. Reused frames and failed inference do not change it. The embedding sidecar receives raw JPEG bytes over a length-prefixed TCP protocol and returns 768 little-endian `f32` values. Images never pass through a JSON or base64 transform. See [Operations](/docs/operations/) and the repository's `docs/deployment.md` for calibration and provider/hardware measurement commands.

## Tiered escalation to a second-pass model

Frames the filter keeps reach inference, and inference itself can be two-tiered. A first-pass model handles the common case. When its output contains a confidence score below the configured threshold, the same frame is sent to a second-pass model. This is an uncalibrated model-reported schema value, not a validated probability of correctness.

The mechanism is three rules:

- Tiering is a configuration of two model IDs and a confidence threshold. When both IDs are the same, tiering is inactive and no second pass ever runs.
- Escalation triggers only when tiering is active and first-pass confidence is strictly below the threshold.
- The second pass has its own output token budget.

The first-pass model handles every selected frame. The second-pass model runs only when the first pass reports low confidence. The `/v1/models` catalog reports each model's tier and runtime availability.

Calibrate semantic reuse on more than one natural sequence. Include frozen
content, slow drift, hard scene cuts, overlays, low-light frames, and repeated
motion, then validate the selected threshold with live shadow samples. Batch
file analysis does not traverse the live semantic novelty filter, so its novelty
counters are not applicable.

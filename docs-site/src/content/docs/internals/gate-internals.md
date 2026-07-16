---
title: Gate internals
description: What the gate computes per frame, the branchless decision dispatch, the deferred commit protocol, and tiered escalation in run_tiered.
---

The gate is the deterministic filter that decides, per decoded frame, whether to spend a VLM call. It lives in `crates/vidarax-core/src/gate.rs`, with its input signals computed in `webrtc/signals.rs`, its windowed second pass in `pipeline.rs`, and the model escalation decision in `tiered_vlm.rs`. It guarantees determinism (same frames in, same decisions out, which is what the replay gate in [Allocation discipline](/docs/internals/allocation-discipline/) pins), zero heap allocation on the per-frame path, and exactly one decision per frame. This page is the code-level companion to [The gate](/docs/gate/).

## What is computed per frame

Before the gate sees anything, `yuv_to_frame_signal` in `signals.rs` reduces a decoded YUV frame to a small `Copy` struct, `FrameSignal`:

| Field | Computation |
|---|---|
| `perceptual_hash: u64` | `perceptual_hash_y`: block-average the luma plane onto an 8x8 grid (sampling every 4th pixel in each axis), take the grid mean, set bit `i` when cell `i` is above the mean |
| `luma_mean: f32` | Single-pass mean over the active luma samples, subsampled by 4, normalized to [0, 1] |
| `noise_variance_score: f32` | Variance from the same pass via `E[X^2] - E[X]^2`, normalized with a fixed 4096 cap |
| `flicker_score: f32` | Absolute `luma_mean` delta versus the previous frame's signal (0.0 on the first frame) |
| `ghosting_score: f32` | Hamming distance between this and the previous perceptual hash, divided by 64 (0.0 on the first frame) |
| `frame_index`, `pts_ms` | Carried through from the stream |

The frame must first clear `check_frame`, which verifies even non-zero dimensions and that each plane holds at least the samples the dimensions imply. Everything downstream indexes planes by dimension with no bounds guarding, so this check is the safety boundary; a frame that fails it is dropped before it can touch the previous-signal state and poison the temporal deltas.

## The decision: branchless dispatch

`GateEngine` holds the config plus four fields of state: an `initialized` flag and the frame index, hash, and luma of the last kept frame. `process` evaluates all seven trigger conditions at once, packs them into a bitmask where a lower bit is a higher-priority trigger, and lets `trailing_zeros` pick the winner:

```rust
let mask: u8 = (!self.initialized as u8)
    | ((hash_distance >= self.config.scene_cut_hamming_threshold) as u8) << 1
    | ((frames_since_keep >= self.config.keepalive_every_frames) as u8) << 2
    | ((luma_shift >= self.config.luma_shift_threshold) as u8) << 3
    | ((s.flicker_score >= self.config.flicker_threshold) as u8) << 4
    | ((s.ghosting_score >= self.config.ghosting_threshold) as u8) << 5
    | ((s.noise_variance_score >= self.config.noise_variance_threshold) as u8) << 6
    | 0x80; // sentinel: NoTrigger at index 7

let idx = mask.trailing_zeros() as usize;
```

Bit 7 is always set, so `trailing_zeros` always yields a valid index into three fixed lookup tables (event type, reason code, confidence). There is no branch per condition, no allocation, and no string: `GateReasonCode` is an enum whose `as_str` reproduces the canonical labels (`initial_frame`, `scene_cut`, `periodic_keepalive`, `exposure_shift`, `flicker_suspected`, `ghosting_suspected`, `noise_variance_spike`, `no_trigger`), which shrank `GateEvent` from 48 to 32 bytes versus the earlier `&'static str` field. A unit test (`branchless_matches_reference_for_all_trigger_paths`) checks the bitmask dispatch against a plain if-return reference implementation for every trigger path.

Priority order and outcomes:

| Bit | Reason code | Event type | Confidence |
|---|---|---|---|
| 0 | `initial_frame` | `KeepKeyframe` | 1.0 |
| 1 | `scene_cut` | `KeepKeyframe` | hash distance / 64 |
| 2 | `periodic_keepalive` | `KeepKeyframe` | 1.0 |
| 3 | `exposure_shift` | `SuspectArtifact` | luma shift, clamped |
| 4 | `flicker_suspected` | `SuspectArtifact` | flicker score, clamped |
| 5 | `ghosting_suspected` | `SuspectArtifact` | ghosting score, clamped |
| 6 | `noise_variance_spike` | `SuspectArtifact` | noise score, clamped |
| 7 | `no_trigger` | `Skip` | 0.0 |

Defaults live as constants at the top of `gate.rs`: `GATE_KEEPALIVE_EVERY_FRAMES` (30), `GATE_SCENE_CUT_HAMMING_THRESHOLD` (18), `GATE_LUMA_SHIFT_THRESHOLD` (0.15), and 0.55 for each of the flicker, ghosting, and noise thresholds.

## The deferred commit protocol

`process` reads state but never writes it. Advancing the reference frame is a separate call, `commit_keyframe`, which stores the kept frame's index, hash, and luma and sets `initialized`. The split exists because in the live path a "keep" decision is provisional until a JPEG actually exists: `GateStreamState::on_frame` in `workers.rs` runs the gate through `TwoPassPipeline::analyze_batch_defer_gate_commit`, and only if the decision is `KeepKeyframe` does it invoke the caller's `encode` closure; only if that closure returns non-empty bytes does it call `commit_gate_keyframe`.

```rust
if meta.gate_event != GateEventType::KeepKeyframe {
    return None;
}
let jpeg_bytes = encode().filter(|b| !b.is_empty())?;
self.pipeline.commit_gate_keyframe(signal);
```

Two properties fall out. A dropped frame never pays for JPEG encoding; the encoder runs at most once and only for kept frames, which in the live decode worker is the difference between encoding every decoded frame and encoding only what survives the gate. And a keep whose thumbnail failed does not become the reference frame, so the next good frame is compared against the last frame that actually produced output. A dedicated test (`keep_decision_does_not_advance_reference_until_committed`) pins the non-mutation contract.

## The windowed second pass

`TwoPassPipeline` (`pipeline.rs`) wraps the gate and adds context from a bounded sliding window of recent samples (`window_size` 16, ring-overwritten, no allocation after construction). For each frame, `window_metrics` derives `novelty_score` (mean normalized hash distance across the window) and `temporal_stability` (mean absolute luma drift across the window) in one fused pass, and `motion_score` is the normalized hash distance to the immediately previous frame. The three fuse into a confidence with fixed weights: `TWO_PASS_CONFIDENCE_NOVELTY_WEIGHT` 0.45, `TWO_PASS_CONFIDENCE_INSTABILITY_WEIGHT` 0.35, `TWO_PASS_CONFIDENCE_MOTION_WEIGHT` 0.20. Frames are also bucketed into fixed segments (`segment_ms` 250). The resulting `FrameMetadata` travels with the keyframe into `KeyframeWork` and is embedded in the VLM prompt as gate context (`trigger`, `confidence`, `novelty`, `motion`, `pts_ms`).

Beside the gate, `LoopDetector` (`loop_detector.rs`) keeps a fixed ring of the 8 most recent perceptual hashes and fires when at least `repeat_trigger` of them sit within a Hamming threshold of the current hash. Entry into the looping state emits a single `loop_detected` event; while the state holds, kept keyframes carry `loop_active` and the VLM worker skips inference for them.

## Escalation to the second-pass model

Frames that survive the gate reach `run_tiered` in `tiered_vlm.rs`. The decision surface is `TieredVlmConfig`:

```rust
pub fn is_tiered(&self) -> bool {
    self.first_pass_model != self.second_pass_model
}

pub fn needs_second_pass(&self, first_pass_confidence: f32) -> bool {
    self.is_tiered() && first_pass_confidence < self.second_pass_threshold
}
```

`is_tiered` makes tiering purely configurational: when both model IDs match (the `single_model` constructor, and the default config), no second pass can ever run. `needs_second_pass` escalates only when tiering is active and first-pass confidence is strictly below `second_pass_threshold` (default 0.7).

`run_tiered` walks the tiers in order:

1. Rewrite the request's model to `first_pass_model` and call the provider. A provider error here is the caller's error; there is nothing to fall back to.
2. Parse confidence from the first pass's output text: `parse_confidence_from_output` looks for a `confidence` field in a JSON body and returns 0.5 when the output is not JSON or has no such field, so free-text output lands exactly at the "uncertain" midpoint and escalates only if the threshold is above 0.5.
3. If `needs_second_pass` says so, rewrite the request to `second_pass_model` with `second_pass_max_tokens`, a second-pass timeout, and a forced `guided_json` of `teacher_label_schema()`, so the second-pass model always returns structured labels regardless of what schema the caller set. On success, the second result wins, with the first pass's token usage and latency folded in so accounting spans the whole analysis. On failure, the first-pass result is returned and `used_second_pass` stays false: a broken teacher degrades quality, never availability.

The caller in `workers.rs` records which tier answered in the event kind: `vlm` for a first-pass answer, `vlm_tiered` when the second pass ran (`clip_vlm` / `clip_vlm_tiered` on the clip path). Before tiering, live keyframe mode can ask the semantic-novelty gate to reuse the last successful description.

## Edge cases and limits

- The gate keys keepalive on frame index deltas, not wall time: `keepalive_every_frames` counts frames since the last commit, so a paused stream produces no keepalives.
- Confidence from `parse_confidence_from_output` trusts the model's self-report; a model that always emits high confidence disables escalation in practice. There is no provider-side contract on the value: the parsed number is compared against the threshold as-is, with no clamp to [0, 1] and no finiteness check, so an out-of-range or NaN confidence changes escalation behavior in whatever direction the float comparison happens to take.
- `GateStreamState::on_frame` emits `loop_detected` through `emit_event_nonblocking` before the gate decision, so the event reaches the sink even for frames the gate then drops.
- The word-overlap check that gates `state_transition` events (`jaccard_word_overlap` in `workers.rs`, threshold `STATE_CHANGE_JACCARD_MAX` 0.5) compares word sets on the stack with FNV hashes; it only decides whether to emit an extra event, never whether to call the VLM.
- Signal computation subsamples (stride 4 in the hash grid and the luma statistics); this is a deliberate approximation that preserves determinism because the stride is fixed.

# Vidarax -- Product Pitch

## SLIDE 1: The Problem

**Video is the last unstructured frontier.**

- Most video is still reviewed manually or reduced to coarse motion events.
- Sending every frame to a cloud vision API gets expensive quickly.
- Local motion detection is cheap, but it misses semantic context.
- Open-source teams need a self-hosted path between raw video and VLMs.

## SLIDE 2: How It Works

**Any stream in, semantic events out.** Three steps:

1. **Ingest** -- File upload or WebRTC live stream. ffmpeg decode with adaptive sampling.
2. **Analyze** -- Deterministic gate engine (scene cuts, artifacts, novelty) at O(1) per frame, plus per-chunk sampling. Only a fraction of frames reach the VLM. Tiered routing: fast model first, larger model when uncertain.
3. **Emit** -- Structured events into SpacetimeDB in real time. Queryable and searchable.

Result: most frames never reach a VLM, so inference cost tracks the sampled
frames rather than the full frame rate.

## SLIDE 3: Live Demo

1. Upload a 10-second video and start analysis.
2. Run detail: markers, keyframes, VLM descriptions on a scrubable timeline.
3. Tracing: full pipeline observability with per-stage latency.
4. Settings: tiered model routing, gate engine tuning.

## SLIDE 4: Differentiators

**Self-hosted.** Video never leaves your infrastructure. Deploy on-prem, cloud, or edge.

**Deterministic gate engine.** A two-pass sliding window in Rust flags redundant frames so the analyze path can skip them.

**Open VLMs.** Qwen3-VL (2B/4B/8B), InternVL3.5, LFM2.5-VL. No vendor lock-in. Tiered routing across vLLM and SGLang with automatic fallback.

**Chunked real-time reasoning.** Bounded-lag chunking keeps results flowing while a stream is still live. Prometheus metrics built in.

**Full observability.** WAL-backed timeline, SpacetimeDB sync, pipeline tracing, Vue 3 command center.

## Use Cases

- Gameplay QA and automated testing
- Surveillance and security monitoring
- Manufacturing visual inspection
- Robotics and autonomous systems

## Q&A

**"How is this different from cloud video APIs?"**
Runs entirely on your hardware with open-source models. Only a sampled subset of frames reaches a model; the gate engine turns the rest into lightweight deterministic markers -- you pay for the moments that matter, not the frame rate.

**"Why not send everything to a large model?"**
Economics. Continuous multi-camera inference is dominated by model calls. Vidarax filters first, infers second, and routes between model tiers.

**"Business model?"**
Open-source engine with commercial support. Managed cloud for teams that prefer not to self-host.

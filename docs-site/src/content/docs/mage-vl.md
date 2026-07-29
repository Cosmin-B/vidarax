---
title: Mage-VL debug modes
description: Compare codec-native video tokens with frame sampling and inspect proactive streaming decisions.
---

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) can read H.264 and HEVC
structure directly. Its codec processor keeps anchor-frame patches and the
predicted-frame regions where the codec spent bits. The same checkpoint also
contains a cognition model that scores whether a completed segment warrants a
response.

Vidarax exposes two debug commands through `scripts/mage_vl_debug.py`. They use
the public Mage-VL Space by default and send the video as a file upload. Set
`--space` to point at a private deployment.

## Tokenizer comparison

```bash
python -m pip install "gradio_client>=2,<3"

python scripts/mage_vl_debug.py tokenizer clip.mp4 \
  --num-frames 32 \
  --max-new-tokens 96 \
  --output-dir /tmp/vidarax-mage
```

The debug client requires Python 3.10 or newer.

The command runs the same video twice. One pass uses codec-native canvases. The
other uniformly samples frames. Its JSON result includes each visual-token
count, the model answer, and the measured reduction. With `--output-dir`, the
gallery is copied there so you can inspect the canvases and frames that reached
the vision encoder.

The output directory is optional. No gallery image, model weight, recording,
or result file is written into the repository.

## Proactive streaming

```bash
python scripts/mage_vl_debug.py proactive clip.mp4 \
  --segment-seconds 8 \
  --gate-threshold 0.50 \
  --max-segments 4 \
  --max-new-tokens 96
```

Mage splits the file into non-overlapping segments and reports
`p(respond)` for each one. Segments below the threshold remain silent. Segments
at or above it invoke the full decoder and return commentary.

`cognition_gate_score` is part of the Vidarax trigger ISA, so captured Mage
scores can be replayed beside audio, novelty, detector, or geometry signals.
Use a dedicated Mage deployment for production event delivery.

## Runtime boundary

The model appears in `GET /v1/models` with tier `experimental`. Standard
OpenAI-compatible servers can handle its image or sampled-frame interface.
Codec-native video and proactive streaming need Mage's own processor and gate
weights. They run through the debug command or a dedicated Mage deployment.
Vidarax rejects requests when the configured runtime cannot provide the
selected capability.

The current upstream implementation is Python and Apache-2.0. Integration of a
C++ runtime starts when a compatible upstream release becomes available.

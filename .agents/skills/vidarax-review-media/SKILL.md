---
name: vidarax-review-media
description: Review recorded video or live streams with Vidarax, including grounded speech, physical sound events, synchronized visual moments, and trigger-ready output. Use for MP4 gameplay or development-session review, camera footage, incident finding, audio-video correlation, event verification, or any request to install and run Vidarax without manually assembling its local audio dependencies.
---

# Vidarax Review Media

Turn media into a compact list of timestamped, replayable moments. Prefer local
audio observations for exact speech and sound claims. Use a visual provider to
describe or verify ambiguous moments, never to invent missing audio.

## 1. Work in an isolated runtime

Resolve the repository root with `git rev-parse --show-toplevel`. Keep runtime
data, logs, screenshots, clips, and test outputs outside that directory. Use a
temporary directory or the user's configured Vidarax data directory.

Do not reuse an unknown process already listening on port 8080. It may run an
older build. Start an isolated API from the current checkout when the requested
review must reflect current code.

## 2. Prepare local audio

Inspect the selected profile:

```bash
python3 scripts/audio_runtime.py check --profile whisper --json
```

Start the sidecar with install-on-first-use:

```bash
python3 scripts/audio_runtime.py run --profile whisper
```

The command provisions Python 3.12, syncs the locked environment, checks out the
pinned EfficientAT revision, and reuses package and model caches on later runs.
Use `--offline` after the selected profile has been cached. Set
`VIDARAX_CACHE_DIR` to move all caches. Set `VIDARAX_AUDIO_VENV_DIR` or
`VIDARAX_MODEL_CACHE_DIR` only when the deployment owns those paths.

Use another profile only when the deployment calls for it:

- `whisper`: default grounded English transcription
- `moonshine`: small English speech path
- `qwen`: multilingual ASR
- `sensevoice`: language and vocal-emotion tags
- `lfm`: speech plus optional spoken feedback
- `core`: sound events and voice activity without transcription

## 3. Start the current API

Build the current API and CLI if their release binaries are absent:

```bash
cargo build --release -p vidarax-api -p vidarax-cli
```

Start the API with an isolated data directory and the sidecar address:

```bash
VIDARAX_DATA_DIR="$RUNTIME_DIR/data" \
VIDARAX_BIND_ADDR=127.0.0.1:18080 \
VIDARAX_API_KEYS=local-review-key \
VIDARAX_AUDIO_SIDECAR_ADDR=127.0.0.1:7790 \
target/release/vidarax-api
```

Choose unused loopback ports when those ports are occupied. Wait for
`GET /v1/health` before analysis.

## 4. Run the smallest useful review

For a local audio pass that makes no provider call:

```bash
target/release/vidarax analyze "$VIDEO" \
  --url http://127.0.0.1:18080 \
  --api-key local-review-key \
  --media audio-video \
  --local-audio \
  --speech-engine whisper \
  --audio-profile screen-recording \
  --no-vlm \
  --json
```

Select `gameplay`, `physical-world`, `screen-recording`, or `general` from the
source. Add a configured native-media provider only when the user needs visual
description or cross-modal verification. Keep `--include-frame-metadata` off
unless frame-level debugging is the task.

## 5. Check before reporting

Apply these rules to the returned moments:

- Treat a local transcript as the source for quoted wording. Mark an unclear
  word as uncertain until replay or a second transcription resolves it.
- Require a local sound observation before asserting that a sound occurred.
- Do not use speech content or visible motion as proof that a sound effect
  played.
- Use the visual provider for visible actions and audio-video relationships.
- Reject moments whose end timestamp is not greater than the start timestamp.
- Collapse duplicate moments that describe the same interval and occurrence.
- Distinguish commentary, character dialogue, sound effects, and inferred intent.
- State when console logs, output logs, calibration, or ground truth were absent.
- Preserve event IDs, source-relative timestamps, model IDs, and media hashes.
- Return media references. Never put image, clip, or WAV bytes in JSON or base64.

Report supported moments first. Follow with uncertain candidates and the exact
reason each remains uncertain. Include runtime, response size, retained media
bytes, moment count, and failure count when the API exposes them.

## 6. Clean up

Stop only the isolated processes started for this review. Leave shared services
alone. Remove the temporary runtime directory when the user did not request
retention. Keep cached dependencies and model weights so the next run starts
quickly.

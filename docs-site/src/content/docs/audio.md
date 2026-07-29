---
title: Local audio perception
description: Event tagging, selective speech recognition, and spoken feedback for recorded audio-video windows.
---

Recorded audio-video analysis can run a local audio pass before the VLM. Each
source window becomes a mono 16 kHz PCM WAV. The API sends those bytes to a
bounded TCP sidecar and receives MessagePack metadata. WAV bytes never enter
JSON or base64.

```
MP4 window
    |
    +--> raw MP4 ------------------------------> VLM
    |
    +--> mono PCM WAV --> Silero VAD
                           |
                           +--> EfficientAT sound events
                           |
                           +--> speech present? --> one ASR engine
                                                    |
sound events + transcript hypotheses ---------------+--> timeline moments
```

Silero VAD decides whether speech work is needed. EfficientAT labels non-speech
sounds on every window. The selected ASR model runs only when Silero finds a
speech interval.

## Model roles

The default local stack uses
[Silero VAD v6](https://github.com/snakers4/silero-vad),
[EfficientAT](https://github.com/fschmid56/EfficientAT) `mn10_as`, and
[SenseVoice](https://github.com/QwenAudio/SenseVoice) for selective
transcription. The other speech models are explicit deployment choices:

| Engine | Use it for | Cost and scope |
|---|---|---|
| [SenseVoice](https://github.com/QwenAudio/SenseVoice) | Multilingual speech, language ID, and vocal emotion | Default selective ASR |
| [Moonshine Streaming Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) | English screen recordings on small edge hardware | 34M parameters and English only |
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Multilingual speech, accents, singing, and noisy audio | Larger runtime with Transformers 5.13 or newer |
| [LFM2.5-Audio 1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) | English speech input and optional spoken feedback | Optional because it is much larger than the event front end |

LFM is never loaded by the default stack. `voice_feedback: true` asks that
backend to synthesize a short summary. The resulting PCM WAV is stored in the
content-addressed media directory and the event carries its hash and reference.

## Install and run

The setup script places Python packages and model source outside the tracked
tree:

```bash
scripts/setup_audio_models.sh sensevoice

VIDARAX_EFFICIENTAT_REPO=.vidarax-models/source/EfficientAT \
  .venv-audio/bin/python scripts/audio_perception_server.py
```

Point the API at the sidecar:

```bash
export VIDARAX_AUDIO_SIDECAR_ADDR=127.0.0.1:7790
```

`core`, `moonshine`, `qwen`, `lfm`, and `all` are also valid setup profiles.
Moonshine, Qwen, and LFM profiles require Python 3.10 or newer. Set
`VIDARAX_AUDIO_PYTHON=python3.12` when the system `python3` is older.
The server defaults to one active inference request. Raise
`--max-in-flight` only after measuring memory and latency on the target device.

## Request

Local audio requires `media.mode: "audio_video"`.

```json
{
  "source_uri": "/srv/vidarax-media/gameplay.mp4",
  "model": "gemini-3.5-flash-lite",
  "semantic_inference": true,
  "media": {
    "mode": "audio_video",
    "window_ms": 8000,
    "persist_evidence": true
  },
  "local_audio": {
    "profile": "gameplay",
    "speech_engine": "auto",
    "min_confidence": 0.35,
    "max_events": 32,
    "voice_feedback": false
  }
}
```

Set `semantic_inference: false` for a local-only pass. That mode emits audio
moments without making a VLM call.

The profile controls label normalization:

- `gameplay` keeps explosions, gunfire, impacts, alarms, vehicles, music,
  speech, typing, and similar game cues.
- `screen_recording` favors speech, typing, clicks, notifications, and music.
- `physical_world` and `general` retain normalized AudioSet labels.

The sidecar limits each request to 4 MiB of WAV, 64 observations, and one
minute of source time. A model error stays on its chunk. Decode continues and
an MP4 window already retained remains available.

## Triggers

Audio observations can be replayed through the trigger VM:

```text
trigger game-highlight version 1
when audio_event:explosion >= 0.75
and cognition_gate_score >= 0.60
cooldown 3000ms
emit game_highlight
capture clip 2000ms 4000ms
notify webhook
end
```

`audio_event:<label>`, `speech_confidence`, `audio_novelty_score`, and
`cognition_gate_score` are observation signals. Missing observations fail
closed.

Live WHIP audio is still receive-only at the peer boundary. This audio path
currently runs on recorded files and server-reachable media sources.

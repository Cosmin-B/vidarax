---
title: Local audio perception
description: Event tagging, selective speech recognition, and spoken feedback for recorded files and live WebRTC audio.
---

Recorded files and live WebRTC sessions can run local audio analysis. Recorded
MP4 windows and live Opus windows both become mono 16 kHz PCM WAV. The API sends
the WAV bytes to a bounded TCP sidecar and receives MessagePack metadata. WAV
bytes never enter JSON or base64.

```
Recorded MP4 window                 Live WebRTC Opus window
         |                                      |
         +----------> mono PCM WAV <------------+
                              |
                    +---------+----------+
                    |                    |
                 Silero VAD          EfficientAT
                    |                sound labels
          speech intervals               |
                    |                    |
                 Whisper                 |
                    +---------+----------+
                              |
                 timestamped WAL moments
```

Silero VAD decides whether speech work is needed. EfficientAT labels non-speech
sounds on every window. The selected ASR model runs only when Silero finds a
speech interval. A provider such as Gemini may inspect the synchronized MP4,
but local transcripts remain the source of speech wording. Provider-only audio
claims are discarded when the local pass has no matching observation.

## Model roles

The default local stack uses
[Silero VAD v6](https://github.com/snakers4/silero-vad),
[EfficientAT](https://github.com/fschmid56/EfficientAT) `mn10_as`, and
[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
for selective transcription. The other speech models are explicit deployment
choices:

| Engine | Use it for | Cost and scope |
|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | English speech where wording matters | Default selective ASR |
| [SenseVoice](https://github.com/QwenAudio/SenseVoice) | Multilingual speech, language ID, and vocal emotion | Optional compact speech model |
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
scripts/setup_audio_models.sh whisper

VIDARAX_EFFICIENTAT_REPO=.vidarax-models/source/EfficientAT \
  .venv-audio/bin/python scripts/audio_perception_server.py
```

Point the API at the sidecar:

```bash
export VIDARAX_AUDIO_SIDECAR_ADDR=127.0.0.1:7790
```

`core`, `sensevoice`, `moonshine`, `qwen`, `lfm`, and `all` are also valid
setup profiles. Whisper, Moonshine, Qwen, and LFM profiles require Python 3.10
or newer. Set
`VIDARAX_AUDIO_PYTHON=python3.12` when the system `python3` is older.
The server defaults to one active inference request and eight queued requests.
Use `--max-in-flight` and `--max-queued` to set both bounds after measuring
memory and latency on the target device. A full queue returns a typed
`overloaded` failure. A request that exhausts its queue deadline returns
`timeout`.

## Request

Local audio requires `media.mode: "audio_video"`.

```json
{
  "source_uri": "/srv/vidarax-media/gameplay.mp4",
  "model": "gemini-3.5-flash-lite",
  "semantic_inference": true,
  "media": {
    "mode": "audio_video",
    "window_ms": 20000,
    "persist_evidence": true
  },
  "local_audio": {
    "profile": "gameplay",
    "speech_engine": "whisper",
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

## Live WebRTC audio

Add `local_audio` to the base64url-encoded `x-attach-config` sent with the WHIP
offer:

```json
{
  "local_audio": {
    "profile": "physical_world",
    "speech_engine": "whisper",
    "min_confidence": 0.35,
    "max_events": 32
  }
}
```

The offer returns 503 when local audio is requested without
`VIDARAX_AUDIO_SIDECAR_ADDR`. Inbound Opus access units enter a bounded queue.
Four-second windows are Ogg-framed in memory, decoded by ffmpeg, and sent to the
same sidecar used for recorded files. A slow sidecar can drop whole analysis
windows. It cannot block RTP video ingest.

Live timestamps are relative to each audio track's RTP clock. They are not
cross-track RTCP wall-clock synchronization. Events carry `session_id`,
`track_id`, and source-relative millisecond offsets so consumers can keep
multiple audio tracks distinct.

## Telemetry

`GET /v1/metrics` separates each part of the audio path:

- WAV extraction latency, encoded bytes, and source duration
- extraction-to-result latency and real-time factor
- VAD, sound-classifier, ASR, and TTS latency
- sidecar active capacity and bounded queue depth
- TTS attempts, successes, failures, output bytes, and latency
- fixed `decode`, `timeout`, `overloaded`, `model_load`,
  `malformed_response`, `reconnect`, and `inference` failure reasons
- live WebRTC track count, Opus access units, encoded bytes, RTP duration,
  receive failures, and bounded-queue drops

Audio tracing links work to run, request, stream, and chunk IDs. Media stays
outside span attributes.

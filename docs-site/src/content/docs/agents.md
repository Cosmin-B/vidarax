---
title: Agent workflows
description: Run grounded media reviews from a compatible agent harness without hand-building the local audio environment.
---

<!-- status: draft, needs Cosmin's rewrite pass before publication -->

Vidarax includes the platform-neutral `vidarax-review-media` Agent Skill under
`.agents/skills/`. It gives a compatible harness one workflow for recorded
media, camera footage, gameplay review, and development-session review.

The skill keeps each review isolated from an unknown server already running on
port 8080. Runtime data, logs, and review artifacts stay outside the source
tree. Cached dependencies and model weights remain available to later runs.

## Install-on-first-use audio

The skill starts the local audio sidecar through:

```bash
python3 scripts/audio_runtime.py run --profile whisper
```

That command performs setup only when the selected environment is missing or
stale. It provisions Python 3.12, syncs `audio/uv.lock`, checks out the pinned
EfficientAT revision, then runs the sidecar. Inspect the resolved state with:

```bash
python3 scripts/audio_runtime.py check --profile whisper --json
```

Set `VIDARAX_CACHE_DIR` to choose a cache root. Once a profile is cached, add
`--offline` to require a network-free start.

## How the skill handles model output

The skill assigns each claim to the component that can support it:

- Local ASR supplies exact spoken wording.
- Local sound observations support claims about audible events.
- The visual provider describes visible actions and audio-video relationships.
- Invalid time ranges and duplicate moments are removed before reporting.
- Event IDs, source timestamps, model IDs, and media hashes remain attached.
- JPEG, MP4, and WAV payloads remain in the binary store. Events carry references.

The result separates supported moments from uncertain candidates. If a matching
console log, output log, calibration, or labelled reference was not supplied,
the report says so directly.

## Manual invocation

An agent harness that does not discover repository skills automatically can
load:

```text
.agents/skills/vidarax-review-media/SKILL.md
```

The skill contains no provider-specific credentials or global installation
steps. The repository checkout, environment variables, and current Vidarax
configuration remain the source of those choices.

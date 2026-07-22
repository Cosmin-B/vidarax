---
title: Trigger programs
description: A bounded instruction set for turning perception signals into durable actions.
---

Trigger programs define when a live stream emits an operational assertion. The
v1 instruction set is deliberately small: load a signal, compare values,
combine booleans, require persistence, detect a rising edge, apply a cooldown,
emit one durable event, request a keyframe or clip, and notify a webhook or
allowlisted local output. Programs contain at most 64 forward-only instructions,
16 stack values, 16 state slots, and 8 actions, so execution is bounded.

```text
trigger loading-bay-entry version 1
when motion_score >= 0.40 for 2 frames
edge rising
cooldown 5000ms
emit loading_bay_entry
capture keyframe
notify webhook
end
```

Compile, validate, and replay through either the REST API, TypeScript SDK, or
CLI. `POST /v1/triggers/compile` returns typed JSON bytecode;
`POST /v1/triggers/validate` checks the version, stack, jumps, state, and action
bounds; `POST /v1/triggers/evaluate` replays timestamped samples through the
same state machine used by live capture.

Replay samples keep scalar signals at the top level:

```json
[
  { "pts_ms": 0, "motion_score": 0.20 },
  { "pts_ms": 100, "motion_score": 0.55 },
  { "pts_ms": 200, "motion_score": 0.60 }
]
```

Attach the resulting `trigger_program` in the WHIP attach configuration. The
program is fixed for one pipeline generation. The current live scalar path
supports `motion_score`, `novelty_score`, and `confidence`, requires one
keyframe capture, and stores the binary JPEG before committing the metadata
event. Detector, tracking, world-geometry, uncertainty, teacher-disagreement,
and clip actions are already represented in the ISA but are rejected at live
attach until their corresponding producers are connected. Replay supports all
signals through explicit observations.

An `emit loading_bay_entry` instruction commits the event as
`trigger.loading_bay_entry`; trigger programs cannot impersonate run lifecycle
events. `notify webhook` opts that assertion into the run's registered webhook
deliveries. `notify local_output` sends a metadata-only Unix datagram to the
absolute path in `VIDARAX_TRIGGER_LOCAL_OUTPUT_SOCKET`. It carries the stable
event ID, run, stream, sequence, timestamp, kind, and pipeline generation—never
image bytes. The receiving process must bind the socket before the stream is
attached.

Missing signals fail closed. A slow binary writer or consumer cannot block
decode: the handoff is bounded and its drop/failure counters are exposed in
Prometheus metrics.

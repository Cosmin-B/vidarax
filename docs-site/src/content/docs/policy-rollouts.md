---
title: Policy rollouts
description: Durable feedback, candidate replay, and generation-safe policy promotion.
---

Vidarax keeps the operator control loop beside the observations that informed it. Feedback, immutable policy revisions, replay results, promotions, and rollbacks are timeline events in the local WAL. A standalone node retains its review and rollout history without SpacetimeDB or another control-plane database.

## The lifecycle

```text
draft → shadow → canary → active
                         ↓
             older active revision ← rollback
```

Create a revision with `POST /v1/runs/:id/policies`. The response's `revision` is the sequence of the durable creation event. The next revision must include that value as `parent_revision`. A stale parent receives `409 Conflict`.

Promotions use `POST /v1/runs/:id/policies/:revision/activate`. Shadow and canary are replay-only stages in this release. They record the operator decision without mutating the live media worker. On a live WHIP run, active promotion requires the current `expected_generation`. The API sends prompt and schema through the bounded session command channel and records an acknowledgement only after the owning VLM worker accepts it. A closed or replaced generation rejects the request.

Restricted-zone parameters are constructed when a pipeline generation starts. They are preserved in the revision and available for replay, but a live response lists `parameters.restricted_zone` in `deferred_fields` until a new generation starts. This avoids claiming that a threshold changed while an older detector is still running.

## Candidate replay

`POST /v1/runs/:id/policies/:revision/replay` reads `restricted_zone_activity_entered` events already persisted on the run. It compares their recorded confidence with the revision's `enter_motion_score` and reports accepted, rejected, and missing-score counts.

Replay shows how a revised threshold treats candidates that the original pipeline captured. Missed incidents and behavior on another camera require retained source media and labelled evaluation data.

## Operator feedback

`POST /v1/runs/:id/feedback` commits `operator_feedback_submitted` before returning success. `GET /v1/feedback` reconstructs entries from the local WAL and filters them to caller-owned runs. When SpacetimeDB is configured, it receives a best-effort mirror after the local commit. Collection and review still use the local WAL.

See the [API reference](/docs/api/#policy-revisions-and-replay) for request and response fields.

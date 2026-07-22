---
title: Edge deployment
description: Signed binary model releases with shadow, canary, activation, and rollback.
---

The edge package runs the normal Vidarax API beside a local model server and a
small signed-update controller. Model bytes never travel in JSON. A manifest
references a binary artifact by URI, length, SHA-256, and monotonically
increasing sequence and is signed with Ed25519; the device retains only the
update public key.

Use `vidarax edge enroll` once, then run `vidarax edge watch` from the supplied
systemd service. A verified shadow candidate advances to canary only after its
reported sample count, success rate, p95 latency, and RSS meet the signed
acceptance bounds. A passing canary becomes the current local model; a failed
candidate is discarded while the previous active release remains available.
Network loss therefore stops updates, not perception.

Each new release must begin in shadow. The device persists a signed-sequence
high-water mark, and a release ID is permanently bound to its first signed
manifest. Rollback is issued as a new higher-sequence shadow release pointing
at the old artifact, so replaying an earlier valid signature cannot downgrade a
device.

Health reports carry the candidate release ID and expected stage. Delayed data
from an older release or stage cannot promote the current candidate. Replacing
a staged candidate first requires the hook to acknowledge rollback of the old
one.

Enrollment pins both the hardware cohort and an absolute transition-hook path.
The hook receives an action, the verified model path, and the release ID. It
must acknowledge `stage_shadow`, `stage_canary`, `activate`, and `rollback`
against the exact release. The controller journals the transition before the
hook runs and replays it after a crash, so hooks must be idempotent. It changes
the current pointer only after `activate` succeeds, and clears a failed
candidate only after `rollback` succeeds. Rejected release IDs are remembered
so a polling device does not repeatedly download the same bad artifact.

The update controller runs as a separate `vidarax-update` system user. The API
service cannot write updater state or replace the enrolled trust root and hook.
Artifact size and free-space checks run before download, and retention is
bounded to the current, previous, and candidate releases.

The complete command sequence, manifest schema, and systemd files are in
[`docs/edge-deployment.md`](https://github.com/Cosmin-B/vidarax/blob/main/docs/edge-deployment.md).

# Edge deployment and signed model rollout

Vidarax can run the API and media pipeline on a Linux gateway while a separate
update controller keeps the last accepted model available during network loss.
The checked-in systemd units under `deploy/edge/` are the first supported
package. They target Jetson-class and x86 gateways. Camera-native firmware
packaging remains hardware-specific.

The model artifact is always binary. A manifest carries only its HTTPS or local
file URI, byte length, SHA-256 digest, monotonic release sequence, rollout
stage, hardware cohort, and acceptance bounds. Ed25519 signs the canonical
manifest. The device stores only the public key.

```bash
# Control-plane setup. Keep update-private.key offline.
vidarax edge keygen \
  --private-key update-private.key \
  --public-key update-public.key
sudo install -o vidarax-update -g vidarax -m 0400 update-public.key \
  /etc/vidarax/update-public.key

# One-time device provisioning. The updater owns this state; the API cannot.
sudo -u vidarax-update vidarax edge enroll \
  --state-dir /var/lib/vidarax-edge \
  --device-id warehouse-17-gateway-2 \
  --hardware-cohort jetson-orin \
  --public-key /etc/vidarax/update-public.key \
  --activation-hook /usr/local/libexec/vidarax-activate-model

# Sign, then verify/download/stage a release.
vidarax edge sign --manifest release.json \
  --private-key update-private.key --output release.signed.json
vidarax edge apply --state-dir /var/lib/vidarax-edge \
  --manifest release.signed.json
```

An unsigned manifest has this shape:

```json
{
  "schema_version": 2,
  "sequence": 42,
  "release_id": "forklift-detector-2026-07-21",
  "model_id": "forklift-pedestrian-detector",
  "hardware_cohort": "jetson-orin",
  "stage": "shadow",
  "artifact": {
    "uri": "https://models.example.com/forklift-detector.bin",
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "size_bytes": 12345678
  },
  "acceptance": {
    "minimum_samples": 1000,
    "minimum_success_rate": 0.98,
    "maximum_p95_ms": 50,
    "maximum_rss_bytes": 1073741824
  },
  "signature": ""
}
```

After a shadow interval, report measured candidate health. A passing shadow
candidate advances to canary. A passing canary becomes current. Any failed
stage removes the candidate and leaves the last active release untouched.

```bash
vidarax edge report --state-dir /var/lib/vidarax-edge \
  --release-id forklift-detector-2026-07-21 --stage shadow \
  --samples 1200 --success-rate 0.991 --p95-ms 42 --rss-bytes 805306368
```

Every report names the release and stage that produced its measurements. A
late report for a replaced candidate or earlier stage is rejected, leaving the
current state unchanged.

`vidarax edge watch` polls an HTTPS desired-manifest URL and applies only a new
release ID with a sequence above the device's per-cohort high-water mark.
Release IDs are permanently bound to their first signed manifest. An intended
rollback is a new signed shadow release with a higher sequence that references
the previously accepted artifact hash. Replaying an old signed manifest cannot
downgrade a device. `deploy/edge/vidarax-update.service` runs that controller.
The current model pointer and verified release files are local, so update-server
loss does not interrupt the active pipeline.

The current package is the device half of a fleet system. Enrollment happens
locally, and `watch` polls an operator-provided HTTPS URL without a Vidarax
fleet identity protocol. The repository has no hosted device registry, remote
enrollment endpoint, cohort editor, rollout dashboard, or server-side health
collector. A control plane can be added by publishing signed manifests and
collecting the output of `vidarax edge status` and `vidarax edge report`.

Every rollout transition is acknowledged by the enrolled hook. It receives
`ACTION MODEL_PATH RELEASE_ID`, where the action is `stage_shadow`,
`stage_canary`, `activate`, or `rollback`. Shadow and canary actions must load
the exact candidate into the corresponding serving lane before returning, so
the next health report measures a running candidate. `activate` moves that
release onto the live lane. `rollback` removes the candidate and restores the
current release when necessary. The current release and model path are also
available as `VIDARAX_EDGE_CURRENT_RELEASE` and
`VIDARAX_EDGE_CURRENT_MODEL` when one exists.

Every new release must begin in shadow. A manifest cannot jump directly to
canary or active. The hook must be idempotent because Vidarax writes the
anti-replay high-water mark and pending transition together before invoking it,
then replays that transition after a crash. Vidarax changes
`current-model` only after the hook acknowledges `activate`. A non-zero exit or
30-second timeout starts `rollback`. Candidate state is cleared only after that
rollback succeeds. Keep the hook root-owned and non-writable by the `vidarax`
service account. The supplied units run the controller as the separate
`vidarax-update` user and do not expose `/var/lib/vidarax-edge` as writable to
the network-facing API service. Provision that system user before enrollment.

The device state records a rejected release ID. Repeated polls do not download
that release again. The control plane must issue a new release ID after fixing
it. Manifest and artifact downloads are streamed into private temporary files,
checked against their signed byte length and SHA-256 digest, and then renamed
into place. The controller refuses artifacts above
`VIDARAX_EDGE_MAX_ARTIFACT_BYTES` (32 GiB by default), reserves 256 MiB of free
space beyond the incoming artifact, and retains only the current, previous, and
candidate releases.

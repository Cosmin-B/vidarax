# Offline archive and restore

`vidarax-archive` backs up the timeline WAL and its referenced keyframes while
the API is stopped. It takes the same WAL lock as the API, checks the events
and keyframes, uploads objects named by their SHA-256 hashes, and writes a
manifest listing them. If a keyframe is missing or corrupt, the command stops
before writing the manifest. A failed run can leave uploaded objects that no
manifest references. Local data is kept. The snapshot includes JPEGs referenced
by keyframe, restricted-zone, and trigger events.

Opening the WAL also runs recovery. It removes an incomplete final record
before taking the snapshot and stops on a corrupt complete record. The
snapshot excludes retained MP4 and WAV files, webhook delivery state,
temporary uploaded videos, inference caches, and optional SpacetimeDB state.

The archive splits the WAL at record boundaries. The default target is 8 MiB
per object. `--chunk-mib` accepts 1-64. A single record can be up to about 64
MiB, so one object can exceed the target. Keyframes are limited to 16 MiB each.
The manifest is limited to 16 MiB and 100,000 WAL chunks and 100,000 distinct
keyframes. The archive checks the complete manifest size before uploading any
object. Repeated archives create complete WAL and keyframe manifests and reuse
objects with identical content. You must run each archive yourself and keep
its manifest key. The command does not delete local data or make archived
events available to API readers.

Use a bucket whose credentials are supplied by the standard `AWS_*`
environment variables understood by `object_store`. For example:

```sh
cargo run -p vidarax-archive -- archive \
  --data-dir /var/lib/vidarax \
  --bucket my-vidarax-archive \
  --region us-west-2 \
  --prefix deployment-a
```

The command prints `manifest_key` and the last archived sequence number in
`event_coverage_seq` and `evidence_coverage_seq`. Save the manifest key outside
the original machine so you can restore that snapshot if the machine is lost.
Both sequence numbers are equal because the archive fails if any referenced
keyframe is missing.

Restore while the API is stopped into a path that does **not** exist:

```sh
cargo run -p vidarax-archive -- restore \
  --target-dir /var/lib/vidarax-restored \
  --bucket my-vidarax-archive \
  --region us-west-2 \
  --manifest-key deployment-a/manifests/<sha256>.json
```

Restore downloads into a temporary directory next to the target. It checks
object hashes, event sequence numbers, and whether the manifest lists every
referenced JPEG before moving the directory into place. It refuses an existing
target. Run the API with
`VIDARAX_DATA_DIR=/var/lib/vidarax-restored` only after restore succeeds.

The final move supports macOS and Linux. On other platforms, restore stops
before creating the target directory.

For a local S3-compatible endpoint, add `--endpoint URL`. Use `--allow-http`
only for a trusted local test service. The repository currently has no bundled
MinIO service. Tests use `object_store::memory::InMemory`.

A restore includes only events from the last successful archive. Events written
after that archive stay on the original machine until you archive again. The
API continues to use the local WAL until you restart it with a restored data
directory.

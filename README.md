# vidarax

Vidarax is a low-latency video interpretation framework.
The point is simple: deterministic outputs, strict validation, and enough observability to trust it in production.

## What we are focused on right now

- Ship a strong v1 that covers practical video interpretation workflows.
- Keep model support explicit for the requested medium/small VLM set.
- Hold a high reliability bar: validation, retries, idempotency, traceability, and frame-level metadata.

## Main reference docs

- `docs/v1-world-class-support-spec.md`
- `docs/lockfree-queue-topology.md`
- `docs/memory-allocation-strategy.md`
- `docs/deterministic-gate-engine.md`
- `docs/provider-routing.md`
- `docs/provider-transport-decision.md`
- `docs/processing-modes.md`
- `docs/api-surface.md`
- `docs/api-architecture.md`
- `docs/deployment.md`
- `docs/ops-runbook.md`
- `docs/inference-observability-slo.md`
- `docs/wal-dual-write.md`
- `docs/victoria-correlation.md`
- `docs/release-gates.md`
- `schemas/processing-config.schema.json`
- `schemas/frame-metadata.schema.json`

## Workspace layout

- `crates/vidarax-core`
  Lock-free primitives and memory foundations.
- `crates/vidarax-contracts`
  Shared contracts for models, lifecycle behavior, and error mapping.
- `crates/vidarax-cli`
  Small CLI entrypoint used for contract checks and tooling.
- `crates/vidarax-api`
  h1/h2 and h3 server, WAL-backed run lifecycle, and external provider inference.

## Validation and release gates

- Replay + schema checks:
  `./scripts/validate_replay_and_schema.sh`
- Contention + allocation probe:
  `./scripts/bench_regression.sh`
- Provider transport benchmark + decision write-up:
  `./scripts/bench_provider_transport.sh`
- Full release gate run:
  `./scripts/release_gates.sh`
- CI definition:
  `.github/workflows/ci.yml`
- End-to-end smoke test:
  `./scripts/smoke_v1.sh`
- MP4 ingest/decode/analyze smoke test:
  `./scripts/smoke_mp4_pipeline.sh`
- Model artifact provisioning:
  `./scripts/provision_models.sh`

## Transport modes

- Default h1/h2:
  `VIDARAX_TRANSPORT=h1h2 cargo run -p vidarax-api --bin vidarax-api`
- HTTP/3:
  `VIDARAX_TRANSPORT=h3 cargo run -p vidarax-api --bin vidarax-api --features h3-experimental`
- HTTP/3 TLS files:
  - `VIDARAX_H3_TLS_CERT_PATH` (default `deploy/certs/dev.crt`)
  - `VIDARAX_H3_TLS_KEY_PATH` (default `deploy/certs/dev.key`)
  - Generate local dev files with: `make dev-cert`
- External inference endpoints:
  - `VIDARAX_VLLM_BASE_URL`
  - `VIDARAX_SGLANG_BASE_URL`
  - must be `https://` for non-loopback hosts; `http://127.0.0.1` / `http://localhost` allowed for local testing
- Ingest file-root allowlist:
  - `VIDARAX_INGEST_FILE_ROOTS` (comma-separated absolute roots; default `cwd` + system temp dir)
- Optional security controls:
  - `VIDARAX_REQUIRE_API_KEY` (default `true`)
  - `VIDARAX_API_KEYS`
  - `VIDARAX_REQUIRE_TENANT_ID`
  - `VIDARAX_RATE_LIMIT_GLOBAL_RPS`
  - `VIDARAX_RATE_LIMIT_TENANT_RPS`
  - `VIDARAX_RATE_LIMIT_TENANT_SLOTS`
  - `VIDARAX_STREAM_TTL_SECS`
  - `VIDARAX_ACTIVE_STREAM_LIMIT`

## Runtime data

- Event WAL location:
  `VIDARAX_DATA_DIR/timeline.wal`
- Default data directory:
  `.vidarax-data`

## Local deployment

- Build API image:
  `docker build -f deploy/Dockerfile.api -t vidarax-api:local .`
- Start local stack (API + Victoria services):
  `docker compose -f deploy/docker-compose.local.yml up --build`
- Local runtime defaults bind to loopback:
  - `VIDARAX_BIND_ADDR=127.0.0.1:8080`
  - `VIDARAX_H3_BIND_ADDR=127.0.0.1:8443`

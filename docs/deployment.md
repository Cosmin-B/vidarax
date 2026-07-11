# Deployment

This document covers server configuration and the supported local deployment
paths. Values below come from the current Rust source, mainly
`crates/vidarax-api/src/config.rs`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDARAX_BIND_ADDR` | `127.0.0.1:8080` | HTTP/1.1 and HTTP/2 bind address. |
| `VIDARAX_TRANSPORT` | `h1h2` | Transport mode. Accepts `h1`, `h2`, `h1h2`, `http`, `http2`, `h3`, or `http3`. |
| `VIDARAX_H3_BIND_ADDR` | `127.0.0.1:8443` | HTTP/3 UDP bind address when `VIDARAX_TRANSPORT=h3`. |
| `VIDARAX_H3_TLS_CERT_PATH` | `deploy/certs/dev.crt` | TLS certificate for experimental HTTP/3. |
| `VIDARAX_H3_TLS_KEY_PATH` | `deploy/certs/dev.key` | TLS private key for experimental HTTP/3. |
| `VIDARAX_DATA_DIR` | `.vidarax-data` | Runtime data directory. The WAL is `${VIDARAX_DATA_DIR}/timeline.wal`. |
| `VIDARAX_INGEST_FILE_ROOTS` | empty | Comma-separated roots allowed for local path and `file://` ingest. Paths are canonicalized at startup. |
| `VIDARAX_CONFIG` | `vidarax.toml` | Backend TOML path, used when explicit backend URLs are not set. |
| `VIDARAX_VLLM_BASE_URL` | unset | vLLM OpenAI-compatible base URL. When set, it is used as priority 1. |
| `VIDARAX_SGLANG_BASE_URL` | unset | SGLang OpenAI-compatible base URL. When set, it is used as priority 2. |
| `VIDARAX_DECODE_BACKEND` | `auto` | Video decode backend. Accepts `cpu`, `ffmpeg`, `cpu-ffmpeg`, `nvdec`, `cuda`, `nvdec-cuda`, `gpu`, `mlx`, `apple`, `metal`, or `videotoolbox`. |
| `VIDARAX_FFMPEG_PATH` | `ffmpeg` | ffmpeg binary path. |
| `VIDARAX_FFPROBE_PATH` | `ffprobe` | ffprobe binary path. |
| `VIDARAX_NVIDIA_SMI_PATH` | `nvidia-smi` | Binary used by decode auto-detection to find NVIDIA hardware. |
| `VIDARAX_REQUIRE_API_KEY` | `true` | Require `x-api-key` on API routes other than `/v1/health`. If true, `VIDARAX_API_KEYS` must contain at least one key. |
| `VIDARAX_API_KEYS` | empty | Comma-separated API keys. |
| `VIDARAX_REQUIRE_TENANT_ID` | `false` | Require `x-tenant-id` metadata. The header is not an authorization boundary. The server rejects this when API keys are disabled. |
| `VIDARAX_METRICS_REQUIRE_API_KEY` | `true` | Require `x-api-key` for `/v1/metrics`. |
| `VIDARAX_RATE_LIMIT_GLOBAL_RPS` | unset | Optional global request limit per second. Parsed as an unsigned integer and clamped internally to at least 1 if set. |
| `VIDARAX_RATE_LIMIT_TENANT_RPS` | unset | Optional per-principal request limit per second. Despite the historical name, the bucket key is the resolved principal. |
| `VIDARAX_RATE_LIMIT_TENANT_SLOTS` | `2048` | Maximum tracked principal buckets for the per-principal limiter. Must be at least 1 when the limiter is enabled. |
| `VIDARAX_CORS_ALLOWED_ORIGINS` | empty | Comma-separated browser origins. Empty means no origin is allowed. `*` is rejected when API keys are required. |
| `VIDARAX_STREAM_TTL_SECS` | `3600` | Idle TTL for active runs. Must be in `[60, 86400]`. |
| `VIDARAX_ACTIVE_STREAM_LIMIT` | `5` | Max active runs per resolved principal. Clamped to `[1, 1024]`. |
| `VIDARAX_WEBRTC_STUN_SERVERS` | `stun:stun.l.google.com:19302` | Comma-separated STUN server URIs. |
| `VIDARAX_WEBRTC_TURN_URL` | unset | Optional TURN relay URL. |
| `VIDARAX_WEBRTC_TURN_USERNAME` | unset | Optional TURN username. |
| `VIDARAX_WEBRTC_TURN_CREDENTIAL` | unset | Optional TURN credential. |
| `VIDARAX_WEBRTC_MAX_OUTPUT_TOKENS_PER_SECOND` | `128` | VLM output token rate cap per WebRTC session. |
| `VIDARAX_WEBRTC_DECODE_WORKERS` | `1` | Loaded for compatibility, clamped to 1. One ordered stream uses one stateful decoder. |
| `VIDARAX_WEBRTC_ANALYSIS_WORKERS` | `1` | Loaded for compatibility, clamped to 1. Analysis owns stream-order state. |
| `VIDARAX_WEBRTC_VLM_WORKERS` | `1` | Loaded for compatibility, clamped to 1. VLM processing owns temporal state. |
| `VIDARAX_WEBRTC_FIRST_PASS_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Local first-pass VLM for WebRTC keyframes. Must be a supported model id. |
| `VIDARAX_WEBRTC_SECOND_PASS_MODEL` | unset | Escalation model id. Set to a distinct supported id (e.g. a `gemini` backend's model) to enable tiering; unset or equal to the first pass keeps sessions local-only. |
| `VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD` | `0.7` | Escalate when first-pass confidence is below this. Clamped to `[0.0, 1.0]`; non-finite values fall back to the default. |
| `VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS` | `256` | Output token cap for the escalation pass. |
| `VIDARAX_DISTILL_ENABLED` | `false` | Enable distillation sample collection. |
| `VIDARAX_DISTILL_EMBEDDING_URL` | unset | Optional embedding server URL. |
| `VIDARAX_DISTILL_TEACHER_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Teacher VLM model used for distillation labels. |
| `VIDARAX_DISTILL_MAX_PAIRS` | `10000` | Max distillation pairs per tenant. Clamped to `[100, 1000000]`. |
| `VIDARAX_DISTILL_COLLECTION_RATE` | `0.1` | Fraction of keyframes sampled for distillation. Clamped to `[0.0, 1.0]`. |
| `VIDARAX_DISTILL_DISTANCE_THRESHOLD` | `0.2` | KNN distance accept threshold. Clamped to `[0.0, 2.0]`. |
| `VIDARAX_DISTILL_KNN_K` | `7` | K for KNN classification. Clamped to `[1, 100]`. |
| `VIDARAX_ALLOW_REMOTE_HLS` | `false` | Allow remote HLS manifests. Keep disabled unless manifests are trusted. |
| `VIDARAX_ALLOW_INSECURE_HTTP` | `false` | Allow `http://` media sources and redirects. |
| `VIDARAX_ALLOW_UNENCRYPTED_RTSP` | `false` | Allow `rtsp://` camera sources. |
| `VIDARAX_ALLOW_INSECURE_TLS` | `false` | Omit ffmpeg TLS verification arguments for supported live sources. |
| `VIDARAX_TENANT_LABEL_MAPS_PATH` | unset | Optional JSON file for event and object label mapping by tenant metadata. |
| `VIDARAX_SPACETIMEDB_URL` | unset | SpacetimeDB base URL (for example `http://127.0.0.1:3000`). When set, the feedback endpoints are enabled and WHIP stream events are written to SpacetimeDB; when unset, stream events use the local WAL and the feedback endpoints return an error. |
| `VIDARAX_SPACETIMEDB_MODULE` | `vidarax` | SpacetimeDB database/module name. Only used when `VIDARAX_SPACETIMEDB_URL` is set. |
| `RUST_LOG` | `info` | Tracing filter used by `tracing_subscriber`. |
| `VIDARAX_TRACES_ENDPOINT` | unset | Optional OTLP gRPC endpoint for trace export. |

The `VIDARAX_STAGING_*` names in the repository are live-test fixtures, not
server deployment configuration. Set `VIDARAX_SPACETIMEDB_URL` to enable the
SpacetimeDB feedback endpoints and route WHIP stream events to SpacetimeDB;
leave it unset to keep events in the local WAL.

## Build and run

Build the API server:

```bash
cargo build --release -p vidarax-api
```

Run with API-key authentication enabled:

```bash
VIDARAX_API_KEYS=dev-key \
VIDARAX_VLLM_BASE_URL=http://127.0.0.1:8000 \
target/release/vidarax-api
```

For local open-mode development, disable API-key checks explicitly:

```bash
VIDARAX_REQUIRE_API_KEY=false \
VIDARAX_METRICS_REQUIRE_API_KEY=false \
VIDARAX_VLLM_BASE_URL=http://127.0.0.1:8000 \
cargo run --release -p vidarax-api
```

The server creates `VIDARAX_DATA_DIR` if needed and stores the primary timeline
at `timeline.wal`. Uploaded files are stored under a dedicated upload directory
below the process temp directory. Shared local media paths are not enabled by
default; set `VIDARAX_INGEST_FILE_ROOTS` to the directories operators trust.

`GET /v1/health` returns readiness for the running HTTP server. It does not
check model backend availability.

## Backend configuration

There are two backend configuration paths.

Set explicit URLs for the common case:

```bash
VIDARAX_VLLM_BASE_URL=http://127.0.0.1:8000
VIDARAX_SGLANG_BASE_URL=http://127.0.0.1:30000
```

If either explicit URL is set, the server builds a provider chain from those
URLs and does not read `VIDARAX_CONFIG`.

When neither URL is set, the server reads `VIDARAX_CONFIG`, defaulting to
`vidarax.toml`. The current parser supports `openai_compat` backends and
`gemini` backends. String fields in that file support `${ENV_VAR}`
interpolation.

The video decode backend is separate from the VLM backend. `cpu-ffmpeg` works
with ffmpeg and ffprobe on `PATH`. `nvdec-cuda` uses ffmpeg with NVDEC for the
JPEG phase. `mlx`, `apple`, `metal`, and `videotoolbox` are aliases for the
same backend, which uses ffmpeg's `-hwaccel videotoolbox` for the JPEG phase
on Apple Silicon. As with `nvdec-cuda`, the frame-signal phase runs on CPU
regardless of backend: it decodes to `framemd5` text output, which has no
hardware-decode benefit. This backend requires an ffmpeg binary built with
VideoToolbox support; if the configured `ffmpeg` lacks it, decode fails at
runtime with an error rather than silently falling back to the CPU pipeline.

## Deployment dependencies

A useful deployment needs:

- An OpenAI-compatible VLM backend, usually vLLM or SGLang, reachable at the
  configured base URL. Without a backend, inference routes fail and WHIP uses a
  no-provider fallback path for live sessions.
- `ffmpeg` and `ffprobe` on `PATH`, or paths set through `VIDARAX_FFMPEG_PATH`
  and `VIDARAX_FFPROBE_PATH`.
- Network egress controls for untrusted remote media. Application-level SSRF
  checks are documented in [security.md](security.md).
- Optional HTTP/3 TLS certificate and key when `VIDARAX_TRANSPORT=h3`. The
  binary must be built with `--features h3-experimental`; otherwise the server
  rejects H3 transport at startup.
- Optional SpacetimeDB. Set `VIDARAX_SPACETIMEDB_URL` to attach a client at
  startup: feedback endpoints are enabled and WHIP stream events go to
  SpacetimeDB. Leave it unset and `run()` keeps events in the local WAL with the
  feedback endpoints disabled.

## Docker and compose

`deploy/Dockerfile.api` builds `vidarax-api` in a Rust builder image and copies
the binary into a Debian runtime image. The runtime image sets:

```bash
VIDARAX_BIND_ADDR=0.0.0.0:8080
VIDARAX_H3_BIND_ADDR=0.0.0.0:8443
VIDARAX_DATA_DIR=/var/lib/vidarax
```

`deploy/docker-compose.local.yml` builds that Dockerfile, exposes the API on
`127.0.0.1:8080`, mounts a named volume for `/var/lib/vidarax`, and starts
VictoriaMetrics, VictoriaLogs, and VictoriaTraces. The compose file sets an
OTLP traces endpoint and a local-dev placeholder API key (the same value the
metrics scrape sends as its `x-api-key` header), but it does not configure an
inference backend. Point `VIDARAX_VLLM_BASE_URL`/`VIDARAX_SGLANG_BASE_URL` at a
running backend, and replace the placeholder key, before using the API anywhere
reachable by anyone but you.

Check readiness with:

```bash
curl -fsS http://127.0.0.1:8080/v1/health
```

## Security

Read [security.md](security.md) before exposing a deployment to untrusted
callers or untrusted media.

The main hardening knobs are:

- Keep `VIDARAX_REQUIRE_API_KEY=true` and issue separate API keys for isolated
  principals.
- Set `VIDARAX_CORS_ALLOWED_ORIGINS` to exact browser origins. Do not use `*`
  with authenticated deployments.
- Configure `VIDARAX_RATE_LIMIT_GLOBAL_RPS` and
  `VIDARAX_RATE_LIMIT_TENANT_RPS` for public endpoints. The per-principal
  limiter still uses the historical `TENANT` variable name.
- Terminate TLS at a proxy or use the experimental HTTP/3 TLS settings.
- Keep insecure media toggles disabled unless the source network is trusted.

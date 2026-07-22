# Deployment

Vidarax reads server configuration from environment variables and model backend
configuration from either explicit URLs or `vidarax.toml`. The defaults below
come from `crates/vidarax-api/src/config.rs`.

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
| `VIDARAX_INFERENCE_GLOBAL_LIMIT` | `8` | Maximum concurrent provider calls across the process. |
| `VIDARAX_INFERENCE_PER_PRINCIPAL_LIMIT` | `4` | Maximum concurrent provider calls for one authenticated principal. |
| `VIDARAX_INFERENCE_WAITER_LIMIT` | `128` | Maximum queued provider calls. |
| `VIDARAX_INFERENCE_WAIT_TIMEOUT_MS` | `5000` | Maximum admission wait before provider dispatch. |
| `VIDARAX_INFERENCE_TOKEN_BUDGET` | `32768` | Aggregate output-token reservation across active calls. |
| `VIDARAX_INFERENCE_BYTE_BUDGET` | `268435456` | Aggregate encoded image/video bytes reserved by active calls. |
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
| `VIDARAX_MEDIA_MEMORY_BUDGET_BYTES` | `8589934592` | Process-wide live media reservation budget. Must be between 256 MiB and 1 TiB. Session admission reserves a conservative payload envelope before workers start. |
| `VIDARAX_MEDIA_WORKER_THREAD_BUDGET` | `64` | Process-wide live media OS-thread budget. Clamped to `[1, 4096]`. H.264/H.265 keyframe generations reserve 4 threads. Clip generations reserve 5, including the ffmpeg stdout reader. |
| `VIDARAX_WEBRTC_STUN_SERVERS` | `stun:stun.l.google.com:19302` | Comma-separated STUN server URIs. |
| `VIDARAX_WEBRTC_TURN_URL` | unset | Optional TURN relay URL. |
| `VIDARAX_WEBRTC_TURN_USERNAME` | unset | Optional TURN username. |
| `VIDARAX_WEBRTC_TURN_CREDENTIAL` | unset | Optional TURN credential. |
| `VIDARAX_WEBRTC_CROP` | unset | Optional `x,y,width,height` analysis crop as normalized fractions. |
| `VIDARAX_WEBRTC_MAX_OUTPUT_TOKENS_PER_SECOND` | `128` | VLM output token rate cap per WebRTC session. |
| `VIDARAX_WEBRTC_DECODE_WORKERS` | `1` | Loaded for compatibility, clamped to 1. One ordered stream uses one stateful decoder. |
| `VIDARAX_WEBRTC_ANALYSIS_WORKERS` | `1` | Loaded for compatibility, clamped to 1. Analysis owns stream-order state. |
| `VIDARAX_WEBRTC_VLM_WORKERS` | `1` | Loaded for compatibility, clamped to 1. VLM processing owns temporal state. |
| `VIDARAX_WEBRTC_FIRST_PASS_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Local first-pass VLM for WebRTC keyframes. Must be a supported model id. |
| `VIDARAX_WEBRTC_SECOND_PASS_MODEL` | unset | Escalation model id. Set to a distinct supported id (e.g. a `gemini` backend's model) to enable tiering. Unset or equal to the first pass keeps sessions local-only. |
| `VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD` | `0.7` | Escalate when first-pass confidence is below this. Clamped to `[0.0, 1.0]`. Non-finite values fall back to the default. |
| `VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS` | `256` | Output token cap for the escalation pass. |
| `VIDARAX_GATE_KEEPALIVE_EVERY_FRAMES` | `30` | Select a periodic frame after this many frames since the last committed selection. |
| `VIDARAX_GATE_SCENE_CUT_HAMMING_THRESHOLD` | `18` | Perceptual-hash Hamming distance that selects a scene cut. |
| `VIDARAX_GATE_LUMA_SHIFT_THRESHOLD` | `0.15` | Mean-luma change that marks an exposure shift. |
| `VIDARAX_GATE_FLICKER_THRESHOLD` | `0.55` | Flicker score that marks a suspected artifact. |
| `VIDARAX_GATE_GHOSTING_THRESHOLD` | `0.55` | Ghosting score that marks a suspected artifact. |
| `VIDARAX_GATE_NOISE_VARIANCE_THRESHOLD` | `0.55` | Noise-variance score that marks a suspected artifact. |
| `VIDARAX_NOVELTY_EMBEDDING_ADDR` | unset | Binary TCP embedding sidecar (`host:port` or `tcp://host:port`). Setting it enables live novelty. |
| `VIDARAX_NOVELTY_MAX_REUSE_MS` | `2000` | Maximum capture-time gap from the last described frame. `0` disables reuse. |
| `VIDARAX_NOVELTY_MAX_CUMULATIVE_DRIFT` | `0.50` | Refresh after reuse scores accumulate to this value. |
| `VIDARAX_NOVELTY_SHADOW_SAMPLE_RATE` | `0.01` | Fraction of reuse decisions sampled through the VLM. |
| `VIDARAX_NOVELTY_EMBEDDING_TIMEOUT_MS` | `2000` | Sidecar deadline. Failure runs the VLM. |
| `VIDARAX_NOVELTY_REUSE_THRESHOLD` | `0.01` | Reuse at or below this embedding-distance score. Must be in `[0,1)`. Treat the default as a conservative starting point and calibrate it on labelled deployment traffic. |
| `VIDARAX_ALLOW_REMOTE_HLS` | `false` | Allow remote HLS manifests. Keep disabled unless manifests are trusted. |
| `VIDARAX_ALLOW_INSECURE_HTTP` | `false` | Allow `http://` media sources and redirects. |
| `VIDARAX_ALLOW_UNENCRYPTED_RTSP` | `false` | Allow `rtsp://` camera sources. |
| `VIDARAX_ALLOW_INSECURE_TLS` | `false` | Omit ffmpeg TLS verification arguments for supported live sources. |
| `VIDARAX_TENANT_LABEL_MAPS_PATH` | unset | Optional JSON file for event and object label mapping by tenant metadata. |
| `VIDARAX_SPACETIMEDB_URL` | unset | Adds a best-effort feedback and blocking-description mirror after local WAL commit. Feedback works without it. Raw keyframes stay local. |
| `VIDARAX_SPACETIMEDB_MODULE` | `vidarax` | SpacetimeDB database/module name. Only used when `VIDARAX_SPACETIMEDB_URL` is set. |
| `VIDARAX_WEBHOOK_SECRET` | unset | Enables outbound webhooks when set to at least 32 bytes. It derives a distinct signing key per webhook and is never written to the timeline WAL or returned by the API. |
| `VIDARAX_TRIGGER_LOCAL_OUTPUT_SOCKET` | unset | Absolute Unix datagram socket for metadata-only `notify local_output` trigger actions. Required when such a program is attached. |
| `RUST_LOG` | `info` | Tracing filter used by `tracing_subscriber`. |
| `VIDARAX_TRACES_ENDPOINT` | unset | Optional OTLP gRPC endpoint for trace export. |

For an OpenAI-compatible backend that serves a converted model id, set both
`model` and `upstream_model` in its TOML entry. `model` is the curated Vidarax
id exposed to clients. `upstream_model` is the backend-specific id sent on the
wire. This is required for mlx-vlm quantized conversions. `GET /v1/models`
probes configured backends and reports readiness per curated model.

The `VIDARAX_STAGING_*` names are live-test fixtures, not server settings.
WHIP events always commit locally. SpacetimeDB receives descriptions only.

Webhook registration is disabled until `VIDARAX_WEBHOOK_SECRET` is configured.
The creation response returns each derived verification key once. Save it with
that receiver. Rotate the root as an application credential and recreate hooks
before restarting Vidarax. Webhook targets must
use HTTPS and resolve only to public addresses. Vidarax pins the validated
addresses for each attempt, disables proxies and redirects, and applies a
three-second total attempt deadline.

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
default. Set `VIDARAX_INGEST_FILE_ROOTS` to the directories operators trust.

Keyframe JPEGs are stored at
`keyframes/blobs/<sha-prefix>/<sha256>.jpg` through an atomic rename before the
`keyframe_stored` event is appended. `image_ref` is relative to
`VIDARAX_DATA_DIR`. The event also records media type, byte count, and SHA-256.
Identical JPEGs share a blob. Writes are flushed but not fsynced per keyframe.

## Live semantic novelty calibration

Start the bundled SigLIP2 sidecar on the embedding accelerator:

```bash
python3 scripts/embedding_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --device auto \
  --batch-size 8 \
  --batch-wait-ms 3 \
  --max-queue-mb 64
```

Set `VIDARAX_NOVELTY_EMBEDDING_ADDR=127.0.0.1:8765`. The client sends raw JPEGs
and receives 768 little-endian f32 values over a persistent TCP connection.
The sidecar batches requests across streams. Any sidecar error runs the VLM.
Reuse is limited by capture time and cumulative drift.

The default 1% shadow sample checks reuse decisions without updating state or
emitting events. Compare sampled and completed totals to find provider failures.
Treat `vidarax_pipeline_novelty_shadow_change_ratio` as a warning because
description overlap can classify paraphrases as changes.

Activation is not calibration. Label at least 30 ordered JPEGs as `novel` or
`redundant`. Include frozen content, slow drift, hard cuts, overlays, low-light
frames, and repeated motion. A single natural sequence is not enough to expose
false reuse. Measure the deployment's VLM p50, and run:

```bash
CARGO_TARGET_DIR=/tmp/vidarax-calibration-target \
cargo run -p vidarax-core --release --example novelty_live_calibration -- \
  127.0.0.1:8765 \
  /tmp/vidarax-novelty-labels.tsv \
  800 2000 0.50 0.98 1.10 \
  > /tmp/vidarax-novelty-calibration.txt
```

The final values are VLM p50 ms, maximum reuse ms, drift budget, minimum recall,
and minimum speedup. The command fails if no threshold meets both floors. Set
`VIDARAX_NOVELTY_REUSE_THRESHOLD` to the selected value, then confirm it with
live shadow samples. A file-based `/reason` benchmark does not exercise the
live semantic filter. Its report must mark novelty as not applicable.

Provider and hardware results must remain separate. Create a TSV outside the
repository with one configured deployment per row:

```text
name<TAB>provider<TAB>hardware<TAB>decode_backend<TAB>api<TAB>model<TAB>resolution
```

Then run:

```bash
VIDARAX_API_KEY='deployment-specific-key' \
python3 benchmarks/provider_hardware_matrix.py \
  --matrix /tmp/vidarax-provider-hardware.tsv \
  --preset clip_balanced \
  --warmups 1 \
  --repeats 3 \
  --min-f1 0.50 \
  --max-errors 0 \
  --output /tmp/vidarax-provider-hardware-matrix.json
```

Omit `VIDARAX_API_KEY` only for an explicitly open local server. The output
keeps every measured run plus aggregate quality, tokens, wall-clock p50/p95,
provider-latency histogram bounds, request counts, decoded-frame and gate
selection counts, decode and gate mean latency, semantic-novelty counters when
applicable, and errors for each row. The command fails on missing provider
calls, excess errors, or low mean F1. Keep unavailable providers and
accelerators as explicit gaps. Never infer their results from another row.

`GET /v1/health` returns readiness for the running HTTP server. It does not
check model backend availability.

Live-session admission also enforces a process capacity plan. Each negotiated
generation reserves bounded RTP queue bytes, decoded-frame pool bytes, JPEG
payload bytes, provider scratch space, a 64 MiB ffmpeg-process allowance, and
its fixed worker count before `run_created` is appended. The reservation is
released with the session. Watch the `vidarax_media_capacity_*` metrics when
sizing the two media budget settings. A rejected reservation returns `503
media process capacity exhausted` without creating a durable run.

Provider admission is a separate process-wide scheduler. Urgent live
keyframes, normal live clips/direct calls, and offline timeline work have fixed
latency classes. Selection preserves each stream's own ordered worker, rotates
away from the last principal and stream when peers are waiting, and ages older
classes so offline work cannot starve. Dispatch is refused when its remaining
deadline cannot cover the estimated provider service time, or when its token or
encoded-media reservation cannot fit. The corresponding active reservations,
deadline misses, budget rejections, and acquisitions by class are exported at
`GET /v1/metrics`.

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

The same file may define one device-level restricted-zone policy. It applies
to WHIP sessions that do not provide a stream-specific policy:

```toml
[restricted_zone]
policy_id = "loading-dock-after-hours"
policy_version = 3
device_id = "camera-west-02"
enter_motion_score = 0.12
exit_motion_score = 0.04
enter_after_frames = 2
exit_after_frames = 4

[restricted_zone.region]
x = 0.10
y = 0.20
width = 0.70
height = 0.60
```

Coordinates are normalized to the decoded image. The region becomes the exact
analysis crop for that pipeline generation. A WHIP attach may replace the
device policy, but a separately supplied crop must match the selected region
and clip mode cannot be enabled with restricted-zone activity detection.
Policy changes take effect on a new generation. An active worker never reads
mutable process-wide policy state.

The video decode backend is separate from the VLM backend. `cpu-ffmpeg` works
with ffmpeg and ffprobe on `PATH`. `nvdec-cuda` uses ffmpeg with NVDEC for the
JPEG phase. `mlx`, `apple`, `metal`, and `videotoolbox` are aliases for the
same backend, which uses ffmpeg's `-hwaccel videotoolbox` for the JPEG phase
on Apple Silicon. As with `nvdec-cuda`, the frame-signal phase runs on CPU
regardless of backend: it decodes to `framemd5` text output, which has no
hardware-decode benefit. This backend requires an ffmpeg build with VideoToolbox
support to accelerate the selective JPEG phase. ffmpeg may fall back to software
decode when VideoToolbox recognizes the accelerator but cannot initialize it for
the current input. The server records that fallback in logs.

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
  binary must be built with `--features h3-experimental`. The server rejects H3
  transport at startup when that feature is absent.
- Optional SpacetimeDB as an additive feedback and description mirror. The
  local WAL and JPEG blobs remain the source of record.

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

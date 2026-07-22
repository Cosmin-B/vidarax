# Security Notes

## Principal-Based Ownership

This release introduces principal-based ownership for runs and uploaded files.
Authenticated API-key callers own data as `api-key:<sha256(api_key)>`.
Unauthenticated/open-mode callers use the shared `public` principal.

The caller-controlled `x-tenant-id` header is not an ownership input and must
not be used as a compatibility path for authorization. It can be logged as
request metadata, but ownership, file visibility, rate limiting, and tenant
label-map selection derive from the authenticated principal.

Tenant isolation requires `VIDARAX_REQUIRE_API_KEY=true` and distinct API keys
issued per tenant. `VIDARAX_REQUIRE_TENANT_ID=true` only requires callers to
include tenant metadata; it does not create isolation without authenticated API
keys, so the server rejects that configuration.

Events and files created before ownership existed do not carry a
`principal_key`. Those records are treated as the shared `public` namespace:
visible in open mode, and not attributed to any authenticated API-key principal
when API-key authentication is enabled.

The earlier `api-key:<fnv1a64>` and `tenant:<id>` principal formats were
introduced only within this same unreleased change set. They were never
shipped, so there is no production data in those formats to migrate or trust.

## Outbound Webhooks

Webhook registration inherits run ownership: a principal can list, create, or
delete hooks only for its own run. `VIDARAX_WEBHOOK_SECRET` is a process-level
HMAC-SHA256 derivation root and must contain at least 32 bytes. It is held in
memory and never persisted in the timeline. Each webhook receives a distinct
derived key, returned only in its creation response. Each request carries
`x-vidarax-event-id: <run_id>:<seq>` and
`x-vidarax-signature: v1=<hex-hmac>` over the exact request body.

Targets must be HTTPS URLs without credentials or fragments. Registration and
every delivery attempt resolve the hostname and reject private, loopback,
link-local, documentation, multicast, and cloud-metadata address ranges. The
HTTP client pins the validated addresses, bypasses ambient proxies, refuses all
redirects, and limits the complete attempt to three seconds. Operators should
still restrict process egress to expected receiver networks as defense in
depth.

Webhook bodies are CloudEvents-compatible event metadata. Binary keyframes and
clips are never embedded or base64-encoded; the existing content-addressed
sidecar reference remains in `data`. Receivers must fetch media through the
authenticated blob route when needed.

## Ingest File Ownership

Files uploaded through `POST /v1/upload` are owned by the authenticated
principal that uploaded them. The server encodes that owner in the stored
filename prefix and only serves or ingests the file for that same principal.

Uploads are stored under a dedicated upload root below the process temp
directory. Upload-root ownership always takes precedence: if an
operator-configured ingest root contains or overlaps the upload root, files
under the upload root still require the upload owner prefix or the shared
`public` namespace.

`VIDARAX_INGEST_FILE_ROOTS` defaults to empty. The system temp directory is not
an implicit shared ingest root. Operators must explicitly configure any shared
media roots that should be readable or ingestable by authenticated callers.

Operator-configured ingest roots, meaning non-upload paths listed in
`VIDARAX_INGEST_FILE_ROOTS`, are admin-trusted shared media roots. Files under
those roots may be served or ingested by any authenticated caller. Do not point
an operator shared root at untrusted per-user writeable storage.

Legacy unprefixed files in the upload temp root predate the ownership model and
are not auto-claimed by authenticated callers. Re-upload them through
`POST /v1/upload` to attach principal ownership.

Legacy files written directly under the system temp directory are quarantined
by default: they are neither under the dedicated upload root nor under an
operator-configured shared root. If a deployment intentionally depended on
direct temp-dir media, migrate those files into a dedicated directory and list
that directory in `VIDARAX_INGEST_FILE_ROOTS`.

## Remote Media SSRF Residual

For downloadable `http://` and `https://` media, VidaraX does not let ffmpeg
perform remote network I/O. The API first validates the original URL, fetches it
with a controlled blocking HTTP client, validates every redirect target by
resolving the target host and rejecting non-global addresses, enforces a small
redirect limit, rejects non-HTTP(S) redirect targets, rejects HTTPS-to-HTTP
downgrades, streams the response to a bounded temp file, rejects `#EXTM3U`
playlist bodies, rejects playlist demuxers with file-only `ffprobe`, and then
decodes the local temp file with a file-only ffmpeg protocol whitelist. Plain
HTTP sources and redirects require `VIDARAX_ALLOW_INSECURE_HTTP=true`. The
prefetch temp file is removed after decode on a best-effort basis.

Current prefetch limits:

- maximum downloaded body: 256 MiB
- total client timeout: 30 seconds
- maximum redirects: 5

Application-level mitigations already in place:

- Initial source validation rejects embedded credentials, localhost/local
  domains, private/loopback/link-local/metadata IP literals, and hostnames that
  resolve to blocked IP ranges.
- IPv4-mapped IPv6 addresses are unwrapped before blocked-range checks.
- ffmpeg keeps the original hostname instead of rewriting to a resolved IP, so
  TLS SNI, certificate identity, and HTTP Host routing remain intact for live
  stream paths.
- Direct ffmpeg-handled `rtsps://` inputs pass `-tls_verify 1` and
  `-verifyhost <original-host>` by default. Opt-in `https://` remote HLS
  inputs pass `-tls_verify 1` without manifest-host pinning, so absolute
  variant, segment, key, and map URLs on other CDN hosts can validate against
  their own peer certificates. Operators with self-signed/internal cameras can
  set `VIDARAX_ALLOW_INSECURE_TLS=true` to omit those ffmpeg TLS verification
  arguments for those sources.
- ffmpeg protocol whitelists are scoped by source type; `file:`, plain
  `http:`, and remote HLS are disabled by default.
- downloadable HTTP(S) media is decoded only after validated local prefetch, so
  content-sniffed HLS manifests cannot make ffmpeg fetch segment, key, map, or
  variant subresources.

Residual accepted risk: RTSP/RTSPS live streams cannot be prefetched. Remote HLS
is disabled by default and remains opt-in for trusted manifests. For live
streams and any opt-in remote-HLS ingestion, fully preventing redirect or
nested-resource SSRF to private IPs over allowed schemes requires NETWORK-LEVEL
EGRESS CONTROL: run those media paths behind an egress proxy, firewall, or
resolver policy that blocks RFC1918, link-local, loopback, and cloud metadata
ranges on every outbound connection.

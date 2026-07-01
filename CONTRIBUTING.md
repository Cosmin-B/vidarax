# Contributing

Vidarax is a Rust workspace with `vidarax-core` for ingest, decode, gate, and
VLM pipeline logic, `vidarax-api` for the Axum API and WHIP handlers,
`vidarax-contracts` for shared request and response contracts, and
`vidarax-cli` for CLI tools. The Vue 3 UI lives in `ui/`, the TypeScript SDK in
`packages/vidarax-sdk/`, and the SpacetimeDB module in `spacetime-module/`.

## Build and test

Rust workspace:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
```

API release build:

```bash
cargo build --release -p vidarax-api
```

UI:

```bash
cd ui
npm install
npm run build
npm run test:e2e
```

SDK:

```bash
cd packages/vidarax-sdk
npm install
npm run build
npm test
```

Live tests need the matching local services: a VLM backend such as vLLM or
SGLang for inference, ffmpeg and ffprobe on `PATH` for decode, and SpacetimeDB
when running the module or parity tests that depend on it. MLX can be selected
as a local decode backend on Apple Silicon, but the current registered backend
falls back to CPU ffmpeg.

## Review and style

Changes are reviewed before merge. Keep changes scoped and include the command
output or failure reason for the checks you ran.

Comments should explain invariants, constraints, or non-obvious performance
decisions. Do not narrate line-by-line behavior.

Do not add performance numbers without the benchmark setup and input data. The
decode, analysis, and VLM per-frame paths should avoid per-frame allocation,
locks, or CAS unless the change includes a measured reason.

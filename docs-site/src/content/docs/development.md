---
title: Development
description: Workspace layout, how to build and test, and contributing basics.
---

Vidarax is a Rust workspace with a TypeScript SDK, a Vue 3 UI, and a SpacetimeDB module beside it.

## Workspace layout

```
crates/
  vidarax-core/         Lock-free primitives, gate engine, ingest pipeline
  vidarax-contracts/    Shared model contracts and error mapping
  vidarax-api/          Axum HTTP server, handlers, WHIP, security
  vidarax-cli/          CLI tooling
ui/                     Vue 3 frontend
packages/vidarax-sdk/   TypeScript SDK
spacetime-module/       SpacetimeDB server module
docs/                   Architecture docs, runbooks, specs
deploy/                 Docker, compose, certificates
scripts/                Benchmarks, smoke tests, release gates
schemas/                JSON Schemas for frame metadata and processing config
examples/               SDK usage examples
```

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

Live tests need the matching local services: a VLM backend such as vLLM or SGLang for inference, `ffmpeg` and `ffprobe` on `PATH` for decode, and SpacetimeDB when running the module or the parity tests that depend on it.

Before shipping, run the release gates described in [Operations](/operations/#release-gates).

## Contributing basics

Changes are reviewed before merge. Keep changes scoped, and include the command output or the failure reason for the checks you ran.

Comments should explain invariants, constraints, or non-obvious decisions, not narrate line-by-line behavior. Performance claims in code or docs must come with the benchmark setup and input data that produced them.

The project has an explicit concurrency and memory policy for hot paths:

- Favor lock-free and wait-free structures on hot paths; if a lock is unavoidable, document why lock-free was rejected and what the contention model is.
- Prefer bounded FIFO, SPSC, and MPSC queues with explicit backpressure.
- Avoid per-frame heap allocation in the core ingest and gate loops; pre-allocate for throughput pipelines. Any new hot-path allocation needs explicit justification and measurement.

When weighing a change, the project's ordered checklist applies: avoid the work entirely, do it once, do it fewer times, approximate safely, use a lookup table or bounded queue, constrain the problem, delete dead code, and only then reach for vectorization. Back decisions with data.

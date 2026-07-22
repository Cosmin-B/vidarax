---
title: Development
description: Workspace layout, how to build and test, and contributing basics.
---

Vidarax is a Rust workspace with a TypeScript SDK, a Vue 3 UI, and a SpacetimeDB module beside it.

## Workspace layout

```
crates/
  vidarax-core/         Frame filter, media primitives, ingest pipeline
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

Live tests need the matching local services. Inference tests need a VLM backend
such as vLLM or SGLang. Decode tests need `ffmpeg` and `ffprobe` on `PATH`.
SpacetimeDB is required only for its module and mirror parity tests.

Before shipping, run the [release checks](/docs/operations/#release-checks).

## Contributing basics

Changes are reviewed before merge. Keep changes scoped. Include the command
output or the failure reason for the checks you ran.

Comments should explain invariants, constraints, or non-obvious decisions.
Line-by-line narration adds no useful context. Performance claims in code or
docs must include the benchmark setup and input data that produced them.

The project has an explicit concurrency and memory policy for hot paths:

- Favor lock-free and wait-free structures on hot paths. If a lock is
  unavoidable, document the contention model and why the lock stayed.
- Prefer bounded FIFO, SPSC, and MPSC queues with explicit backpressure.
- Avoid selected-global-allocator calls per frame after warmup in the core
  ingest and filter loops. Reuse bounded buffers in throughput pipelines. Any
  new hot-path allocation needs explicit justification and measurement.

When weighing a change, the project's ordered checklist applies: avoid the work entirely, do it once, do it fewer times, approximate safely, use a lookup table or bounded queue, constrain the problem, delete dead code, and only then reach for vectorization. Back decisions with data.

## Documentation site

The documentation is an Astro/Starlight static build:

```bash
cd docs-site
npm ci
npm run build
```

The build writes `docs-site/dist`. The canonical site mounts that directory at
`https://vidarax.cosminbararu.com/docs/` inside one Cloudflare Workers static
asset deployment with the product landing page. The Cloudflare Worker serves
documentation only. Vidarax API traffic, control-plane state, and media never
pass through it.

Deploy the combined asset tree from the website workspace. Deploying a
standalone landing-page directory would remove `/docs/` from the same Worker
asset set.

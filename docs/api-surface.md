# API surface

The maintained endpoint reference is the docs-site source at
[`docs-site/src/content/docs/api.md`](../docs-site/src/content/docs/api.md).
Keeping the contracts in one place avoids a second route table drifting away
from `crates/vidarax-api/src/router.rs` and the handler models.

For related details:

- Deployment and environment variables: [`deployment.md`](deployment.md)
- Authentication and ingest hardening: [`security.md`](security.md)
- Event payloads and SDK methods:
  [`docs-site/src/content/docs/events.md`](../docs-site/src/content/docs/events.md)
- Trigger language, validation, and replay:
  [`docs-site/src/content/docs/triggers.md`](../docs-site/src/content/docs/triggers.md)
- Policy revision and rollout semantics:
  [`docs-site/src/content/docs/policy-rollouts.md`](../docs-site/src/content/docs/policy-rollouts.md)
- Signed edge release workflow: [`edge-deployment.md`](edge-deployment.md)

Build the rendered reference from `docs-site/` with `npm run build`.

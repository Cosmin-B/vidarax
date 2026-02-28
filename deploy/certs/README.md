Development TLS assets are generated locally and never committed.

Generate a short-lived cert/key pair:

```bash
make dev-cert
```

Generated files:

- `deploy/certs/dev.crt`
- `deploy/certs/dev.key`

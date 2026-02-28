.PHONY: lint test validate bench bench-api bench-provider staging-e2e release-gates h3-check smoke smoke-mp4 provision-models dev-cert

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

validate:
	./scripts/validate_replay_and_schema.sh

bench:
	./scripts/bench_regression.sh

bench-api:
	./scripts/bench_api_path.sh

bench-provider:
	./scripts/bench_provider_transport.sh

staging-e2e:
	./scripts/staging_provider_e2e.sh

release-gates:
	./scripts/release_gates.sh

h3-check:
	cargo check -p vidarax-api --features h3-experimental

smoke:
	./scripts/smoke_v1.sh

smoke-mp4:
	./scripts/smoke_mp4_pipeline.sh

provision-models:
	./scripts/provision_models.sh

dev-cert:
	@mkdir -p deploy/certs
	@openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 7 \
		-keyout deploy/certs/dev.key \
		-out deploy/certs/dev.crt \
		-subj "/CN=localhost" >/dev/null 2>&1
	@echo "generated deploy/certs/dev.crt and deploy/certs/dev.key"

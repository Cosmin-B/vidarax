//! Telemetry initialisation for vidarax-api.
//!
//! Stacks four layers on a `tracing_subscriber::Registry`:
//!
//! 1. `EnvFilter`  — `RUST_LOG`-driven level filter (default: `info`)
//! 2. `fmt` console layer — human-readable to stdout (dev ergonomics)
//! 3. `fmt` JSON layer  — structured JSON to stderr (VictoriaLogs sidecar ingest)
//! 4. OpenTelemetry layer — span bridge; OTLP exporter added in Phase 3
//!
//! Call once at process start (safe to call multiple times — idempotent via
//! `try_init`).

use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialise the global tracing subscriber.
///
/// # Examples
///
/// ```no_run
/// vidarax_api::telemetry::init_telemetry();
/// ```
pub fn init_telemetry() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Human-readable output to stdout (dev / local runs).
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_writer(std::io::stdout);

    // Structured JSON to stderr (consumed by VictoriaLogs log-shipper sidecar).
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(std::io::stderr);

    // Phase 1: no-op OTel provider (no exporter, spans are collected but
    // immediately dropped).  Phase 3 upgrades this to the OTLP gRPC exporter.
    let otel_provider = SdkTracerProvider::builder().build();
    let tracer = opentelemetry::trace::TracerProvider::tracer(&otel_provider, "vidarax-api");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Publish the provider globally so it can be shut down on exit.
    opentelemetry::global::set_tracer_provider(otel_provider);

    // `try_init` is idempotent: a second call (e.g. in integration tests that
    // also call `run()`) returns an error that we deliberately ignore.
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(json_layer)
        .with(otel_layer)
        .try_init();
}

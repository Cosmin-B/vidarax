//! Telemetry initialisation for vidarax-api.
//!
//! Stacks four layers on a `tracing_subscriber::Registry`:
//!
//! 1. `EnvFilter`  — `RUST_LOG`-driven level filter (default: `info`)
//! 2. `fmt` console layer — human-readable to stdout (dev ergonomics)
//! 3. `fmt` JSON layer  — structured JSON to stderr (VictoriaLogs sidecar ingest)
//! 4. OpenTelemetry layer — span bridge; OTLP exporter wired when
//!    `VIDARAX_TRACES_ENDPOINT` is set (Phase 3)
//!
//! # Usage
//!
//! ```no_run
//! // At process start:
//! vidarax_api::telemetry::init_telemetry();
//!
//! // On graceful shutdown — flushes any buffered spans:
//! vidarax_api::telemetry::shutdown_telemetry();
//! ```

use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Holds the active provider so `shutdown_telemetry` can flush it.
static OTEL_PROVIDER: std::sync::OnceLock<SdkTracerProvider> = std::sync::OnceLock::new();

/// Initialise the global tracing subscriber.
///
/// Safe to call multiple times — idempotent via `try_init`.
///
/// Environment variables:
/// - `RUST_LOG` — tracing filter (default: `info`)
/// - `VIDARAX_TRACES_ENDPOINT` — OTLP gRPC endpoint for trace export
///   (e.g. `http://localhost:4317`).  When unset, spans are collected
///   but immediately dropped (no-op Phase 1 provider).
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

    // Build the OTel provider.  When VIDARAX_TRACES_ENDPOINT is set, wire the
    // OTLP gRPC exporter (Phase 3).  Otherwise fall back to a no-op provider so
    // the binary starts cleanly without external infra.
    let otel_provider = build_otel_provider();
    let tracer = opentelemetry::trace::TracerProvider::tracer(&otel_provider, "vidarax-api");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Store a clone for shutdown_telemetry(); then publish globally.
    let _ = OTEL_PROVIDER.set(otel_provider.clone());
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

/// Flush and shut down the global OTel tracer provider.
///
/// Call once during graceful process shutdown (after serving stops).
/// Safe to call if `init_telemetry` was never called (no-op).
pub fn shutdown_telemetry() {
    if let Some(provider) = OTEL_PROVIDER.get() {
        if let Err(e) = provider.shutdown() {
            tracing::warn!(%e, "OTel provider shutdown error");
        }
    }
}

/// Build the `SdkTracerProvider` appropriate for the current environment.
///
/// - If `VIDARAX_TRACES_ENDPOINT` is set: wire an OTLP gRPC exporter.
/// - Otherwise: return a no-op provider (Phase 1 default).
fn build_otel_provider() -> SdkTracerProvider {
    if let Ok(endpoint) = std::env::var("VIDARAX_TRACES_ENDPOINT") {
        match build_otlp_provider(&endpoint) {
            Ok(provider) => {
                tracing::info!(endpoint, "OTel OTLP exporter configured");
                return provider;
            }
            Err(err) => {
                // Gracefully fall back rather than crashing on bad config.
                tracing::warn!(
                    endpoint,
                    %err,
                    "OTel OTLP exporter init failed; falling back to no-op provider"
                );
            }
        }
    }
    // No endpoint configured — collect spans locally but discard them.
    SdkTracerProvider::builder().build()
}

/// Build an OTLP gRPC exporter pointing at `endpoint`.
///
/// Returns an error if the exporter cannot be constructed (e.g. bad URI).
fn build_otlp_provider(endpoint: &str) -> Result<SdkTracerProvider, Box<dyn std::error::Error>> {
    use opentelemetry_otlp::SpanExporter;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::BatchSpanProcessor;

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(opentelemetry_sdk::trace::BatchConfig::default())
        .build();

    Ok(SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build())
}

#![forbid(unsafe_code)]

use std::{io, sync::Arc};

mod auth;
pub mod config;
mod handlers;
mod ids;
mod inference_metrics;
mod models;
mod response;
mod router;
mod security;
mod semantic;
mod semantic_infer;
mod server;
// Stays pub: integration tests and doctests construct SpacetimeClient directly.
pub mod spacetime_client;
mod state;
pub mod telemetry;
mod tenant_labels;
mod validation;
pub(crate) mod wal_sink;
mod whip;

pub use config::{resolve_wal_path, ServerConfig, TransportMode};
pub use models::AttachStreamRequest;
pub use router::app_router;
pub use state::{AppState, StreamSlotGuard};
use vidarax_core::ingest::pipeline::{build_decode_pipeline, DecodePipeline, PipelineBackend};
use vidarax_core::webrtc::decode::VideoCodec;
use vidarax_core::webrtc::resources::MediaSessionResources;
use vidarax_core::webrtc::session::{
    TurnServer, WebRtcConfig, MAX_RTP_ACCESS_UNIT_BYTES, RTP_FRAME_QUEUE_CAPACITY,
};
use vidarax_core::webrtc::workers::{per_stream_analysis_workers, WorkerPoolConfig};

pub async fn run(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    telemetry::init_telemetry();

    // Install the TLS crypto provider once, before any WebRTC sessions are
    // created.  rustrtc uses rustls for DTLS; it requires an installed
    // CryptoProvider.  `ok()` silences the error when a provider is already
    // installed (e.g. in tests).
    rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider()).ok();

    let wal_path = resolve_wal_path(&config).map_err(invalid_input)?;
    let backend_config = if let Some(backends) = backend_entries_from_explicit_urls(&config) {
        vidarax_core::backends::VidaraxConfig { backends }
    } else {
        let config_path =
            std::env::var("VIDARAX_CONFIG").unwrap_or_else(|_| "vidarax.toml".to_string());
        config::load_backend_config(&config_path).map_err(invalid_input)?
    };
    let provider = if backend_config.backends.is_empty() {
        None
    } else {
        let backends = backend_config.backends;
        Some(
            tokio::task::spawn_blocking(move || {
                // reqwest::blocking builds and drops an internal runtime.
                //
                // Use the model-routing builder, not the plain fallback
                // chain: tiered VLM inference (see
                // vidarax_core::tiered_vlm::run_tiered) swaps the model id on
                // an escalated request and needs that id to reach whichever
                // backend actually serves it, not just retry the same
                // primary backend picked at startup.
                vidarax_core::backends::build_provider_with_model_routing(&backends)
            })
            .await
            .map_err(|e| invalid_input(format!("failed to build provider chain: {e}")))?
            .map_err(|e| invalid_input(format!("failed to build provider chain: {e}")))?,
        )
    };
    let security_policy = security::SecurityPolicy::from_config(&config).map_err(invalid_input)?;
    let webrtc_config = build_webrtc_config(&config);
    let capacity_worker_config = WorkerPoolConfig::from(&webrtc_config);
    let keyframe_capacity =
        MediaSessionResources::for_pipeline(&capacity_worker_config, VideoCodec::H264, false);
    let clip_capacity =
        MediaSessionResources::for_pipeline(&capacity_worker_config, VideoCodec::H264, true);
    tracing::info!(
        maximum_live_sessions = config.active_stream_limit,
        worker_thread_budget = config.media_worker_thread_budget,
        memory_budget_bytes = config.media_memory_budget_bytes,
        rtp_queue_slots_per_session = RTP_FRAME_QUEUE_CAPACITY,
        maximum_access_unit_bytes = MAX_RTP_ACCESS_UNIT_BYTES,
        estimated_keyframe_session_bytes = keyframe_capacity.reserved_bytes,
        estimated_keyframe_session_threads = keyframe_capacity.worker_threads,
        estimated_clip_session_bytes = clip_capacity.reserved_bytes,
        estimated_clip_session_threads = clip_capacity.worker_threads,
        "live media capacity plan"
    );
    let decode_pipeline = build_configured_decode_pipeline(&config.decode_backend)?;
    let state = AppState::from_wal(
        wal_path,
        config.ingest_file_roots.clone(),
        provider,
        decode_pipeline,
        security_policy,
        config.stream_ttl_secs,
        config.active_stream_limit,
        webrtc_config,
        config.novelty.clone(),
        vidarax_core::admission::AdmissionLimits {
            global_in_flight: config.inference_global_limit,
            per_principal_in_flight: config.inference_per_principal_limit,
            global_waiters: config.inference_waiter_limit,
            wait_timeout: std::time::Duration::from_millis(config.inference_wait_timeout_ms),
        },
        config.media_memory_budget_bytes,
        config.media_worker_thread_budget,
    )
    .map_err(invalid_input)?;
    let state = attach_spacetime_client(state, &config);
    let app = app_router(state);

    tracing::info!(transport = config.transport.label(), "vidarax-api startup");
    let serve_result = match config.transport {
        TransportMode::H1H2 => server::serve_h1h2(&config.bind_addr, app).await,
        TransportMode::H3Experimental => server::serve_h3_experimental(&config, app).await,
    };
    telemetry::shutdown_telemetry();
    tracing::info!("vidarax-api shutdown complete");
    serve_result?;
    Ok(())
}

/// Attach a SpacetimeDB client when VIDARAX_SPACETIMEDB_URL is configured.
/// With no URL the state is returned unchanged: stream events use the local
/// WAL and the feedback endpoints stay disabled.
fn attach_spacetime_client(state: AppState, config: &ServerConfig) -> AppState {
    match config.spacetimedb_url.as_deref() {
        Some(url) => {
            let module = config.spacetimedb_module.as_deref().unwrap_or("vidarax");
            tracing::info!(
                spacetimedb_url = url,
                spacetimedb_module = module,
                "SpacetimeDB integration enabled"
            );
            state.with_spacetime_client(crate::spacetime_client::SpacetimeClient::new(url, module))
        }
        None => {
            tracing::info!(
                "SpacetimeDB not configured; stream events use the local WAL and feedback endpoints are disabled"
            );
            state
        }
    }
}

pub fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn backend_entries_from_explicit_urls(
    config: &ServerConfig,
) -> Option<Vec<vidarax_core::backends::BackendEntry>> {
    let mut backends = Vec::new();
    if let Some(base_url) = &config.inference_vllm_base_url {
        backends.push(vidarax_core::backends::BackendEntry {
            name: "vllm".to_string(),
            backend_type: "openai_compat".to_string(),
            base_url: Some(base_url.clone()),
            api_key: None,
            model: None,
            upstream_model: None,
            openai_kind: Some("vllm".to_string()),
            priority: 1,
        });
    }
    if let Some(base_url) = &config.inference_sglang_base_url {
        backends.push(vidarax_core::backends::BackendEntry {
            name: "sglang".to_string(),
            backend_type: "openai_compat".to_string(),
            base_url: Some(base_url.clone()),
            api_key: None,
            model: None,
            upstream_model: None,
            openai_kind: Some("sglang".to_string()),
            priority: 2,
        });
    }
    (!backends.is_empty()).then_some(backends)
}

fn build_configured_decode_pipeline(
    configured_backend: &str,
) -> Result<Arc<dyn DecodePipeline>, io::Error> {
    let decode_backend = if configured_backend == "auto" {
        PipelineBackend::auto_detect().label()
    } else {
        configured_backend
    };
    build_decode_pipeline(decode_backend)
        .map_err(|e| invalid_input(format!("failed to build decode pipeline: {e}")))
}

fn build_webrtc_config(config: &ServerConfig) -> WebRtcConfig {
    let mut turn_servers = Vec::new();
    if let Some(url) = &config.webrtc_turn_url {
        turn_servers.push(TurnServer {
            url: url.clone(),
            username: config.webrtc_turn_username.clone().unwrap_or_default(),
            credential: config.webrtc_turn_credential.clone().unwrap_or_default(),
        });
    }
    WebRtcConfig {
        stun_servers: config.webrtc_stun_servers.clone(),
        turn_servers,
        max_output_tokens_per_second: config.webrtc_max_output_tokens_per_second,
        decode_workers: config.webrtc_decode_workers,
        analysis_workers: per_stream_analysis_workers(config.webrtc_analysis_workers),
        vlm_workers: config.webrtc_vlm_workers,
        gate_config: config.gate_config.clone(),
        // Resolved once here from the ServerConfig fields, not the process
        // environment, so it is the single source of truth every WHIP
        // session on this AppState clones. See config::build_webrtc_vlm_config
        // for why that distinction matters.
        vlm_tiering: config::build_webrtc_vlm_config(config),
        crop: config.webrtc_crop,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        app_router, attach_spacetime_client, build_webrtc_config, AppState, ServerConfig,
        TransportMode,
    };
    use crate::security::SecurityPolicy;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;
    use vidarax_core::backends::BackendEntry;
    use vidarax_core::gate::GateConfig;
    use vidarax_core::ingest::pipeline::{
        register_decode_backend, CpuFfmpegPipeline, PipelineBackend,
    };
    use vidarax_core::webrtc::session::WebRtcConfig;
    #[cfg(feature = "h3-experimental")]
    use {
        tokio_quiche::http3::driver::{ClientH3Event, H3Event, InboundFrame, NewClientRequest},
        tokio_quiche::quic::connect,
        tokio_quiche::quiche::h3::{Header, NameValue},
    };

    fn test_state() -> AppState {
        test_state_with_provider(None)
    }

    fn default_test_server_config() -> ServerConfig {
        ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        }
    }

    fn build_provider_from_url(
        base_url: &str,
    ) -> Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync> {
        let entry = BackendEntry {
            name: "test".to_string(),
            backend_type: "openai_compat".to_string(),
            base_url: Some(base_url.to_string()),
            api_key: None,
            model: None,
            upstream_model: None,
            openai_kind: None,
            priority: 1,
        };
        vidarax_core::backends::build_provider_chain(&[entry]).unwrap()
    }

    fn test_state_with_endpoints(base_url: Option<&str>) -> AppState {
        // reqwest::blocking::Client can't be created or dropped inside a tokio async
        // context (it internally owns a tokio runtime; nested runtimes are forbidden).
        // We create the provider on a dedicated OS thread so its lifetime is fully
        // outside the async executor.
        let provider = base_url.map(|url| {
            let url = url.to_string();
            std::thread::spawn(move || build_provider_from_url(&url))
                .join()
                .unwrap()
        });
        test_state_with_provider_and_policy(provider, SecurityPolicy::from_config_for_tests())
    }

    fn test_state_with_provider(
        provider: Option<Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync>>,
    ) -> AppState {
        test_state_with_provider_and_policy(provider, SecurityPolicy::from_config_for_tests())
    }

    fn test_state_with_provider_and_policy(
        provider: Option<Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync>>,
        policy: SecurityPolicy,
    ) -> AppState {
        test_state_with_runtime(provider, policy, 3600, 5)
    }

    fn test_state_with_runtime(
        provider: Option<Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync>>,
        policy: SecurityPolicy,
        stream_ttl_secs: u64,
        active_stream_limit: usize,
    ) -> AppState {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let wal_path = std::env::temp_dir().join(format!("vidarax-api-test-{nanos}.wal"));
        AppState::with_wal_for_tests_runtime(
            wal_path,
            provider,
            policy,
            stream_ttl_secs,
            active_stream_limit,
        )
    }

    fn spawn_mock_provider_http_server(
        status: u16,
        body: String,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut req_buf = [0u8; 4096];
                        let _ = stream.read(&mut req_buf);
                        let reason = match status {
                            200 => "OK",
                            503 => "Service Unavailable",
                            _ => "Error",
                        };
                        let response = format!(
                            "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.flush();
                        break;
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://{addr}"), handle)
    }

    fn spawn_mock_provider_http_server_n(
        status: u16,
        body: String,
        requests: usize,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut served = 0usize;
            while served < requests && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut req_buf = [0u8; 4096];
                        let _ = stream.read(&mut req_buf);
                        let reason = match status {
                            200 => "OK",
                            503 => "Service Unavailable",
                            _ => "Error",
                        };
                        let response = format!(
                            "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.flush();
                        served += 1;
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://{addr}"), handle)
    }

    #[test]
    fn startup_decode_backend_resolution_uses_registry_name() {
        register_decode_backend("api-test-custom-backend", || {
            Arc::new(CpuFfmpegPipeline::new())
        });

        let pipeline = super::build_configured_decode_pipeline("api-test-custom-backend")
            .expect("registered backend should build");

        assert!(matches!(pipeline.backend(), PipelineBackend::CpuFfmpeg));
    }

    #[test]
    fn build_webrtc_config_clamps_programmatic_analysis_workers() {
        let mut config = default_test_server_config();
        config.webrtc_analysis_workers = 8;

        let webrtc = build_webrtc_config(&config);

        assert_eq!(webrtc.analysis_workers, 1);
    }

    #[test]
    fn build_webrtc_config_carries_resolved_gate_config() {
        let mut config = default_test_server_config();
        config.gate_config.scene_cut_hamming_threshold = 7;

        let webrtc = build_webrtc_config(&config);

        assert_eq!(webrtc.gate_config.scene_cut_hamming_threshold, 7);
    }

    #[test]
    fn build_webrtc_config_carries_resolved_tiering_config() {
        // Set the tiering knobs on the ServerConfig directly, not via env, so
        // this proves build_webrtc_config resolves TieredVlmConfig from the
        // struct a caller passed to `run`, not whatever the process
        // environment happens to hold.
        let mut config = default_test_server_config();
        config.webrtc_first_pass_model = "local-model".to_string();
        config.webrtc_second_pass_model = Some("escalation-model".to_string());
        config.webrtc_second_pass_threshold = 0.42;
        config.webrtc_second_pass_max_tokens = 512;

        let webrtc = build_webrtc_config(&config);

        assert!(
            webrtc.vlm_tiering.is_tiered(),
            "a ServerConfig with a distinct second-pass model must resolve to tiered routing"
        );
        assert_eq!(&*webrtc.vlm_tiering.first_pass_model, "local-model");
        assert_eq!(&*webrtc.vlm_tiering.second_pass_model, "escalation-model");
        assert_eq!(webrtc.vlm_tiering.second_pass_threshold, 0.42);
        assert_eq!(webrtc.vlm_tiering.second_pass_max_tokens, 512);
    }

    #[test]
    fn attach_spacetime_client_respects_env_gated_config() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let disabled_wal_path =
            std::env::temp_dir().join(format!("vidarax-spacetime-client-disabled-{nanos}.wal"));
        let enabled_wal_path =
            std::env::temp_dir().join(format!("vidarax-spacetime-client-enabled-{nanos}.wal"));
        let mut config = default_test_server_config();

        let state = AppState::with_wal_for_tests(disabled_wal_path);
        let state = attach_spacetime_client(state, &config);
        assert!(state.spacetime_client().is_none());

        config.spacetimedb_url = Some("http://127.0.0.1:3000".to_string());

        let state = AppState::with_wal_for_tests(enabled_wal_path);
        let state = attach_spacetime_client(state, &config);
        assert!(state.spacetime_client().is_some());
    }

    #[test]
    fn explicit_inference_urls_build_provider_backend_entries() {
        let config = ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: Some("http://127.0.0.1:8081/v1".to_string()),
            inference_sglang_base_url: Some("http://127.0.0.1:8082/v1".to_string()),
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        };

        let entries = super::backend_entries_from_explicit_urls(&config).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "vllm");
        assert_eq!(entries[0].backend_type, "openai_compat");
        assert_eq!(
            entries[0].base_url.as_deref(),
            Some("http://127.0.0.1:8081/v1")
        );
        assert_eq!(entries[0].priority, 1);
        assert_eq!(entries[1].name, "sglang");
        assert_eq!(entries[1].backend_type, "openai_compat");
        assert_eq!(
            entries[1].base_url.as_deref(),
            Some("http://127.0.0.1:8082/v1")
        );
        assert_eq!(entries[1].priority, 2);
    }

    fn ffmpeg_available() -> bool {
        Command::new(vidarax_core::ingest::ffmpeg_path())
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn create_test_mp4() -> Option<String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let path = std::env::temp_dir().join(format!("vidarax-mp4-test-{nanos}-{seq}-{tid:?}.mp4"));
        let status = Command::new(vidarax_core::ingest::ffmpeg_path())
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=12",
                "-t",
                "1.2",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-y",
            ])
            .arg(path.as_os_str())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        Some(path.to_string_lossy().to_string())
    }

    #[cfg(feature = "h3-experimental")]
    fn ensure_test_tls_assets(
        workspace_root: &std::path::Path,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let cert_dir = workspace_root.join("deploy/certs");
        std::fs::create_dir_all(&cert_dir).expect("cert dir");
        let cert_path = cert_dir.join("dev.crt");
        let key_path = cert_dir.join("dev.key");
        if cert_path.exists() && key_path.exists() {
            return (cert_path, key_path);
        }

        let status = Command::new("openssl")
            .args([
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-sha256",
                "-nodes",
                "-days",
                "7",
                "-subj",
                "/CN=localhost",
                "-keyout",
            ])
            .arg(key_path.as_os_str())
            .arg("-out")
            .arg(cert_path.as_os_str())
            .status()
            .expect("openssl must be available to generate test TLS assets");
        assert!(
            status.success(),
            "openssl failed to generate test TLS assets"
        );
        (cert_path, key_path)
    }

    #[tokio::test]
    async fn health_endpoint_works() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/health")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn run_creation_returns_id() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();
        assert!(run_id.starts_with("run-"));
        assert_eq!(run_id.len(), 36);
    }

    #[tokio::test]
    async fn run_scoped_endpoint_enforces_principal_ownership() {
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let allowed_with_forged_tenant = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-b")
            .body(Body::empty())
            .unwrap();
        let allowed_with_forged_tenant_resp = app
            .clone()
            .oneshot(allowed_with_forged_tenant)
            .await
            .unwrap();
        assert_eq!(allowed_with_forged_tenant_resp.status(), StatusCode::OK);

        let denied_other_key = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "key-b")
            .header("x-tenant-id", "tenant-a")
            .body(Body::empty())
            .unwrap();
        let denied_other_key_resp = app.clone().oneshot(denied_other_key).await.unwrap();
        assert_eq!(denied_other_key_resp.status(), StatusCode::NOT_FOUND);

        let allowed = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-a")
            .body(Body::empty())
            .unwrap();
        let allowed_resp = app.oneshot(allowed).await.unwrap();
        assert_eq!(allowed_resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn open_mode_uses_shared_public_principal_without_tenant_isolation() {
        let app = app_router(test_state());
        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "unvalidated-key")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let read_from_other_tenant_header = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "other-unvalidated-key")
            .header("x-tenant-id", "tenant-b")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(read_from_other_tenant_header).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn forged_tenant_id_cannot_cross_api_key_principals() {
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-shared")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let forged = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "key-b")
            .header("x-tenant-id", "tenant-shared")
            .body(Body::empty())
            .unwrap();
        let forged_resp = app.clone().oneshot(forged).await.unwrap();
        assert_eq!(forged_resp.status(), StatusCode::NOT_FOUND);

        let allowed = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-shared")
            .body(Body::empty())
            .unwrap();
        let allowed_resp = app.oneshot(allowed).await.unwrap();
        assert_eq!(allowed_resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn file_endpoint_requires_owner_prefix_and_video_extension() {
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let principal = format!("api-key:{}", crate::auth::strong_hash_hex("key-a"));
        let prefix = format!("{}__", crate::auth::strong_hash_hex(&principal));
        let upload_root = std::env::temp_dir().join(crate::config::UPLOAD_DIR_NAME);
        fs::create_dir_all(&upload_root).unwrap();
        let filename = format!("{prefix}owned.mp4");
        let path = upload_root.join(&filename);
        fs::write(&path, b"video").unwrap();

        let forged = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .header("x-api-key", "key-b")
            .body(Body::empty())
            .unwrap();
        let forged_resp = app.clone().oneshot(forged).await.unwrap();
        assert_eq!(forged_resp.status(), StatusCode::NOT_FOUND);

        let allowed = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .header("x-api-key", "key-a")
            .body(Body::empty())
            .unwrap();
        let allowed_resp = app.clone().oneshot(allowed).await.unwrap();
        assert_eq!(allowed_resp.status(), StatusCode::OK);

        let bad_ext = format!("{prefix}owned.txt");
        fs::write(upload_root.join(&bad_ext), b"text").unwrap();
        let rejected = Request::builder()
            .uri(format!("/v1/files/{bad_ext}"))
            .method("GET")
            .header("x-api-key", "key-a")
            .body(Body::empty())
            .unwrap();
        let rejected_resp = app.oneshot(rejected).await.unwrap();
        assert_eq!(rejected_resp.status(), StatusCode::BAD_REQUEST);

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(upload_root.join(bad_ext));
    }

    #[tokio::test]
    async fn legacy_temp_dir_file_is_not_shared_by_default() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = format!("vidarax-legacy-temp-{nanos}.mp4");
        let legacy_path = std::env::temp_dir().join(&filename);
        fs::write(&legacy_path, b"legacy-temp-video").unwrap();

        let state = AppState::from_wal(
            std::env::temp_dir().join(format!("vidarax-legacy-temp-{nanos}.wal")),
            vec![],
            None,
            super::build_configured_decode_pipeline("cpu-ffmpeg").unwrap(),
            SecurityPolicy::from_config_for_tests(),
            3600,
            5,
            WebRtcConfig::default(),
            vidarax_core::novelty::LiveNoveltyConfig::default(),
            vidarax_core::admission::AdmissionLimits {
                global_in_flight: 8,
                per_principal_in_flight: 4,
                global_waiters: 128,
                wait_timeout: std::time::Duration::from_secs(5),
            },
            u64::MAX,
            usize::MAX,
        )
        .unwrap();
        let app = app_router(state);

        let served = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let served_resp = app.clone().oneshot(served).await.unwrap();
        assert_eq!(
            served_resp.status(),
            StatusCode::NOT_FOUND,
            "temp_dir files must not be served unless an operator configures temp_dir as a shared root"
        );

        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let ingest = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "source_uri": legacy_path,
                    "sampling_policy": "fixed",
                    "fixed_fps": 1.0,
                    "max_frames": 1
                }))
                .unwrap(),
            ))
            .unwrap();
        let ingest_resp = app.oneshot(ingest).await.unwrap();
        assert_eq!(ingest_resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let _ = fs::remove_file(legacy_path);
    }

    #[tokio::test]
    async fn operator_root_file_is_shared_while_uploaded_owner_prefix_stays_private() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let operator_root = std::env::temp_dir().join(format!("vidarax-operator-root-{nanos}"));
        fs::create_dir_all(&operator_root).unwrap();
        let operator_root = operator_root.canonicalize().unwrap();
        let shared_name = "operator-shared.mp4";
        fs::write(operator_root.join(shared_name), b"operator-video").unwrap();

        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let state = AppState::from_wal(
            std::env::temp_dir().join(format!("vidarax-operator-root-{nanos}.wal")),
            vec![operator_root.clone(), std::env::temp_dir()],
            None,
            super::build_configured_decode_pipeline("cpu-ffmpeg").unwrap(),
            policy,
            3600,
            5,
            WebRtcConfig::default(),
            vidarax_core::novelty::LiveNoveltyConfig::default(),
            vidarax_core::admission::AdmissionLimits {
                global_in_flight: 8,
                per_principal_in_flight: 4,
                global_waiters: 128,
                wait_timeout: std::time::Duration::from_secs(5),
            },
            u64::MAX,
            usize::MAX,
        )
        .unwrap();
        let app = app_router(state);

        let shared = Request::builder()
            .uri(format!("/v1/files/{shared_name}"))
            .method("GET")
            .header("x-api-key", "key-b")
            .body(Body::empty())
            .unwrap();
        let shared_resp = app.clone().oneshot(shared).await.unwrap();
        assert_eq!(
            shared_resp.status(),
            StatusCode::OK,
            "operator-configured roots are shared for authenticated callers"
        );

        let principal = format!("api-key:{}", crate::auth::strong_hash_hex("key-a"));
        let owner_prefix = format!("{}__", crate::auth::strong_hash_hex(&principal));
        let uploaded_name = format!("{owner_prefix}private.mp4");
        let upload_root = std::env::temp_dir().join(crate::config::UPLOAD_DIR_NAME);
        fs::create_dir_all(&upload_root).unwrap();
        fs::write(upload_root.join(&uploaded_name), b"uploaded-video").unwrap();
        let forged = Request::builder()
            .uri(format!("/v1/files/{uploaded_name}"))
            .method("GET")
            .header("x-api-key", "key-b")
            .body(Body::empty())
            .unwrap();
        let forged_resp = app.oneshot(forged).await.unwrap();
        assert_eq!(
            forged_resp.status(),
            StatusCode::NOT_FOUND,
            "another principal's uploaded temp-root file remains private"
        );

        let _ = fs::remove_file(operator_root.join(shared_name));
        let _ = fs::remove_dir(operator_root);
        let _ = fs::remove_file(upload_root.join(uploaded_name));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upload_root_symlink_alias_is_not_served_or_ingested() {
        use std::os::unix::fs::symlink;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let state = AppState::from_wal(
            std::env::temp_dir().join(format!("vidarax-upload-alias-{nanos}.wal")),
            vec![std::env::temp_dir()],
            None,
            super::build_configured_decode_pipeline("cpu-ffmpeg").unwrap(),
            policy,
            3600,
            5,
            WebRtcConfig::default(),
            vidarax_core::novelty::LiveNoveltyConfig::default(),
            vidarax_core::admission::AdmissionLimits {
                global_in_flight: 8,
                per_principal_in_flight: 4,
                global_waiters: 128,
                wait_timeout: std::time::Duration::from_secs(5),
            },
            u64::MAX,
            usize::MAX,
        )
        .unwrap();
        let app = app_router(state);

        let principal_a = format!("api-key:{}", crate::auth::strong_hash_hex("key-a"));
        let owner_prefix_a = format!("{}__", crate::auth::strong_hash_hex(&principal_a));
        let principal_b = format!("api-key:{}", crate::auth::strong_hash_hex("key-b"));
        let owner_prefix_b = format!("{}__", crate::auth::strong_hash_hex(&principal_b));
        let upload_root = std::env::temp_dir().join(crate::config::UPLOAD_DIR_NAME);
        fs::create_dir_all(&upload_root).unwrap();

        let victim_name = format!("{owner_prefix_b}victim-{nanos}.mp4");
        let victim_path = upload_root.join(&victim_name);
        fs::write(&victim_path, b"victim-video").unwrap();
        let alias_to_victim = format!("{owner_prefix_a}alias-to-victim-{nanos}.mp4");
        let alias_to_victim_path = upload_root.join(&alias_to_victim);
        let _ = fs::remove_file(&alias_to_victim_path);
        symlink(&victim_path, &alias_to_victim_path).unwrap();

        let forged_serve = Request::builder()
            .uri(format!("/v1/files/{alias_to_victim}"))
            .method("GET")
            .header("x-api-key", "key-a")
            .body(Body::empty())
            .unwrap();
        let forged_serve_resp = app.clone().oneshot(forged_serve).await.unwrap();
        assert_eq!(
            forged_serve_resp.status(),
            StatusCode::NOT_FOUND,
            "serve_file must authorize the resolved upload target, not the symlink basename"
        );

        let owned_name = format!("{owner_prefix_a}owned-{nanos}.mp4");
        let owned_path = upload_root.join(&owned_name);
        let fixture = create_test_mp4().expect("ffmpeg should generate an upload fixture");
        fs::copy(&fixture, &owned_path).unwrap();
        let owned_alias = format!("{owner_prefix_a}owned-alias-{nanos}.mp4");
        let owned_alias_path = upload_root.join(&owned_alias);
        let _ = fs::remove_file(&owned_alias_path);
        symlink(&owned_path, &owned_alias_path).unwrap();

        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let ingest = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "source_uri": owned_alias_path,
                    "sampling_policy": "fixed",
                    "fixed_fps": 1.0,
                    "max_frames": 1
                }))
                .unwrap(),
            ))
            .unwrap();
        let ingest_resp = app.oneshot(ingest).await.unwrap();
        assert_eq!(
            ingest_resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "upload-root symlinks must not be ingestable even when they point at an owned upload"
        );
        let ingest_body = ingest_resp.into_body().collect().await.unwrap().to_bytes();
        let ingest_json: serde_json::Value = serde_json::from_slice(&ingest_body).unwrap();
        let details = ingest_json["error"]["details"].as_array().unwrap();
        assert!(
            details.iter().any(|d| {
                d["field"].as_str() == Some("source_uri")
                    && d["message"].as_str().unwrap_or("").contains("not visible")
            }),
            "expected upload alias visibility rejection, got: {details:?}"
        );

        let _ = fs::remove_file(alias_to_victim_path);
        let _ = fs::remove_file(owned_alias_path);
        let _ = fs::remove_file(victim_path);
        let _ = fs::remove_file(owned_path);
        let _ = fs::remove_file(fixture);
    }

    #[tokio::test]
    async fn file_prefixes_ignore_caller_controlled_tenant_header() {
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["shared-key".to_string()],
            false,
            false,
            vec![],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let principal = format!("api-key:{}", crate::auth::strong_hash_hex("shared-key"));
        let owner_prefix = format!("{}__", crate::auth::strong_hash_hex(&principal));
        let video_path = create_test_mp4().expect("ffmpeg should generate an upload fixture");
        let video = fs::read(&video_path).expect("upload fixture should be readable");

        let upload_for_tenant = |tenant: &'static str, filename: &'static str| {
            let boundary = format!("vidarax-{tenant}-{filename}");
            let mut body = Vec::new();
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "content-disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
                     content-type: video/mp4\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(&video);
            body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
            Request::builder()
                .uri("/v1/upload")
                .method("POST")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .header("x-api-key", "shared-key")
                .header("x-tenant-id", tenant)
                .body(Body::from(body))
                .unwrap()
        };

        let upload_a = app
            .clone()
            .oneshot(upload_for_tenant("tenant-a", "a.mp4"))
            .await
            .unwrap();
        assert_eq!(upload_a.status(), StatusCode::OK);
        let body_a = upload_a.into_body().collect().await.unwrap().to_bytes();
        let json_a: serde_json::Value = serde_json::from_slice(&body_a).unwrap();
        let path_a = json_a["file_path"].as_str().unwrap().to_string();
        let filename = std::path::Path::new(&path_a)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(filename.starts_with(&owner_prefix));

        let upload_b = app
            .clone()
            .oneshot(upload_for_tenant("tenant-b", "b.mp4"))
            .await
            .unwrap();
        assert_eq!(upload_b.status(), StatusCode::OK);
        let body_b = upload_b.into_body().collect().await.unwrap().to_bytes();
        let json_b: serde_json::Value = serde_json::from_slice(&body_b).unwrap();
        let path_b = json_b["file_path"].as_str().unwrap().to_string();
        let tenant_b_filename = std::path::Path::new(&path_b)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(tenant_b_filename.starts_with(&owner_prefix));

        let allowed_with_different_tenant_header = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .header("x-api-key", "shared-key")
            .header("x-tenant-id", "tenant-b")
            .body(Body::empty())
            .unwrap();
        let allowed_with_different_tenant_resp = app
            .clone()
            .oneshot(allowed_with_different_tenant_header)
            .await
            .unwrap();
        assert_eq!(allowed_with_different_tenant_resp.status(), StatusCode::OK);

        let allowed = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .header("x-api-key", "shared-key")
            .header("x-tenant-id", "tenant-a")
            .body(Body::empty())
            .unwrap();
        let allowed_resp = app.oneshot(allowed).await.unwrap();
        assert_eq!(allowed_resp.status(), StatusCode::OK);

        let _ = fs::remove_file(path_a);
        let _ = fs::remove_file(path_b);
        let _ = fs::remove_file(video_path);
    }

    #[tokio::test]
    async fn public_uploads_use_shared_public_namespace_and_can_be_served() {
        let app = app_router(test_state());
        let boundary = "vidarax-test-boundary";
        let video_path = create_test_mp4().expect("ffmpeg should generate an upload fixture");
        let video = fs::read(&video_path).expect("upload fixture should be readable");
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"content-disposition: form-data; name=\"file\"; filename=\"clip.mp4\"\r\ncontent-type: video/mp4\r\n\r\n",
        );
        body.extend_from_slice(&video);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let upload = Request::builder()
            .uri("/v1/upload")
            .method("POST")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .header("x-api-key", "unvalidated-key")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(body))
            .unwrap();
        let upload_resp = app.clone().oneshot(upload).await.unwrap();
        assert_eq!(upload_resp.status(), StatusCode::OK);
        let body = upload_resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let file_path = json["file_path"].as_str().unwrap();
        let filename = std::path::Path::new(file_path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(filename.starts_with("public__"));

        let serve = Request::builder()
            .uri(format!("/v1/files/{filename}"))
            .method("GET")
            .header("x-tenant-id", "tenant-b")
            .body(Body::empty())
            .unwrap();
        let serve_resp = app.oneshot(serve).await.unwrap();
        assert_eq!(serve_resp.status(), StatusCode::OK);

        let _ = fs::remove_file(file_path);
        let _ = fs::remove_file(video_path);
    }

    #[tokio::test]
    async fn rejects_invalid_mode_in_run_creation() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"turbo"}"#))
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"code\":\"validation_error\""));
        assert!(text.contains("\"field\":\"mode\""));
    }

    #[tokio::test]
    async fn rejects_invalid_model_in_run_creation() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"bad/model"}"#))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"field\":\"model\""));
    }

    #[tokio::test]
    async fn query_returns_request_id_when_valid() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        assert_eq!(create_response.status(), StatusCode::OK);
        let body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let req = Request::builder()
            .uri("/v1/query")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({ "run_id": run_id })).unwrap(),
            ))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"request_id\":\"req-"));
    }

    #[tokio::test]
    async fn ingest_rejects_unknown_fields_and_missing_source() {
        let app = app_router(test_state());

        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let ingest_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"frame_index":1}"#))
            .unwrap();
        let ingest_response = app.clone().oneshot(ingest_req).await.unwrap();
        assert_eq!(ingest_response.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let state_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let state_response = app.oneshot(state_req).await.unwrap();
        assert_eq!(state_response.status(), StatusCode::OK);
        let state_body = state_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let text = std::str::from_utf8(&state_body).unwrap();
        assert!(text.contains("\"state\":\"pending\""));
    }

    #[tokio::test]
    async fn stop_transitions_state_to_cancelled() {
        let app = app_router(test_state());

        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let stop_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/stop"))
            .method("POST")
            .body(Body::empty())
            .unwrap();
        let stop_response = app.clone().oneshot(stop_req).await.unwrap();
        assert_eq!(stop_response.status(), StatusCode::OK);

        let state_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let state_response = app.oneshot(state_req).await.unwrap();
        assert_eq!(state_response.status(), StatusCode::OK);
        let state_body = state_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let text = std::str::from_utf8(&state_body).unwrap();
        assert!(text.contains("\"state\":\"cancelled\""));
    }

    #[tokio::test]
    async fn keepalive_updates_run_and_state() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        assert_eq!(create_response.status(), StatusCode::OK);
        let body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let keepalive_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/keepalive"))
            .method("POST")
            .body(Body::empty())
            .unwrap();
        let keepalive_response = app.clone().oneshot(keepalive_req).await.unwrap();
        assert_eq!(keepalive_response.status(), StatusCode::OK);

        let state_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let state_response = app.oneshot(state_req).await.unwrap();
        let state_body = state_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let state_text = std::str::from_utf8(&state_body).unwrap();
        assert!(state_text.contains("\"state\":\"processing\""));
    }

    #[tokio::test]
    async fn run_state_expires_after_ttl() {
        let app = app_router(test_state_with_runtime(
            None,
            SecurityPolicy::from_config_for_tests(),
            1,
            5,
        ));
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        assert_eq!(create_response.status(), StatusCode::OK);
        let body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
        let state_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let state_response = app.oneshot(state_req).await.unwrap();
        assert_eq!(state_response.status(), StatusCode::OK);
        let body = state_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"state\":\"expired\""));
    }

    #[tokio::test]
    async fn active_stream_limit_is_enforced_per_principal() {
        let app = app_router(test_state_with_runtime(
            None,
            SecurityPolicy::from_config_for_tests(),
            3600,
            1,
        ));
        let make_req = || {
            Request::builder()
                .uri("/v1/runs")
                .method("POST")
                .header("content-type", "application/json")
                .header("x-tenant-id", "tenant-a")
                .body(Body::from(r#"{"mode":"balanced"}"#))
                .unwrap()
        };

        let first = app.clone().oneshot(make_req()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let second = app.clone().oneshot(make_req()).await.unwrap();
        assert_eq!(second.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn deleted_run_is_not_addressable_and_releases_active_slot() {
        let app = app_router(test_state_with_runtime(
            None,
            SecurityPolicy::from_config_for_tests(),
            3600,
            1,
        ));
        let create = || {
            Request::builder()
                .uri("/v1/runs")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"mode":"balanced"}"#))
                .unwrap()
        };

        let first = app.clone().oneshot(create()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let body = first.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let delete = Request::builder()
            .uri(format!("/v1/runs/{run_id}"))
            .method("DELETE")
            .body(Body::empty())
            .unwrap();
        let delete_response = app.clone().oneshot(delete).await.unwrap();
        assert_eq!(delete_response.status(), StatusCode::OK);

        let ingest = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"source_uri":"/tmp/not-present.mp4"}"#))
            .unwrap();
        let ingest_response = app.clone().oneshot(ingest).await.unwrap();
        assert_eq!(ingest_response.status(), StatusCode::NOT_FOUND);

        let analyze = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":1,"pts_ms":33,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.clone().oneshot(analyze).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::NOT_FOUND);

        let feedback = Request::builder()
            .uri(format!("/v1/runs/{run_id}/feedback"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"rating":5,"category":"quality"}"#))
            .unwrap();
        let feedback_response = app.clone().oneshot(feedback).await.unwrap();
        assert_eq!(feedback_response.status(), StatusCode::NOT_FOUND);

        let query = Request::builder()
            .uri("/v1/query")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({ "run_id": run_id })).unwrap(),
            ))
            .unwrap();
        let query_response = app.clone().oneshot(query).await.unwrap();
        assert_eq!(query_response.status(), StatusCode::NOT_FOUND);

        let state = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let state_response = app.clone().oneshot(state).await.unwrap();
        assert_eq!(state_response.status(), StatusCode::NOT_FOUND);

        let second = app.clone().oneshot(create()).await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn infer_uses_external_provider_and_returns_text() {
        let completion =
            "{\"choices\":[{\"message\":{\"content\":\"hello from provider\"}}]}".to_string();
        let (base_url, server) = spawn_mock_provider_http_server(200, completion);
        let state = test_state_with_endpoints(Some(base_url.as_str()));
        let app = app_router(state);

        let req = Request::builder()
            .uri("/v1/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"hello","primary_provider":"vllm"}"#,
            ))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"provider\":\"vllm\""));
        assert!(text.contains("\"output_text\":\"hello from provider\""));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn infer_batch_returns_ordered_results() {
        let completion =
            "{\"choices\":[{\"message\":{\"content\":\"hello from provider\"}}]}".to_string();
        let (base_url, server) = spawn_mock_provider_http_server_n(200, completion, 2);
        let state = test_state_with_endpoints(Some(base_url.as_str()));
        let app = app_router(state);

        let req = Request::builder()
            .uri("/v1/infer/batch")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "max_parallel": 2,
                    "requests": [
                        {"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"hello","primary_provider":"vllm"},
                        {"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"hello again","primary_provider":"sglang"}
                    ]
                }"#,
            ))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json.get("processed").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(json.get("succeeded").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(json.get("failed").and_then(|v| v.as_u64()), Some(0));
        let results = json.get("results").and_then(|v| v.as_array()).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get("index").and_then(|v| v.as_u64()),
            Some(0),
            "first entry should retain request index ordering"
        );
        assert_eq!(results[0].get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            results[1].get("index").and_then(|v| v.as_u64()),
            Some(1),
            "second entry should retain request index ordering"
        );
        assert_eq!(results[1].get("ok").and_then(|v| v.as_bool()), Some(true));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn infer_batch_rejects_out_of_range_max_parallel() {
        let state = test_state_with_endpoints(Some("http://127.0.0.1:9"));
        let app = app_router(state);

        let req = Request::builder()
            .uri("/v1/infer/batch")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "max_parallel": 0,
                    "requests": [
                        {"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"hello","primary_provider":"vllm"}
                    ]
                }"#,
            ))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("\"field\":\"max_parallel\""));
    }

    #[tokio::test]
    async fn metrics_include_provider_series_after_infer() {
        let completion =
            "{\"choices\":[{\"message\":{\"content\":\"hello from provider\"}}]}".to_string();
        let (base_url, server) = spawn_mock_provider_http_server_n(200, completion, 8);
        let state = test_state_with_endpoints(Some(base_url.as_str()));
        let app = app_router(state);

        let infer_req = Request::builder()
            .uri("/v1/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"hello","primary_provider":"vllm"}"#,
            ))
            .unwrap();
        let infer_resp = app.clone().oneshot(infer_req).await.unwrap();
        assert_eq!(infer_resp.status(), StatusCode::OK);

        let metrics_req = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let metrics_resp = app.oneshot(metrics_req).await.unwrap();
        assert_eq!(metrics_resp.status(), StatusCode::OK);
        let metrics_body = metrics_resp.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&metrics_body).unwrap();
        assert!(text.contains("vidarax_infer_requests_total{provider=\"vllm\",status=\"ok\"} 1"));
        assert!(text.contains("vidarax_infer_latency_ms_count{provider=\"vllm\"} 1"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn models_catalog_reports_unavailable_without_provider_config() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json.get("models").and_then(|v| v.as_array()).unwrap();
        assert!(!models.is_empty());
        assert_eq!(
            models[0].get("availability").and_then(|v| v.as_str()),
            Some("unavailable")
        );
    }

    #[tokio::test]
    async fn models_catalog_reports_ready_with_reachable_providers() {
        let (base_url, server) = spawn_mock_provider_http_server_n(200, "{}".to_string(), 2);
        let state = test_state_with_endpoints(Some(base_url.as_str()));
        let app = app_router(state);

        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json.get("models").and_then(|v| v.as_array()).unwrap();
        assert!(!models.is_empty());
        assert_eq!(
            models[0].get("availability").and_then(|v| v.as_str()),
            Some("ready")
        );
        let providers = models[0]
            .get("providers_available")
            .and_then(|v| v.as_array())
            .unwrap();
        // New config-driven backend system exposes a single chain entry (vllm).
        assert!(
            !providers.is_empty(),
            "at least one provider must be reported"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn analyze_endpoint_generates_metadata() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":1,"pts_ms":33,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::OK);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let analyze_json: serde_json::Value = serde_json::from_slice(&analyze_body).unwrap();
        assert_eq!(
            analyze_json.get("generated").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert!(analyze_json
            .get("metadata")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .and_then(|v| v.get("ordering_key"))
            .is_some());
        assert!(analyze_json
            .get("metadata")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .and_then(|v| v.get("sampling_policy"))
            .is_some());
    }

    #[tokio::test]
    async fn analyze_label_maps_key_on_authenticated_principal_not_tenant_header() {
        let principal_a = format!("api-key:{}", crate::auth::strong_hash_hex("key-a"));
        let principal_b = format!("api-key:{}", crate::auth::strong_hash_hex("key-b"));
        let maps = crate::tenant_labels::TenantLabelMaps::from_test_file(
            HashMap::new(),
            HashMap::new(),
            HashMap::from([
                (
                    principal_a,
                    (
                        HashMap::from([("keyframe_keep".to_string(), "event.a".to_string())]),
                        HashMap::from([("keyframe_candidate".to_string(), "object.a".to_string())]),
                    ),
                ),
                (
                    principal_b.clone(),
                    (
                        HashMap::from([("keyframe_keep".to_string(), "event.b".to_string())]),
                        HashMap::from([("keyframe_candidate".to_string(), "object.b".to_string())]),
                    ),
                ),
            ]),
        );
        let policy = SecurityPolicy::from_test_policy(
            true,
            vec!["key-a".to_string(), "key-b".to_string()],
            false,
            false,
            vec![],
        );
        let state = test_state_with_provider_and_policy(None, policy)
            .with_tenant_label_maps_for_tests(maps);
        let app = app_router(state);

        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        assert_eq!(create_response.status(), StatusCode::OK);
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", principal_b)
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::OK);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let analyze_json: serde_json::Value = serde_json::from_slice(&analyze_body).unwrap();
        let first = analyze_json["metadata"]
            .as_array()
            .and_then(|rows| rows.first())
            .expect("metadata row");
        assert_eq!(
            first["annotations"]["events"][0]["type"].as_str(),
            Some("event.a")
        );
        assert_eq!(
            first["annotations"]["objects"][0]["label"].as_str(),
            Some("object.a")
        );
    }

    #[tokio::test]
    async fn analyze_fixed_sampling_requires_fixed_fps() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "sampling_policy":"fixed",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn analyze_fixed_sampling_emits_policy_and_rate() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "sampling_policy":"fixed",
                    "fixed_fps": 3.0,
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":1,"pts_ms":333,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::OK);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let analyze_json: serde_json::Value = serde_json::from_slice(&analyze_body).unwrap();
        let first = analyze_json
            .get("metadata")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .unwrap();
        assert_eq!(
            first.get("sampling_policy").and_then(|v| v.as_str()),
            Some("fixed")
        );
        assert_eq!(first.get("sample_fps").and_then(|v| v.as_f64()), Some(3.0));
    }

    #[tokio::test]
    async fn ingest_mp4_then_analyze_without_frames_uses_decoded_signals() {
        assert!(
            ffmpeg_available(),
            "ffmpeg must be available for ingest tests"
        );
        let video_path = create_test_mp4().expect("ffmpeg should generate fixture");

        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let ingest_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "source_uri": video_path,
                    "sample_fps": 2.0,
                    "max_frames": 32
                }))
                .unwrap(),
            ))
            .unwrap();
        let ingest_response = app.clone().oneshot(ingest_req).await.unwrap();
        assert_eq!(ingest_response.status(), StatusCode::OK);
        let ingest_body = ingest_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let ingest_json: serde_json::Value = serde_json::from_slice(&ingest_body).unwrap();
        assert!(
            ingest_json
                .get("decoded_frames")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                >= 1
        );

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"Qwen/Qwen3-VL-4B-Instruct","window_size":8}"#,
            ))
            .unwrap();
        let analyze_response = app.clone().oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::OK);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let analyze_json: serde_json::Value = serde_json::from_slice(&analyze_body).unwrap();
        assert!(
            analyze_json
                .get("generated")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                >= 1
        );
        let first_metadata = analyze_json
            .get("metadata")
            .and_then(|value| value.as_array())
            .and_then(|rows| rows.first())
            .expect("decoded analysis should return frame metadata");
        assert_eq!(
            first_metadata
                .get("coordinate_schema")
                .and_then(|value| value.as_str()),
            Some("vidarax.image.v1")
        );
        let coordinates = first_metadata
            .get("coordinates")
            .expect("decoded analysis should preserve image provenance");
        assert_eq!(
            coordinates.pointer("/requested_region/width"),
            Some(&json!(1.0))
        );
        assert_eq!(
            coordinates.pointer("/analysis_extent"),
            coordinates.pointer("/source_extent")
        );

        let query_req = Request::builder()
            .uri("/v1/query")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "run_id": run_id,
                    "kind": "frames_decoded"
                }))
                .unwrap(),
            ))
            .unwrap();
        let query_response = app.clone().oneshot(query_req).await.unwrap();
        assert_eq!(query_response.status(), StatusCode::OK);
        let query_body = query_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let query_json: serde_json::Value = serde_json::from_slice(&query_body).unwrap();
        assert_eq!(
            query_json
                .get("matches")
                .and_then(|v| v.as_array())
                .map(|items| items.len()),
            Some(1)
        );

        let _ = fs::remove_file(video_path);
    }

    #[tokio::test]
    async fn analyze_without_frames_requires_decoded_ingest_signals() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"Qwen/Qwen3-VL-4B-Instruct"}"#))
            .unwrap();
        let analyze_response = app.oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let text = std::str::from_utf8(&analyze_body).unwrap();
        assert!(text.contains("\"field\":\"frames\""));
        assert!(text.contains("no decoded ingest frames exist"));
    }

    #[tokio::test]
    async fn analyze_endpoint_emits_marker_lifecycle() {
        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":1,"pts_ms":33,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":3,"pts_ms":99,"perceptual_hash":3,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .unwrap();
        let analyze_response = app.clone().oneshot(analyze_req).await.unwrap();
        assert_eq!(analyze_response.status(), StatusCode::OK);
        let analyze_body = analyze_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let analyze_json: serde_json::Value = serde_json::from_slice(&analyze_body).unwrap();
        let markers = analyze_json
            .get("markers")
            .and_then(|v| v.as_array())
            .expect("markers array");
        assert!(!markers.is_empty());

        let marker_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/markers"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let marker_resp = app.oneshot(marker_req).await.unwrap();
        assert_eq!(marker_resp.status(), StatusCode::OK);
        let marker_body = marker_resp.into_body().collect().await.unwrap().to_bytes();
        let marker_json: serde_json::Value = serde_json::from_slice(&marker_body).unwrap();
        assert!(marker_json
            .get("markers")
            .and_then(|v| v.as_array())
            .map(|v| !v.is_empty())
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn realtime_reason_endpoint_generates_markers_and_lag_stats() {
        assert!(
            ffmpeg_available(),
            "ffmpeg must be available for realtime tests"
        );
        let video_path = create_test_mp4().expect("ffmpeg should generate fixture");

        let app = app_router(test_state());
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let reason_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/reason"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "source_uri": video_path,
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "sampling_policy": "fixed",
                    "fixed_fps": 2.0,
                    "max_frames": 32,
                    "chunk_size": 16
                }))
                .unwrap(),
            ))
            .unwrap();
        let reason_resp = app.oneshot(reason_req).await.unwrap();
        let status = reason_resp.status();
        let reason_body = reason_resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            std::str::from_utf8(&reason_body).unwrap_or("<invalid utf8>")
        );
        let reason_json: serde_json::Value = serde_json::from_slice(&reason_body).unwrap();
        assert!(
            reason_json
                .get("markers_emitted")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                >= 1
        );
        assert!(reason_json
            .get("lag_p95_ms")
            .and_then(|v| v.as_u64())
            .is_some());
        assert!(reason_json
            .get("lag_p99_ms")
            .and_then(|v| v.as_u64())
            .is_some());
        let _ = fs::remove_file(video_path);
    }

    #[tokio::test]
    async fn realtime_reason_uses_provider_semantic_overlay_when_configured() {
        assert!(
            ffmpeg_available(),
            "ffmpeg must be available for realtime provider tests"
        );
        let video_path = create_test_mp4().expect("ffmpeg should generate fixture");

        let completion = "{\"choices\":[{\"message\":{\"content\":\"{\\\"event_type\\\":\\\"scene_cut\\\",\\\"object_label\\\":\\\"temple\\\",\\\"summary\\\":\\\"Temple visible\\\",\\\"description\\\":\\\"Temple appears in this chunk\\\",\\\"confidence\\\":0.91}\"}}]}".to_string();
        let (base_url, server) = spawn_mock_provider_http_server(200, completion);
        let state = test_state_with_endpoints(Some(base_url.as_str()));
        let app = app_router(state);

        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_response = app.clone().oneshot(create_req).await.unwrap();
        let create_body = create_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let reason_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/reason"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "source_uri": video_path,
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "sampling_policy": "fixed",
                    "fixed_fps": 2.0,
                    "max_frames": 16,
                    "chunk_size": 8,
                    "semantic_inference": true,
                    "semantic_frames_per_chunk": 1
                }))
                .unwrap(),
            ))
            .unwrap();
        let reason_resp = app.oneshot(reason_req).await.unwrap();
        let status = reason_resp.status();
        let reason_body = reason_resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            std::str::from_utf8(&reason_body).unwrap_or("<invalid utf8>")
        );
        let reason_json: serde_json::Value = serde_json::from_slice(&reason_body).unwrap();
        let first = reason_json
            .get("metadata")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .expect("metadata row");
        assert_eq!(
            first["annotations"]["events"][0]["type"].as_str(),
            Some("scene_cut")
        );
        assert_eq!(
            first["annotations"]["objects"][0]["label"].as_str(),
            Some("temple")
        );
        assert_eq!(first["fallback"]["used"].as_bool(), Some(false));

        let _ = fs::remove_file(video_path);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn security_policy_rejects_missing_api_key() {
        let policy = SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: true,
            security_api_keys: vec!["test-key".to_string()],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let req_ok = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("x-api-key", "test-key")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let response_ok = app.oneshot(req_ok).await.unwrap();
        assert_eq!(response_ok.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn security_policy_enforces_global_rate_limit() {
        let policy = SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: false,
            security_global_rps: Some(1),
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));
        let mut saw_rate_limited = false;
        for _ in 0..20 {
            let req = Request::builder()
                .uri("/v1/runs")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"mode":"balanced"}"#))
                .unwrap();
            let response = app.clone().oneshot(req).await.unwrap();
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                saw_rate_limited = true;
                break;
            }
        }
        assert!(saw_rate_limited);
    }

    #[tokio::test]
    async fn tenant_rate_limit_keys_on_authenticated_principal_not_tenant_header() {
        let policy = SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: true,
            security_api_keys: vec!["key-a".to_string(), "key-b".to_string()],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: Some(1),
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let first = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(first).await.unwrap().status(),
            StatusCode::OK
        );

        let rotated_tenant = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-a")
            .header("x-tenant-id", "tenant-b")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(rotated_tenant).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS,
            "same API key must not bypass quota by rotating x-tenant-id"
        );

        let other_key = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-api-key", "key-b")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        assert_eq!(
            app.oneshot(other_key).await.unwrap().status(),
            StatusCode::OK,
            "a different API key gets an independent quota bucket"
        );
    }

    #[tokio::test]
    async fn security_policy_requires_tenant_header_when_enabled() {
        let policy = SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: true,
            security_api_keys: vec!["test-key".to_string()],
            security_require_tenant_id: true,
            security_global_rps: None,
            security_tenant_rps: Some(10),
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req_missing = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("x-api-key", "test-key")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let resp_missing = app.clone().oneshot(req_missing).await.unwrap();
        assert_eq!(resp_missing.status(), StatusCode::UNAUTHORIZED);

        let req_present = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("x-api-key", "test-key")
            .header("x-tenant-id", "tenant-a")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let resp_present = app.oneshot(req_present).await.unwrap();
        assert_eq!(resp_present.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn security_policy_rejects_required_tenant_without_api_keys() {
        let err = match SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: true,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        }) {
            Ok(_) => panic!("required-tenant without required API keys should be rejected"),
            Err(err) => err,
        };

        assert!(
            err.contains("VIDARAX_REQUIRE_TENANT_ID requires VIDARAX_REQUIRE_API_KEY=true"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_requires_api_key_when_enabled() {
        let policy = SecurityPolicy::from_config(&ServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            h3_bind_addr: "127.0.0.1:8443".to_string(),
            h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
            h3_tls_key_path: "deploy/certs/dev.key".to_string(),
            data_dir: ".vidarax-data".to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec!["metrics-key".to_string()],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 16,
            security_metrics_require_api_key: true,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req_missing = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp_missing = app.clone().oneshot(req_missing).await.unwrap();
        assert_eq!(resp_missing.status(), StatusCode::UNAUTHORIZED);

        let req_ok = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .header("x-api-key", "metrics-key")
            .body(Body::empty())
            .unwrap();
        let resp_ok = app.oneshot(req_ok).await.unwrap();
        assert_eq!(resp_ok.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn authenticated_metrics_endpoint_uses_principal_rate_limit() {
        let mut config = default_test_server_config();
        config.security_api_keys = vec!["metrics-key".to_string(), "metrics-key-b".to_string()];
        config.security_tenant_rps = Some(1);
        config.security_metrics_require_api_key = true;
        let policy = SecurityPolicy::from_config(&config).unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let first = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .header("x-api-key", "metrics-key")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(first).await.unwrap().status(),
            StatusCode::OK
        );

        let second = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .header("x-api-key", "metrics-key")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(second).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS
        );

        let other_key = Request::builder()
            .uri("/v1/metrics")
            .method("GET")
            .header("x-api-key", "metrics-key-b")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(other_key).await.unwrap().status(),
            StatusCode::OK,
            "a different metrics API key gets an independent quota bucket"
        );
    }

    #[tokio::test]
    async fn cors_preflight_rejects_unlisted_origin() {
        let policy = SecurityPolicy::from_test_policy(
            false,
            vec![],
            false,
            false,
            vec!["https://app.example.com".to_string()],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req = Request::builder()
            .uri("/v1/runs")
            .method("OPTIONS")
            .header("origin", "https://evil.example.com")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn cors_preflight_allows_listed_origin() {
        let policy = SecurityPolicy::from_test_policy(
            false,
            vec![],
            false,
            false,
            vec!["https://app.example.com".to_string()],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req = Request::builder()
            .uri("/v1/runs")
            .method("OPTIONS")
            .header("origin", "https://app.example.com")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("https://app.example.com")
        );
    }

    #[tokio::test]
    async fn whip_offer_cors_exposes_location_and_run_id_headers() {
        let policy = SecurityPolicy::from_test_policy(
            false,
            vec![],
            false,
            false,
            vec!["https://app.example.com".to_string()],
        );
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req = Request::builder()
            .uri("/v1/stream/whip")
            .method("POST")
            .header("origin", "https://app.example.com")
            .header("content-type", "application/sdp")
            .body(Body::from("v=0\r\n"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let exposed = resp
            .headers()
            .get("access-control-expose-headers")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");

        assert!(exposed.split(',').any(|header| header.trim() == "location"));
        assert!(exposed
            .split(',')
            .any(|header| header.trim() == "x-vidarax-run-id"));
    }

    #[tokio::test]
    async fn responses_include_security_headers() {
        let app = app_router(test_state());
        let req = Request::builder()
            .uri("/v1/health")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-content-type-options")
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        assert_eq!(
            resp.headers()
                .get("x-frame-options")
                .and_then(|v| v.to_str().ok()),
            Some("DENY")
        );
        assert_eq!(
            resp.headers()
                .get("referrer-policy")
                .and_then(|v| v.to_str().ok()),
            Some("no-referrer")
        );
        assert_eq!(
            resp.headers()
                .get("cache-control")
                .and_then(|v| v.to_str().ok()),
            Some("no-store")
        );
    }

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn h3_health_over_http3() {
        let bind_port = reserve_tcp_port();
        let h3_port = reserve_udp_port();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let data_dir = std::env::temp_dir().join(format!("vidarax-h3-test-{nanos}"));
        std::fs::create_dir_all(&data_dir).unwrap();
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let (cert_path, key_path) = ensure_test_tls_assets(workspace_root);

        let config = ServerConfig {
            bind_addr: format!("127.0.0.1:{bind_port}"),
            h3_bind_addr: format!("127.0.0.1:{h3_port}"),
            h3_tls_cert_path: cert_path.to_string_lossy().to_string(),
            h3_tls_key_path: key_path.to_string_lossy().to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 2048,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H3Experimental,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        };

        let server_task =
            tokio::spawn(async move { super::run(config).await.map_err(|e| e.to_string()) });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let client_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        client_socket
            .connect(format!("127.0.0.1:{h3_port}"))
            .await
            .unwrap();
        let (_conn, mut controller) = connect(client_socket, Some("test.com"))
            .await
            .expect("h3 client must connect");

        let headers = vec![
            Header::new(b":method", b"GET"),
            Header::new(b":scheme", b"https"),
            Header::new(b":authority", b"test.com"),
            Header::new(b":path", b"/v1/health"),
        ];
        controller
            .request_sender()
            .send(NewClientRequest {
                request_id: 1,
                headers,
                body_writer: None,
            })
            .expect("request should enqueue");

        let mut saw_headers = false;
        let mut saw_ok_body = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

        while tokio::time::Instant::now() < deadline {
            let next = tokio::time::timeout(
                std::time::Duration::from_millis(500),
                controller.event_receiver_mut().recv(),
            )
            .await
            .ok()
            .flatten();

            let Some(event) = next else {
                continue;
            };

            if let ClientH3Event::Core(H3Event::IncomingHeaders(incoming)) = event {
                let status = incoming
                    .headers
                    .iter()
                    .find_map(|h| {
                        if h.name() == b":status" {
                            std::str::from_utf8(h.value()).ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or("");
                saw_headers = status == "200";

                let mut body = Vec::new();
                let mut recv = incoming.recv;
                while let Some(frame) = recv.recv().await {
                    if let InboundFrame::Body(chunk, fin) = frame {
                        body.extend_from_slice(chunk.as_ref());
                        if fin {
                            break;
                        }
                    }
                }
                saw_ok_body = std::str::from_utf8(&body)
                    .map(|v| v.contains("\"status\":\"ok\""))
                    .unwrap_or(false);
                break;
            }
        }

        server_task.abort();
        let _ = server_task.await;

        assert!(saw_headers, "expected h3 :status=200 response");
        assert!(saw_ok_body, "expected h3 body payload containing status=ok");
    }

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn h3_metrics_endpoint_serves_over_http3() {
        let bind_port = reserve_tcp_port();
        let h3_port = reserve_udp_port();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let data_dir = std::env::temp_dir().join(format!("vidarax-h3-metrics-test-{nanos}"));
        std::fs::create_dir_all(&data_dir).unwrap();
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let (cert_path, key_path) = ensure_test_tls_assets(workspace_root);

        let config = ServerConfig {
            bind_addr: format!("127.0.0.1:{bind_port}"),
            h3_bind_addr: format!("127.0.0.1:{h3_port}"),
            h3_tls_cert_path: cert_path.to_string_lossy().to_string(),
            h3_tls_key_path: key_path.to_string_lossy().to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            ingest_file_roots: vec![std::env::temp_dir()],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
            inference_global_limit: 8,
            inference_per_principal_limit: 4,
            inference_waiter_limit: 128,
            inference_wait_timeout_ms: 5_000,
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: false,
            security_global_rps: None,
            security_tenant_rps: None,
            security_tenant_slots: 2048,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            media_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            media_worker_thread_budget: 64,
            transport: TransportMode::H3Experimental,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            webrtc_crop: None,
            gate_config: GateConfig::default(),
            novelty: vidarax_core::novelty::LiveNoveltyConfig::default(),
        };

        let server_task =
            tokio::spawn(async move { super::run(config).await.map_err(|e| e.to_string()) });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let client_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        client_socket
            .connect(format!("127.0.0.1:{h3_port}"))
            .await
            .unwrap();
        let (_conn, mut controller) = connect(client_socket, Some("test.com"))
            .await
            .expect("h3 client must connect");

        let headers = vec![
            Header::new(b":method", b"GET"),
            Header::new(b":scheme", b"https"),
            Header::new(b":authority", b"test.com"),
            Header::new(b":path", b"/v1/metrics"),
        ];
        controller
            .request_sender()
            .send(NewClientRequest {
                request_id: 1,
                headers,
                body_writer: None,
            })
            .expect("request should enqueue");

        let mut saw_headers = false;
        let mut saw_metrics_body = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

        while tokio::time::Instant::now() < deadline {
            let next = tokio::time::timeout(
                std::time::Duration::from_millis(500),
                controller.event_receiver_mut().recv(),
            )
            .await
            .ok()
            .flatten();

            let Some(event) = next else {
                continue;
            };

            if let ClientH3Event::Core(H3Event::IncomingHeaders(incoming)) = event {
                let status = incoming
                    .headers
                    .iter()
                    .find_map(|h| {
                        if h.name() == b":status" {
                            std::str::from_utf8(h.value()).ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or("");
                saw_headers = status == "200";

                let mut body = Vec::new();
                let mut recv = incoming.recv;
                while let Some(frame) = recv.recv().await {
                    if let InboundFrame::Body(chunk, fin) = frame {
                        body.extend_from_slice(chunk.as_ref());
                        if fin {
                            break;
                        }
                    }
                }
                saw_metrics_body = std::str::from_utf8(&body)
                    .map(|v| v.contains("vidarax_runs_created_total"))
                    .unwrap_or(false);
                break;
            }
        }

        server_task.abort();
        let _ = server_task.await;

        assert!(saw_headers, "expected h3 :status=200 response");
        assert!(
            saw_metrics_body,
            "expected h3 metrics payload containing vidarax_runs_created_total"
        );
    }

    #[cfg(feature = "h3-experimental")]
    fn reserve_tcp_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[cfg(feature = "h3-experimental")]
    fn reserve_udp_port() -> u16 {
        std::net::UdpSocket::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn parses_transport_mode() {
        assert_eq!(TransportMode::parse(None).unwrap(), TransportMode::H1H2);
        assert_eq!(
            TransportMode::parse(Some("h3")).unwrap(),
            TransportMode::H3Experimental
        );
        assert!(TransportMode::parse(Some("bad-mode")).is_err());
    }
}

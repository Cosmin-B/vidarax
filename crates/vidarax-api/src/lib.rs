#![forbid(unsafe_code)]

use std::{io, sync::Arc};

pub mod config;
mod auth;
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
pub mod spacetime_client;
mod state;
mod tenant_labels;
pub mod telemetry;
mod validation;
pub mod wal_sink;
mod whip;

pub use config::{resolve_wal_path, ServerConfig, TransportMode};
pub use models::AttachStreamRequest;
pub use router::app_router;
pub use state::AppState;
use vidarax_core::ingest::pipeline::{build_decode_pipeline, DecodePipeline, PipelineBackend};
use vidarax_core::webrtc::session::{TurnServer, WebRtcConfig};

pub async fn run(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    telemetry::init_telemetry();

    // Install the TLS crypto provider once, before any WebRTC sessions are
    // created.  rustrtc uses rustls for DTLS; it requires an installed
    // CryptoProvider.  `ok()` silences the error when a provider is already
    // installed (e.g. in tests).
    rustls::crypto::CryptoProvider::install_default(
        rustls::crypto::ring::default_provider(),
    )
    .ok();

    let wal_path = resolve_wal_path(&config).map_err(invalid_input)?;
    let config_path = std::env::var("VIDARAX_CONFIG").unwrap_or_else(|_| "vidarax.toml".to_string());
    let backend_config = config::load_backend_config(&config_path).map_err(invalid_input)?;
    let provider = if backend_config.backends.is_empty() {
        None
    } else {
        let backends = backend_config.backends;
        Some(
            tokio::task::spawn_blocking(move || {
                // reqwest::blocking builds and drops an internal runtime.
                vidarax_core::backends::build_provider_chain(&backends)
            })
            .await
            .map_err(|e| invalid_input(format!("failed to build provider chain: {e}")))?
                .map_err(|e| invalid_input(format!("failed to build provider chain: {e}")))?,
        )
    };
    let security_policy = security::SecurityPolicy::from_config(&config).map_err(invalid_input)?;
    let webrtc_config = build_webrtc_config(&config);
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
        config.distillation.clone(),
    )
    .map_err(invalid_input)?;
    let app = app_router(state);

    tracing::info!(transport = config.transport.label(), "vidarax-api startup");
    match config.transport {
        TransportMode::H1H2 => server::serve_h1h2(&config.bind_addr, app).await?,
        TransportMode::H3Experimental => server::serve_h3_experimental(&config, app).await?,
    }
    Ok(())
}

pub fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
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
            username: config
                .webrtc_turn_username
                .clone()
                .unwrap_or_default(),
            credential: config
                .webrtc_turn_credential
                .clone()
                .unwrap_or_default(),
        });
    }
    WebRtcConfig {
        stun_servers: config.webrtc_stun_servers.clone(),
        turn_servers,
        max_output_tokens_per_second: config.webrtc_max_output_tokens_per_second,
        decode_workers: config.webrtc_decode_workers,
        analysis_workers: config.webrtc_analysis_workers,
        vlm_workers: config.webrtc_vlm_workers,
    }
}


#[cfg(test)]
mod tests {
    use super::{app_router, AppState, ServerConfig, TransportMode};
    use vidarax_core::ingest::pipeline::{register_decode_backend, CpuFfmpegPipeline, PipelineBackend};
    use vidarax_core::tiered_vlm::DistillationConfig;
    use crate::security::SecurityPolicy;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use std::sync::Arc;
    use tower::ServiceExt;
    use vidarax_core::backends::BackendEntry;
    #[cfg(feature = "h3-experimental")]
    use {
        tokio_quiche::http3::driver::{ClientH3Event, H3Event, InboundFrame, NewClientRequest},
        tokio_quiche::quic::connect,
        tokio_quiche::quiche::h3::{Header, NameValue},
    };

    fn test_state() -> AppState {
        test_state_with_provider(None)
    }

    fn build_provider_from_url(base_url: &str) -> Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync> {
        let entry = BackendEntry {
            name: "test".to_string(),
            backend_type: "openai_compat".to_string(),
            base_url: Some(base_url.to_string()),
            api_key: None,
            model: None,
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

    fn test_state_with_provider(provider: Option<Arc<dyn vidarax_core::provider::InferenceProvider + Send + Sync>>) -> AppState {
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
        let app = app_router(test_state());
        let create = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .header("x-tenant-id", "tenant-a")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let create_resp = app.clone().oneshot(create).await.unwrap();
        assert_eq!(create_resp.status(), StatusCode::OK);
        let body = create_resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let run_id = json.get("run_id").and_then(|v| v.as_str()).unwrap();

        let denied = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-tenant-id", "tenant-b")
            .body(Body::empty())
            .unwrap();
        let denied_resp = app.clone().oneshot(denied).await.unwrap();
        assert_eq!(denied_resp.status(), StatusCode::NOT_FOUND);

        let allowed = Request::builder()
            .uri(format!("/v1/runs/{run_id}/state"))
            .method("GET")
            .header("x-tenant-id", "tenant-a")
            .body(Body::empty())
            .unwrap();
        let allowed_resp = app.oneshot(allowed).await.unwrap();
        assert_eq!(allowed_resp.status(), StatusCode::OK);
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
    async fn ingest_transitions_state_to_processing() {
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
        assert_eq!(ingest_response.status(), StatusCode::OK);

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
        assert!(text.contains("\"state\":\"processing\""));
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
        assert!(!providers.is_empty(), "at least one provider must be reported");
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
        if !ffmpeg_available() {
            eprintln!("skipping test: ffmpeg unavailable");
            return;
        }
        let Some(video_path) = create_test_mp4() else {
            eprintln!("skipping test: ffmpeg failed to generate fixture");
            return;
        };

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
        if !ffmpeg_available() {
            eprintln!("skipping test: ffmpeg unavailable");
            return;
        }
        let Some(video_path) = create_test_mp4() else {
            eprintln!("skipping test: ffmpeg failed to generate fixture");
            return;
        };

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
        if !ffmpeg_available() {
            eprintln!("skipping test: ffmpeg unavailable");
            return;
        }
        let Some(video_path) = create_test_mp4() else {
            eprintln!("skipping test: ffmpeg failed to generate fixture");
            return;
        };

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
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
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
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
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
            security_require_api_key: false,
            security_api_keys: vec![],
            security_require_tenant_id: true,
            security_global_rps: None,
            security_tenant_rps: Some(10),
            security_tenant_slots: 16,
            security_metrics_require_api_key: false,
            cors_allowed_origins: vec![],
            stream_ttl_secs: 3600,
            active_stream_limit: 5,
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
        })
        .unwrap();
        let app = app_router(test_state_with_provider_and_policy(None, policy));

        let req_missing = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let resp_missing = app.clone().oneshot(req_missing).await.unwrap();
        assert_eq!(resp_missing.status(), StatusCode::UNAUTHORIZED);

        let req_present = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("x-tenant-id", "tenant-a")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .unwrap();
        let resp_present = app.oneshot(req_present).await.unwrap();
        assert_eq!(resp_present.status(), StatusCode::OK);
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
            transport: TransportMode::H1H2,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
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
            transport: TransportMode::H3Experimental,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
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
            transport: TransportMode::H3Experimental,
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            distillation: DistillationConfig::default(),
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

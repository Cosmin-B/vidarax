use std::time::Duration;

use reqwest::StatusCode;
use serde_json::Value;
use vidarax_api::{run, ServerConfig, TransportMode};
use vidarax_core::ingest::pipeline::PipelineBackend;

#[tokio::test]
async fn staging_live_provider_e2e_opt_in() {
    let vllm = std::env::var("VIDARAX_STAGING_VLLM_BASE_URL").ok();
    let sglang = std::env::var("VIDARAX_STAGING_SGLANG_BASE_URL").ok();
    if vllm.is_none() || sglang.is_none() {
        eprintln!(
            "skipping staging live provider e2e: set VIDARAX_STAGING_VLLM_BASE_URL and VIDARAX_STAGING_SGLANG_BASE_URL"
        );
        return;
    }

    let bind_port = reserve_tcp_port();
    let data_dir = std::env::temp_dir().join(format!("vidarax-staging-e2e-{bind_port}"));
    std::fs::create_dir_all(&data_dir).unwrap();

    let config = ServerConfig {
        bind_addr: format!("127.0.0.1:{bind_port}"),
        h3_bind_addr: "127.0.0.1:18443".to_string(),
        h3_tls_cert_path: "deploy/certs/dev.crt".to_string(),
        h3_tls_key_path: "deploy/certs/dev.key".to_string(),
        data_dir: data_dir.to_string_lossy().to_string(),
        ingest_file_roots: vec![std::env::temp_dir()],
        inference_vllm_base_url: vllm,
        inference_sglang_base_url: sglang,
        security_require_api_key: false,
        security_api_keys: vec![],
        security_require_tenant_id: false,
        security_global_rps: None,
        security_tenant_rps: None,
        security_tenant_slots: 128,
        security_metrics_require_api_key: false,
        cors_allowed_origins: vec![],
        stream_ttl_secs: 3600,
        active_stream_limit: 5,
        transport: TransportMode::H1H2,
        decode_backend: PipelineBackend::CpuFfmpeg,
    };

    let server_task = tokio::spawn(async move { run(config).await.map_err(|e| e.to_string()) });
    tokio::time::sleep(Duration::from_millis(400)).await;

    let client = reqwest::Client::new();
    let create = client
        .post(format!("http://127.0.0.1:{bind_port}/v1/runs"))
        .json(&serde_json::json!({"mode":"balanced"}))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), StatusCode::OK);
    let create_json: Value = create.json().await.unwrap();
    let run_id = create_json.get("run_id").and_then(|v| v.as_str()).unwrap();

    let model = std::env::var("VIDARAX_STAGING_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-2B-Instruct".to_string());
    let prompt = std::env::var("VIDARAX_STAGING_PROMPT")
        .unwrap_or_else(|_| "Provide a single concise sentence.".to_string());
    let infer = client
        .post(format!("http://127.0.0.1:{bind_port}/v1/infer"))
        .json(&serde_json::json!({
            "run_id": run_id,
            "model": model,
            "prompt": prompt,
            "primary_provider": "vllm",
            "timeout_ms": 30000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(infer.status(), StatusCode::OK);
    let infer_json: Value = infer.json().await.unwrap();
    let output = infer_json
        .get("output_text")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        !output.trim().is_empty(),
        "live provider returned empty output"
    );

    server_task.abort();
    let _ = server_task.await;
}

fn reserve_tcp_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

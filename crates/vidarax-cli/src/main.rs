#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use reqwest::Method;
use serde_json::{Map, Value};
use vidarax_contracts::lifecycle::StreamState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .with_target(false)
        .without_time()
        .init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("run-create") => {
            cmd_run_create(RunCreateOpts::parse(args)).await;
        }
        Some("health") => {
            cmd_health(ApiOpts::from_env(args)).await;
        }
        Some("models") => {
            cmd_models(ApiOpts::from_env(args)).await;
        }
        Some("states") => {
            let states = [
                StreamState::Pending,
                StreamState::Processing,
                StreamState::Completed,
                StreamState::Failed,
                StreamState::Cancelled,
                StreamState::Expired,
            ];
            for state in states {
                println!("{state:?}\tterminal={}", state.is_terminal());
            }
        }
        Some("distill") => {
            let sub = args.next();
            let opts = DistillOpts::parse(args);
            match sub.as_deref() {
                Some("status") => cmd_distill_status(opts).await,
                Some("train") => cmd_distill_train(opts).await,
                Some("export") => cmd_distill_export(opts).await,
                Some("deploy") => cmd_distill_deploy(opts).await,
                _ => {
                    eprintln!("Usage:");
                    eprintln!("  vidarax-cli distill status  --tenant-id <id> [--data-dir <dir>]");
                    eprintln!("  vidarax-cli distill train   --tenant-id <id> [--data-dir <dir>] [--mlx] [--resume]");
                    eprintln!("  vidarax-cli distill export  --tenant-id <id> [--data-dir <dir>] [--format gguf|onnx|mlx]");
                    eprintln!("  vidarax-cli distill deploy  --tenant-id <id> [--data-dir <dir>]");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  vidarax-cli run-create [--url <base>] [--api-key <key>] [--tenant-id <id>] [--mode <mode>] [--model <model>]");
            eprintln!("  vidarax-cli health [--url <base>] [--api-key <key>] [--tenant-id <id>]");
            eprintln!("  vidarax-cli models [--url <base>] [--api-key <key>] [--tenant-id <id>]");
            eprintln!("  vidarax-cli states");
            eprintln!("  vidarax-cli distill <status|train|export|deploy> ...");
        }
    }
}

// ─── Options ──────────────────────────────────────────────────────────────────

const DEFAULT_API_URL: &str = "http://127.0.0.1:8080";
const API_TIMEOUT_SECS: u64 = 10;

struct ApiOpts {
    base_url: String,
    api_key: Option<String>,
    tenant_id: Option<String>,
}

impl ApiOpts {
    fn from_env(args: impl Iterator<Item = String>) -> Self {
        Self::parse(
            args,
            std::env::var("VIDARAX_API_URL").ok(),
            std::env::var("VIDARAX_API_KEY").ok(),
            std::env::var("VIDARAX_TENANT_ID").ok(),
        )
    }

    fn parse(
        args: impl Iterator<Item = String>,
        env_url: Option<String>,
        env_api_key: Option<String>,
        env_tenant_id: Option<String>,
    ) -> Self {
        let args: Vec<String> = args.collect();
        let mut flag_url = None;
        let mut flag_api_key = None;
        let mut flag_tenant_id = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--url" => {
                    flag_url = non_empty(flag_value_or_exit(&args, i, "--url"));
                    i += 2;
                }
                "--api-key" => {
                    flag_api_key = non_empty(flag_value_or_exit(&args, i, "--api-key"));
                    i += 2;
                }
                "--tenant-id" => {
                    flag_tenant_id = non_empty(flag_value_or_exit(&args, i, "--tenant-id"));
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }

        let base_url = choose_config_value(flag_url, env_url)
            .unwrap_or_else(|| DEFAULT_API_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        Self {
            base_url,
            api_key: choose_config_value(flag_api_key, env_api_key),
            tenant_id: choose_config_value(flag_tenant_id, env_tenant_id),
        }
    }
}

struct RunCreateOpts {
    api: ApiOpts,
    mode: Option<String>,
    model: Option<String>,
}

impl RunCreateOpts {
    fn parse(args: impl Iterator<Item = String>) -> Self {
        let args: Vec<String> = args.collect();
        let api = ApiOpts::from_env(args.clone().into_iter());
        let mut mode = None;
        let mut model = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--mode" => {
                    mode = non_empty(flag_value_or_exit(&args, i, "--mode"));
                    i += 2;
                }
                "--model" => {
                    model = non_empty(flag_value_or_exit(&args, i, "--model"));
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }

        Self { api, mode, model }
    }

    fn request_body(&self) -> Value {
        let mut body = Map::new();
        if let Some(mode) = &self.mode {
            body.insert("mode".to_string(), Value::String(mode.clone()));
        }
        if let Some(model) = &self.model {
            body.insert("model".to_string(), Value::String(model.clone()));
        }
        Value::Object(body)
    }
}

fn choose_config_value(flag: Option<String>, env: Option<String>) -> Option<String> {
    flag.or_else(|| env.and_then(non_empty))
}

fn flag_value(args: &[String], i: usize, flag: &str) -> Result<String, String> {
    match args.get(i + 1) {
        Some(value) if !value.starts_with("--") => Ok(value.clone()),
        _ => Err(format!("{flag} requires a value")),
    }
}

fn flag_value_or_exit(args: &[String], i: usize, flag: &str) -> String {
    flag_value(args, i, flag).unwrap_or_else(|e| {
        exit_with_error(e);
    })
}

fn non_empty(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

struct DistillOpts {
    tenant_id: Option<String>,
    data_dir: PathBuf,
    mlx: bool,
    resume: bool,
    format: String,
}

impl DistillOpts {
    fn parse(args: impl Iterator<Item = String>) -> Self {
        let mut opts = Self {
            tenant_id: None,
            data_dir: PathBuf::from(".vidarax-data"),
            mlx: false,
            resume: false,
            format: "gguf".to_string(),
        };
        let args: Vec<String> = args.collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--tenant-id" if i + 1 < args.len() => {
                    opts.tenant_id = Some(args[i + 1].clone());
                    i += 2;
                }
                "--data-dir" if i + 1 < args.len() => {
                    opts.data_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                }
                "--mlx" => {
                    opts.mlx = true;
                    i += 1;
                }
                "--resume" => {
                    opts.resume = true;
                    i += 1;
                }
                "--format" if i + 1 < args.len() => {
                    opts.format = args[i + 1].clone();
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }
        opts
    }

    fn require_tenant_id(&self) -> &str {
        self.tenant_id.as_deref().unwrap_or_else(|| {
            eprintln!("error: --tenant-id is required");
            std::process::exit(1);
        })
    }
}

// ─── API commands ─────────────────────────────────────────────────────────────

struct ApiClient {
    opts: ApiOpts,
    http: reqwest::Client,
}

impl ApiClient {
    fn new(opts: ApiOpts) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(API_TIMEOUT_SECS))
            .build()
            .map_err(|e| format!("failed to create HTTP client: {e}"))?;
        Ok(Self { opts, http })
    }

    async fn get_json(&self, path: &str) -> Result<Value, String> {
        self.request_json(Method::GET, path, None).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.request_json(Method::POST, path, Some(body)).await
    }

    async fn request_json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, String> {
        let url = format!("{}{}", self.opts.base_url, path);
        let mut request = self.http.request(method, &url);
        if let Some(api_key) = &self.opts.api_key {
            request = request.header("x-api-key", api_key);
        }
        if let Some(tenant_id) = &self.opts.tenant_id {
            request = request.header("x-tenant-id", tenant_id);
        }
        if let Some(body) = body {
            request = request.json(body);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("request to {url} failed: {e}"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| format!("request to {url} failed while reading response body: {e}"))?;

        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|body| extract_error_message(&body))
                .or_else(|| non_empty(text.clone()))
                .unwrap_or_else(|| "empty response body".to_string());
            return Err(format!("request to {url} returned {status}: {message}"));
        }

        serde_json::from_str::<Value>(&text)
            .map_err(|e| format!("request to {url} returned invalid JSON: {e}"))
    }
}

async fn cmd_health(opts: ApiOpts) {
    let base_url = opts.base_url.clone();
    let client = api_client_or_exit(opts);
    let body = client.get_json("/v1/health").await.unwrap_or_else(|e| {
        exit_with_error(e);
    });

    if body.get("status").and_then(Value::as_str).is_none() {
        exit_with_error(format!(
            "response from {base_url}/v1/health is missing string field status"
        ));
    }

    println!("{body}");
}

async fn cmd_models(opts: ApiOpts) {
    let base_url = opts.base_url.clone();
    let client = api_client_or_exit(opts);
    let body = client.get_json("/v1/models").await.unwrap_or_else(|e| {
        exit_with_error(e);
    });
    let model_ids = extract_model_ids(&body).unwrap_or_else(|e| {
        exit_with_error(format!("response from {base_url}/v1/models {e}"));
    });

    for model_id in model_ids {
        println!("{model_id}");
    }
}

async fn cmd_run_create(opts: RunCreateOpts) {
    let base_url = opts.api.base_url.clone();
    let body = opts.request_body();
    let client = api_client_or_exit(opts.api);
    let response = client
        .post_json("/v1/runs", &body)
        .await
        .unwrap_or_else(|e| {
            exit_with_error(e);
        });

    validate_run_create_response(&response).unwrap_or_else(|e| {
        exit_with_error(format!("response from {base_url}/v1/runs {e}"));
    });

    println!("{response}");
}

fn api_client_or_exit(opts: ApiOpts) -> ApiClient {
    ApiClient::new(opts).unwrap_or_else(|e| {
        exit_with_error(e);
    })
}

fn exit_with_error(message: String) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}

fn validate_run_create_response(response: &Value) -> Result<(), String> {
    for field in ["run_id", "request_id", "status", "mode"] {
        if response.get(field).and_then(Value::as_str).is_none() {
            return Err(format!("is missing string field {field}"));
        }
    }

    if response
        .get("model")
        .is_some_and(|model| !model.is_null() && !model.is_string())
    {
        return Err("has non-string field model".to_string());
    }

    Ok(())
}

fn extract_model_ids(body: &Value) -> Result<Vec<String>, String> {
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "is missing array field models".to_string())?;

    models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            model
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("is missing string field models[{i}].id"))
        })
        .collect()
}

fn extract_error_message(body: &Value) -> Option<String> {
    let error = body.get("error")?;
    let message = error.get("message")?.as_str()?.to_string();
    let details = error
        .get("details")
        .and_then(Value::as_array)
        .map(|details| {
            details
                .iter()
                .filter_map(|detail| {
                    let detail_message = detail.get("message")?.as_str()?;
                    match detail.get("field").and_then(Value::as_str) {
                        Some(field) if !field.is_empty() => {
                            Some(format!("{field}: {detail_message}"))
                        }
                        _ => Some(detail_message.to_string()),
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if details.is_empty() {
        Some(message)
    } else {
        Some(format!("{message}: {}", details.join("; ")))
    }
}

// ─── distill status ───────────────────────────────────────────────────────────

/// Show the number of stored training pairs and database location.
#[cfg(feature = "training")]
async fn cmd_distill_status(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    match vidarax_core::training_data::TrainingStore::new(&opts.data_dir) {
        Ok(store) => {
            let count = store.pair_count(tenant_id).unwrap_or(0);
            println!("tenant_id : {tenant_id}");
            println!("pairs     : {count}");
            println!("data_dir  : {}", opts.data_dir.display());
        }
        Err(e) => {
            eprintln!("error opening training store: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "training"))]
async fn cmd_distill_status(_opts: DistillOpts) {
    eprintln!("error: training support is not compiled in (rebuild with --features training)");
    std::process::exit(1);
}

// ─── distill train ────────────────────────────────────────────────────────────

/// Spawn the Python training script and stream its output.
#[cfg(feature = "training")]
async fn cmd_distill_train(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    // Export a fresh JSONL snapshot before kicking off training.
    let jsonl_path = opts.data_dir.join(format!("{tenant_id}-training.jsonl"));
    match vidarax_core::training_data::TrainingStore::new(&opts.data_dir) {
        Ok(store) => match store.export_training_jsonl(tenant_id, &jsonl_path) {
            Ok(n) => {
                tracing::info!(count = n, path = %jsonl_path.display(), "exported training pairs")
            }
            Err(e) => {
                eprintln!("error exporting training data: {e}");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("error opening training store: {e}");
            std::process::exit(1);
        }
    }

    let script = if opts.mlx {
        "scripts/train_specialist_mlx.py"
    } else {
        "scripts/train_specialist.py"
    };

    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(script)
        .arg("--data-dir")
        .arg(&opts.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if opts.resume {
        cmd.arg("--resume");
    }

    tracing::info!(script, tenant_id, "starting training subprocess");
    run_subprocess(cmd).await;
}

#[cfg(not(feature = "training"))]
async fn cmd_distill_train(_opts: DistillOpts) {
    eprintln!("error: training support is not compiled in (rebuild with --features training)");
    std::process::exit(1);
}

// ─── distill export ───────────────────────────────────────────────────────────

/// Spawn the Python export script to convert the fine-tuned model.
async fn cmd_distill_export(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    let format = opts.format.as_str();
    if !matches!(format, "gguf" | "onnx" | "mlx") {
        eprintln!("error: --format must be one of: gguf, onnx, mlx");
        std::process::exit(1);
    }

    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg("scripts/export_specialist.py")
        .arg("--tenant-id")
        .arg(tenant_id)
        .arg("--format")
        .arg(format)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    tracing::info!(tenant_id, format, "starting export subprocess");
    run_subprocess(cmd).await;
}

// ─── distill deploy ───────────────────────────────────────────────────────────

/// Write a `.reload-specialist` sentinel file that the running API polls to
/// trigger a hot-reload of the specialist model weights.
async fn cmd_distill_deploy(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    let sentinel = opts.data_dir.join(".reload-specialist");
    match std::fs::write(&sentinel, tenant_id) {
        Ok(()) => {
            tracing::info!(
                path = %sentinel.display(),
                tenant_id,
                "reload sentinel written — running API will pick this up"
            );
            println!("sentinel written: {}", sentinel.display());
        }
        Err(e) => {
            eprintln!("error writing sentinel: {e}");
            std::process::exit(1);
        }
    }
}

// ─── Subprocess runner ────────────────────────────────────────────────────────

/// Spawn the command, stream its stdout/stderr via tracing, then wait for exit.
async fn run_subprocess(mut cmd: tokio::process::Command) {
    use tokio::io::AsyncBufReadExt as _;

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn subprocess: {e}");
            std::process::exit(1);
        }
    };

    // Stream stdout
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("[subprocess] {}", line);
            }
        });
    }

    // Stream stderr
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::warn!("[subprocess] {}", line);
            }
        });
    }

    match child.wait().await {
        Ok(status) if status.success() => {
            tracing::info!("subprocess exited successfully");
        }
        Ok(status) => {
            eprintln!("subprocess exited with: {status}");
            std::process::exit(status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("subprocess wait error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_opts_prefers_non_empty_flags_over_env_and_defaults() {
        let opts = ApiOpts::parse(
            [
                "--url",
                "http://localhost:9000/",
                "--api-key",
                "flag-key",
                "--tenant-id",
                "tenant-a",
            ]
            .into_iter()
            .map(String::from),
            Some("http://env.example".to_string()),
            Some("env-key".to_string()),
            Some("env-tenant".to_string()),
        );

        assert_eq!(opts.base_url, "http://localhost:9000");
        assert_eq!(opts.api_key.as_deref(), Some("flag-key"));
        assert_eq!(opts.tenant_id.as_deref(), Some("tenant-a"));
    }

    #[test]
    fn api_opts_uses_env_when_flags_are_empty() {
        let opts = ApiOpts::parse(
            ["--url", "", "--api-key", "", "--tenant-id", ""]
                .into_iter()
                .map(String::from),
            Some("http://env.example/".to_string()),
            Some("env-key".to_string()),
            Some("env-tenant".to_string()),
        );

        assert_eq!(opts.base_url, "http://env.example");
        assert_eq!(opts.api_key.as_deref(), Some("env-key"));
        assert_eq!(opts.tenant_id.as_deref(), Some("env-tenant"));
    }

    #[test]
    fn flag_value_rejects_flag_shaped_value() {
        let good = ["--url", "http://localhost:9000"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let flag_shaped = ["--url", "--api-key", "k"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        assert_eq!(
            flag_value(&good, 0, "--url").unwrap(),
            "http://localhost:9000"
        );
        assert_eq!(
            flag_value(&flag_shaped, 0, "--url").unwrap_err(),
            "--url requires a value"
        );
    }

    #[test]
    fn run_create_opts_omits_absent_mode_and_model() {
        let opts = RunCreateOpts::parse(std::iter::empty::<String>());

        assert_eq!(opts.mode, None);
        assert_eq!(opts.model, None);
        assert_eq!(opts.request_body(), serde_json::json!({}));
    }

    #[test]
    fn run_create_response_accepts_null_or_absent_model() {
        let with_null_model = serde_json::json!({
            "run_id": "run-1",
            "request_id": "req-1",
            "status": "pending",
            "mode": "balanced",
            "model": null
        });
        let without_model = serde_json::json!({
            "run_id": "run-1",
            "request_id": "req-1",
            "status": "pending",
            "mode": "balanced"
        });

        assert!(validate_run_create_response(&with_null_model).is_ok());
        assert!(validate_run_create_response(&without_model).is_ok());
    }

    #[test]
    fn run_create_response_rejects_malformed_required_fields() {
        let missing_run_id = serde_json::json!({
            "request_id": "req-1",
            "status": "pending",
            "mode": "balanced",
            "model": null
        });
        let non_string_status = serde_json::json!({
            "run_id": "run-1",
            "request_id": "req-1",
            "status": 200,
            "mode": "balanced",
            "model": null
        });
        let non_string_model = serde_json::json!({
            "run_id": "run-1",
            "request_id": "req-1",
            "status": "pending",
            "mode": "balanced",
            "model": 123
        });

        assert!(validate_run_create_response(&missing_run_id).is_err());
        assert!(validate_run_create_response(&non_string_status).is_err());
        assert_eq!(
            validate_run_create_response(&non_string_model).unwrap_err(),
            "has non-string field model"
        );
    }

    #[test]
    fn live_models_extracts_ids_in_order() {
        let body = serde_json::json!({
            "request_id": "req-1",
            "models": [
                {"id": "model-a", "tier": "fast"},
                {"id": "model-b", "tier": "balanced"}
            ]
        });

        assert_eq!(
            extract_model_ids(&body).unwrap(),
            vec!["model-a".to_string(), "model-b".to_string()]
        );
    }

    #[test]
    fn error_envelope_includes_message_and_field_details() {
        let body = serde_json::json!({
            "error": {
                "message": "invalid request",
                "details": [
                    {"field": "mode", "message": "unsupported mode"}
                ]
            }
        });

        assert_eq!(
            extract_error_message(&body).as_deref(),
            Some("invalid request: mode: unsupported mode")
        );
    }
}

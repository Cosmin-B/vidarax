#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use comfy_table::{presets, Cell, Table};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use owo_colors::OwoColorize;
use reqwest::Method;
use serde_json::{json, Map, Value};
use vidarax_contracts::lifecycle::StreamState;

const DEFAULT_API_URL: &str = "http://127.0.0.1:8080";
const API_TIMEOUT_SECS: u64 = 10;
const DEFAULT_ANALYZE_MODEL: &str = "vidarax-medium";
const ANALYZE_PROGRESS_POLL_TIMEOUT: Duration = Duration::from_secs(5);

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

    let cli = Cli::parse();
    let global = cli.global.clone();
    let color = !global.json && should_use_color(&global);
    let output = OutputMode {
        json: global.json,
        color,
    };

    let config = match RuntimeConfig::from_global_args(&global) {
        Ok(config) => config,
        Err(e) => exit_with_error(e),
    };

    if let Err(e) = dispatch(cli.command, &global, config, output).await {
        exit_with_error(e);
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "vidarax",
    about = "Inspect and manage Vidarax API runs",
    after_long_help = "Examples:
  vidarax health --url http://127.0.0.1:8080
  vidarax runs list --tenant-id demo
  vidarax search \"person entering lobby\" --run run_01 --limit 10
"
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug, Clone)]
struct GlobalArgs {
    /// API base URL.
    #[arg(long, value_name = "URL", global = true)]
    url: Option<String>,
    /// API key sent as x-api-key.
    #[arg(long, value_name = "KEY", global = true)]
    api_key: Option<String>,
    /// Tenant ID sent as x-tenant-id.
    #[arg(long, value_name = "ID", global = true)]
    tenant_id: Option<String>,
    /// Print raw API JSON responses.
    #[arg(long, global = true)]
    json: bool,
    /// Disable ANSI color.
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Work with analysis runs.
    #[command(subcommand)]
    Runs(RunsCommands),
    /// List events for a run.
    Events(EventsArgs),
    /// List markers for a run.
    Markers(MarkersArgs),
    /// Search indexed run events.
    #[command(after_long_help = "Examples:
  vidarax search \"red car turning\" --limit 5
  vidarax search \"door opened\" --run run_01 --json
")]
    Search(SearchArgs),
    /// Upload a local video and run the full analysis pipeline.
    Analyze(AnalyzeArgs),
    /// Submit and list feedback.
    #[command(subcommand)]
    Feedback(FeedbackCommands),
    /// List model availability.
    Models,
    /// Check API health.
    Health,
    /// Check local config and API readiness.
    #[command(after_long_help = "Examples:
  vidarax doctor
  vidarax doctor --url http://127.0.0.1:8080
")]
    Doctor,
    /// Show resolved CLI configuration.
    #[command(subcommand)]
    Config(ConfigCommands),
    /// List stream states and terminal status.
    States,
    /// Run local distillation helpers.
    #[command(subcommand)]
    Distill(DistillCommands),
    /// Create a run.
    #[command(name = "run-create", hide = true)]
    RunCreate(RunCreateArgs),
}

#[derive(Subcommand, Debug)]
enum RunsCommands {
    /// List runs.
    List,
    /// Show one run.
    Show { run_id: String },
    /// Create a run.
    Create(RunCreateArgs),
    /// Delete a run.
    Rm { run_id: String },
}

#[derive(Args, Debug, Clone)]
struct RunCreateArgs {
    /// Run mode.
    #[arg(long, value_name = "MODE")]
    mode: Option<String>,
    /// Model ID.
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
}

impl RunCreateArgs {
    fn request_body(&self) -> Value {
        let mut body = Map::new();
        if let Some(mode) = self.mode.as_deref().and_then(non_empty_opt) {
            body.insert("mode".to_string(), Value::String(mode.to_string()));
        }
        if let Some(model) = self.model.as_deref().and_then(non_empty_opt) {
            body.insert("model".to_string(), Value::String(model.to_string()));
        }
        Value::Object(body)
    }
}

#[derive(Args, Debug)]
struct EventsArgs {
    /// Run ID.
    run_id: String,
    /// Filter events by kind after fetching.
    #[arg(long, value_name = "KIND")]
    kind: Option<String>,
    /// Event index name.
    #[arg(long, value_name = "NAME")]
    index: Option<String>,
}

#[derive(Args, Debug)]
struct MarkersArgs {
    /// Run ID.
    run_id: String,
    /// Marker status filter.
    #[arg(long, value_name = "S")]
    status: Option<String>,
    /// Event type filter.
    #[arg(long, value_name = "T")]
    event_type: Option<String>,
    /// Start frame filter.
    #[arg(long, value_name = "N")]
    from_frame: Option<u64>,
    /// End frame filter.
    #[arg(long, value_name = "N")]
    to_frame: Option<u64>,
}

#[derive(Args, Debug)]
struct SearchArgs {
    /// Search query.
    query: String,
    /// Limit results to one run.
    #[arg(long, value_name = "RUN_ID")]
    run: Option<String>,
    /// Maximum number of results.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,
}

/// A region of interest parsed from the `--crop X,Y,W,H` flag, as fractions of
/// the frame. Shape is validated here; the server enforces the range and
/// in-frame bounds so both sides agree on one rule.
#[derive(Debug, Clone, Copy)]
struct CropArg {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl std::str::FromStr for CropArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(
                "crop must be four comma-separated fractions: X,Y,WIDTH,HEIGHT".to_string(),
            );
        }
        let mut vals = [0f32; 4];
        for (slot, part) in vals.iter_mut().zip(parts.iter()) {
            *slot = part
                .parse::<f32>()
                .map_err(|_| format!("crop value '{part}' is not a number"))?;
        }
        Ok(CropArg {
            x: vals[0],
            y: vals[1],
            width: vals[2],
            height: vals[3],
        })
    }
}

impl CropArg {
    fn to_json(self) -> Value {
        json!({
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        })
    }
}

#[derive(Args, Debug, Clone)]
struct AnalyzeArgs {
    /// Local video file to analyze.
    file: PathBuf,
    /// Optional semantic prompt.
    #[arg(long, value_name = "TEXT")]
    prompt: Option<String>,
    /// Model ID.
    #[arg(long, value_name = "ID", default_value = DEFAULT_ANALYZE_MODEL)]
    model: String,
    /// Optional run mode.
    #[arg(long, value_name = "MODE")]
    mode: Option<String>,
    /// Reason decode FPS.
    #[arg(long, value_name = "F", default_value_t = 1.0)]
    fixed_fps: f32,
    /// Reason chunk size.
    #[arg(long, value_name = "N", default_value_t = 25)]
    chunk_size: usize,
    /// Cap frames for ingest and reason.
    #[arg(long, value_name = "N")]
    max_frames: Option<u64>,
    /// Reason index name and events poll filter.
    #[arg(long, value_name = "NAME")]
    index_name: Option<String>,
    /// Optional ingest sampling policy.
    #[arg(long, value_name = "P")]
    sampling_policy: Option<String>,
    /// Analyze only a region of the frame: X,Y,WIDTH,HEIGHT as fractions in
    /// [0,1] (e.g. 0.25,0.25,0.5,0.5 for the center half). Restricts both the
    /// gate and the VLM to that part of the screen. Omit to analyze the whole frame.
    #[arg(long, value_name = "X,Y,W,H")]
    crop: Option<CropArg>,
}

impl AnalyzeArgs {
    fn create_run_body(&self) -> Value {
        let mut body = Map::new();
        if let Some(mode) = self.mode.as_deref().and_then(non_empty_opt) {
            body.insert("mode".to_string(), Value::String(mode.to_string()));
        }
        body.insert("model".to_string(), Value::String(self.model.clone()));
        Value::Object(body)
    }

    fn ingest_body(&self, source_uri: &str) -> Value {
        let mut body = Map::new();
        body.insert(
            "source_uri".to_string(),
            Value::String(source_uri.to_string()),
        );
        if let Some(policy) = self.sampling_policy.as_deref().and_then(non_empty_opt) {
            body.insert(
                "sampling_policy".to_string(),
                Value::String(policy.to_string()),
            );
            if policy.eq_ignore_ascii_case("fixed") {
                body.insert("fixed_fps".to_string(), json!(self.fixed_fps));
            }
        }
        if let Some(max_frames) = self.max_frames {
            body.insert("max_frames".to_string(), json!(max_frames));
        }
        Value::Object(body)
    }

    fn reason_body(&self, source_uri: &str) -> Value {
        let mut body = Map::new();
        body.insert(
            "source_uri".to_string(),
            Value::String(source_uri.to_string()),
        );
        body.insert("model".to_string(), Value::String(self.model.clone()));
        if let Some(mode) = self.mode.as_deref().and_then(non_empty_opt) {
            body.insert("mode".to_string(), Value::String(mode.to_string()));
        }
        body.insert(
            "sampling_policy".to_string(),
            Value::String("fixed".to_string()),
        );
        body.insert("fixed_fps".to_string(), json!(self.fixed_fps));
        body.insert("chunk_size".to_string(), json!(self.chunk_size));
        body.insert("semantic_inference".to_string(), Value::Bool(true));
        if let Some(max_frames) = self.max_frames {
            body.insert("max_frames".to_string(), json!(max_frames));
        }
        if let Some(index_name) = self.index_name.as_deref().and_then(non_empty_opt) {
            body.insert(
                "index_name".to_string(),
                Value::String(index_name.to_string()),
            );
        }
        if let Some(prompt) = self.prompt.as_deref().and_then(non_empty_opt) {
            body.insert(
                "semantic_prompt".to_string(),
                Value::String(prompt.to_string()),
            );
        }
        if let Some(crop) = self.crop {
            body.insert("crop".to_string(), crop.to_json());
        }
        Value::Object(body)
    }
}

#[derive(Subcommand, Debug)]
enum FeedbackCommands {
    /// Submit feedback for a run.
    #[command(after_long_help = "Examples:
  vidarax feedback submit run_01 --rating 8 --category useful
  vidarax feedback submit run_01 --rating 3 --category miss --note \"missed the doorway event\"
")]
    Submit(FeedbackSubmitArgs),
    /// List submitted feedback.
    List,
}

#[derive(Args, Debug)]
struct FeedbackSubmitArgs {
    /// Run ID.
    run_id: String,
    /// Rating from 0 to 10.
    #[arg(long, value_name = "0-10")]
    rating: u32,
    /// Feedback category.
    #[arg(long, value_name = "CATEGORY")]
    category: String,
    /// Optional feedback note.
    #[arg(long, value_name = "TEXT")]
    note: Option<String>,
}

#[derive(Subcommand, Debug)]
enum ConfigCommands {
    /// Show resolved configuration.
    Show,
}

#[derive(Subcommand, Debug)]
enum DistillCommands {
    /// Show stored training data status.
    Status(DistillCliArgs),
    /// Train a local specialist model.
    Train(DistillCliArgs),
    /// Export a local specialist model.
    Export(DistillCliArgs),
    /// Ask the API to reload specialist weights.
    Deploy(DistillCliArgs),
}

#[derive(Args, Debug, Clone)]
struct DistillCliArgs {
    /// Training data directory.
    #[arg(long, value_name = "DIR", default_value = ".vidarax-data")]
    data_dir: PathBuf,
    /// Use the MLX training script.
    #[arg(long)]
    mlx: bool,
    /// Resume training.
    #[arg(long)]
    resume: bool,
    /// Export format.
    #[arg(long, value_name = "gguf|onnx|mlx", default_value = "gguf")]
    format: String,
}

impl DistillCliArgs {
    fn to_opts(&self, tenant_id: Option<String>) -> DistillOpts {
        DistillOpts {
            tenant_id,
            data_dir: self.data_dir.clone(),
            mlx: self.mlx,
            resume: self.resume,
            format: self.format.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct OutputMode {
    json: bool,
    color: bool,
}

async fn dispatch(
    command: Commands,
    global: &GlobalArgs,
    config: RuntimeConfig,
    output: OutputMode,
) -> Result<(), String> {
    match command {
        Commands::Runs(command) => match command {
            RunsCommands::List => cmd_runs_list(config, output).await,
            RunsCommands::Show { run_id } => cmd_runs_show(config, output, &run_id).await,
            RunsCommands::Create(args) => cmd_run_create(config, output, &args).await,
            RunsCommands::Rm { run_id } => cmd_runs_rm(config, output, &run_id).await,
        },
        Commands::Events(args) => cmd_events(config, output, &args).await,
        Commands::Markers(args) => cmd_markers(config, output, &args).await,
        Commands::Search(args) => cmd_search(config, output, &args).await,
        Commands::Analyze(args) => cmd_analyze(config, output, &args).await,
        Commands::Feedback(command) => match command {
            FeedbackCommands::Submit(args) => cmd_feedback_submit(config, output, &args).await,
            FeedbackCommands::List => cmd_feedback_list(config, output).await,
        },
        Commands::Models => cmd_models(config, output).await,
        Commands::Health => cmd_health(config, output).await,
        Commands::Doctor => cmd_doctor(config, output).await,
        Commands::Config(ConfigCommands::Show) => cmd_config_show(config, output),
        Commands::States => cmd_states(output),
        Commands::Distill(command) => {
            let tenant_id = global.tenant_id.clone();
            match command {
                DistillCommands::Status(args) => cmd_distill_status(args.to_opts(tenant_id)).await,
                DistillCommands::Train(args) => cmd_distill_train(args.to_opts(tenant_id)).await,
                DistillCommands::Export(args) => cmd_distill_export(args.to_opts(tenant_id)).await,
                DistillCommands::Deploy(args) => cmd_distill_deploy(args.to_opts(tenant_id)).await,
            }
            Ok(())
        }
        Commands::RunCreate(args) => cmd_run_create(config, output, &args).await,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeConfig {
    base_url: ResolvedValue,
    api_key: ResolvedValue,
    tenant_id: ResolvedValue,
}

impl RuntimeConfig {
    fn from_global_args(global: &GlobalArgs) -> Result<Self, String> {
        let file = load_config_file()?;
        let env_url = env::var("VIDARAX_API_URL").ok();
        let env_api_key = env::var("VIDARAX_API_KEY").ok();
        let env_tenant_id = env::var("VIDARAX_TENANT_ID").ok();

        let base_url = resolve_config_value(
            global.url.as_deref(),
            env_url.as_deref(),
            file.get("api_url").map(String::as_str),
            Some(DEFAULT_API_URL),
        )
        .trim_trailing_slash();
        let api_key = resolve_config_value(
            global.api_key.as_deref(),
            env_api_key.as_deref(),
            file.get("api_key").map(String::as_str),
            None,
        );
        let tenant_id = resolve_config_value(
            global.tenant_id.as_deref(),
            env_tenant_id.as_deref(),
            file.get("tenant_id").map(String::as_str),
            None,
        );

        Ok(Self {
            base_url,
            api_key,
            tenant_id,
        })
    }

    fn api_opts(&self) -> ApiOpts {
        ApiOpts {
            base_url: self
                .base_url
                .value
                .clone()
                .unwrap_or_else(|| DEFAULT_API_URL.to_string()),
            api_key: self.api_key.value.clone(),
            tenant_id: self.tenant_id.value.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedValue {
    value: Option<String>,
    source: ConfigSource,
}

impl ResolvedValue {
    fn trim_trailing_slash(mut self) -> Self {
        if let Some(value) = &mut self.value {
            *value = value.trim_end_matches('/').to_string();
        }
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigSource {
    Flag,
    Env,
    File,
    Default,
}

impl ConfigSource {
    fn as_str(self) -> &'static str {
        match self {
            ConfigSource::Flag => "flag",
            ConfigSource::Env => "env",
            ConfigSource::File => "file",
            ConfigSource::Default => "default",
        }
    }
}

fn resolve_config_value(
    flag: Option<&str>,
    env: Option<&str>,
    file: Option<&str>,
    default: Option<&str>,
) -> ResolvedValue {
    for (source, value) in [
        (ConfigSource::Flag, flag),
        (ConfigSource::Env, env),
        (ConfigSource::File, file),
        (ConfigSource::Default, default),
    ] {
        if let Some(value) = value.and_then(non_empty_opt) {
            return ResolvedValue {
                value: Some(value.to_string()),
                source,
            };
        }
    }

    ResolvedValue {
        value: None,
        source: ConfigSource::Default,
    }
}

fn parse_config_file(contents: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if matches!(key.as_str(), "api_url" | "api_key" | "tenant_id") {
            values.insert(key, value.trim().to_string());
        }
    }
    values
}

fn load_config_file() -> Result<HashMap<String, String>, String> {
    let Some(path) = config_file_path() else {
        return Ok(HashMap::new());
    };

    match fs::read_to_string(&path) {
        Ok(contents) => Ok(parse_config_file(&contents)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(e) => Err(format!("failed to read {}: {e}", path.display())),
    }
}

fn config_file_path() -> Option<PathBuf> {
    env::var_os("VIDARAX_CONFIG")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME").map(|home| {
                PathBuf::from(home)
                    .join(".config")
                    .join("vidarax")
                    .join("config")
            })
        })
}

fn non_empty_opt(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn should_use_color(global: &GlobalArgs) -> bool {
    !global.no_color && env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

fn fmt_pts(pts_ms: u64) -> String {
    let minutes = pts_ms / 60_000;
    let seconds = (pts_ms / 1_000) % 60;
    let millis = pts_ms % 1_000;
    format!("{minutes}:{seconds:02}.{millis:03}")
}

#[derive(Clone)]
struct ApiOpts {
    base_url: String,
    api_key: Option<String>,
    tenant_id: Option<String>,
}

#[derive(Clone)]
struct ApiClient {
    opts: ApiOpts,
    http: reqwest::Client,
}

impl ApiClient {
    fn new(opts: ApiOpts) -> Result<Self, String> {
        Self::new_with_timeout(opts, Some(Duration::from_secs(API_TIMEOUT_SECS)))
    }

    fn new_without_timeout(opts: ApiOpts) -> Result<Self, String> {
        Self::new_with_timeout(opts, None)
    }

    fn new_with_timeout(opts: ApiOpts, timeout: Option<Duration>) -> Result<Self, String> {
        let mut builder = reqwest::Client::builder();
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        let http = builder
            .build()
            .map_err(|e| format!("failed to create HTTP client: {e}"))?;
        Ok(Self { opts, http })
    }

    async fn get(&self, path: &str) -> Result<Value, String> {
        self.request_json(Method::GET, path, &[], None).await
    }

    async fn get_with_query(&self, path: &str, query: &[(&str, String)]) -> Result<Value, String> {
        self.request_json(Method::GET, path, query, None).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.request_json(Method::POST, path, &[], Some(body)).await
    }

    async fn post_multipart(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<Value, String> {
        let url = format!("{}{}", self.opts.base_url, path);
        let mut request = self.http.post(&url).multipart(form);
        if let Some(api_key) = &self.opts.api_key {
            request = request.header("x-api-key", api_key);
        }
        if let Some(tenant_id) = &self.opts.tenant_id {
            request = request.header("x-tenant-id", tenant_id);
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("request to {url} failed: {e}"))?;
        decode_json_response(&url, response).await
    }

    async fn delete(&self, path: &str) -> Result<Value, String> {
        self.request_json(Method::DELETE, path, &[], None).await
    }

    async fn request_json(
        &self,
        method: Method,
        path: &str,
        query: &[(&str, String)],
        body: Option<&Value>,
    ) -> Result<Value, String> {
        let url = format!("{}{}", self.opts.base_url, path);
        let mut request = self.http.request(method, &url);
        if !query.is_empty() {
            request = request.query(query);
        }
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
        decode_json_response(&url, response).await
    }
}

async fn decode_json_response(url: &str, response: reqwest::Response) -> Result<Value, String> {
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| format!("request to {url} failed while reading response body: {e}"))?;

    if !status.is_success() {
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|body| extract_error_message(&body))
            .or_else(|| non_empty_opt(&text).map(str::to_string))
            .unwrap_or_else(|| "empty response body".to_string());
        return Err(format!("request to {url} returned {status}: {message}"));
    }

    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("request to {url} returned invalid JSON: {e}"))
}

async fn cmd_health(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    let opts = config.api_opts();
    let base_url = opts.base_url.clone();
    let client = ApiClient::new(opts)?;
    match client.get("/v1/health").await {
        Ok(body) => {
            if body.get("status").and_then(Value::as_str).is_none() {
                return Err(format!(
                    "response from {base_url}/v1/health is missing string field status"
                ));
            }
            if output.json {
                print_json(&body)?;
            } else {
                println!(
                    "{} at {base_url}",
                    colorize("API OK", "green", output.color)
                );
            }
            Ok(())
        }
        Err(e) => {
            let message = format!("API unreachable at {base_url}: {e}");
            if output.color {
                eprintln!("{}", message.red());
            } else {
                eprintln!("{message}");
            }
            std::process::exit(1);
        }
    }
}

async fn cmd_models(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let body = client.get("/v1/models").await?;
    if output.json {
        return print_json(&body);
    }

    let models = body
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "response from /v1/models is missing array field models".to_string())?;
    let mut table = table(output.color);
    table.set_header(["ID", "TIER", "AVAILABILITY", "PROVIDERS"]);
    for model in models {
        let id = string_field(model, "id").unwrap_or("-");
        let tier = string_field(model, "tier").unwrap_or("-");
        let availability = string_field(model, "availability").unwrap_or("-");
        let providers = string_array_field(model, "providers_available");
        let providers = if providers.is_empty() {
            "-".to_string()
        } else {
            providers.join(",")
        };
        table.add_row([
            Cell::new(id),
            Cell::new(tier),
            Cell::new(color_availability(availability, output.color)),
            Cell::new(providers),
        ]);
    }
    println!("{table}");
    Ok(())
}

async fn cmd_runs_list(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let body = client.get("/v1/runs").await?;
    if output.json {
        return print_json(&body);
    }

    let rows = parse_run_list_rows(&body)?;
    if rows.is_empty() {
        println!("{}", colorize("no runs", "dim", output.color));
        return Ok(());
    }

    let mut table = table(output.color);
    table.set_header(["RUN ID", "STATUS", "MODE", "MODEL", "SOURCE", "CREATED"]);
    for row in rows {
        table.add_row([
            Cell::new(row.run_id),
            Cell::new(color_status(&row.status, output.color)),
            Cell::new(row.mode),
            Cell::new(row.model),
            Cell::new(row.source),
            Cell::new(row.created),
        ]);
    }
    println!("{table}");
    Ok(())
}

async fn cmd_runs_show(
    config: RuntimeConfig,
    output: OutputMode,
    run_id: &str,
) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let body = client.get(&format!("/v1/runs/{run_id}")).await?;
    if output.json {
        return print_json(&body);
    }

    let shown_id = string_field(&body, "run_id").unwrap_or(run_id);
    println!("Run {shown_id}");
    print_kv(
        "Status",
        &color_status(string_field(&body, "status").unwrap_or("-"), output.color),
    );
    print_kv("Mode", string_field(&body, "mode").unwrap_or("-"));
    print_kv("Model", string_field(&body, "model").unwrap_or("-"));
    print_kv("Source", string_field(&body, "source_uri").unwrap_or("-"));
    print_kv("Created", string_field(&body, "created_at").unwrap_or("-"));
    print_kv("Updated", string_field(&body, "updated_at").unwrap_or("-"));
    Ok(())
}

async fn cmd_run_create(
    config: RuntimeConfig,
    output: OutputMode,
    args: &RunCreateArgs,
) -> Result<(), String> {
    let base_url = config
        .base_url
        .value
        .clone()
        .unwrap_or_else(|| DEFAULT_API_URL.to_string());
    let body = args.request_body();
    let client = ApiClient::new(config.api_opts())?;
    let response = client.post_json("/v1/runs", &body).await?;
    validate_run_create_response(&response)
        .map_err(|e| format!("response from {base_url}/v1/runs {e}"))?;

    if output.json {
        return print_json(&response);
    }

    let run_id = string_field(&response, "run_id").unwrap_or("-");
    let status = string_field(&response, "status").unwrap_or("-");
    let mode = string_field(&response, "mode").unwrap_or("-");
    let model = response.get("model").and_then(Value::as_str).unwrap_or("-");
    println!(
        "created run {run_id} status={} mode={mode} model={model}",
        color_status(status, output.color)
    );
    Ok(())
}

async fn cmd_runs_rm(
    config: RuntimeConfig,
    output: OutputMode,
    run_id: &str,
) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let body = client.delete(&format!("/v1/runs/{run_id}")).await?;
    if output.json {
        return print_json(&body);
    }
    let deleted = string_field(&body, "run_id").unwrap_or(run_id);
    println!("deleted run {deleted}");
    Ok(())
}

async fn cmd_events(
    config: RuntimeConfig,
    output: OutputMode,
    args: &EventsArgs,
) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let mut query = Vec::new();
    if let Some(index) = &args.index {
        query.push(("index", index.clone()));
    }
    let body = client
        .get_with_query(&format!("/v1/runs/{}/events", args.run_id), &query)
        .await?;
    if output.json {
        return print_json(&body);
    }

    let events = body
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "response from events route is missing array field events".to_string())?;
    let mut table = table(output.color);
    table.set_header(["SEQ", "PTS", "KIND", "DETAIL"]);
    let mut shown = 0usize;
    for event in events {
        let kind = string_field(event, "kind").unwrap_or("-");
        if args.kind.as_deref().is_some_and(|wanted| wanted != kind) {
            continue;
        }
        let payload = event.get("payload").unwrap_or(&Value::Null);
        table.add_row([
            Cell::new(u64_field(event, "seq").unwrap_or(0)),
            Cell::new(fmt_pts(u64_field(event, "pts_ms").unwrap_or(0))),
            Cell::new(kind),
            Cell::new(event_detail(payload)),
        ]);
        shown += 1;
    }
    if shown == 0 {
        println!("{}", colorize("no events", "dim", output.color));
    } else {
        println!("{table}");
    }
    Ok(())
}

async fn cmd_markers(
    config: RuntimeConfig,
    output: OutputMode,
    args: &MarkersArgs,
) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let mut query = Vec::new();
    if let Some(status) = &args.status {
        query.push(("status", status.clone()));
    }
    if let Some(event_type) = &args.event_type {
        query.push(("event_type", event_type.clone()));
    }
    if let Some(from_frame) = args.from_frame {
        query.push(("from_frame", from_frame.to_string()));
    }
    if let Some(to_frame) = args.to_frame {
        query.push(("to_frame", to_frame.to_string()));
    }
    let body = client
        .get_with_query(&format!("/v1/runs/{}/markers", args.run_id), &query)
        .await?;
    if output.json {
        return print_json(&body);
    }

    let markers = body
        .get("markers")
        .and_then(Value::as_array)
        .ok_or_else(|| "response from markers route is missing array field markers".to_string())?;
    if markers.is_empty() {
        println!("{}", colorize("no markers", "dim", output.color));
        return Ok(());
    }

    let mut table = table(output.color);
    table.set_header(["MARKER", "EVENT TYPE", "STATUS", "FRAMES", "PTS", "CONF"]);
    for marker in markers {
        let marker_id = string_field(marker, "marker_id").unwrap_or("-");
        let start_frame = u64_field(marker, "start_frame").unwrap_or(0);
        let end_frame = u64_field(marker, "end_frame").unwrap_or(0);
        let start_pts = u64_field(marker, "start_pts_ms").unwrap_or(0);
        let end_pts = u64_field(marker, "end_pts_ms").unwrap_or(0);
        let confidence = marker
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        table.add_row([
            Cell::new(short_id(marker_id, 8)),
            Cell::new(string_field(marker, "event_type").unwrap_or("-")),
            Cell::new(color_status(
                string_field(marker, "status").unwrap_or("-"),
                output.color,
            )),
            Cell::new(format!("{start_frame}-{end_frame}")),
            Cell::new(format!("{}-{}", fmt_pts(start_pts), fmt_pts(end_pts))),
            Cell::new(format!("{confidence:.2}")),
        ]);
    }
    println!("{table}");
    Ok(())
}

async fn cmd_search(
    config: RuntimeConfig,
    output: OutputMode,
    args: &SearchArgs,
) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let mut body = Map::new();
    body.insert("query".to_string(), Value::String(args.query.clone()));
    if let Some(run_id) = &args.run {
        body.insert("run_id".to_string(), Value::String(run_id.clone()));
    }
    if let Some(limit) = args.limit {
        body.insert("limit".to_string(), json!(limit));
    }
    let response = client.post_json("/v1/search", &Value::Object(body)).await?;
    if output.json {
        return print_json(&response);
    }

    let hits = response
        .get("hits")
        .and_then(Value::as_array)
        .ok_or_else(|| "response from /v1/search is missing array field hits".to_string())?;
    if hits.is_empty() {
        println!("{}", colorize("no matches", "dim", output.color));
    } else {
        let mut table = table(output.color);
        table.set_header(["SEQ", "RUN", "PTS", "KIND", "DESCRIPTION"]);
        for hit in hits {
            table.add_row([
                Cell::new(u64_field(hit, "seq").unwrap_or(0)),
                Cell::new(string_field(hit, "run_id").unwrap_or("-")),
                Cell::new(fmt_pts(u64_field(hit, "pts_ms").unwrap_or(0))),
                Cell::new(string_field(hit, "kind").unwrap_or("-")),
                Cell::new(truncate(
                    string_field(hit, "description").unwrap_or("-"),
                    70,
                )),
            ]);
        }
        println!("{table}");
    }
    let scanned = u64_field(&response, "scanned").unwrap_or(0);
    let total_hits = u64_field(&response, "total_hits").unwrap_or(hits.len() as u64);
    println!(
        "{}",
        colorize(
            &format!(
                "scanned {scanned}, {total_hits} hits (showing {})",
                hits.len()
            ),
            "dim",
            output.color,
        )
    );
    Ok(())
}

async fn cmd_analyze(
    config: RuntimeConfig,
    output: OutputMode,
    args: &AnalyzeArgs,
) -> Result<(), String> {
    validate_analyze_args(args)?;
    let client = ApiClient::new_without_timeout(config.api_opts())?;

    status_line(output, "uploading...");
    let file_path = upload_analyze_file(&client, &args.file)
        .await
        .map_err(|e| format!("upload failed: {e}"))?;

    status_line(output, "creating run...");
    let run_response = client
        .post_json("/v1/runs", &args.create_run_body())
        .await
        .map_err(|e| format!("create run failed: {e}"))?;
    let run_id = string_field(&run_response, "run_id")
        .ok_or_else(|| "create run failed: response missing string field run_id".to_string())?
        .to_string();

    status_line(output, "ingesting...");
    let ingest_response = client
        .post_json(
            &format!("/v1/runs/{run_id}/ingest"),
            &args.ingest_body(&file_path),
        )
        .await
        .map_err(|e| format!("ingest failed: {e}"))?;
    let decoded_frames = u64_field(&ingest_response, "decoded_frames");
    let source_fps = ingest_response.get("source_fps").and_then(Value::as_f64);
    let estimated_chunks =
        estimate_reason_chunks(decoded_frames, source_fps, args.fixed_fps, args.chunk_size);

    status_line(output, "reasoning...");
    let progress = AnalyzeProgress::new(output, estimated_chunks);
    let stop_polling = Arc::new(AtomicBool::new(false));
    let poller = {
        let client = client.clone();
        let run_id = run_id.clone();
        let index_name = args.index_name.clone();
        let progress = progress.clone();
        let stop_polling = Arc::clone(&stop_polling);
        tokio::spawn(async move {
            poll_reason_progress(client, run_id, index_name, progress, stop_polling).await
        })
    };
    let reason = {
        let client = client.clone();
        let path = format!("/v1/runs/{run_id}/reason");
        let body = args.reason_body(&file_path);
        tokio::spawn(async move { client.post_json(&path, &body).await })
    };

    let reason_result = match reason.await {
        Ok(result) => result,
        Err(e) => Err(format!("reason task failed: {e}")),
    };
    stop_polling.store(true, Ordering::Release);
    let polled_chunks = poller.await.unwrap_or(0);
    let reason_response = match reason_result {
        Ok(response) => response,
        Err(e) => {
            progress.clear();
            return Err(format!("reason failed: {e}"));
        }
    };
    let generated = u64_field(&reason_response, "generated").unwrap_or(polled_chunks as u64);
    progress.finish(generated);

    if output.json {
        return print_json(&reason_response);
    }

    print_analyze_human(&reason_response, output, generated, polled_chunks as u64);
    Ok(())
}

fn validate_analyze_args(args: &AnalyzeArgs) -> Result<(), String> {
    if !args.file.is_file() {
        return Err(format!("{} is not a file", args.file.display()));
    }
    if args.fixed_fps <= 0.0 {
        return Err("--fixed-fps must be greater than 0".to_string());
    }
    if args.chunk_size == 0 {
        return Err("--chunk-size must be greater than 0".to_string());
    }
    if non_empty_opt(&args.model).is_none() {
        return Err("--model must not be empty".to_string());
    }
    Ok(())
}

async fn upload_analyze_file(client: &ApiClient, file: &Path) -> Result<String, String> {
    let bytes = fs::read(file).map_err(|e| format!("failed to read {}: {e}", file.display()))?;
    let file_name = file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("upload")
        .to_string();
    let part = reqwest::multipart::Part::bytes(bytes).file_name(file_name);
    let form = reqwest::multipart::Form::new().part("file", part);
    let response = client.post_multipart("/v1/upload", form).await?;
    string_field(&response, "file_path")
        .map(str::to_string)
        .ok_or_else(|| "response missing string field file_path".to_string())
}

#[derive(Clone)]
enum AnalyzeProgress {
    Hidden,
    Determinate { bar: ProgressBar, len: u64 },
    Spinner { bar: ProgressBar },
}

impl AnalyzeProgress {
    fn new(output: OutputMode, estimated_chunks: Option<u64>) -> Self {
        if output.json || !output.color {
            return Self::Hidden;
        }

        match estimated_chunks {
            Some(len) => {
                let bar = ProgressBar::new(len);
                bar.set_draw_target(ProgressDrawTarget::stderr());
                let template = "{bar:40.cyan/blue} {pos}/{len} chunks {elapsed}";
                if let Ok(style) = ProgressStyle::with_template(template) {
                    bar.set_style(style.progress_chars("=>-"));
                }
                Self::Determinate { bar, len }
            }
            None => {
                let bar = ProgressBar::new_spinner();
                bar.set_draw_target(ProgressDrawTarget::stderr());
                if let Ok(style) = ProgressStyle::with_template("{spinner} {msg} {elapsed}") {
                    bar.set_style(style);
                }
                bar.enable_steady_tick(Duration::from_millis(120));
                bar.set_message("reasoning - 0 chunks");
                Self::Spinner { bar }
            }
        }
    }

    fn update(&self, chunks: u64) {
        match self {
            Self::Hidden => {}
            Self::Determinate { bar, len } => {
                let max_before_done = len.saturating_sub(1);
                bar.set_position(chunks.min(max_before_done));
            }
            Self::Spinner { bar } => {
                bar.set_message(format!("reasoning - {chunks} chunks"));
            }
        }
    }

    fn finish(&self, _generated: u64) {
        match self {
            Self::Hidden => {}
            Self::Determinate { bar, len } => {
                bar.set_position(*len);
                bar.finish_and_clear();
            }
            Self::Spinner { bar } => {
                bar.finish_and_clear();
            }
        }
    }

    fn clear(&self) {
        match self {
            Self::Hidden => {}
            Self::Determinate { bar, .. } | Self::Spinner { bar } => {
                bar.finish_and_clear();
            }
        }
    }
}

async fn poll_reason_progress(
    client: ApiClient,
    run_id: String,
    index_name: Option<String>,
    progress: AnalyzeProgress,
    stop: Arc<AtomicBool>,
) -> usize {
    let mut max_chunks = 0usize;
    while !stop.load(Ordering::Acquire) {
        if let Some(chunks) =
            poll_semantic_chunk_count_bounded(&client, &run_id, index_name.as_deref()).await
        {
            max_chunks = max_chunks.max(chunks);
            progress.update(max_chunks as u64);
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    if let Some(chunks) =
        poll_semantic_chunk_count_bounded(&client, &run_id, index_name.as_deref()).await
    {
        max_chunks = max_chunks.max(chunks);
        progress.update(max_chunks as u64);
    }
    max_chunks
}

async fn poll_semantic_chunk_count_bounded(
    client: &ApiClient,
    run_id: &str,
    index_name: Option<&str>,
) -> Option<usize> {
    match tokio::time::timeout(
        ANALYZE_PROGRESS_POLL_TIMEOUT,
        poll_semantic_chunk_count(client, run_id, index_name),
    )
    .await
    {
        Ok(Ok(chunks)) => Some(chunks),
        Ok(Err(_)) | Err(_) => None,
    }
}

async fn poll_semantic_chunk_count(
    client: &ApiClient,
    run_id: &str,
    index_name: Option<&str>,
) -> Result<usize, String> {
    let mut query = Vec::new();
    if let Some(index_name) = index_name {
        query.push(("index", index_name.to_string()));
    }
    let response = client
        .get_with_query(&format!("/v1/runs/{run_id}/events"), &query)
        .await?;
    Ok(count_semantic_chunks(&response, index_name))
}

fn count_semantic_chunks(body: &Value, index_name: Option<&str>) -> usize {
    let Some(events) = body.get("events").and_then(Value::as_array) else {
        return 0;
    };
    let mut chunk_indexes = HashSet::new();
    let mut without_index = 0usize;
    for event in events {
        if string_field(event, "kind") != Some("semantic_chunk_generated") {
            continue;
        }
        let payload = event.get("payload").unwrap_or(&Value::Null);
        if let Some(index_name) = index_name {
            if payload.get("index_name").and_then(Value::as_str) != Some(index_name) {
                continue;
            }
        }
        if let Some(chunk_index) = u64_field(payload, "chunk_index") {
            chunk_indexes.insert(chunk_index);
        } else {
            without_index += 1;
        }
    }
    chunk_indexes.len() + without_index
}

fn estimate_reason_chunks(
    decoded_frames: Option<u64>,
    source_fps: Option<f64>,
    fixed_fps: f32,
    chunk_size: usize,
) -> Option<u64> {
    let decoded_frames = decoded_frames?;
    let source_fps = source_fps?;
    if decoded_frames == 0 || source_fps <= 0.0 || fixed_fps <= 0.0 || chunk_size == 0 {
        return None;
    }
    let duration_s = decoded_frames as f64 / source_fps;
    let est_reason_frames = duration_s * f64::from(fixed_fps);
    let chunks = (est_reason_frames / chunk_size as f64).ceil() as u64;
    Some(chunks.max(1))
}

#[derive(Debug, PartialEq, Eq)]
struct TimelineEntry {
    start_pts: u64,
    end_pts: u64,
    summary: String,
}

fn collapse_semantic_timeline(metadata: &[Value]) -> Vec<TimelineEntry> {
    let mut entries: Vec<TimelineEntry> = Vec::new();
    let mut previous_had_summary = false;
    for frame in metadata {
        let Some(summary) = frame
            .get("annotations")
            .and_then(|annotations| annotations.get("summary"))
            .and_then(Value::as_str)
            .and_then(non_empty_opt)
        else {
            previous_had_summary = false;
            continue;
        };
        let pts_ms = u64_field(frame, "pts_ms").unwrap_or(0);
        if let Some(last) = entries.last_mut() {
            if previous_had_summary && last.summary == summary {
                last.end_pts = pts_ms;
                previous_had_summary = true;
                continue;
            }
        }
        entries.push(TimelineEntry {
            start_pts: pts_ms,
            end_pts: pts_ms,
            summary: summary.to_string(),
        });
        previous_had_summary = true;
    }
    entries
}

fn print_analyze_human(response: &Value, output: OutputMode, generated: u64, chunks: u64) {
    let decoded_frames = u64_field(response, "decoded_frames").unwrap_or(0);
    let markers_emitted = u64_field(response, "markers_emitted").unwrap_or(0);
    let sample_fps = response
        .get("sample_fps")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    println!(
        "decoded_frames={decoded_frames} frames={generated} chunks={chunks} markers_emitted={markers_emitted} sample_fps={sample_fps:.2}"
    );

    // Token + latency cost of the analysis (e2e model spend across all chunks).
    if let Some(tokens) = response.get("tokens") {
        let total = u64_field(tokens, "total_tokens").unwrap_or(0);
        if total > 0 {
            let prompt = u64_field(tokens, "prompt_tokens").unwrap_or(0);
            let completion = u64_field(tokens, "completion_tokens").unwrap_or(0);
            let thinking = u64_field(tokens, "thinking_tokens").unwrap_or(0);
            let infer_ms = u64_field(tokens, "inference_latency_ms").unwrap_or(0);
            let analyzed = u64_field(tokens, "chunks_analyzed").unwrap_or(0);
            let per_chunk = if analyzed > 0 { total / analyzed } else { 0 };
            let mut cost = format!("tokens total={total} (prompt={prompt} completion={completion}");
            if thinking > 0 {
                cost.push_str(&format!(" thinking={thinking}"));
            }
            cost.push_str(&format!(
                ") ~{per_chunk}/chunk infer_latency={infer_ms}ms across {analyzed} chunks"
            ));
            println!("{}", colorize(&cost, "dim", output.color));
        }
    }

    let metadata = response
        .get("metadata")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let timeline = collapse_semantic_timeline(metadata);
    if timeline.is_empty() {
        println!(
            "{}",
            colorize("no semantic annotations were produced", "dim", output.color)
        );
    } else {
        println!();
        println!("{}", colorize("SEMANTIC TIMELINE", "green", output.color));
        for entry in timeline {
            println!(
                "{}-{}  {}",
                fmt_pts(entry.start_pts),
                fmt_pts(entry.end_pts),
                truncate(&entry.summary, 100)
            );
        }
    }

    let markers = response
        .get("markers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if markers.is_empty() {
        println!();
        println!("{}", colorize("no markers", "dim", output.color));
        return;
    }

    println!();
    let mut table = table(output.color);
    table.set_header(["EVENT TYPE", "STATUS", "START->END", "CONFIDENCE"]);
    for marker in markers {
        let start_pts = u64_field(marker, "start_pts_ms")
            .or_else(|| u64_field(marker, "start_pts"))
            .unwrap_or(0);
        let end_pts = u64_field(marker, "end_pts_ms")
            .or_else(|| u64_field(marker, "end_pts"))
            .unwrap_or(0);
        let confidence = marker
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        table.add_row([
            Cell::new(string_field(marker, "event_type").unwrap_or("-")),
            Cell::new(color_status(
                string_field(marker, "status").unwrap_or("-"),
                output.color,
            )),
            Cell::new(format!("{}->{}", fmt_pts(start_pts), fmt_pts(end_pts))),
            Cell::new(format!("{confidence:.2}")),
        ]);
    }
    println!("{table}");
}

fn status_line(output: OutputMode, message: &str) {
    if !output.json {
        eprintln!("{}", colorize(message, "dim", output.color));
    }
}

async fn cmd_feedback_submit(
    config: RuntimeConfig,
    output: OutputMode,
    args: &FeedbackSubmitArgs,
) -> Result<(), String> {
    if args.rating > 10 {
        return Err("--rating must be between 0 and 10".to_string());
    }
    if non_empty_opt(&args.category).is_none() {
        return Err("--category must not be empty".to_string());
    }

    let client = ApiClient::new(config.api_opts())?;
    let response = client
        .post_json(
            &format!("/v1/runs/{}/feedback", args.run_id),
            &json!({
                "rating": args.rating,
                "category": args.category,
                "feedback": args.note.clone().unwrap_or_default(),
            }),
        )
        .await?;
    if output.json {
        return print_json(&response);
    }
    let run_id = string_field(&response, "run_id").unwrap_or(&args.run_id);
    println!("feedback submitted for run {run_id}");
    Ok(())
}

async fn cmd_feedback_list(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let body = client.get("/v1/feedback").await?;
    if output.json {
        return print_json(&body);
    }

    let feedback = body
        .get("feedback")
        .and_then(Value::as_array)
        .ok_or_else(|| "response from /v1/feedback is missing array field feedback".to_string())?;
    if feedback.is_empty() {
        println!("{}", colorize("no feedback", "dim", output.color));
        return Ok(());
    }

    let mut table = table(output.color);
    table.set_header(["ID", "RUN", "RATING", "CATEGORY", "FEEDBACK"]);
    for item in feedback {
        table.add_row([
            Cell::new(string_field(item, "id").unwrap_or("-")),
            Cell::new(string_field(item, "run_id").unwrap_or("-")),
            Cell::new(u64_field(item, "rating").unwrap_or(0)),
            Cell::new(string_field(item, "category").unwrap_or("-")),
            Cell::new(truncate(string_field(item, "feedback").unwrap_or(""), 50)),
        ]);
    }
    println!("{table}");
    Ok(())
}

async fn cmd_doctor(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    let client = ApiClient::new(config.api_opts())?;
    let mut hard_failed = false;
    let mut checks = Map::new();

    let api_key_status = mask_key(config.api_key.value.as_deref());
    checks.insert(
        "config".to_string(),
        json!({
            "base_url": config.base_url.value.as_deref().unwrap_or(DEFAULT_API_URL),
            "api_key": api_key_status.clone(),
        }),
    );

    if !output.json {
        println!(
            "{} base_url={} api_key={}",
            colorize("config", "green", output.color),
            config.base_url.value.as_deref().unwrap_or(DEFAULT_API_URL),
            api_key_status,
        );
    }

    match client.get("/v1/health").await {
        Ok(_) => {
            checks.insert("api_reachable".to_string(), json!({"status": "ok"}));
            if !output.json {
                println!("{} API reachable", colorize("ok", "green", output.color));
            }
        }
        Err(e) => {
            hard_failed = true;
            checks.insert(
                "api_reachable".to_string(),
                json!({"status": "unreachable", "error": e}),
            );
            if !output.json {
                println!("{} API unreachable", colorize("fail", "red", output.color));
            }
        }
    }

    match client.get("/v1/models").await {
        Ok(body) => {
            let models = body
                .get("models")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    "response from /v1/models is missing array field models".to_string()
                })?;
            let mut ready = 0usize;
            let mut saturated = 0usize;
            let mut unavailable = 0usize;
            let mut unavailable_ids = Vec::new();
            for model in models {
                match string_field(model, "availability").unwrap_or("") {
                    "ready" => ready += 1,
                    "saturated" => saturated += 1,
                    "unavailable" => {
                        unavailable += 1;
                        unavailable_ids.push(string_field(model, "id").unwrap_or("-").to_string());
                    }
                    _ => {}
                }
            }
            checks.insert(
                "models".to_string(),
                json!({
                    "ready": ready,
                    "saturated": saturated,
                    "unavailable": unavailable,
                    "unavailable_ids": unavailable_ids.clone(),
                    "warning": if ready == 0 {
                        Some("analyze/reason need a ready VLM")
                    } else {
                        None
                    },
                }),
            );
            if !output.json {
                println!(
                    "{} models ready={ready} saturated={saturated} unavailable={unavailable}",
                    colorize("ok", "green", output.color)
                );
                if !unavailable_ids.is_empty() {
                    println!(
                        "{} unavailable: {}",
                        colorize("warn", "yellow", output.color),
                        unavailable_ids.join(", ")
                    );
                }
                if ready == 0 {
                    println!(
                        "{} analyze/reason need a ready VLM",
                        colorize("warn", "yellow", output.color)
                    );
                }
            }
        }
        Err(e) => {
            let warning = e;
            checks.insert("models".to_string(), json!({"warning": warning.clone()}));
            if !output.json {
                println!(
                    "{} models check failed: {}",
                    colorize("warn", "yellow", output.color),
                    warning
                );
            }
        }
    }

    if output.json {
        print_json(&Value::Object(checks))?;
    }
    if hard_failed {
        Err("doctor found failing checks".to_string())
    } else {
        Ok(())
    }
}

fn cmd_config_show(config: RuntimeConfig, output: OutputMode) -> Result<(), String> {
    if output.json {
        return print_json(&json!({
            "base_url": {
                "value": config.base_url.value.as_deref().unwrap_or(DEFAULT_API_URL),
                "source": config.base_url.source.as_str(),
            },
            "api_key": {
                "value": mask_key(config.api_key.value.as_deref()),
                "source": config.api_key.source.as_str(),
            },
            "tenant_id": {
                "value": config.tenant_id.value.as_deref(),
                "source": config.tenant_id.source.as_str(),
            },
        }));
    }

    println!(
        "base_url  : {} ({})",
        config.base_url.value.as_deref().unwrap_or(DEFAULT_API_URL),
        config.base_url.source.as_str()
    );
    println!(
        "api_key   : {} ({})",
        mask_key(config.api_key.value.as_deref()),
        config.api_key.source.as_str()
    );
    println!(
        "tenant_id : {} ({})",
        config.tenant_id.value.as_deref().unwrap_or("not set"),
        config.tenant_id.source.as_str()
    );
    Ok(())
}

fn cmd_states(output: OutputMode) -> Result<(), String> {
    let states = [
        StreamState::Pending,
        StreamState::Processing,
        StreamState::Completed,
        StreamState::Failed,
        StreamState::Cancelled,
        StreamState::Expired,
    ];

    if output.json {
        let body = Value::Array(
            states
                .iter()
                .map(|state| {
                    json!({
                        "state": format!("{state:?}"),
                        "terminal": state.is_terminal(),
                    })
                })
                .collect(),
        );
        return print_json(&body);
    }

    let mut table = table(output.color);
    table.set_header(["STATE", "TERMINAL"]);
    for state in states {
        table.add_row([
            Cell::new(format!("{state:?}")),
            Cell::new(state.is_terminal()),
        ]);
    }
    println!("{table}");
    Ok(())
}

fn table(color: bool) -> Table {
    let mut table = Table::new();
    if color {
        table.load_preset(presets::UTF8_FULL);
    } else {
        table.load_preset(presets::ASCII_MARKDOWN);
    }
    table
}

fn print_json(value: &Value) -> Result<(), String> {
    let text =
        serde_json::to_string_pretty(value).map_err(|e| format!("failed to render JSON: {e}"))?;
    println!("{text}");
    Ok(())
}

fn print_kv(label: &str, value: &str) {
    println!("{label:<8} {value}");
}

fn string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn string_array_field(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn u64_field(value: &Value, field: &str) -> Option<u64> {
    value.get(field).and_then(Value::as_u64)
}

fn color_status(status: &str, color: bool) -> String {
    match status {
        "completed" => colorize(status, "green", color),
        "processing" | "pending" => colorize(status, "yellow", color),
        "failed" | "cancelled" | "expired" => colorize(status, "red", color),
        _ => status.to_string(),
    }
}

fn color_availability(availability: &str, color: bool) -> String {
    match availability {
        "ready" => colorize(availability, "green", color),
        "saturated" => colorize(availability, "yellow", color),
        "unavailable" => colorize(availability, "red_dim", color),
        _ => availability.to_string(),
    }
}

fn colorize(text: &str, style: &str, enabled: bool) -> String {
    if !enabled {
        return text.to_string();
    }
    match style {
        "green" => text.green().to_string(),
        "yellow" => text.yellow().to_string(),
        "red" => text.red().to_string(),
        "red_dim" => text.red().dimmed().to_string(),
        "dim" => text.dimmed().to_string(),
        _ => text.to_string(),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let take = max_chars.saturating_sub(3);
    format!("{}...", value.chars().take(take).collect::<String>())
}

fn event_detail(payload: &Value) -> String {
    for field in ["summary", "description"] {
        if let Some(value) = payload.get(field).and_then(Value::as_str) {
            return truncate(value, 80);
        }
    }
    let compact = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    truncate(&compact, 80)
}

fn basename_uri(uri: &str) -> String {
    if uri.is_empty() {
        return "-".to_string();
    }
    Path::new(uri)
        .file_name()
        .and_then(|name| name.to_str())
        .or_else(|| uri.split('/').filter(|part| !part.is_empty()).next_back())
        .unwrap_or(uri)
        .to_string()
}

fn short_id(id: &str, len: usize) -> String {
    let count = id.chars().count();
    if count <= len {
        id.to_string()
    } else {
        id.chars().skip(count - len).collect()
    }
}

fn mask_key(key: Option<&str>) -> String {
    let Some(key) = key.and_then(non_empty_opt) else {
        return "not set".to_string();
    };
    let count = key.chars().count();
    if count < 4 {
        "set (…)".to_string()
    } else {
        format!("set (…{})", key.chars().skip(count - 4).collect::<String>())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RunListRow {
    run_id: String,
    status: String,
    mode: String,
    model: String,
    source: String,
    created: String,
}

fn parse_run_list_rows(body: &Value) -> Result<Vec<RunListRow>, String> {
    let runs = body
        .as_array()
        .ok_or_else(|| "response from /v1/runs must be a JSON array".to_string())?;
    Ok(runs
        .iter()
        .map(|run| RunListRow {
            run_id: string_field(run, "run_id").unwrap_or("-").to_string(),
            status: string_field(run, "status").unwrap_or("-").to_string(),
            mode: string_field(run, "mode").unwrap_or("-").to_string(),
            model: string_field(run, "model").unwrap_or("-").to_string(),
            source: string_field(run, "source_uri")
                .map(basename_uri)
                .unwrap_or_else(|| "-".to_string()),
            created: string_field(run, "created_at").unwrap_or("-").to_string(),
        })
        .collect())
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

fn exit_with_error(message: String) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}

struct DistillOpts {
    tenant_id: Option<String>,
    data_dir: PathBuf,
    mlx: bool,
    resume: bool,
    format: String,
}

impl DistillOpts {
    fn require_tenant_id(&self) -> &str {
        self.tenant_id.as_deref().unwrap_or_else(|| {
            eprintln!("error: --tenant-id is required");
            std::process::exit(1);
        })
    }
}

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

/// Spawn the Python training script and stream its output.
#[cfg(feature = "training")]
async fn cmd_distill_train(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

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
async fn cmd_distill_train(opts: DistillOpts) {
    let _ = (opts.mlx, opts.resume);
    eprintln!("error: training support is not compiled in (rebuild with --features training)");
    std::process::exit(1);
}

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

/// Write a `.reload-specialist` sentinel file for the running API.
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

/// Spawn the command, stream stdout and stderr, then wait for exit.
async fn run_subprocess(mut cmd: tokio::process::Command) {
    use tokio::io::AsyncBufReadExt as _;

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn subprocess: {e}");
            std::process::exit(1);
        }
    };

    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("[subprocess] {}", line);
            }
        });
    }

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
    use clap::CommandFactory;

    #[test]
    fn fmt_pts_formats_expected_values() {
        assert_eq!(fmt_pts(0), "0:00.000");
        assert_eq!(fmt_pts(3_723), "0:03.723");
        assert_eq!(fmt_pts(65_000), "1:05.000");
        assert_eq!(fmt_pts(3_600_000), "60:00.000");
    }

    #[test]
    fn config_precedence_reports_source() {
        let resolved =
            resolve_config_value(Some("flag"), Some("env"), Some("file"), Some("default"));
        assert_eq!(resolved.value.as_deref(), Some("flag"));
        assert_eq!(resolved.source, ConfigSource::Flag);

        let resolved = resolve_config_value(None, Some("env"), Some("file"), Some("default"));
        assert_eq!(resolved.value.as_deref(), Some("env"));
        assert_eq!(resolved.source, ConfigSource::Env);

        let resolved = resolve_config_value(None, None, Some("file"), Some("default"));
        assert_eq!(resolved.value.as_deref(), Some("file"));
        assert_eq!(resolved.source, ConfigSource::File);

        let resolved = resolve_config_value(None, None, None, Some("default"));
        assert_eq!(resolved.value.as_deref(), Some("default"));
        assert_eq!(resolved.source, ConfigSource::Default);
    }

    #[test]
    fn config_file_parser_trims_and_filters_keys() {
        let parsed = parse_config_file(
            "
            # comment
            API_URL = http://localhost:9000/
            api_key = key-1

            Tenant_ID = tenant-a
            ignored = value
            malformed
            ",
        );

        assert_eq!(
            parsed.get("api_url").map(String::as_str),
            Some("http://localhost:9000/")
        );
        assert_eq!(parsed.get("api_key").map(String::as_str), Some("key-1"));
        assert_eq!(
            parsed.get("tenant_id").map(String::as_str),
            Some("tenant-a")
        );
        assert!(!parsed.contains_key("ignored"));
    }

    #[test]
    fn run_create_args_omits_absent_mode_and_model() {
        let opts = RunCreateArgs {
            mode: None,
            model: None,
        };

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
    fn estimate_reason_chunks_uses_duration_and_reason_sampling() {
        assert_eq!(
            estimate_reason_chunks(Some(300), Some(30.0), 1.0, 25),
            Some(1)
        );
        assert_eq!(
            estimate_reason_chunks(Some(3_000), Some(30.0), 1.0, 25),
            Some(4)
        );
        assert_eq!(
            estimate_reason_chunks(Some(3_600), Some(30.0), 2.0, 25),
            Some(10)
        );
        assert_eq!(estimate_reason_chunks(Some(300), Some(0.0), 1.0, 25), None);
        assert_eq!(estimate_reason_chunks(Some(0), Some(30.0), 1.0, 25), None);
    }

    #[test]
    fn semantic_timeline_collapses_consecutive_identical_summaries() {
        let empty = collapse_semantic_timeline(&[]);
        assert!(empty.is_empty());

        let all_identical = serde_json::json!([
            {"pts_ms": 0, "annotations": {"summary": "same"}},
            {"pts_ms": 1_000, "annotations": {"summary": "same"}},
            {"pts_ms": 2_000, "annotations": {"summary": "same"}}
        ]);
        let rows =
            collapse_semantic_timeline(all_identical.as_array().map(Vec::as_slice).unwrap_or(&[]));
        assert_eq!(
            rows,
            vec![TimelineEntry {
                start_pts: 0,
                end_pts: 2_000,
                summary: "same".to_string()
            }]
        );

        let alternating = serde_json::json!([
            {"pts_ms": 0, "annotations": {"summary": "a"}},
            {"pts_ms": 1_000, "annotations": {"summary": "a"}},
            {"pts_ms": 2_000, "annotations": {"summary": ""}},
            {"pts_ms": 3_000, "annotations": {"summary": "a"}},
            {"pts_ms": 4_000, "annotations": {"summary": "b"}},
            {"pts_ms": 5_000, "annotations": {"summary": "a"}}
        ]);
        let rows =
            collapse_semantic_timeline(alternating.as_array().map(Vec::as_slice).unwrap_or(&[]));
        assert_eq!(
            rows,
            vec![
                TimelineEntry {
                    start_pts: 0,
                    end_pts: 1_000,
                    summary: "a".to_string()
                },
                TimelineEntry {
                    start_pts: 3_000,
                    end_pts: 3_000,
                    summary: "a".to_string()
                },
                TimelineEntry {
                    start_pts: 4_000,
                    end_pts: 4_000,
                    summary: "b".to_string()
                },
                TimelineEntry {
                    start_pts: 5_000,
                    end_pts: 5_000,
                    summary: "a".to_string()
                }
            ]
        );
    }

    fn analyze_args() -> AnalyzeArgs {
        AnalyzeArgs {
            file: PathBuf::from("clip.mp4"),
            prompt: None,
            model: "vidarax-medium".to_string(),
            mode: None,
            fixed_fps: 1.0,
            chunk_size: 25,
            max_frames: None,
            index_name: None,
            sampling_policy: None,
            crop: None,
        }
    }

    #[test]
    fn crop_arg_parses_four_fractions_and_rejects_malformed() {
        let c: CropArg = "0.1, 0.2 ,0.3,0.4".parse().expect("valid crop");
        assert_eq!(c.x, 0.1);
        assert_eq!(c.y, 0.2);
        assert_eq!(c.width, 0.3);
        assert_eq!(c.height, 0.4);
        assert!("0.1,0.2,0.3".parse::<CropArg>().is_err());
        assert!("a,b,c,d".parse::<CropArg>().is_err());
        assert!("".parse::<CropArg>().is_err());
    }

    #[test]
    fn reason_body_includes_crop_only_when_supplied() {
        let mut args = analyze_args();
        assert!(args.reason_body("/tmp/x.mp4").get("crop").is_none());
        args.crop = Some(CropArg {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.5,
        });
        let body = args.reason_body("/tmp/x.mp4");
        let crop = body.get("crop").expect("crop present");
        assert_eq!(crop.get("x").and_then(Value::as_f64), Some(0.25));
        assert_eq!(crop.get("width").and_then(Value::as_f64), Some(0.5));
    }

    #[test]
    fn ingest_body_adds_fixed_fps_only_for_explicit_fixed_sampling() {
        let mut args = analyze_args();

        let default_body = args.ingest_body("/tmp/uploaded.mp4");
        assert!(default_body.get("sampling_policy").is_none());
        assert!(default_body.get("fixed_fps").is_none());

        args.sampling_policy = Some("fixed".to_string());
        args.fixed_fps = 2.5;
        let fixed_body = args.ingest_body("/tmp/uploaded.mp4");
        assert_eq!(
            fixed_body.get("sampling_policy").and_then(Value::as_str),
            Some("fixed")
        );
        assert_eq!(
            fixed_body.get("fixed_fps").and_then(Value::as_f64),
            Some(2.5)
        );

        args.sampling_policy = Some("adaptive".to_string());
        let adaptive_body = args.ingest_body("/tmp/uploaded.mp4");
        assert_eq!(
            adaptive_body.get("sampling_policy").and_then(Value::as_str),
            Some("adaptive")
        );
        assert!(adaptive_body.get("fixed_fps").is_none());
    }

    #[test]
    fn reason_body_omits_absent_semantic_prompt_and_includes_supplied_prompt() {
        let mut args = analyze_args();

        let without_prompt = args.reason_body("/tmp/uploaded.mp4");
        assert!(without_prompt.get("semantic_prompt").is_none());
        assert_eq!(
            without_prompt
                .get("sampling_policy")
                .and_then(Value::as_str),
            Some("fixed")
        );
        assert_eq!(
            without_prompt.get("fixed_fps").and_then(Value::as_f64),
            Some(1.0)
        );
        assert_eq!(
            without_prompt
                .get("semantic_inference")
                .and_then(Value::as_bool),
            Some(true)
        );

        args.prompt = Some("custom semantic prompt".to_string());
        let with_prompt = args.reason_body("/tmp/uploaded.mp4");
        assert_eq!(
            with_prompt.get("semantic_prompt").and_then(Value::as_str),
            Some("custom semantic prompt")
        );
    }

    #[test]
    fn clap_tree_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn run_list_rows_parse_bare_array() {
        let body = serde_json::json!([
            {
                "run_id": "run-1",
                "status": "completed",
                "mode": "fast",
                "model": "vlm-a",
                "source_uri": "file:///tmp/video-a.mp4",
                "created_at": "2026-01-01T00:00:00Z"
            },
            {
                "run_id": "run-2",
                "status": "pending",
                "mode": "balanced",
                "created_at": "2026-01-02T00:00:00Z"
            }
        ]);

        let rows = parse_run_list_rows(&body).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].run_id, "run-1");
        assert_eq!(rows[0].source, "video-a.mp4");
        assert_eq!(rows[1].model, "-");
        assert_eq!(rows[1].source, "-");
    }

    #[test]
    fn run_list_rows_parse_empty_array() {
        let rows = parse_run_list_rows(&serde_json::json!([])).unwrap();
        assert!(rows.is_empty());
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

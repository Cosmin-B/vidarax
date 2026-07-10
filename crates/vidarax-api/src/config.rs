use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use vidarax_core::gate::GateConfig;
use vidarax_core::tiered_vlm::{DistillationConfig, TieredVlmConfig};

pub(crate) const UPLOAD_DIR_NAME: &str = "vidarax-uploads";

/// Load the backend config from a TOML file, falling back to legacy env vars.
///
/// Reads `config_path` (default: `"vidarax.toml"`).  If the file is absent or
/// unreadable the function synthesises a [`vidarax_core::backends::VidaraxConfig`]
/// from the legacy `VIDARAX_VLLM_BASE_URL` / `VIDARAX_SGLANG_BASE_URL` env vars
/// so that existing deployments continue to work unchanged.
pub fn load_backend_config(
    config_path: &str,
) -> Result<vidarax_core::backends::VidaraxConfig, String> {
    match std::fs::read_to_string(config_path) {
        Ok(contents) => vidarax_core::backends::parse_config(&contents),
        Err(_) => {
            // Fallback: synthesise config from legacy env vars so that deployments
            // that don't have a vidarax.toml continue to work.
            let vllm = std::env::var("VIDARAX_VLLM_BASE_URL").ok();
            let sglang = std::env::var("VIDARAX_SGLANG_BASE_URL").ok();
            let mut backends = Vec::new();
            if let Some(url) = vllm {
                backends.push(vidarax_core::backends::BackendEntry {
                    name: "vllm".to_string(),
                    backend_type: "openai_compat".to_string(),
                    base_url: Some(url),
                    api_key: None,
                    model: None,
                    openai_kind: Some("vllm".to_string()),
                    priority: 1,
                });
            }
            if let Some(url) = sglang {
                backends.push(vidarax_core::backends::BackendEntry {
                    name: "sglang".to_string(),
                    backend_type: "openai_compat".to_string(),
                    base_url: Some(url),
                    api_key: None,
                    model: None,
                    openai_kind: Some("sglang".to_string()),
                    priority: 2,
                });
            }
            Ok(vidarax_core::backends::VidaraxConfig { backends })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    H1H2,
    H3Experimental,
}

impl TransportMode {
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw.unwrap_or("h1h2").to_ascii_lowercase().as_str() {
            "h1" | "h2" | "h1h2" | "http" | "http2" => Ok(Self::H1H2),
            "h3" | "http3" => Ok(Self::H3Experimental),
            other => Err(format!(
                "unsupported transport mode '{other}', expected one of: h1h2, h3"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::H1H2 => "h1h2",
            Self::H3Experimental => "h3-experimental",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub h3_bind_addr: String,
    pub h3_tls_cert_path: String,
    pub h3_tls_key_path: String,
    pub data_dir: String,
    pub ingest_file_roots: Vec<PathBuf>,
    pub inference_vllm_base_url: Option<String>,
    pub inference_sglang_base_url: Option<String>,
    pub security_require_api_key: bool,
    pub security_api_keys: Vec<String>,
    pub security_require_tenant_id: bool,
    pub security_global_rps: Option<u64>,
    pub security_tenant_rps: Option<u64>,
    pub security_tenant_slots: usize,
    pub security_metrics_require_api_key: bool,
    pub cors_allowed_origins: Vec<String>,
    pub stream_ttl_secs: u64,
    pub active_stream_limit: usize,
    pub transport: TransportMode,
    pub decode_backend: String,
    /// STUN server URIs (comma-separated). Defaults to Google's public STUN server.
    pub webrtc_stun_servers: Vec<String>,
    /// Optional TURN relay URL (`VIDARAX_WEBRTC_TURN_URL`).
    pub webrtc_turn_url: Option<String>,
    /// TURN username (`VIDARAX_WEBRTC_TURN_USERNAME`).
    pub webrtc_turn_username: Option<String>,
    /// TURN credential (`VIDARAX_WEBRTC_TURN_CREDENTIAL`).
    pub webrtc_turn_credential: Option<String>,
    /// Optional SpacetimeDB base URL (`VIDARAX_SPACETIMEDB_URL`). When set, the
    /// feedback endpoints and the WHIP event sink use SpacetimeDB; when unset,
    /// stream events use the local WAL and the feedback endpoints are disabled.
    pub spacetimedb_url: Option<String>,
    /// SpacetimeDB database/module name (`VIDARAX_SPACETIMEDB_MODULE`). Only
    /// used when `spacetimedb_url` is set; defaults to "vidarax".
    pub spacetimedb_module: Option<String>,
    /// VLM output token rate cap per session in tokens/s (`VIDARAX_WEBRTC_MAX_OUTPUT_TOKENS_PER_SECOND`).
    pub webrtc_max_output_tokens_per_second: u32,
    pub webrtc_decode_workers: usize,
    pub webrtc_analysis_workers: usize,
    pub webrtc_vlm_workers: usize,
    /// WHIP VLM tiering: first-pass model ID (`VIDARAX_WEBRTC_FIRST_PASS_MODEL`).
    /// Every live keyframe is analyzed with this model by default.
    pub webrtc_first_pass_model: String,
    /// Optional second-pass (escalation) model ID
    /// (`VIDARAX_WEBRTC_SECOND_PASS_MODEL`). Unset, empty, or equal to the
    /// first-pass model keeps live streams on a single model with no
    /// escalation; this is the default so tiering only turns on when an
    /// operator opts in.
    pub webrtc_second_pass_model: Option<String>,
    /// Confidence threshold below which the second-pass model runs
    /// (`VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD`), clamped to [0.0, 1.0].
    pub webrtc_second_pass_threshold: f32,
    /// Max output tokens for the second-pass model
    /// (`VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS`).
    pub webrtc_second_pass_max_tokens: u32,
    pub gate_config: GateConfig,
    pub distillation: DistillationConfig,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, String> {
        let transport = TransportMode::parse(env::var("VIDARAX_TRANSPORT").ok().as_deref())?;
        let decode_backend =
            env::var("VIDARAX_DECODE_BACKEND").unwrap_or_else(|_| "auto".to_string());
        let bind_addr = env::var("VIDARAX_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
        let h3_bind_addr =
            env::var("VIDARAX_H3_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8443".into());
        let h3_tls_cert_path =
            env::var("VIDARAX_H3_TLS_CERT_PATH").unwrap_or_else(|_| "deploy/certs/dev.crt".into());
        let h3_tls_key_path =
            env::var("VIDARAX_H3_TLS_KEY_PATH").unwrap_or_else(|_| "deploy/certs/dev.key".into());
        let data_dir = env::var("VIDARAX_DATA_DIR").unwrap_or_else(|_| ".vidarax-data".into());
        let ingest_file_roots = parse_ingest_roots_env("VIDARAX_INGEST_FILE_ROOTS")?;
        let inference_vllm_base_url = env::var("VIDARAX_VLLM_BASE_URL").ok();
        let inference_sglang_base_url = env::var("VIDARAX_SGLANG_BASE_URL").ok();
        let security_require_api_key = parse_bool_env("VIDARAX_REQUIRE_API_KEY", true)?;
        let security_api_keys = parse_csv_env("VIDARAX_API_KEYS");
        let security_require_tenant_id = parse_bool_env("VIDARAX_REQUIRE_TENANT_ID", false)?;
        let security_global_rps = parse_u64_env("VIDARAX_RATE_LIMIT_GLOBAL_RPS")?;
        let security_tenant_rps = parse_u64_env("VIDARAX_RATE_LIMIT_TENANT_RPS")?;
        let security_tenant_slots = parse_usize_env("VIDARAX_RATE_LIMIT_TENANT_SLOTS", 2048)?;
        let security_metrics_require_api_key =
            parse_bool_env("VIDARAX_METRICS_REQUIRE_API_KEY", true)?;
        let cors_allowed_origins = parse_csv_env("VIDARAX_CORS_ALLOWED_ORIGINS");
        if security_require_api_key && cors_allowed_origins.iter().any(|o| o.trim() == "*") {
            return Err(
                "VIDARAX_CORS_ALLOWED_ORIGINS must not contain '*' when VIDARAX_REQUIRE_API_KEY is enabled"
                    .to_string(),
            );
        }
        validate_tenant_auth_config(security_require_api_key, security_require_tenant_id)?;
        let stream_ttl_secs = parse_u64_env_with_default("VIDARAX_STREAM_TTL_SECS", 3600)?;
        if !(60..=86_400).contains(&stream_ttl_secs) {
            return Err("VIDARAX_STREAM_TTL_SECS must be in [60, 86400]".to_string());
        }
        let active_stream_limit = parse_usize_env("VIDARAX_ACTIVE_STREAM_LIMIT", 5)?.clamp(1, 1024);
        let webrtc_stun_servers = {
            let v = parse_csv_env("VIDARAX_WEBRTC_STUN_SERVERS");
            if v.is_empty() {
                vec!["stun:stun.l.google.com:19302".to_string()]
            } else {
                v
            }
        };
        let webrtc_turn_url = env::var("VIDARAX_WEBRTC_TURN_URL").ok();
        let webrtc_turn_username = env::var("VIDARAX_WEBRTC_TURN_USERNAME").ok();
        let webrtc_turn_credential = env::var("VIDARAX_WEBRTC_TURN_CREDENTIAL").ok();
        let spacetimedb_url = env::var("VIDARAX_SPACETIMEDB_URL").ok();
        let spacetimedb_module = env::var("VIDARAX_SPACETIMEDB_MODULE").ok();
        let webrtc_max_output_tokens_per_second =
            parse_usize_env("VIDARAX_WEBRTC_MAX_OUTPUT_TOKENS_PER_SECOND", 128)? as u32;
        // One ordered stream is decoded by one stateful decoder. Keep the env
        // knob for compatibility but clamp above 1 at config load.
        let webrtc_decode_workers = parse_usize_env("VIDARAX_WEBRTC_DECODE_WORKERS", 1)?
            .clamp(1, 64)
            .min(1);
        // Analysis owns stream-order gate/loop state; more workers make one
        // shared stream's decisions nondeterministic.
        let webrtc_analysis_workers = parse_usize_env("VIDARAX_WEBRTC_ANALYSIS_WORKERS", 1)?
            .clamp(1, 64)
            .min(1);
        // VLM keyframe analysis carries temporal/dedup state; do not split one
        // stream across racing workers.
        let webrtc_vlm_workers = parse_usize_env("VIDARAX_WEBRTC_VLM_WORKERS", 1)?
            .clamp(1, 64)
            .min(1);
        let (
            webrtc_first_pass_model,
            webrtc_second_pass_model,
            webrtc_second_pass_threshold,
            webrtc_second_pass_max_tokens,
        ) = parse_webrtc_vlm_tiering()?;
        let gate_config = parse_gate_config()?;
        let distillation = parse_distillation_config()?;
        Ok(Self {
            bind_addr,
            h3_bind_addr,
            h3_tls_cert_path,
            h3_tls_key_path,
            data_dir,
            ingest_file_roots,
            inference_vllm_base_url,
            inference_sglang_base_url,
            security_require_api_key,
            security_api_keys,
            security_require_tenant_id,
            security_global_rps,
            security_tenant_rps,
            security_tenant_slots,
            security_metrics_require_api_key,
            cors_allowed_origins,
            stream_ttl_secs,
            active_stream_limit,
            transport,
            decode_backend,
            webrtc_stun_servers,
            webrtc_turn_url,
            webrtc_turn_username,
            webrtc_turn_credential,
            spacetimedb_url,
            spacetimedb_module,
            webrtc_max_output_tokens_per_second,
            webrtc_decode_workers,
            webrtc_analysis_workers,
            webrtc_vlm_workers,
            webrtc_first_pass_model,
            webrtc_second_pass_model,
            webrtc_second_pass_threshold,
            webrtc_second_pass_max_tokens,
            gate_config,
            distillation,
        })
    }
}

fn parse_ingest_roots_env(var: &str) -> Result<Vec<PathBuf>, String> {
    let values = env::var(var)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(values.len());
    for root in values {
        out.push(
            root.canonicalize().map_err(|err| {
                format!("{var} contains invalid root '{}': {err}", root.display())
            })?,
        );
    }
    Ok(out)
}

pub fn resolve_wal_path(config: &ServerConfig) -> Result<PathBuf, String> {
    let data_dir = PathBuf::from(&config.data_dir);
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;
    Ok(data_dir.join("timeline.wal"))
}

fn validate_tenant_auth_config(
    security_require_api_key: bool,
    security_require_tenant_id: bool,
) -> Result<(), String> {
    if security_require_tenant_id && !security_require_api_key {
        return Err(
            "VIDARAX_REQUIRE_TENANT_ID requires VIDARAX_REQUIRE_API_KEY=true: tenant isolation is derived from authenticated API keys"
                .to_string(),
        );
    }
    Ok(())
}

fn parse_bool_env(var: &str, default: bool) -> Result<bool, String> {
    let Some(raw) = env::var(var).ok() else {
        return Ok(default);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{var} must be one of: 1,true,yes,on,0,false,no,off"
        )),
    }
}

fn parse_u64_env(var: &str) -> Result<Option<u64>, String> {
    match env::var(var) {
        Ok(raw) => raw
            .parse::<u64>()
            .map(Some)
            .map_err(|_| format!("{var} must be an unsigned integer")),
        Err(_) => Ok(None),
    }
}

fn parse_u64_env_with_default(var: &str, default: u64) -> Result<u64, String> {
    match env::var(var) {
        Ok(raw) => raw
            .parse::<u64>()
            .map_err(|_| format!("{var} must be an unsigned integer")),
        Err(_) => Ok(default),
    }
}

fn parse_usize_env(var: &str, default: usize) -> Result<usize, String> {
    match env::var(var) {
        Ok(raw) => raw
            .parse::<usize>()
            .map_err(|_| format!("{var} must be an unsigned integer")),
        Err(_) => Ok(default),
    }
}

fn parse_csv_env(var: &str) -> Vec<String> {
    env::var(var)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_f32_env(var: &str, default: f32) -> Result<f32, String> {
    match env::var(var) {
        Ok(raw) => raw
            .parse::<f32>()
            .map_err(|_| format!("{var} must be a floating-point number")),
        Err(_) => Ok(default),
    }
}

fn parse_gate_config() -> Result<GateConfig, String> {
    let default = GateConfig::default();
    Ok(GateConfig {
        keepalive_every_frames: parse_u64_env_with_default(
            "VIDARAX_GATE_KEEPALIVE_EVERY_FRAMES",
            default.keepalive_every_frames,
        )?,
        scene_cut_hamming_threshold: parse_u32_env_with_default(
            "VIDARAX_GATE_SCENE_CUT_HAMMING_THRESHOLD",
            default.scene_cut_hamming_threshold,
        )?,
        luma_shift_threshold: parse_f32_env(
            "VIDARAX_GATE_LUMA_SHIFT_THRESHOLD",
            default.luma_shift_threshold,
        )?,
        flicker_threshold: parse_f32_env(
            "VIDARAX_GATE_FLICKER_THRESHOLD",
            default.flicker_threshold,
        )?,
        ghosting_threshold: parse_f32_env(
            "VIDARAX_GATE_GHOSTING_THRESHOLD",
            default.ghosting_threshold,
        )?,
        noise_variance_threshold: parse_f32_env(
            "VIDARAX_GATE_NOISE_VARIANCE_THRESHOLD",
            default.noise_variance_threshold,
        )?,
    })
}

fn parse_u32_env_with_default(var: &str, default: u32) -> Result<u32, String> {
    match env::var(var) {
        Ok(raw) => raw
            .parse::<u32>()
            .map_err(|_| format!("{var} must be an unsigned integer")),
        Err(_) => Ok(default),
    }
}

fn parse_distillation_config() -> Result<DistillationConfig, String> {
    let enabled = parse_bool_env("VIDARAX_DISTILL_ENABLED", false)?;
    let embedding_server_url = env::var("VIDARAX_DISTILL_EMBEDDING_URL").ok();
    let teacher_model = env::var("VIDARAX_DISTILL_TEACHER_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-8B-Instruct".to_string());
    let max_pairs_per_tenant =
        parse_usize_env("VIDARAX_DISTILL_MAX_PAIRS", 10_000)?.clamp(100, 1_000_000);
    let collection_rate = parse_f32_env("VIDARAX_DISTILL_COLLECTION_RATE", 0.1)?.clamp(0.0, 1.0);
    let distance_threshold =
        parse_f32_env("VIDARAX_DISTILL_DISTANCE_THRESHOLD", 0.2)?.clamp(0.0, 2.0);
    let knn_k = parse_usize_env("VIDARAX_DISTILL_KNN_K", 7)?.clamp(1, 100);

    Ok(DistillationConfig {
        enabled,
        embedding_server_url,
        teacher_model: Arc::from(teacher_model),
        max_pairs_per_tenant,
        collection_rate,
        distance_threshold,
        knn_k,
    })
}

/// Parse the WHIP VLM tiering knobs from environment variables. This is the
/// population path for the `webrtc_first_pass_model` / `webrtc_second_pass_model`
/// / `webrtc_second_pass_threshold` / `webrtc_second_pass_max_tokens` fields
/// on [`ServerConfig`], called once from [`ServerConfig::from_env`]. After
/// that, [`build_webrtc_vlm_config`] reads those stored fields, not the
/// environment again, so the `ServerConfig` a caller builds and passes to
/// [`crate::run`] is the single source of truth for a session's tiering
/// behavior.
fn parse_webrtc_vlm_tiering() -> Result<(String, Option<String>, f32, u32), String> {
    let first_pass_model = env::var("VIDARAX_WEBRTC_FIRST_PASS_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-8B-Instruct".to_string());
    // An empty (or all-whitespace) value is treated the same as unset: the
    // stream stays on a single model with no escalation.
    let second_pass_model = env::var("VIDARAX_WEBRTC_SECOND_PASS_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let mut second_pass_threshold = parse_f32_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", 0.7)?;
    if !second_pass_threshold.is_finite() {
        // "NaN" and "inf"/"-inf" all parse successfully as f32, and clamp()
        // leaves NaN untouched (every comparison against NaN is false), so a
        // NaN threshold would make `confidence < threshold` always false and
        // silently turn off second-pass escalation forever. An infinite
        // threshold is equally unusable as a confidence cutoff. Neither is a
        // value an operator can have meant, so treat it the same as unset and
        // fall back to the default before clamping.
        second_pass_threshold = 0.7;
    }
    let second_pass_threshold = second_pass_threshold.clamp(0.0, 1.0);
    let second_pass_max_tokens =
        parse_u32_env_with_default("VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS", 256)?;
    Ok((
        first_pass_model,
        second_pass_model,
        second_pass_threshold,
        second_pass_max_tokens,
    ))
}

/// Build the tiered VLM config a WHIP session uses for keyframe inference,
/// resolved from `config`'s `webrtc_first_pass_model` / `webrtc_second_pass_model`
/// / `webrtc_second_pass_threshold` / `webrtc_second_pass_max_tokens` fields.
///
/// This reads only the passed-in `ServerConfig`, never the process
/// environment. `ServerConfig::from_env` is one way to populate those fields
/// (see [`parse_webrtc_vlm_tiering`]), but a caller that builds a
/// `ServerConfig` some other way, such as tests or an embedder driving
/// [`crate::run`] directly, gets exactly the tiering it configured on the
/// struct, with no dependency on what the environment happens to hold at
/// call time.
///
/// [`crate::build_webrtc_config`] calls this once at startup and stores the
/// result on [`vidarax_core::webrtc::session::WebRtcConfig::vlm_tiering`], so
/// every WHIP session clones an already-resolved value instead of
/// re-resolving it per session.
///
/// Defaults to a local-only config (single model, no escalation) whenever
/// `webrtc_second_pass_model` is unset, blank, or equal to the first-pass
/// model.
pub fn build_webrtc_vlm_config(config: &ServerConfig) -> TieredVlmConfig {
    // Defend the programmatic ServerConfig path the same way the environment
    // parser does. A non-finite threshold makes `confidence < threshold` false
    // for every frame, silently disabling escalation, and a caller that builds
    // ServerConfig directly never passes through the env sanitizer. Coerce
    // non-finite back to the default and clamp into range here so the stored
    // config, whatever its origin, cannot turn tiering off by accident.
    let second_pass_threshold = if config.webrtc_second_pass_threshold.is_finite() {
        config.webrtc_second_pass_threshold.clamp(0.0, 1.0)
    } else {
        0.7
    };
    let second_pass_model = config
        .webrtc_second_pass_model
        .as_deref()
        .filter(|s| !s.trim().is_empty());

    match second_pass_model {
        Some(second) if second != config.webrtc_first_pass_model => TieredVlmConfig {
            first_pass_model: Arc::from(config.webrtc_first_pass_model.as_str()),
            second_pass_model: Arc::from(second),
            second_pass_threshold,
            second_pass_max_tokens: config.webrtc_second_pass_max_tokens,
        },
        _ => TieredVlmConfig::single_model(&config.webrtc_first_pass_model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use vidarax_core::webrtc::session::WebRtcConfig;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct EnvRestore {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match self.old.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn set_env(key: &'static str, value: Option<&str>) -> EnvRestore {
        let old = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        EnvRestore { key, old }
    }

    // ─── TransportMode parsing ────────────────────────────────────────────────

    #[test]
    fn transport_mode_accepts_all_h1h2_aliases() {
        for alias in &["h1", "h2", "h1h2", "http", "http2"] {
            assert_eq!(
                TransportMode::parse(Some(alias)),
                Ok(TransportMode::H1H2),
                "alias '{alias}' should map to H1H2"
            );
        }
    }

    #[test]
    fn transport_mode_accepts_h3_aliases() {
        for alias in &["h3", "http3"] {
            assert_eq!(
                TransportMode::parse(Some(alias)),
                Ok(TransportMode::H3Experimental),
                "alias '{alias}' should map to H3Experimental"
            );
        }
    }

    #[test]
    fn transport_mode_defaults_to_h1h2_when_none() {
        assert_eq!(TransportMode::parse(None), Ok(TransportMode::H1H2));
    }

    #[test]
    fn transport_mode_rejects_unknown_value() {
        let result = TransportMode::parse(Some("grpc"));
        assert!(result.is_err(), "unknown transport mode should be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("grpc"),
            "error message should mention the unknown value: {msg}"
        );
    }

    #[test]
    fn server_config_preserves_custom_decode_backend_name() {
        let _guard = env_guard();
        let _decode_backend = set_env("VIDARAX_DECODE_BACKEND", Some("test-custom-backend"));
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));

        let cfg = ServerConfig::from_env().expect("custom decode backend should parse");

        assert_eq!(cfg.decode_backend, "test-custom-backend");
    }

    #[test]
    fn server_config_clamps_webrtc_worker_counts() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _decode_workers = set_env("VIDARAX_WEBRTC_DECODE_WORKERS", Some("999"));
        let _analysis_workers = set_env("VIDARAX_WEBRTC_ANALYSIS_WORKERS", Some("999"));
        let _vlm_workers = set_env("VIDARAX_WEBRTC_VLM_WORKERS", Some("999"));

        let cfg = ServerConfig::from_env().expect("worker counts should parse");

        assert_eq!(cfg.webrtc_decode_workers, 1);
        assert_eq!(cfg.webrtc_analysis_workers, 1);
        assert_eq!(cfg.webrtc_vlm_workers, 1);
    }

    #[test]
    fn server_config_default_ingest_roots_are_empty() {
        let _guard = env_guard();
        let _ingest_roots = set_env("VIDARAX_INGEST_FILE_ROOTS", None);
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));

        let cfg = ServerConfig::from_env().expect("default config should parse");

        assert!(
            cfg.ingest_file_roots.is_empty(),
            "shared ingest roots must be operator-configured; temp_dir is not a default root"
        );
    }

    #[test]
    fn server_config_gate_defaults_match_core_defaults_when_unset() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _keepalive = set_env("VIDARAX_GATE_KEEPALIVE_EVERY_FRAMES", None);
        let _scene_cut = set_env("VIDARAX_GATE_SCENE_CUT_HAMMING_THRESHOLD", None);
        let _luma = set_env("VIDARAX_GATE_LUMA_SHIFT_THRESHOLD", None);
        let _flicker = set_env("VIDARAX_GATE_FLICKER_THRESHOLD", None);
        let _ghosting = set_env("VIDARAX_GATE_GHOSTING_THRESHOLD", None);
        let _noise = set_env("VIDARAX_GATE_NOISE_VARIANCE_THRESHOLD", None);

        let cfg = ServerConfig::from_env().expect("default gate config should parse");

        assert_eq!(cfg.gate_config, GateConfig::default());
    }

    #[test]
    fn server_config_gate_thresholds_can_be_overridden() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _keepalive = set_env("VIDARAX_GATE_KEEPALIVE_EVERY_FRAMES", None);
        let _scene_cut = set_env("VIDARAX_GATE_SCENE_CUT_HAMMING_THRESHOLD", Some("7"));
        let _luma = set_env("VIDARAX_GATE_LUMA_SHIFT_THRESHOLD", None);
        let _flicker = set_env("VIDARAX_GATE_FLICKER_THRESHOLD", None);
        let _ghosting = set_env("VIDARAX_GATE_GHOSTING_THRESHOLD", None);
        let _noise = set_env("VIDARAX_GATE_NOISE_VARIANCE_THRESHOLD", None);

        let cfg = ServerConfig::from_env().expect("gate override should parse");

        assert_eq!(cfg.gate_config.scene_cut_hamming_threshold, 7);
        assert_eq!(
            cfg.gate_config.keepalive_every_frames,
            GateConfig::default().keepalive_every_frames
        );
    }

    #[test]
    fn server_config_reads_spacetimedb_env() {
        let _guard = env_guard();
        let _url = set_env("VIDARAX_SPACETIMEDB_URL", None);
        let _module = set_env("VIDARAX_SPACETIMEDB_MODULE", None);
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));

        let cfg = ServerConfig::from_env().expect("default config should parse");
        assert!(cfg.spacetimedb_url.is_none());
        assert!(cfg.spacetimedb_module.is_none());

        let _url = set_env("VIDARAX_SPACETIMEDB_URL", Some("http://127.0.0.1:3000"));

        let cfg = ServerConfig::from_env().expect("url-only config should parse");
        assert_eq!(
            cfg.spacetimedb_url.as_deref(),
            Some("http://127.0.0.1:3000")
        );
        assert!(cfg.spacetimedb_module.is_none());

        let _module = set_env("VIDARAX_SPACETIMEDB_MODULE", Some("custom_module"));

        let cfg = ServerConfig::from_env().expect("url and module config should parse");
        assert_eq!(
            cfg.spacetimedb_url.as_deref(),
            Some("http://127.0.0.1:3000")
        );
        assert_eq!(cfg.spacetimedb_module.as_deref(), Some("custom_module"));
    }

    #[test]
    fn server_config_rejects_required_tenant_without_required_api_key() {
        let err = validate_tenant_auth_config(false, true)
            .expect_err("required tenant IDs without required API-key auth should be rejected");

        assert!(
            err.contains("VIDARAX_REQUIRE_TENANT_ID requires VIDARAX_REQUIRE_API_KEY=true"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn server_config_explicit_ingest_roots_are_canonicalized() {
        let _guard = env_guard();
        let root =
            std::env::temp_dir().join(format!("vidarax-explicit-root-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let _ingest_roots = set_env(
            "VIDARAX_INGEST_FILE_ROOTS",
            Some(root.to_string_lossy().as_ref()),
        );
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));

        let cfg = ServerConfig::from_env().expect("explicit ingest root should parse");

        assert_eq!(cfg.ingest_file_roots, vec![root.canonicalize().unwrap()]);
        let _ = std::fs::remove_dir(root);
    }

    // ─── WebRtcConfig defaults (STUN / token-rate) ───────────────────────────

    #[test]
    fn webrtc_config_default_stun_is_google() {
        let wc = WebRtcConfig::default();
        assert_eq!(
            wc.stun_servers,
            vec!["stun:stun.l.google.com:19302"],
            "default STUN should point to Google"
        );
    }

    #[test]
    fn webrtc_config_default_has_no_turn_servers() {
        let wc = WebRtcConfig::default();
        assert!(
            wc.turn_servers.is_empty(),
            "default WebRtcConfig should have no TURN servers"
        );
    }

    #[test]
    fn webrtc_config_default_token_rate_is_128() {
        let wc = WebRtcConfig::default();
        assert_eq!(
            wc.max_output_tokens_per_second, 128,
            "default token-rate cap should be 128 t/s"
        );
    }

    // ─── ServerConfig TURN/STUN field round-trip ─────────────────────────────

    #[test]
    fn server_config_turn_fields_roundtrip() {
        let cfg = ServerConfig {
            bind_addr: "127.0.0.1:8080".into(),
            h3_bind_addr: "127.0.0.1:8443".into(),
            h3_tls_cert_path: "dev.crt".into(),
            h3_tls_key_path: "dev.key".into(),
            data_dir: "/tmp".into(),
            ingest_file_roots: vec![],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
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
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:custom.example.com:3478".into()],
            webrtc_turn_url: Some("turn:relay.example.com:3478".into()),
            webrtc_turn_username: Some("alice".into()),
            webrtc_turn_credential: Some("secret".into()),
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 64,
            webrtc_decode_workers: 2,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 2,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            gate_config: GateConfig::default(),
            distillation: DistillationConfig::default(),
        };

        assert_eq!(
            cfg.webrtc_turn_url.as_deref(),
            Some("turn:relay.example.com:3478")
        );
        assert_eq!(cfg.webrtc_turn_username.as_deref(), Some("alice"));
        assert_eq!(cfg.webrtc_turn_credential.as_deref(), Some("secret"));
        assert_eq!(cfg.webrtc_max_output_tokens_per_second, 64);
        assert_eq!(cfg.webrtc_decode_workers, 2);
        assert_eq!(cfg.webrtc_analysis_workers, 1);
        assert_eq!(cfg.webrtc_vlm_workers, 2);
        assert_eq!(
            cfg.webrtc_stun_servers,
            vec!["stun:custom.example.com:3478"]
        );
    }

    #[test]
    fn server_config_turn_absent_fields_are_none() {
        let cfg = ServerConfig {
            bind_addr: "127.0.0.1:8080".into(),
            h3_bind_addr: "127.0.0.1:8443".into(),
            h3_tls_cert_path: "dev.crt".into(),
            h3_tls_key_path: "dev.key".into(),
            data_dir: "/tmp".into(),
            ingest_file_roots: vec![],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
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
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".into()],
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
            gate_config: GateConfig::default(),
            distillation: DistillationConfig::default(),
        };

        assert!(cfg.webrtc_turn_url.is_none(), "turn_url should be None");
        assert!(
            cfg.webrtc_turn_username.is_none(),
            "turn_username should be None"
        );
        assert!(
            cfg.webrtc_turn_credential.is_none(),
            "turn_credential should be None"
        );
    }

    // ─── WHIP VLM tiering env parsing ────────────────────────────────────────

    #[test]
    fn server_config_vlm_tiering_defaults_to_local_only() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _first = set_env("VIDARAX_WEBRTC_FIRST_PASS_MODEL", None);
        let _second = set_env("VIDARAX_WEBRTC_SECOND_PASS_MODEL", None);
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", None);
        let _max_tokens = set_env("VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS", None);

        let cfg = ServerConfig::from_env().expect("default config should parse");

        assert_eq!(cfg.webrtc_first_pass_model, "Qwen/Qwen3-VL-8B-Instruct");
        assert!(
            cfg.webrtc_second_pass_model.is_none(),
            "unset second-pass model must default to local-only (no escalation)"
        );
        assert_eq!(cfg.webrtc_second_pass_threshold, 0.7);
        assert_eq!(cfg.webrtc_second_pass_max_tokens, 256);

        let tiered = build_webrtc_vlm_config(&cfg);
        assert!(
            !tiered.is_tiered(),
            "default TieredVlmConfig must stay local-only so live streams never escalate unasked"
        );
    }

    #[test]
    fn server_config_vlm_tiering_blank_second_pass_model_is_local_only() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _second = set_env("VIDARAX_WEBRTC_SECOND_PASS_MODEL", Some("   "));

        let cfg = ServerConfig::from_env().expect("blank second-pass model should parse");

        assert!(
            cfg.webrtc_second_pass_model.is_none(),
            "an empty/whitespace-only value must be treated the same as unset"
        );
    }

    #[test]
    fn server_config_vlm_tiering_differing_second_pass_model_enables_tiering() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _first = set_env("VIDARAX_WEBRTC_FIRST_PASS_MODEL", None);
        let _second = set_env(
            "VIDARAX_WEBRTC_SECOND_PASS_MODEL",
            Some("gemini-3.1-flash-lite"),
        );

        let cfg = ServerConfig::from_env().expect("tiered config should parse");

        assert_eq!(
            cfg.webrtc_second_pass_model.as_deref(),
            Some("gemini-3.1-flash-lite")
        );
        assert_ne!(
            cfg.webrtc_second_pass_model.as_deref(),
            Some(cfg.webrtc_first_pass_model.as_str())
        );

        let tiered = build_webrtc_vlm_config(&cfg);
        assert!(tiered.is_tiered(), "differing models must enable tiering");
        assert_eq!(&*tiered.second_pass_model, "gemini-3.1-flash-lite");
    }

    #[test]
    fn server_config_vlm_tiering_second_pass_equal_to_first_stays_local_only() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _first = set_env("VIDARAX_WEBRTC_FIRST_PASS_MODEL", None);
        let _second = set_env(
            "VIDARAX_WEBRTC_SECOND_PASS_MODEL",
            Some("Qwen/Qwen3-VL-8B-Instruct"),
        );

        let cfg = ServerConfig::from_env().expect("equal-model config should parse");

        assert_eq!(
            cfg.webrtc_second_pass_model.as_deref(),
            Some(cfg.webrtc_first_pass_model.as_str())
        );

        let tiered = build_webrtc_vlm_config(&cfg);
        assert!(
            !tiered.is_tiered(),
            "a second-pass model equal to the first-pass model must stay local-only"
        );
    }

    #[test]
    fn server_config_vlm_tiering_threshold_is_clamped_to_unit_range() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", Some("5.0"));

        let cfg = ServerConfig::from_env().expect("out-of-range threshold should clamp, not fail");

        assert_eq!(cfg.webrtc_second_pass_threshold, 1.0);
    }

    #[test]
    fn server_config_vlm_tiering_threshold_out_of_range_clamps_to_one() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", Some("1.5"));

        let cfg = ServerConfig::from_env().expect("out-of-range threshold should clamp, not fail");

        assert_eq!(cfg.webrtc_second_pass_threshold, 1.0);
    }

    #[test]
    fn server_config_vlm_tiering_threshold_in_range_is_preserved() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", Some("0.5"));

        let cfg = ServerConfig::from_env().expect("in-range threshold should parse");

        assert_eq!(cfg.webrtc_second_pass_threshold, 0.5);
    }

    #[test]
    fn server_config_vlm_tiering_threshold_nan_falls_back_to_default() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", Some("NaN"));

        let cfg =
            ServerConfig::from_env().expect("NaN threshold should parse and fall back, not fail");

        assert_eq!(
            cfg.webrtc_second_pass_threshold, 0.7,
            "a NaN threshold must fall back to the default instead of silently disabling escalation"
        );
    }

    #[test]
    fn server_config_vlm_tiering_threshold_infinite_falls_back_to_default() {
        let _guard = env_guard();
        let _require_api_key = set_env("VIDARAX_REQUIRE_API_KEY", Some("false"));
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", Some("inf"));

        let cfg = ServerConfig::from_env()
            .expect("infinite threshold should parse and fall back, not fail");

        assert_eq!(
            cfg.webrtc_second_pass_threshold, 0.7,
            "an infinite threshold must fall back to the default before clamping"
        );
    }

    // ─── build_webrtc_vlm_config resolves from ServerConfig, not env ─────────

    #[test]
    fn build_webrtc_vlm_config_reads_server_config_fields_not_env() {
        let _guard = env_guard();
        // The environment carries no tiering opt-in at all; if
        // build_webrtc_vlm_config still reached for the environment, this
        // would stay local-only regardless of what the ServerConfig says.
        let _first = set_env("VIDARAX_WEBRTC_FIRST_PASS_MODEL", None);
        let _second = set_env("VIDARAX_WEBRTC_SECOND_PASS_MODEL", None);
        let _threshold = set_env("VIDARAX_WEBRTC_SECOND_PASS_THRESHOLD", None);
        let _max_tokens = set_env("VIDARAX_WEBRTC_SECOND_PASS_MAX_TOKENS", None);

        let mut cfg = default_test_server_config_for_tiering();
        cfg.webrtc_first_pass_model = "local-model".to_string();
        cfg.webrtc_second_pass_model = Some("escalation-model".to_string());
        cfg.webrtc_second_pass_threshold = 0.42;
        cfg.webrtc_second_pass_max_tokens = 512;

        let tiered = build_webrtc_vlm_config(&cfg);

        assert!(
            tiered.is_tiered(),
            "a ServerConfig with a distinct second-pass model must resolve to tiered, \
             independent of the process environment"
        );
        assert_eq!(&*tiered.first_pass_model, "local-model");
        assert_eq!(&*tiered.second_pass_model, "escalation-model");
        assert_eq!(tiered.second_pass_threshold, 0.42);
        assert_eq!(tiered.second_pass_max_tokens, 512);
    }

    #[test]
    fn build_webrtc_vlm_config_sanitizes_non_finite_programmatic_threshold() {
        // A ServerConfig built directly (not via from_env) can carry a
        // non-finite threshold. `confidence < NaN` is always false, so an
        // unsanitized value would silently disable escalation even with a
        // distinct second-pass model configured. build_webrtc_vlm_config must
        // coerce it back to the default, matching the environment parser.
        let mut cfg = default_test_server_config_for_tiering();
        cfg.webrtc_first_pass_model = "local-model".to_string();
        cfg.webrtc_second_pass_model = Some("escalation-model".to_string());
        cfg.webrtc_second_pass_max_tokens = 256;

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            cfg.webrtc_second_pass_threshold = bad;
            let tiered = build_webrtc_vlm_config(&cfg);
            assert!(tiered.is_tiered());
            assert_eq!(
                tiered.second_pass_threshold, 0.7,
                "a non-finite programmatic threshold must fall back to the default"
            );
        }

        // Finite out-of-range values clamp into [0, 1] rather than falling back.
        cfg.webrtc_second_pass_threshold = 5.0;
        assert_eq!(build_webrtc_vlm_config(&cfg).second_pass_threshold, 1.0);
        cfg.webrtc_second_pass_threshold = -1.0;
        assert_eq!(build_webrtc_vlm_config(&cfg).second_pass_threshold, 0.0);
    }

    fn default_test_server_config_for_tiering() -> ServerConfig {
        ServerConfig {
            bind_addr: "127.0.0.1:8080".into(),
            h3_bind_addr: "127.0.0.1:8443".into(),
            h3_tls_cert_path: "dev.crt".into(),
            h3_tls_key_path: "dev.key".into(),
            data_dir: "/tmp".into(),
            ingest_file_roots: vec![],
            inference_vllm_base_url: None,
            inference_sglang_base_url: None,
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
            decode_backend: "cpu-ffmpeg".to_string(),
            webrtc_stun_servers: vec!["stun:stun.l.google.com:19302".into()],
            webrtc_turn_url: None,
            webrtc_turn_username: None,
            webrtc_turn_credential: None,
            spacetimedb_url: None,
            spacetimedb_module: None,
            webrtc_max_output_tokens_per_second: 128,
            webrtc_decode_workers: 1,
            webrtc_analysis_workers: 1,
            webrtc_vlm_workers: 1,
            webrtc_first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            webrtc_second_pass_model: None,
            webrtc_second_pass_threshold: 0.7,
            webrtc_second_pass_max_tokens: 256,
            gate_config: GateConfig::default(),
            distillation: DistillationConfig::default(),
        }
    }
}

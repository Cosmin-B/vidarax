use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use vidarax_core::tiered_vlm::DistillationConfig;

pub(crate) const UPLOAD_DIR_NAME: &str = "vidarax-uploads";

/// Load the backend config from a TOML file, falling back to legacy env vars.
///
/// Reads `config_path` (default: `"vidarax.toml"`).  If the file is absent or
/// unreadable the function synthesises a [`vidarax_core::backends::VidaraxConfig`]
/// from the legacy `VIDARAX_VLLM_BASE_URL` / `VIDARAX_SGLANG_BASE_URL` env vars
/// so that existing deployments continue to work unchanged.
pub fn load_backend_config(config_path: &str) -> Result<vidarax_core::backends::VidaraxConfig, String> {
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
        let webrtc_decode_workers =
            parse_usize_env("VIDARAX_WEBRTC_DECODE_WORKERS", 1)?.clamp(1, 64).min(1);
        // Analysis owns stream-order gate/loop state; more workers make one
        // shared stream's decisions nondeterministic.
        let webrtc_analysis_workers = parse_usize_env("VIDARAX_WEBRTC_ANALYSIS_WORKERS", 1)?
            .clamp(1, 64)
            .min(1);
        // VLM keyframe analysis carries temporal/dedup state; do not split one
        // stream across racing workers.
        let webrtc_vlm_workers =
            parse_usize_env("VIDARAX_WEBRTC_VLM_WORKERS", 1)?.clamp(1, 64).min(1);
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

fn parse_distillation_config() -> Result<DistillationConfig, String> {
    let enabled = parse_bool_env("VIDARAX_DISTILL_ENABLED", false)?;
    let embedding_server_url = env::var("VIDARAX_DISTILL_EMBEDDING_URL").ok();
    let teacher_model = env::var("VIDARAX_DISTILL_TEACHER_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-8B-Instruct".to_string());
    let max_pairs_per_tenant =
        parse_usize_env("VIDARAX_DISTILL_MAX_PAIRS", 10_000)?.clamp(100, 1_000_000);
    let collection_rate = parse_f32_env("VIDARAX_DISTILL_COLLECTION_RATE", 0.1)?
        .clamp(0.0, 1.0);
    let distance_threshold = parse_f32_env("VIDARAX_DISTILL_DISTANCE_THRESHOLD", 0.2)?
        .clamp(0.0, 2.0);
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
        let root = std::env::temp_dir().join(format!(
            "vidarax-explicit-root-{}",
            std::process::id()
        ));
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
        assert_eq!(cfg.webrtc_stun_servers, vec!["stun:custom.example.com:3478"]);
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
}

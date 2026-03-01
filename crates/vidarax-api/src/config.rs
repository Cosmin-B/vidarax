use std::env;
use std::path::PathBuf;

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
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, String> {
        let transport = TransportMode::parse(env::var("VIDARAX_TRANSPORT").ok().as_deref())?;
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
        let stream_ttl_secs = parse_u64_env_with_default("VIDARAX_STREAM_TTL_SECS", 3600)?;
        if !(60..=86_400).contains(&stream_ttl_secs) {
            return Err("VIDARAX_STREAM_TTL_SECS must be in [60, 86400]".to_string());
        }
        let active_stream_limit = parse_usize_env("VIDARAX_ACTIVE_STREAM_LIMIT", 5)?.clamp(1, 1024);
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
        let cwd = std::env::current_dir().map_err(|err| format!("failed to resolve cwd: {err}"))?;
        let tmp = std::env::temp_dir();
        return Ok(vec![
            cwd.canonicalize()
                .map_err(|err| format!("failed to canonicalize cwd ingest root: {err}"))?,
            tmp.canonicalize()
                .map_err(|err| format!("failed to canonicalize temp ingest root: {err}"))?,
        ]);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteSpec {
    pub method: &'static str,
    pub path: &'static str,
}

const ROUTE_MANIFEST: &[RouteSpec] = &[
    RouteSpec {
        method: "POST",
        path: "/v1/runs",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/runs/{run_id}/ingest",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/runs/{run_id}/analyze",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/runs/{run_id}/reason",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/runs/{run_id}/stop",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/runs/{run_id}/keepalive",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/runs/{run_id}/events",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/runs/{run_id}/markers",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/runs/{run_id}/state",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/query",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/infer",
    },
    RouteSpec {
        method: "POST",
        path: "/v1/infer/batch",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/models",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/health",
    },
    RouteSpec {
        method: "GET",
        path: "/v1/metrics",
    },
];

pub fn route_manifest() -> &'static [RouteSpec] {
    ROUTE_MANIFEST
}

pub fn assert_route_parity() -> Result<(), String> {
    let h1h2_fingerprint = fingerprint(route_manifest());
    let h3_fingerprint = fingerprint(route_manifest());
    if h1h2_fingerprint != h3_fingerprint {
        return Err(format!(
            "transport route parity mismatch: h1h2={h1h2_fingerprint:#x} h3={h3_fingerprint:#x}"
        ));
    }
    Ok(())
}

pub fn resolve_wal_path(config: &ServerConfig) -> Result<PathBuf, String> {
    let data_dir = PathBuf::from(&config.data_dir);
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;
    Ok(data_dir.join("timeline.wal"))
}

fn fingerprint(routes: &[RouteSpec]) -> u64 {
    // FNV-1a gives deterministic, stable parity checks without heap allocation.
    let mut hash = 1469598103934665603u64;
    for route in routes {
        for b in route.method.bytes().chain([b' ']).chain(route.path.bytes()) {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }
    hash
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

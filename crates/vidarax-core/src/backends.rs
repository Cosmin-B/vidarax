//! Config-driven backend adapter system.
//!
//! Parses `vidarax.toml` and builds an [`crate::provider::InferenceProvider`] chain
//! sorted by priority.  String values in the config support `${ENV_VAR}` interpolation.
//!
//! # Example
//!
//! ```toml
//! [[backends]]
//! name = "vllm"
//! type = "openai_compat"
//! base_url = "${VIDARAX_VLLM_BASE_URL}"
//! priority = 1
//!
//! [[backends]]
//! name = "sglang"
//! type = "openai_compat"
//! base_url = "${VIDARAX_SGLANG_BASE_URL}"
//! priority = 2
//! ```

use std::sync::Arc;

use serde::Deserialize;

use crate::gemini::GeminiProvider;
use crate::provider::{HttpTransport, InferenceProvider, OpenAiCompatProvider, ProviderKind, ProviderRouter};

// ── Config structs ────────────────────────────────────────────────────────────

/// Top-level `vidarax.toml` structure.
#[derive(Debug, Clone, Deserialize)]
pub struct VidaraxConfig {
    #[serde(default)]
    pub backends: Vec<BackendEntry>,
}

/// A single backend entry from `[[backends]]`.
#[derive(Debug, Clone, Deserialize)]
pub struct BackendEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub backend_type: String,
    /// Base URL for `openai_compat` backends (e.g. `http://127.0.0.1:8000`).
    pub base_url: Option<String>,
    /// API key for `gemini` backends.
    pub api_key: Option<String>,
    /// Default model ID for `gemini` backends.
    pub model: Option<String>,
    /// Backends are tried in ascending priority order (lowest value = first).
    #[serde(default = "default_priority")]
    pub priority: u32,
}

fn default_priority() -> u32 {
    100
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a TOML string into a [`VidaraxConfig`].
///
/// All `Option<String>` fields in each [`BackendEntry`] have `${VAR}` patterns
/// expanded via [`interpolate_env_vars`] before being returned.
pub fn parse_config(toml_str: &str) -> Result<VidaraxConfig, String> {
    let mut config: VidaraxConfig =
        toml::from_str(toml_str).map_err(|e| format!("toml parse error: {e}"))?;

    for entry in &mut config.backends {
        entry.name = interpolate_env_vars(&entry.name);
        entry.backend_type = interpolate_env_vars(&entry.backend_type);
        if let Some(s) = &entry.base_url {
            entry.base_url = Some(interpolate_env_vars(s));
        }
        if let Some(s) = &entry.api_key {
            entry.api_key = Some(interpolate_env_vars(s));
        }
        if let Some(s) = &entry.model {
            entry.model = Some(interpolate_env_vars(s));
        }
    }

    Ok(config)
}

/// Expand `${VAR_NAME}` patterns in `s` using environment variables.
///
/// If the variable is not set the literal `${VAR_NAME}` is left in place;
/// this function never panics.
pub fn interpolate_env_vars(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut remaining = s;

    while let Some(start) = remaining.find("${") {
        result.push_str(&remaining[..start]);
        remaining = &remaining[start + 2..]; // skip "${"

        if let Some(end) = remaining.find('}') {
            let var_name = &remaining[..end];
            match std::env::var(var_name) {
                Ok(value) => result.push_str(&value),
                Err(_) => {
                    // Leave the literal placeholder intact.
                    result.push_str("${");
                    result.push_str(var_name);
                    result.push('}');
                }
            }
            remaining = &remaining[end + 1..]; // skip past '}'
        } else {
            // No closing brace — emit the rest verbatim.
            result.push_str("${");
            result.push_str(remaining);
            remaining = "";
        }
    }
    result.push_str(remaining);
    result
}

/// Build a provider chain from a slice of [`BackendEntry`] values.
///
/// Entries are sorted by `priority` (ascending).  The chain is assembled as
/// nested [`ProviderRouter`]s: the lowest-priority entry is the primary, the
/// next is its fallback, and so on.
///
/// Returns `Err` if:
/// - `entries` is empty.
/// - Any entry has an unrecognised `type`.
/// - A required field (e.g. `base_url` for `openai_compat`) is missing.
pub fn build_provider_chain(
    entries: &[BackendEntry],
) -> Result<Arc<dyn InferenceProvider + Send + Sync>, String> {
    if entries.is_empty() {
        return Err("no backends configured".to_string());
    }

    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| e.priority);

    let providers: Vec<Box<dyn InferenceProvider + Send + Sync>> = sorted
        .iter()
        .map(build_single_provider)
        .collect::<Result<_, _>>()?;

    // Fold from the back so that the first entry becomes the outermost primary.
    // With a single provider we skip wrapping entirely.
    let chain: Arc<dyn InferenceProvider + Send + Sync> = if providers.len() == 1 {
        let only = providers.into_iter().next().unwrap();
        Arc::from(only)
    } else {
        fold_into_router(providers)
    };

    Ok(chain)
}

fn build_single_provider(
    entry: &BackendEntry,
) -> Result<Box<dyn InferenceProvider + Send + Sync>, String> {
    match entry.backend_type.as_str() {
        "openai_compat" => {
            // Note: env vars are already interpolated by parse_config().
            let base_url = entry.base_url.as_deref().unwrap_or("");
            if base_url.is_empty() || base_url.contains("${") {
                return Err(format!(
                    "backend '{}': openai_compat requires a valid base_url (unresolved env var?)",
                    entry.name
                ));
            }

            // Warn when a non-loopback backend is configured with plain HTTP.
            if base_url.starts_with("http://") {
                let is_loopback = base_url.contains("://localhost")
                    || base_url.contains("://127.0.0.1")
                    || base_url.contains("://[::1]");
                if !is_loopback {
                    tracing::warn!(
                        backend = %entry.name,
                        url = %base_url,
                        "backend uses plain HTTP on a non-loopback host — consider HTTPS in production"
                    );
                }
            }

            // Derive ProviderKind from the entry name for backwards-compatible
            // telemetry labelling.  Unknown names fall back to Vllm.
            let kind = match entry.name.to_ascii_lowercase().as_str() {
                "sglang" => ProviderKind::Sglang,
                _ => ProviderKind::Vllm,
            };

            let transport = HttpTransport::new(&base_url)
                .map_err(|e| format!("backend '{}': transport error: {e:?}", entry.name))?;

            Ok(Box::new(OpenAiCompatProvider::new(transport, kind)))
        }
        "gemini" => {
            // Note: env vars are already interpolated by parse_config().
            let api_key = entry.api_key.as_deref().unwrap_or("");
            let model = entry
                .model
                .as_deref()
                .unwrap_or("gemini-2.5-flash-preview-05-20");
            if api_key.is_empty() || api_key.contains("${") {
                return Err(format!(
                    "backend '{}': gemini requires api_key",
                    entry.name
                ));
            }
            Ok(Box::new(
                GeminiProvider::new(api_key.to_string(), model.to_string())
                    .map_err(|e| format!("backend '{}': {e:?}", entry.name))?,
            ))
        }
        other => Err(format!("backend '{}': unknown type '{other}'", entry.name)),
    }
}

/// Fold a `Vec<Box<dyn InferenceProvider>>` (length ≥ 2) into a nested
/// [`ProviderRouter`] chain where `providers[0]` is the primary and each
/// subsequent entry is a deeper fallback.
///
/// Because `ProviderRouter<P, F>` is generic we erase the types through
/// `Arc<dyn InferenceProvider + Send + Sync>` at each nesting level.
fn fold_into_router(
    providers: Vec<Box<dyn InferenceProvider + Send + Sync>>,
) -> Arc<dyn InferenceProvider + Send + Sync> {
    // Convert each Box into an Arc so we can use the blanket impl for Arc<dyn InferenceProvider>.
    let arcs: Vec<Arc<dyn InferenceProvider + Send + Sync>> =
        providers.into_iter().map(Arc::from).collect();

    // Fold from the right: the last arc becomes the innermost fallback.
    let mut iter = arcs.into_iter().rev();
    let mut chain: Arc<dyn InferenceProvider + Send + Sync> = iter.next().unwrap();

    for primary in iter {
        chain = Arc::new(ProviderRouter::new(primary, chain));
    }

    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::ENV_TEST_LOCK
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

    #[test]
    fn interpolate_replaces_set_variable() {
        let _guard = env_guard();
        let _var = set_env("_VIDARAX_TEST_VAR", Some("hello"));
        let result = interpolate_env_vars("prefix_${_VIDARAX_TEST_VAR}_suffix");
        assert_eq!(result, "prefix_hello_suffix");
    }

    #[test]
    fn interpolate_leaves_unset_variable_literal() {
        let _guard = env_guard();
        let _var = set_env("_VIDARAX_MISSING_VAR_XYZ", None);
        let result = interpolate_env_vars("${_VIDARAX_MISSING_VAR_XYZ}");
        assert_eq!(result, "${_VIDARAX_MISSING_VAR_XYZ}");
    }

    #[test]
    fn interpolate_no_placeholders_is_identity() {
        let s = "http://127.0.0.1:8000";
        assert_eq!(interpolate_env_vars(s), s);
    }

    #[test]
    fn interpolate_multiple_placeholders() {
        let _guard = env_guard();
        let _a = set_env("_VDX_A", Some("foo"));
        let _b = set_env("_VDX_B", Some("bar"));
        let result = interpolate_env_vars("${_VDX_A}/${_VDX_B}");
        assert_eq!(result, "foo/bar");
    }

    // ── parse_config ──────────────────────────────────────────────────────────

    #[test]
    fn parse_config_minimal() {
        let toml = r#"
[[backends]]
name = "vllm"
type = "openai_compat"
base_url = "http://localhost:8000"
priority = 1
"#;
        let config = parse_config(toml).expect("parse");
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.backends[0].name, "vllm");
        assert_eq!(config.backends[0].priority, 1);
    }

    #[test]
    fn parse_config_empty_backends() {
        let config = parse_config("").expect("empty toml is valid");
        assert!(config.backends.is_empty());
    }

    #[test]
    fn parse_config_default_priority() {
        let toml = r#"
[[backends]]
name = "x"
type = "openai_compat"
base_url = "http://localhost:8000"
"#;
        let config = parse_config(toml).expect("parse");
        assert_eq!(config.backends[0].priority, 100);
    }

    #[test]
    fn parse_config_interpolates_base_url() {
        let _guard = env_guard();
        let _url = set_env("_VDX_TEST_URL", Some("http://myhost:1234"));
        let toml = r#"
[[backends]]
name = "vllm"
type = "openai_compat"
base_url = "${_VDX_TEST_URL}"
"#;
        let config = parse_config(toml).expect("parse");
        assert_eq!(
            config.backends[0].base_url.as_deref(),
            Some("http://myhost:1234")
        );
    }

    // ── build_provider_chain ──────────────────────────────────────────────────

    #[test]
    fn build_chain_empty_returns_err() {
        assert!(build_provider_chain(&[]).is_err());
    }

    #[test]
    fn build_chain_unknown_type_returns_err() {
        let entry = BackendEntry {
            name: "x".into(),
            backend_type: "unknown_type".into(),
            base_url: None,
            api_key: None,
            model: None,
            priority: 1,
        };
        assert!(build_provider_chain(&[entry]).is_err());
    }

    #[test]
    fn build_chain_gemini_succeeds_with_api_key() {
        let entry = BackendEntry {
            name: "gemini".into(),
            backend_type: "gemini".into(),
            base_url: None,
            api_key: Some("test-api-key".into()),
            model: Some("gemini-2.5-flash".into()),
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_ok(), "gemini with api_key should succeed, got: {:?}", result.err());
        assert_eq!(result.unwrap().kind(), ProviderKind::Gemini);
    }

    #[test]
    fn build_chain_gemini_missing_api_key_returns_err() {
        let entry = BackendEntry {
            name: "gemini".into(),
            backend_type: "gemini".into(),
            base_url: None,
            api_key: None,
            model: Some("gemini-2.5-flash".into()),
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_err(), "gemini without api_key should fail");
        let err = result.err().unwrap();
        assert!(err.contains("api_key"), "expected api_key error, got: {err}");
    }

    #[test]
    fn build_chain_gemini_unresolved_env_var_api_key_returns_err() {
        let _guard = env_guard();
        let _key = set_env("_VDX_NONEXISTENT_GEMINI_KEY", None);
        let entry = BackendEntry {
            name: "gemini".into(),
            backend_type: "gemini".into(),
            base_url: None,
            api_key: Some("${_VDX_NONEXISTENT_GEMINI_KEY}".into()),
            model: None,
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_err(), "unresolved api_key should fail");
    }

    #[test]
    fn build_chain_missing_base_url_returns_err() {
        let entry = BackendEntry {
            name: "vllm".into(),
            backend_type: "openai_compat".into(),
            base_url: None,
            api_key: None,
            model: None,
            priority: 1,
        };
        assert!(build_provider_chain(&[entry]).is_err());
    }

    #[test]
    fn build_chain_openai_compat_unresolved_env_var_base_url_returns_err() {
        let _guard = env_guard();
        let _url = set_env("_VDX_NONEXISTENT_VLLM_URL", None);
        let entry = BackendEntry {
            name: "vllm".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("${_VDX_NONEXISTENT_VLLM_URL}".into()),
            api_key: None,
            model: None,
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_err(), "unresolved base_url should fail");
        let err = result.err().unwrap();
        assert!(err.contains("base_url"), "expected base_url error, got: {err}");
    }

    #[test]
    fn build_chain_single_provider() {
        let entry = BackendEntry {
            name: "vllm".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://localhost:8000".into()),
            api_key: None,
            model: None,
            priority: 1,
        };
        let provider = build_provider_chain(&[entry]).expect("single provider");
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }

    #[test]
    fn build_chain_two_providers_primary_is_lower_priority() {
        let entries = vec![
            BackendEntry {
                name: "sglang".into(),
                backend_type: "openai_compat".into(),
                base_url: Some("http://localhost:8001".into()),
                api_key: None,
                model: None,
                priority: 2,
            },
            BackendEntry {
                name: "vllm".into(),
                backend_type: "openai_compat".into(),
                base_url: Some("http://localhost:8000".into()),
                api_key: None,
                model: None,
                priority: 1,
            },
        ];
        let provider = build_provider_chain(&entries).expect("two providers");
        // Primary should be the vllm entry (priority = 1).
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }
}

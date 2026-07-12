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
use crate::provider::{
    HttpTransport, InferenceProvider, ModelRoutingProvider, OpenAiCompatProvider, ProviderKind,
    ProviderRouter,
};

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
    /// Provider flavor for `openai_compat` backends, used for telemetry
    /// labelling. Accepts "vllm", "sglang", or "mlx" (mlx-vlm's OpenAI-compatible
    /// server, typically run on-device on Apple Silicon). When unset the flavor
    /// is guessed from the backend name for backward compatibility, which is why
    /// setting it explicitly is preferred: the name is a free-form label, not a
    /// type.
    #[serde(default)]
    pub openai_kind: Option<String>,
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
        if let Some(s) = &entry.openai_kind {
            entry.openai_kind = Some(interpolate_env_vars(s));
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
/// The chain is returned behind `Arc<dyn InferenceProvider>` on purpose. The set
/// of providers is decided at runtime from config, and the fallback structure is
/// recursive — a router wraps a router wraps a leaf — so a closed enum could not
/// describe it without boxing the nested case anyway. Static dispatch would also
/// force every caller and worker to carry the concrete chain type in its
/// signature. Neither cost buys anything back: each call through the chain ends in
/// a network round-trip to the model, so the virtual dispatch is not what bounds
/// it, and the `Arc` is cloned once per worker at spawn rather than per frame.
/// This is the deliberate exception to the ownership-first, static-dispatch
/// preference used on the hot per-frame paths.
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
        // The empty-entries guard above ran, and every entry yields one provider,
        // so a length of one means exactly one element is waiting here.
        let only = providers
            .into_iter()
            .next()
            .expect("provider count checked to be exactly one");
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

            // Prefer the explicit flavor. Fall back to guessing from the backend
            // name only when it is unset, so existing configs that named their
            // backend "sglang" keep their old telemetry label. Unknown values
            // fall back to Vllm.
            let flavor = entry
                .openai_kind
                .as_deref()
                .unwrap_or(entry.name.as_str())
                .to_ascii_lowercase();
            let kind = match flavor.as_str() {
                "sglang" => ProviderKind::Sglang,
                "mlx" => ProviderKind::Mlx,
                _ => ProviderKind::Vllm,
            };

            let transport = HttpTransport::new(base_url)
                .map_err(|e| format!("backend '{}': transport error: {e:?}", entry.name))?;

            Ok(Box::new(OpenAiCompatProvider::new(transport, kind)))
        }
        "gemini" => Ok(Box::new(build_gemini_provider(entry)?)),
        other => Err(format!("backend '{}': unknown type '{other}'", entry.name)),
    }
}

/// Construct a [`GeminiProvider`] from a `gemini` [`BackendEntry`].
///
/// Shared by [`build_single_provider`] (for the ordinary fallback chain) and
/// [`build_model_routes`] (for the model-id routing table), so the api-key
/// validation and default-model logic live in exactly one place.
fn build_gemini_provider(entry: &BackendEntry) -> Result<GeminiProvider, String> {
    // Note: env vars are already interpolated by parse_config().
    let api_key = entry.api_key.as_deref().unwrap_or("");
    let model = entry.model.as_deref().unwrap_or("gemini-3.1-flash-lite");
    if api_key.is_empty() || api_key.contains("${") {
        return Err(format!("backend '{}': gemini requires api_key", entry.name));
    }
    GeminiProvider::new(api_key.to_string(), model.to_string())
        .map_err(|e| format!("backend '{}': {e:?}", entry.name))
}

/// Pick, for each `gemini` backend, the single entry that claims its model
/// id in the routing table.
///
/// Entries are sorted ascending by `priority` first, matching the precedence
/// convention [`build_provider_chain`] already uses (lowest priority number
/// wins). The sort is stable, so equal priorities keep their config order.
/// Walking the sorted list and keeping only the first entry seen for each
/// model id means that when two backends are configured to serve the exact
/// same model, the higher-precedence one deterministically claims it (lower
/// priority number, or the earlier config entry on a tie), and the later
/// duplicate is dropped here rather than left to whichever order a `HashMap`
/// insert happens to run in.
///
/// Dropping the duplicate is intentional: routed models are single-target
/// (see [`crate::provider::ModelRoutingProvider::infer`]), so a second backend
/// naming the same model id adds no failover for routed traffic. That is not
/// obvious from config, so each dropped entry is logged at warn.
fn select_model_route_entries(entries: &[BackendEntry]) -> Vec<&BackendEntry> {
    let mut gemini_entries: Vec<&BackendEntry> = entries
        .iter()
        .filter(|entry| entry.backend_type == "gemini")
        .collect();
    gemini_entries.sort_by_key(|entry| entry.priority);

    let mut claimed_models: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut selected = Vec::with_capacity(gemini_entries.len());
    for entry in gemini_entries {
        let model = entry.model.as_deref().unwrap_or("gemini-3.1-flash-lite");
        if claimed_models.insert(model) {
            selected.push(entry);
        } else {
            tracing::warn!(
                backend = %entry.name,
                model = %model,
                "backend excluded from model routing: a higher-precedence gemini backend already claims this model id (lower priority number, or the earlier config entry on a priority tie), and routed models are single-target, so this backend will not receive routed traffic for it"
            );
        }
    }
    selected
}

/// Build the model-id routing table used by
/// [`build_provider_with_model_routing`]: one entry per `gemini` backend,
/// keyed on its configured `model` (defaulting the same way
/// [`build_gemini_provider`] does when `model` is unset).
///
/// Factored out from `build_provider_with_model_routing` so tests can assert
/// on the routing keys directly, without downcasting through the `Arc<dyn
/// InferenceProvider>` the public function returns.
fn build_model_routes(
    entries: &[BackendEntry],
) -> Result<std::collections::HashMap<String, Arc<dyn InferenceProvider + Send + Sync>>, String> {
    let mut routes: std::collections::HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> =
        std::collections::HashMap::new();
    for entry in select_model_route_entries(entries) {
        let model = entry
            .model
            .clone()
            .unwrap_or_else(|| "gemini-3.1-flash-lite".to_string());
        let provider = build_gemini_provider(entry)?;
        routes.insert(model, Arc::new(provider));
    }
    Ok(routes)
}

/// Test-only view of [`select_model_route_entries`] that reports which
/// backend *name* claimed each model id, so a test can tell which of two
/// same-model backends won without downcasting the built provider.
#[cfg(test)]
fn model_route_winners_for_tests(
    entries: &[BackendEntry],
) -> std::collections::HashMap<String, String> {
    select_model_route_entries(entries)
        .into_iter()
        .map(|entry| {
            let model = entry
                .model
                .clone()
                .unwrap_or_else(|| "gemini-3.1-flash-lite".to_string());
            (model, entry.name.clone())
        })
        .collect()
}

/// Build a provider that dispatches on `request.model` to whichever backend
/// was configured to serve that exact model id, falling back to the same
/// fallback chain [`build_provider_chain`] builds for any model with no
/// explicit route.
///
/// This exists because `build_provider_chain` builds a *fallback* chain
/// (primary, then retry on the next entry after a retryable error) — it is
/// not a dispatcher, so a request lands on the highest-priority backend
/// regardless of which model id it names. Tiered inference (see
/// `crate::tiered_vlm::run_tiered`) swaps `request.model` between a
/// first pass and a second pass and needs that second-pass model id to reach
/// the backend that actually serves it, not just retry the same primary
/// backend that already ran the first pass.
///
/// Only `gemini` backends get an explicit route, keyed on their configured
/// `model` field. This is deliberately not a name-based heuristic: a model is
/// routable because some backend's config names it via `model`, not because
/// its id looks like a particular vendor's naming scheme.
///
/// When at least one gemini backend exists, its `GeminiProvider` is built
/// twice: once inside `default` (as part of the ordinary fallback chain) and
/// once more for the routing table. Neither `GeminiProvider::new` nor
/// `HttpTransport::new` perform any I/O at construction, so this duplication
/// costs a couple of allocations at startup, not a network round-trip, and it
/// keeps `default`'s fallback semantics — and `build_provider_chain` itself —
/// completely unchanged rather than threading one shared instance through two
/// different structures.
///
/// Returns `Err` under the same conditions as [`build_provider_chain`], plus
/// a gemini construction error surfaced from building the routing table.
pub fn build_provider_with_model_routing(
    entries: &[BackendEntry],
) -> Result<Arc<dyn InferenceProvider + Send + Sync>, String> {
    let default = build_provider_chain(entries)?;
    let routes = build_model_routes(entries)?;

    if routes.is_empty() {
        // No gemini backend is configured, so there is nothing to route to
        // beyond what `default` already does. Return it unwrapped: zero
        // extra indirection, identical behavior to `build_provider_chain`
        // today.
        return Ok(default);
    }

    Ok(Arc::new(ModelRoutingProvider::new(routes, default)))
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
    // The caller only reaches here with two or more providers (see the doc
    // above), so the reversed iterator always yields a first element.
    let mut iter = arcs.into_iter().rev();
    let mut chain: Arc<dyn InferenceProvider + Send + Sync> = iter
        .next()
        .expect("router chain is built from two or more providers");

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
            openai_kind: None,
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
            model: Some("gemini-3.1-flash-lite".into()),
            openai_kind: None,
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(
            result.is_ok(),
            "gemini with api_key should succeed, got: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().kind(), ProviderKind::Gemini);
    }

    #[test]
    fn build_chain_gemini_missing_api_key_returns_err() {
        let entry = BackendEntry {
            name: "gemini".into(),
            backend_type: "gemini".into(),
            base_url: None,
            api_key: None,
            model: Some("gemini-3.1-flash-lite".into()),
            openai_kind: None,
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_err(), "gemini without api_key should fail");
        let err = result.err().unwrap();
        assert!(
            err.contains("api_key"),
            "expected api_key error, got: {err}"
        );
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
            openai_kind: None,
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
            openai_kind: None,
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
            openai_kind: None,
            priority: 1,
        };
        let result = build_provider_chain(&[entry]);
        assert!(result.is_err(), "unresolved base_url should fail");
        let err = result.err().unwrap();
        assert!(
            err.contains("base_url"),
            "expected base_url error, got: {err}"
        );
    }

    #[test]
    fn build_chain_single_provider() {
        let entry = BackendEntry {
            name: "vllm".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://localhost:8000".into()),
            api_key: None,
            model: None,
            openai_kind: None,
            priority: 1,
        };
        let provider = build_provider_chain(&[entry]).expect("single provider");
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }

    #[test]
    fn openai_kind_overrides_the_name_guess() {
        // An explicit flavor wins even when the free-form name would guess wrong.
        let entry = BackendEntry {
            name: "prod-inference".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://localhost:8000".into()),
            api_key: None,
            model: None,
            openai_kind: Some("sglang".into()),
            priority: 1,
        };
        let provider = build_provider_chain(&[entry]).expect("single provider");
        assert_eq!(provider.kind(), ProviderKind::Sglang);
    }

    #[test]
    fn openai_kind_mlx_maps_to_mlx_provider_kind() {
        // mlx-vlm speaks the same protocol as vLLM/SGLang, so it is built as
        // an ordinary OpenAiCompatProvider; only the telemetry label differs.
        let entry = BackendEntry {
            name: "mlx".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://127.0.0.1:8080".into()),
            api_key: None,
            model: None,
            openai_kind: Some("mlx".into()),
            priority: 1,
        };
        let provider = build_provider_chain(&[entry]).expect("single provider");
        assert_eq!(provider.kind(), ProviderKind::Mlx);
    }

    #[test]
    fn openai_kind_unset_falls_back_to_name_guess() {
        // Existing configs that named their backend "sglang" keep their label.
        let entry = BackendEntry {
            name: "sglang".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://localhost:8001".into()),
            api_key: None,
            model: None,
            openai_kind: None,
            priority: 1,
        };
        let provider = build_provider_chain(&[entry]).expect("single provider");
        assert_eq!(provider.kind(), ProviderKind::Sglang);
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
                openai_kind: None,
                priority: 2,
            },
            BackendEntry {
                name: "vllm".into(),
                backend_type: "openai_compat".into(),
                base_url: Some("http://localhost:8000".into()),
                api_key: None,
                model: None,
                openai_kind: None,
                priority: 1,
            },
        ];
        let provider = build_provider_chain(&entries).expect("two providers");
        // Primary should be the vllm entry (priority = 1).
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }

    // ── build_provider_with_model_routing ─────────────────────────────────────

    #[test]
    fn model_routing_vllm_only_has_no_routes_and_returns_the_plain_chain() {
        let entries = vec![BackendEntry {
            name: "vllm".into(),
            backend_type: "openai_compat".into(),
            base_url: Some("http://localhost:8000".into()),
            api_key: None,
            model: None,
            openai_kind: None,
            priority: 1,
        }];

        let routes = build_model_routes(&entries).expect("route build");
        assert!(
            routes.is_empty(),
            "no gemini backend means no routing entries"
        );

        let provider = build_provider_with_model_routing(&entries).expect("provider build");
        // With no gemini backend, build_provider_with_model_routing returns
        // build_provider_chain's result directly (no ModelRoutingProvider
        // wrapper), so its kind matches the plain chain exactly.
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }

    #[test]
    fn model_routing_vllm_plus_gemini_routes_the_configured_model_id() {
        let entries = vec![
            BackendEntry {
                name: "vllm".into(),
                backend_type: "openai_compat".into(),
                base_url: Some("http://localhost:8000".into()),
                api_key: None,
                model: None,
                openai_kind: None,
                priority: 1,
            },
            BackendEntry {
                name: "gemini".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: Some("test-api-key".into()),
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 10,
            },
        ];

        let routes = build_model_routes(&entries).expect("route build");
        assert_eq!(routes.len(), 1);
        assert!(
            routes.contains_key("gemini-3.1-flash-lite"),
            "route key must be the exact configured model id, not a name guess"
        );

        let provider = build_provider_with_model_routing(&entries).expect("provider build");
        // The fallback default still picks vllm as primary (lowest priority);
        // wrapping in ModelRoutingProvider does not disturb that.
        assert_eq!(provider.kind(), ProviderKind::Vllm);
    }

    #[test]
    fn model_routing_gemini_without_explicit_model_uses_default_route_key() {
        let entries = vec![BackendEntry {
            name: "gemini".into(),
            backend_type: "gemini".into(),
            base_url: None,
            api_key: Some("test-api-key".into()),
            model: None,
            openai_kind: None,
            priority: 1,
        }];

        let routes = build_model_routes(&entries).expect("route build");
        assert!(routes.contains_key("gemini-3.1-flash-lite"));
    }

    #[test]
    fn model_routing_duplicate_model_id_lowest_priority_number_wins() {
        // Two gemini backends configured to serve the exact same model id.
        // Before the priority sort, both entries raced to insert into the
        // same HashMap key and whichever ran last in (nondeterministic)
        // iteration order overwrote the other. Priority 1 must win every time.
        let entries = vec![
            BackendEntry {
                name: "gemini-primary".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: Some("key-primary".into()),
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 1,
            },
            BackendEntry {
                name: "gemini-secondary".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: Some("key-secondary".into()),
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 2,
            },
        ];

        let winners = model_route_winners_for_tests(&entries);
        assert_eq!(
            winners.get("gemini-3.1-flash-lite").map(String::as_str),
            Some("gemini-primary"),
            "the priority-1 backend must claim the duplicated model id"
        );

        // Also confirm the routing table this feeds actually collapses to one
        // entry rather than building (and discarding) both providers.
        let routes = build_model_routes(&entries).expect("route build");
        assert_eq!(routes.len(), 1, "duplicate model id must yield one route");
    }

    #[test]
    fn model_routing_duplicate_model_id_is_deterministic_regardless_of_input_order() {
        // Same two backends as above, but listed in the opposite order. The
        // winner must still be the lower priority number, not whichever entry
        // happened to appear first in the input slice.
        let entries = vec![
            BackendEntry {
                name: "gemini-secondary".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: Some("key-secondary".into()),
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 2,
            },
            BackendEntry {
                name: "gemini-primary".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: Some("key-primary".into()),
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 1,
            },
        ];

        let winners = model_route_winners_for_tests(&entries);
        assert_eq!(
            winners.get("gemini-3.1-flash-lite").map(String::as_str),
            Some("gemini-primary"),
        );
    }

    #[test]
    fn model_routing_propagates_gemini_construction_errors() {
        let entries = vec![
            BackendEntry {
                name: "vllm".into(),
                backend_type: "openai_compat".into(),
                base_url: Some("http://localhost:8000".into()),
                api_key: None,
                model: None,
                openai_kind: None,
                priority: 1,
            },
            BackendEntry {
                name: "gemini".into(),
                backend_type: "gemini".into(),
                base_url: None,
                api_key: None,
                model: Some("gemini-3.1-flash-lite".into()),
                openai_kind: None,
                priority: 10,
            },
        ];

        let result = build_provider_with_model_routing(&entries);
        assert!(
            result.is_err(),
            "missing api_key on the gemini entry should fail construction"
        );
    }
}

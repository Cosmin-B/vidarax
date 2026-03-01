use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::config::ServerConfig;
use crate::state::AppState;

const HEADER_API_KEY: &str = "x-api-key";
const HEADER_TENANT_ID: &str = "x-tenant-id";

#[derive(Clone)]
pub struct SecurityPolicy {
    require_api_key: bool,
    api_keys: Arc<Vec<String>>,
    require_tenant_id: bool,
    global_limiter: Option<Arc<FixedWindowLimiter>>,
    tenant_limiter: Option<Arc<TenantWindowLimiter>>,
    metrics_require_api_key: bool,
    cors_allowed_origins: Arc<Vec<String>>,
}

impl SecurityPolicy {
    pub fn from_config(config: &ServerConfig) -> Result<Self, String> {
        if config.security_require_api_key && config.security_api_keys.is_empty() {
            return Err(
                "VIDARAX_REQUIRE_API_KEY=true requires at least one VIDARAX_API_KEYS value"
                    .to_string(),
            );
        }
        if config.security_tenant_rps.is_some() && config.security_tenant_slots == 0 {
            return Err("VIDARAX_RATE_LIMIT_TENANT_SLOTS must be >= 1".to_string());
        }

        Ok(Self {
            require_api_key: config.security_require_api_key,
            api_keys: Arc::new(config.security_api_keys.clone()),
            require_tenant_id: config.security_require_tenant_id,
            global_limiter: config
                .security_global_rps
                .map(|limit| Arc::new(FixedWindowLimiter::new(limit))),
            tenant_limiter: config.security_tenant_rps.map(|limit| {
                Arc::new(TenantWindowLimiter::new(
                    limit,
                    config.security_tenant_slots,
                ))
            }),
            metrics_require_api_key: config.security_metrics_require_api_key,
            cors_allowed_origins: Arc::new(normalize_cors_origins(&config.cors_allowed_origins)),
        })
    }

    pub fn from_config_for_tests() -> Self {
        Self {
            require_api_key: false,
            api_keys: Arc::new(Vec::new()),
            require_tenant_id: false,
            global_limiter: None,
            tenant_limiter: None,
            metrics_require_api_key: false,
            cors_allowed_origins: Arc::new(Vec::new()),
        }
    }

    #[cfg(test)]
    pub fn from_test_policy(
        require_api_key: bool,
        api_keys: Vec<String>,
        require_tenant_id: bool,
        metrics_require_api_key: bool,
        cors_allowed_origins: Vec<String>,
    ) -> Self {
        Self {
            require_api_key,
            api_keys: Arc::new(api_keys),
            require_tenant_id,
            global_limiter: None,
            tenant_limiter: None,
            metrics_require_api_key,
            cors_allowed_origins: Arc::new(normalize_cors_origins(&cors_allowed_origins)),
        }
    }

    pub fn allows_open_access(&self) -> bool {
        !self.require_api_key
            && !self.require_tenant_id
            && self.global_limiter.is_none()
            && self.tenant_limiter.is_none()
    }

    fn metrics_requires_api_key(&self) -> bool {
        self.metrics_require_api_key
    }

    fn api_key_matches(&self, api_key: &str) -> bool {
        self.api_keys
            .iter()
            .any(|k| constant_time_eq(k.as_bytes(), api_key.as_bytes()))
    }

    fn has_api_keys(&self) -> bool {
        !self.api_keys.is_empty()
    }

    fn is_origin_allowed(&self, origin: &str) -> bool {
        if self.cors_allowed_origins.is_empty() {
            return false;
        }
        if self.cors_allowed_origins.iter().any(|value| value == "*") {
            return true;
        }
        self.cors_allowed_origins
            .iter()
            .any(|value| value == origin)
    }

    fn cors_origin_header(&self, origin: Option<&str>) -> Option<(HeaderValue, bool)> {
        let origin = origin?;
        if self.cors_allowed_origins.iter().any(|value| value == "*") {
            return Some((HeaderValue::from_static("*"), false));
        }
        if !self.is_origin_allowed(origin) {
            return None;
        }
        HeaderValue::from_str(origin)
            .ok()
            .map(|value| (value, true))
    }
}

pub async fn enforce_security(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let policy = state.security_policy();
    let origin = header_value(&request, header::ORIGIN.as_str()).map(ToString::to_string);

    if is_cors_preflight(&request) {
        let response = preflight_response(policy, origin.as_deref(), state.next_request_id());
        return finalize_response(policy, origin.as_deref(), response);
    }

    if request.uri().path() == "/v1/health" {
        return finalize_response(policy, origin.as_deref(), next.run(request).await);
    }

    if request.uri().path() == "/v1/metrics" && policy.metrics_requires_api_key() {
        let request_id = state.next_request_id();
        if !policy.has_api_keys() {
            let response = error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "metrics_unavailable",
                "metrics authentication is enabled but no api keys are configured",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
        let Some(api_key) = header_value(&request, HEADER_API_KEY) else {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing x-api-key header",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        };
        if !policy.api_key_matches(api_key) {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "invalid api key",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
        return finalize_response(policy, origin.as_deref(), next.run(request).await);
    }

    if policy.allows_open_access() {
        return finalize_response(policy, origin.as_deref(), next.run(request).await);
    }

    let request_id = state.next_request_id();
    if policy.require_api_key {
        let Some(api_key) = header_value(&request, HEADER_API_KEY) else {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing x-api-key header",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        };
        if !policy.api_key_matches(api_key) {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "invalid api key",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
    }

    let tenant_id = header_value(&request, HEADER_TENANT_ID).map(ToString::to_string);
    if policy.require_tenant_id && tenant_id.is_none() {
        let response = error_response(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing x-tenant-id header",
            request_id,
        );
        return finalize_response(policy, origin.as_deref(), response);
    }

    let now_sec = epoch_seconds();
    if let Some(global) = &policy.global_limiter {
        if !global.allow(now_sec) {
            let response = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "global request rate exceeded",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
    }

    if let Some(tenant_limiter) = &policy.tenant_limiter {
        let Some(tenant_id) = tenant_id.as_deref() else {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "x-tenant-id header is required when tenant rate limiting is enabled",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        };
        if !tenant_limiter.allow(tenant_id, now_sec) {
            let response = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "tenant request rate exceeded",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
    }

    finalize_response(policy, origin.as_deref(), next.run(request).await)
}

fn is_cors_preflight(request: &Request<Body>) -> bool {
    request.method() == Method::OPTIONS
        && request.headers().contains_key(header::ORIGIN)
        && request
            .headers()
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD)
}

fn preflight_response(
    policy: &SecurityPolicy,
    origin: Option<&str>,
    request_id: String,
) -> Response {
    let Some(origin) = origin else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "validation_error",
            "missing origin header",
            request_id,
        );
    };

    if !policy.is_origin_allowed(origin) {
        return error_response(
            StatusCode::FORBIDDEN,
            "cors_forbidden",
            "origin is not allowed by cors policy",
            request_id,
        );
    }

    let mut response = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .expect("building preflight response should not fail");
    add_preflight_headers(policy, &mut response, Some(origin));
    response
}

fn add_preflight_headers(policy: &SecurityPolicy, response: &mut Response, origin: Option<&str>) {
    add_cors_origin_header(policy, response.headers_mut(), origin);
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET,POST,PATCH,DELETE,OPTIONS"),
    );
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type,x-api-key,x-tenant-id"),
    );
    response.headers_mut().insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static("600"),
    );
}

fn add_cors_origin_header(policy: &SecurityPolicy, headers: &mut HeaderMap, origin: Option<&str>) {
    let Some((header_value, vary_origin)) = policy.cors_origin_header(origin) else {
        return;
    };
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, header_value);
    if vary_origin {
        headers.insert(header::VARY, HeaderValue::from_static("origin"));
    }
}

fn finalize_response(
    policy: &SecurityPolicy,
    origin: Option<&str>,
    mut response: Response,
) -> Response {
    apply_security_headers(response.headers_mut());
    add_cors_origin_header(policy, response.headers_mut(), origin);
    response
}

fn apply_security_headers(headers: &mut HeaderMap) {
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static("default-src 'none'; frame-ancestors 'none'"),
    );
    headers.insert(
        HeaderName::from_static("strict-transport-security"),
        HeaderValue::from_static("max-age=63072000; includeSubDomains"),
    );
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
}

fn normalize_cors_origins(origins: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for origin in origins {
        let trimmed = origin.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !normalized.iter().any(|existing| existing == trimmed) {
            normalized.push(trimmed.to_string());
        }
    }
    normalized
}

fn header_value<'a>(request: &'a Request<Body>, name: &str) -> Option<&'a str> {
    request.headers().get(name).and_then(|v| v.to_str().ok())
}

fn error_response(status: StatusCode, code: &str, message: &str, request_id: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "details": []
            }
        })),
    )
        .into_response()
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

struct FixedWindowLimiter {
    limit_per_sec: u64,
    state: AtomicU64,
}

impl FixedWindowLimiter {
    fn new(limit_per_sec: u64) -> Self {
        Self {
            limit_per_sec: limit_per_sec.max(1),
            state: AtomicU64::new(0),
        }
    }

    fn allow(&self, now_sec: u64) -> bool {
        let window = pack_window(now_sec, 0);
        let limit = self.limit_per_sec.min(u32::MAX as u64) as u32;
        loop {
            let observed = self.state.load(Ordering::Acquire);
            let observed_window = unpack_window(observed);
            let observed_count = unpack_count(observed);
            let (next, allowed) = if observed_window != unpack_window(window) {
                (pack_window(now_sec, 1), true)
            } else if observed_count >= limit {
                (observed, false)
            } else {
                (pack_window(now_sec, observed_count + 1), true)
            };
            if self
                .state
                .compare_exchange(observed, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                if !allowed {
                    return false;
                }
                return unpack_count(next) <= limit;
            }
        }
    }
}

struct TenantWindowLimiter {
    limit_per_sec: u64,
    slots: Vec<TenantSlot>,
}

impl TenantWindowLimiter {
    fn new(limit_per_sec: u64, slot_count: usize) -> Self {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(TenantSlot::default());
        }
        Self {
            limit_per_sec: limit_per_sec.max(1),
            slots,
        }
    }

    fn allow(&self, tenant_id: &str, now_sec: u64) -> bool {
        let target = stable_hash(tenant_id);
        let now_window = bounded_window(now_sec);
        let mut stale_candidate: Option<(usize, u64)> = None;

        for probe in 0..self.slots.len() {
            let idx = ((target as usize).wrapping_add(probe)) % self.slots.len();
            let slot = &self.slots[idx];
            let key = slot.hash.load(Ordering::Acquire);
            if key == target {
                return slot.allow(now_sec, self.limit_per_sec);
            }
            if key == 0
                && slot
                    .hash
                    .compare_exchange(0, target, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                slot.state.store(pack_window(now_sec, 0), Ordering::Release);
                return slot.allow(now_sec, self.limit_per_sec);
            }

            if stale_candidate.is_none() {
                let observed_window = unpack_window(slot.state.load(Ordering::Acquire));
                if observed_window != now_window {
                    stale_candidate = Some((idx, key));
                }
            }
        }

        if let Some((idx, key)) = stale_candidate {
            let slot = &self.slots[idx];
            if slot
                .hash
                .compare_exchange(key, target, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                slot.state.store(pack_window(now_sec, 0), Ordering::Release);
                return slot.allow(now_sec, self.limit_per_sec);
            }
            // If replacement lost the race, fail closed instead of reusing a slot whose state may
            // still correspond to a different tenant window.
        }

        false
    }
}

#[derive(Default)]
struct TenantSlot {
    hash: AtomicU64,
    state: AtomicU64,
}

impl TenantSlot {
    fn allow(&self, now_sec: u64, limit_per_sec: u64) -> bool {
        let limit = limit_per_sec.min(u32::MAX as u64) as u32;
        loop {
            let observed = self.state.load(Ordering::Acquire);
            let observed_window = unpack_window(observed);
            let observed_count = unpack_count(observed);
            let (next, allowed) = if observed_window != bounded_window(now_sec) {
                (pack_window(now_sec, 1), true)
            } else if observed_count >= limit {
                (observed, false)
            } else {
                (pack_window(now_sec, observed_count + 1), true)
            };
            if self
                .state
                .compare_exchange(observed, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                if !allowed {
                    return false;
                }
                return unpack_count(next) <= limit;
            }
        }
    }
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for b in value.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash | 1
}

#[inline]
fn pack_window(now_sec: u64, count: u32) -> u64 {
    ((bounded_window(now_sec) as u64) << 32) | count as u64
}

#[inline]
fn bounded_window(now_sec: u64) -> u32 {
    now_sec.min(u32::MAX as u64) as u32
}

#[inline]
fn unpack_window(state: u64) -> u32 {
    (state >> 32) as u32
}

#[inline]
fn unpack_count(state: u64) -> u32 {
    (state & 0xffff_ffff) as u32
}

fn constant_time_eq(lhs: &[u8], rhs: &[u8]) -> bool {
    let mut diff = lhs.len() ^ rhs.len();
    let max = lhs.len().max(rhs.len());
    for i in 0..max {
        let a = lhs.get(i).copied().unwrap_or(0);
        let b = rhs.get(i).copied().unwrap_or(0);
        diff |= (a ^ b) as usize;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::{constant_time_eq, SecurityPolicy, TenantWindowLimiter};

    #[test]
    fn tenant_window_isolated_by_tenant_id() {
        let limiter = TenantWindowLimiter::new(1, 8);
        let now = 100;
        assert!(limiter.allow("tenant-a", now));
        assert!(!limiter.allow("tenant-a", now));
        assert!(limiter.allow("tenant-b", now));
    }

    #[test]
    fn tenant_window_reuses_stale_slot_after_window_rollover() {
        let limiter = TenantWindowLimiter::new(1, 1);
        assert!(limiter.allow("tenant-a", 10));
        assert!(!limiter.allow("tenant-b", 10));
        assert!(limiter.allow("tenant-b", 11));
    }

    #[test]
    fn cors_policy_honors_exact_allowlist() {
        let policy = SecurityPolicy::from_test_policy(
            false,
            vec![],
            false,
            false,
            vec!["https://app.example.com".to_string()],
        );
        assert!(policy.is_origin_allowed("https://app.example.com"));
        assert!(!policy.is_origin_allowed("https://evil.example.com"));
    }

    #[test]
    fn constant_time_key_compare_matches_expected() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"abc", b"abd"));
    }
}

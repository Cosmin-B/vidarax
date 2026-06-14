use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use governor::{DefaultDirectRateLimiter, DefaultKeyedRateLimiter, Quota, RateLimiter};
use serde_json::json;

use crate::auth::{HEADER_API_KEY, HEADER_TENANT_ID};
use crate::config::ServerConfig;
use crate::state::AppState;

const TENANT_LIMITER_RETAIN_EVERY_REQUESTS: u64 = 64;
const IDLE_TENANT_RETENTION: Duration = Duration::from_secs(1);

#[derive(Clone)]
pub struct SecurityPolicy {
    require_api_key: bool,
    api_keys: Arc<Vec<String>>,
    require_tenant_id: bool,
    global_limiter: Option<Arc<GlobalRateLimiter>>,
    tenant_limiter: Option<Arc<TenantRateLimiter>>,
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
                .map(|limit| Arc::new(GlobalRateLimiter::new(limit))),
            tenant_limiter: config
                .security_tenant_rps
                .map(|limit| {
                    Arc::new(TenantRateLimiter::new(
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

    // Only the API key check is skipped for health/preflight, not the rate limit.
    if let Some(global) = &policy.global_limiter {
        if !global.allow() {
            let request_id = state.next_request_id();
            let response = error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "global request rate exceeded",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        }
    }

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

    // Note: global rate limiting is enforced at the top of this function
    // (before the preflight/health bypass) so it is not duplicated here.

    if let Some(tenant_limiter) = &policy.tenant_limiter {
        let Some(tenant_id) = tenant_id.as_ref() else {
            let response = error_response(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "x-tenant-id header is required when tenant rate limiting is enabled",
                request_id,
            );
            return finalize_response(policy, origin.as_deref(), response);
        };
        if !tenant_limiter.allow(tenant_id) {
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
        HeaderValue::from_static("content-type,x-api-key,x-tenant-id,x-attach-config"),
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

struct GlobalRateLimiter {
    limiter: DefaultDirectRateLimiter,
}

impl GlobalRateLimiter {
    fn new(limit_per_sec: u64) -> Self {
        Self {
            limiter: RateLimiter::direct(quota_per_second(limit_per_sec)),
        }
    }

    fn allow(&self) -> bool {
        self.limiter.check().is_ok()
    }
}

struct TenantRateLimiter {
    limiter: DefaultKeyedRateLimiter<String>,
    max_tenants: usize,
    retain_counter: AtomicU64,
    tenants: Mutex<HashMap<String, Instant>>,
}

impl TenantRateLimiter {
    fn new(limit_per_sec: u64, max_tenants: usize) -> Self {
        Self {
            limiter: RateLimiter::keyed(quota_per_second(limit_per_sec)),
            max_tenants: max_tenants.max(1),
            retain_counter: AtomicU64::new(0),
            tenants: Mutex::new(HashMap::new()),
        }
    }

    fn allow(&self, tenant_id: &String) -> bool {
        let now = Instant::now();
        if self.retain_counter.fetch_add(1, Ordering::Relaxed)
            % TENANT_LIMITER_RETAIN_EVERY_REQUESTS
            == 0
        {
            self.retain_recent(now);
        }

        if !self.admit_tenant(tenant_id, now) {
            self.retain_recent(now);
            if !self.admit_tenant(tenant_id, now) {
                return false;
            }
        }

        self.limiter.check_key(tenant_id).is_ok()
    }

    fn admit_tenant(&self, tenant_id: &String, now: Instant) -> bool {
        // Admission needs an exact cap across arbitrary tenant IDs. This mutex is
        // outside the video hot path and guards only a small in-memory index; the
        // governor limiter still handles the per-tenant rate accounting.
        let Ok(mut tenants) = self.tenants.lock() else {
            return false;
        };

        if let Some(last_seen) = tenants.get_mut(tenant_id) {
            *last_seen = now;
            return true;
        }

        if tenants.len() >= self.max_tenants {
            return false;
        }

        tenants.insert(tenant_id.clone(), now);
        true
    }

    fn retain_recent(&self, now: Instant) {
        self.limiter.retain_recent();
        self.limiter.shrink_to_fit();
        if let Ok(mut tenants) = self.tenants.lock() {
            tenants.retain(|_, last_seen| {
                now.duration_since(*last_seen) < IDLE_TENANT_RETENTION
            });
            tenants.shrink_to_fit();
        }
    }

    #[cfg(test)]
    fn tracked_tenants(&self) -> usize {
        self.tenants.lock().map(|tenants| tenants.len()).unwrap_or(0)
    }
}

fn quota_per_second(limit_per_sec: u64) -> Quota {
    let limit = limit_per_sec.max(1).min(u32::MAX as u64) as u32;
    Quota::per_second(NonZeroU32::new(limit).expect("limit is clamped to >= 1"))
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
    use super::{constant_time_eq, GlobalRateLimiter, SecurityPolicy, TenantRateLimiter};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    };
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn governor_global_limiter_enforces_burst_limit() {
        let limiter = GlobalRateLimiter::new(2);
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(!limiter.allow());
    }

    #[test]
    fn keyed_governor_limiter_isolated_by_tenant_id() {
        let limiter = TenantRateLimiter::new(1, 8);
        let tenant_a = "tenant-a".to_string();
        let tenant_b = "tenant-b".to_string();
        assert!(limiter.allow(&tenant_a));
        assert!(!limiter.allow(&tenant_a));
        assert!(limiter.allow(&tenant_b));
    }

    #[test]
    fn keyed_governor_limiter_honors_tenant_slot_cap() {
        let limiter = TenantRateLimiter::new(100, 2);
        let tenant_a = "tenant-a".to_string();
        let tenant_b = "tenant-b".to_string();
        let tenant_c = "tenant-c".to_string();

        assert!(limiter.allow(&tenant_a));
        assert!(limiter.allow(&tenant_b));
        assert!(!limiter.allow(&tenant_c));
        assert!(limiter.allow(&tenant_a));
        assert_eq!(limiter.tracked_tenants(), 2);
    }

    #[test]
    fn keyed_governor_limiter_evicts_idle_tenants() {
        let limiter = TenantRateLimiter::new(100, 1);
        let tenant_a = "tenant-a".to_string();
        let tenant_b = "tenant-b".to_string();

        assert!(limiter.allow(&tenant_a));
        assert!(!limiter.allow(&tenant_b));
        // Sleep well past IDLE_TENANT_RETENTION (1s) so the eviction is not
        // racing the wall clock on a loaded CI machine.
        thread::sleep(IDLE_TENANT_RETENTION + Duration::from_millis(500));
        assert!(limiter.allow(&tenant_b));
        assert_eq!(limiter.tracked_tenants(), 1);
    }

    #[test]
    fn keyed_governor_limiter_does_not_bypass_at_second_boundary() {
        const THREADS: usize = 64;
        const LIMIT: u64 = 8;

        let limiter = Arc::new(TenantRateLimiter::new(LIMIT, 8));
        let tenant = "tenant-boundary".to_string();
        for _ in 0..LIMIT {
            assert!(limiter.allow(&tenant));
        }
        assert!(!limiter.allow(&tenant));

        wait_for_epoch_second_rollover();

        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let allowed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let limiter = Arc::clone(&limiter);
            let barrier = Arc::clone(&barrier);
            let allowed = Arc::clone(&allowed);
            let tenant = tenant.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                if limiter.allow(&tenant) {
                    allowed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }

        let allowed = allowed.load(Ordering::Relaxed);
        assert!(
            allowed <= LIMIT as usize,
            "allowed {allowed} requests across second boundary with limit {LIMIT}"
        );

        let fresh = "tenant-fresh".to_string();
        for _ in 0..LIMIT {
            assert!(limiter.allow(&fresh), "fresh tenant should receive its full quota");
        }
        assert!(!limiter.allow(&fresh));
    }

    fn wait_for_epoch_second_rollover() {
        let start = epoch_second_for_test();
        while epoch_second_for_test() == start {
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn epoch_second_for_test() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
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

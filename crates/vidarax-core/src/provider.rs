use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use base64::Engine as _;
use serde_json::Value;
use vidarax_contracts::models::normalize_model_id;

use crate::admission::{AdmissionRequest, InferenceAdmission, LatencyClass, LimitClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Vllm,
    Sglang,
    Gemini,
    /// mlx-vlm running on Apple Silicon. Speaks the same OpenAI-compatible
    /// protocol as vLLM/SGLang, so it reuses `OpenAiCompatProvider`; this
    /// variant only exists to keep its telemetry and tiering label distinct
    /// from the self-hosted GPU backends.
    Mlx,
}

impl ProviderKind {
    /// Returns a stable lowercase string name suitable for logging and display.
    pub fn name(self) -> &'static str {
        match self {
            ProviderKind::Vllm => "vllm",
            ProviderKind::Sglang => "sglang",
            ProviderKind::Gemini => "gemini",
            ProviderKind::Mlx => "mlx",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub model: Arc<str>,
    pub prompt: Arc<str>,
    pub input_images: Vec<InferenceImage>,
    /// Short video clips (e.g. MP4 segments) to include after image parts.
    pub input_videos: Vec<InferenceVideo>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_ms: u64,
    pub allow_fallback: bool,
    /// Optional JSON schema string for constrained decoding.
    /// Sent as `response_format.json_schema` for vLLM ≥0.15 compatibility.
    pub guided_json: Option<Arc<str>>,
    pub scheduling: InferenceScheduling,
}

impl InferenceRequest {
    /// Return the time still available to this logical request.
    ///
    /// Both the caller timeout and scheduler deadline bound every serial
    /// provider attempt. Routers and providers call this again before a
    /// fallback, upload, poll, or retry so those operations cannot each claim
    /// the original timeout independently.
    pub fn remaining_timeout_ms(&self) -> Result<u64, ProviderError> {
        let remaining = self
            .scheduling
            .deadline_at
            .saturating_duration_since(Instant::now());
        let remaining_ms = remaining.as_millis().min(u128::from(u64::MAX)) as u64;
        let timeout_ms = self.timeout_ms.min(remaining_ms);
        if timeout_ms == 0 {
            Err(ProviderError::DeadlineMissed)
        } else {
            Ok(timeout_ms)
        }
    }

    fn with_remaining_timeout(&self) -> Result<Self, ProviderError> {
        let mut request = self.clone();
        request.timeout_ms = self.remaining_timeout_ms()?;
        Ok(request)
    }
}

#[derive(Debug, Clone)]
pub struct InferenceScheduling {
    pub stream_id: Arc<str>,
    pub class: LatencyClass,
    pub deadline_ms: u64,
    pub estimated_service_ms: u64,
    pub deadline_at: Instant,
}

impl InferenceScheduling {
    pub fn new(
        stream_id: Arc<str>,
        class: LatencyClass,
        deadline_ms: u64,
        estimated_service_ms: u64,
    ) -> Self {
        Self {
            stream_id,
            class,
            deadline_ms,
            estimated_service_ms,
            deadline_at: Instant::now()
                .checked_add(Duration::from_millis(deadline_ms))
                .unwrap_or_else(Instant::now),
        }
    }
}

impl Default for InferenceScheduling {
    fn default() -> Self {
        Self::new(Arc::from("direct"), LatencyClass::Live, 30_000, 500)
    }
}

/// Structured label emitted by the teacher VLM.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TeacherLabel {
    pub event_type: String,
    pub confidence: f32,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

/// JSON schema string for constrained teacher-label decoding.
///
/// Pass this as `guided_json` in an [`InferenceRequest`] to force vLLM/SGLang
/// to emit a valid [`TeacherLabel`] object.
pub fn teacher_label_schema() -> &'static str {
    r#"{
  "type": "object",
  "properties": {
    "event_type":   { "type": "string" },
    "confidence":   { "type": "number", "minimum": 0, "maximum": 1 },
    "description":  { "type": "string" },
    "reasoning":    { "type": "string" }
  },
  "required": ["event_type", "confidence"]
}"#
}

/// Parse a [`TeacherLabel`] from raw VLM output text.
///
/// Returns `None` if the text is not valid JSON or does not match the schema.
pub fn parse_teacher_label(text: &str) -> Option<TeacherLabel> {
    serde_json::from_str(text).ok()
}

#[derive(Debug, Clone)]
pub struct InferenceImage {
    pub media_type: &'static str,
    pub data_base64: String,
}

#[derive(Debug, Clone)]
pub struct InferenceVideo {
    pub media_type: &'static str, // e.g. "video/mp4"
    /// Raw media used by binary-capable providers. Keeping this alongside the
    /// legacy representation lets video pipelines avoid a base64/JSON copy.
    pub raw_bytes: Option<Arc<[u8]>>,
    /// Legacy representation for providers that only accept data URLs.
    pub data_base64: String,
}

/// Provider-reported token usage for one inference. Zeroed when a provider does
/// not report usage. `thinking_tokens` is non-zero only for models that emit
/// hidden reasoning (e.g. Gemini `thoughtsTokenCount`); it is what lets the
/// pipeline detect thinking without hardcoding model names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub thinking_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    /// Fold another usage into this one, saturating on overflow. Used to sum
    /// token spend across multiple inference passes (e.g. tiered first+second
    /// pass) so the surfaced total reflects the whole analysis, not just the
    /// final call.
    pub fn accumulate(&mut self, other: TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(other.thinking_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub provider: ProviderKind,
    pub model: Arc<str>,
    pub output_text: String,
    pub fallback_used: bool,
    pub finish_reason: Option<String>,
    pub inference_latency_ms: u64,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    UnsupportedModel(String),
    HttpStatus(u16),
    Transport(String),
    InvalidResponse(std::borrow::Cow<'static, str>),
    Saturated {
        retry_after_ms: u64,
        blocked_by: LimitClass,
    },
    DeadlineMissed,
    RequestBudget,
}

impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::HttpStatus(code) => matches!(*code, 408 | 429 | 500..=599),
            ProviderError::Transport(_) => true,
            ProviderError::Saturated { .. } => true,
            ProviderError::UnsupportedModel(_)
            | ProviderError::InvalidResponse(_)
            | ProviderError::DeadlineMissed
            | ProviderError::RequestBudget => false,
        }
    }
}

pub struct AdmittedProvider {
    inner: Arc<dyn InferenceProvider + Send + Sync>,
    admission: Arc<InferenceAdmission>,
    principal: Box<str>,
}

impl AdmittedProvider {
    pub fn new(
        inner: Arc<dyn InferenceProvider + Send + Sync>,
        admission: Arc<InferenceAdmission>,
        principal: Box<str>,
    ) -> Self {
        Self {
            inner,
            admission,
            principal,
        }
    }
}

impl InferenceProvider for AdmittedProvider {
    fn kind(&self) -> ProviderKind {
        self.inner.kind()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let media_bytes = request
            .input_images
            .iter()
            .map(|image| image.data_base64.len() as u64)
            .chain(request.input_videos.iter().map(|video| {
                video
                    .raw_bytes
                    .as_ref()
                    .map_or(video.data_base64.len() as u64, |bytes| {
                        (bytes.len() as u64).saturating_add(2) / 3 * 4 + 64
                    })
            }))
            .fold(0_u64, u64::saturating_add);
        let queue_budget = request
            .scheduling
            .deadline_at
            .saturating_duration_since(Instant::now());
        if queue_budget.is_zero() {
            self.admission.record_deadline_missed();
            return Err(ProviderError::DeadlineMissed);
        }
        let _permit = self
            .admission
            .acquire_scheduled(AdmissionRequest {
                principal: &self.principal,
                stream: &request.scheduling.stream_id,
                class: request.scheduling.class,
                deadline: queue_budget,
                estimated_service: Duration::from_millis(request.scheduling.estimated_service_ms),
                tokens: self.inner.reserved_output_tokens(request),
                bytes: media_bytes,
            })
            .map_err(|error| match error {
                crate::admission::AdmissionError::DeadlineMissed => ProviderError::DeadlineMissed,
                crate::admission::AdmissionError::RequestBudget => ProviderError::RequestBudget,
                error => ProviderError::Saturated {
                    retry_after_ms: self
                        .admission
                        .limits()
                        .wait_timeout
                        .as_millis()
                        .min(u128::from(u64::MAX)) as u64,
                    blocked_by: match error {
                        crate::admission::AdmissionError::Timeout { blocked_by } => blocked_by,
                        crate::admission::AdmissionError::WaiterLimit => LimitClass::Global,
                        crate::admission::AdmissionError::DeadlineMissed
                        | crate::admission::AdmissionError::RequestBudget => unreachable!(),
                    },
                },
            })?;
        let dispatched = request.with_remaining_timeout().map_err(|error| {
            if error == ProviderError::DeadlineMissed {
                self.admission.record_deadline_missed();
            }
            error
        })?;
        self.inner.infer(&dispatched)
    }

    fn kind_for_model(&self, model: &str) -> ProviderKind {
        self.inner.kind_for_model(model)
    }

    fn available_kinds(&self) -> Vec<ProviderKind> {
        self.inner.available_kinds()
    }

    fn configured_kinds_for_model(&self, model: &str) -> Vec<ProviderKind> {
        self.inner.configured_kinds_for_model(model)
    }

    fn reserved_output_tokens(&self, request: &InferenceRequest) -> u64 {
        self.inner.reserved_output_tokens(request)
    }
}

pub trait Transport: Send + Sync {
    fn call(&self, endpoint: &str, body: String, timeout_ms: u64) -> Result<String, ProviderError>;

    /// Check whether the transport's backend is reachable without running inference.
    fn probe(&self, _timeout_ms: u64) -> Result<(), ProviderError> {
        Ok(())
    }
}

pub trait InferenceProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError>;

    /// Which provider kind actually serves `model`.
    ///
    /// A leaf provider serves everything itself, so the default is `kind()`.
    /// `ModelRoutingProvider` overrides this so an error recorded after a
    /// routed call lands on the backend that ran the model rather than on the
    /// default leaf. Callers that record a failure use this instead of `kind()`
    /// when they know which model failed.
    fn kind_for_model(&self, _model: &str) -> ProviderKind {
        self.kind()
    }

    /// Provider kinds that can currently accept work.
    ///
    /// Test and in-process providers are available by construction. Networked
    /// providers override this with a bounded transport probe.
    fn available_kinds(&self) -> Vec<ProviderKind> {
        vec![self.kind()]
    }

    /// Provider kinds configured to serve a public Vidarax model id.
    fn configured_kinds_for_model(&self, _model: &str) -> Vec<ProviderKind> {
        vec![self.kind()]
    }

    /// Maximum output-token capacity this logical call can consume while it
    /// holds an admission permit. Providers that may retry with additional
    /// output headroom must override this reservation.
    fn reserved_output_tokens(&self, request: &InferenceRequest) -> u64 {
        u64::from(request.max_tokens)
    }
}

/// Records inference outcomes for `/metrics`, one call per inference pass.
///
/// `vidarax-core` defines this interface rather than depending on the metrics
/// type itself, since `vidarax-api` (where the real recorder lives) depends on
/// `vidarax-core` and not the other way around. Tiered inference calls this
/// once per pass so each pass is attributed to the provider that actually
/// served it, instead of folding a cheap local pass and an expensive escalated
/// pass into one bucket.
pub trait InferenceObserver: Send + Sync {
    fn record_success(
        &self,
        provider: ProviderKind,
        latency_ms: u64,
        fallback_used: bool,
        usage: TokenUsage,
    );
    fn record_error(&self, provider: ProviderKind, latency_ms: u64);
}

#[derive(Clone)]
pub struct HttpTransport {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl HttpTransport {
    pub fn new(base_url: &str) -> Result<Self, ProviderError> {
        let client = reqwest::blocking::Client::builder()
            .pool_max_idle_per_host(8)
            .build()
            .map_err(|err| ProviderError::Transport(err.to_string()))?;

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        })
    }

    fn endpoint_url(&self, endpoint: &str) -> String {
        format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'))
    }
}

impl Transport for HttpTransport {
    fn call(&self, endpoint: &str, body: String, timeout_ms: u64) -> Result<String, ProviderError> {
        let url = self.endpoint_url(endpoint);
        let response = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .timeout(Duration::from_millis(timeout_ms.max(1)))
            .body(body)
            .send()
            .map_err(|err| ProviderError::Transport(err.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::HttpStatus(status.as_u16()));
        }

        response
            .text()
            .map_err(|err| ProviderError::Transport(err.to_string()))
    }

    fn probe(&self, timeout_ms: u64) -> Result<(), ProviderError> {
        let response = self
            .client
            .get(self.endpoint_url("/v1/models"))
            .timeout(Duration::from_millis(timeout_ms.max(1)))
            .send()
            .map_err(|err| ProviderError::Transport(err.to_string()))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(ProviderError::HttpStatus(response.status().as_u16()))
        }
    }
}

/// Unified OpenAI-compatible provider for vLLM, SGLang, and any other
/// backend that speaks the `/v1/chat/completions` API.
///
/// # Examples
///
/// ```
/// use vidarax_core::provider::{OpenAiCompatProvider, ProviderKind};
///
/// // Construct with a mock or HTTP transport:
/// // let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);
/// ```
pub struct OpenAiCompatProvider<T: Transport> {
    transport: T,
    kind: ProviderKind,
    served_model: Option<Arc<str>>,
    upstream_model: Option<Arc<str>>,
    model_cache: ArcSwap<Arc<str>>,
}

impl<T: Transport> OpenAiCompatProvider<T> {
    pub fn new(transport: T, kind: ProviderKind) -> Self {
        Self {
            transport,
            kind,
            served_model: None,
            upstream_model: None,
            model_cache: ArcSwap::from(Arc::new(Arc::from(""))),
        }
    }

    /// Override the model id sent to the OpenAI-compatible backend while
    /// retaining Vidarax's curated model id at its public API boundary.
    pub fn with_model_mapping(
        mut self,
        served_model: Option<String>,
        upstream_model: Option<String>,
    ) -> Self {
        self.served_model = served_model
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(Arc::from);
        self.upstream_model = upstream_model
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(Arc::from);
        self
    }
}

impl<T: Transport> InferenceProvider for OpenAiCompatProvider<T> {
    fn kind(&self) -> ProviderKind {
        self.kind
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let model = canonical_model(&request.model)?;
        if self
            .served_model
            .as_deref()
            .is_some_and(|served| served != model)
        {
            return Err(ProviderError::UnsupportedModel(request.model.to_string()));
        }
        let upstream_model = self.upstream_model.as_deref().unwrap_or(model);
        let body = build_payload(upstream_model, request);
        let t0 = Instant::now();
        let response = self
            .transport
            .call("/v1/chat/completions", body, request.timeout_ms)?;
        let inference_latency_ms = t0.elapsed().as_millis() as u64;
        let (output_text, finish_reason, usage) = parse_completion(&response)?;

        Ok(InferenceResult {
            provider: self.kind,
            model: cached_arc_str(&self.model_cache, model),
            output_text,
            fallback_used: false,
            finish_reason,
            inference_latency_ms,
            usage,
        })
    }

    fn available_kinds(&self) -> Vec<ProviderKind> {
        if self.transport.probe(1_000).is_ok() {
            vec![self.kind]
        } else {
            Vec::new()
        }
    }

    fn configured_kinds_for_model(&self, model: &str) -> Vec<ProviderKind> {
        if self
            .served_model
            .as_deref()
            .is_none_or(|served| served == model)
        {
            vec![self.kind]
        } else {
            Vec::new()
        }
    }
}

pub struct ProviderRouter<P: InferenceProvider, F: InferenceProvider> {
    primary: P,
    fallback: F,
}

impl<P: InferenceProvider, F: InferenceProvider> ProviderRouter<P, F> {
    pub fn new(primary: P, fallback: F) -> Self {
        Self { primary, fallback }
    }

    pub fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        self.infer_routed(request)
    }

    fn infer_routed(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let primary_request = request.with_remaining_timeout()?;
        match self.primary.infer(&primary_request) {
            Ok(result) => Ok(result),
            Err(err) if request.allow_fallback && err.is_retryable() => {
                let fallback_request = request.with_remaining_timeout()?;
                let mut fallback_result = self.fallback.infer(&fallback_request)?;
                fallback_result.fallback_used = true;
                Ok(fallback_result)
            }
            Err(err) => Err(err),
        }
    }
}

/// Allow `ProviderRouter` to be passed as `Arc<dyn InferenceProvider>` to worker pools.
impl<P: InferenceProvider, F: InferenceProvider> InferenceProvider for ProviderRouter<P, F> {
    fn kind(&self) -> ProviderKind {
        self.primary.kind()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        self.infer_routed(request)
    }

    fn available_kinds(&self) -> Vec<ProviderKind> {
        let mut kinds = self.primary.available_kinds();
        for kind in self.fallback.available_kinds() {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
        kinds
    }

    fn configured_kinds_for_model(&self, model: &str) -> Vec<ProviderKind> {
        let mut kinds = self.primary.configured_kinds_for_model(model);
        for kind in self.fallback.configured_kinds_for_model(model) {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
        kinds
    }

    fn reserved_output_tokens(&self, request: &InferenceRequest) -> u64 {
        let primary = self.primary.reserved_output_tokens(request);
        if request.allow_fallback {
            primary.max(self.fallback.reserved_output_tokens(request))
        } else {
            primary
        }
    }
}

/// Forward trait calls through an `Arc<dyn InferenceProvider + Send + Sync>`.
///
/// This allows passing the return value of [`crate::backends::build_provider_chain`]
/// directly to generic functions that accept `Arc<I>` where `I: InferenceProvider`.
impl InferenceProvider for Arc<dyn InferenceProvider + Send + Sync> {
    fn kind(&self) -> ProviderKind {
        (**self).kind()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        (**self).infer(request)
    }

    // Must be forwarded, not left to the trait default. Production wires the
    // provider as this exact `Arc<dyn ...>`, so a bare `provider.kind_for_model()`
    // resolves to this impl. Without this line it would fall through to the
    // default (`self.kind()`) and report the router's default backend, undoing
    // the whole point of kind_for_model. `(**self)` reaches the inner trait
    // object so a router's override actually runs.
    fn kind_for_model(&self, model: &str) -> ProviderKind {
        (**self).kind_for_model(model)
    }

    fn available_kinds(&self) -> Vec<ProviderKind> {
        (**self).available_kinds()
    }

    fn configured_kinds_for_model(&self, model: &str) -> Vec<ProviderKind> {
        (**self).configured_kinds_for_model(model)
    }

    fn reserved_output_tokens(&self, request: &InferenceRequest) -> u64 {
        (**self).reserved_output_tokens(request)
    }
}

/// Forward `request` to `provider`, which holds pre-built transports.
///
/// The provider must be created once (e.g. via [`crate::backends::build_provider_chain`])
/// and reused across calls so that TCP connection pools are shared rather
/// than rebuilt on every inference.
pub fn infer_with_endpoints(
    provider: &dyn InferenceProvider,
    request: &InferenceRequest,
) -> Result<InferenceResult, ProviderError> {
    provider.infer(request)
}

/// Dispatches an inference request to whichever backend was configured to
/// serve `request.model`, falling back to `default` for any model id with no
/// explicit route.
///
/// [`ProviderRouter`] answers a different question: given one request, which
/// backend should try it first and which backend catches a retryable
/// failure. This type answers "which backend actually speaks this model id",
/// which is what tiered inference needs when it swaps `request.model` between
/// a first pass and a second pass — the second pass has to land on the
/// backend that serves *that* model, not just retry the same backend that
/// already ran the first pass.
///
/// Routing keys are exact, explicitly configured model ids (see
/// [`crate::backends::build_provider_with_model_routing`]). There is no
/// name-substring or prefix matching here: a model is routable because some
/// backend's config names it, not because its id looks a certain way.
pub struct ModelRoutingProvider {
    routes: std::collections::HashMap<String, Arc<dyn InferenceProvider + Send + Sync>>,
    default: Arc<dyn InferenceProvider + Send + Sync>,
}

impl ModelRoutingProvider {
    pub fn new(
        routes: std::collections::HashMap<String, Arc<dyn InferenceProvider + Send + Sync>>,
        default: Arc<dyn InferenceProvider + Send + Sync>,
    ) -> Self {
        Self { routes, default }
    }

    /// Model ids with an explicit route, sorted for stable diagnostics and
    /// test assertions.
    pub fn route_models(&self) -> Vec<&str> {
        let mut models: Vec<&str> = self.routes.keys().map(String::as_str).collect();
        models.sort_unstable();
        models
    }
}

impl InferenceProvider for ModelRoutingProvider {
    fn kind(&self) -> ProviderKind {
        self.default.kind()
    }

    fn kind_for_model(&self, model: &str) -> ProviderKind {
        // Mirror the routing `infer` does: a mapped model is served by its
        // route's backend, everything else by the default. Reporting the wrong
        // one here is exactly the bug this override fixes, so the two lookups
        // must stay in lockstep.
        self.routes
            .get(model)
            .map(|p| p.kind())
            .unwrap_or_else(|| self.default.kind())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        // A HashMap lookup costs nanoseconds; the call it guards is a network
        // round-trip to a model that takes hundreds of milliseconds. This is
        // the same reasoning `build_provider_chain`'s doc comment gives for
        // why virtual dispatch through `Arc<dyn InferenceProvider>` is fine
        // here: it is the per-inference path, not the per-frame hot path, so
        // the lookup is nowhere close to what bounds latency.
        //
        // Contract: for a model it has a route for, this router never falls
        // through to `default`. It forwards the request unchanged to the routed
        // provider and returns whatever that provider returns, error included.
        // `request.allow_fallback` is passed through untouched, so whether it
        // does anything is the routed provider's call; the router itself never
        // uses it to try `default` or another route.
        //
        // Routes built by `build_provider_with_model_routing` are single Gemini
        // leaves (`select_model_route_entries` keeps one backend per model id
        // and logs any dropped duplicate), so those routes are single-target by
        // construction: on failure the caller gets that backend's error, never
        // a silent hop to a backend that was not pinned for that model. An
        // unrouted model id still falls through to `default`, which honors
        // `allow_fallback` as usual.
        self.routes
            .get(request.model.as_ref())
            .unwrap_or(&self.default)
            .infer(request)
    }

    fn available_kinds(&self) -> Vec<ProviderKind> {
        let mut kinds = self.default.available_kinds();
        for provider in self.routes.values() {
            for kind in provider.available_kinds() {
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
        }
        kinds
    }

    fn configured_kinds_for_model(&self, model: &str) -> Vec<ProviderKind> {
        self.routes
            .get(model)
            .unwrap_or(&self.default)
            .configured_kinds_for_model(model)
    }

    fn reserved_output_tokens(&self, request: &InferenceRequest) -> u64 {
        self.routes
            .get(request.model.as_ref())
            .unwrap_or(&self.default)
            .reserved_output_tokens(request)
    }
}

fn canonical_model(model: &str) -> Result<&'static str, ProviderError> {
    normalize_model_id(model).ok_or_else(|| ProviderError::UnsupportedModel(model.to_string()))
}

pub(crate) fn new_arc_str_cache() -> ArcSwap<Arc<str>> {
    ArcSwap::from(Arc::new(Arc::from("")))
}

pub(crate) fn cached_arc_str(cache: &ArcSwap<Arc<str>>, value: &str) -> Arc<str> {
    let cached = cache.load_full();
    if cached.as_ref().as_ref() == value {
        return Arc::clone(&*cached);
    }

    let updated: Arc<str> = Arc::from(value);
    cache.store(Arc::new(Arc::clone(&updated)));
    updated
}

fn build_payload(model: &str, request: &InferenceRequest) -> String {
    let has_media = !request.input_images.is_empty() || !request.input_videos.is_empty();
    let user_content = if !has_media {
        Value::String(request.prompt.to_string())
    } else {
        let mut content =
            Vec::with_capacity(1 + request.input_images.len() + request.input_videos.len());
        content.push(serde_json::json!({
            "type": "text",
            "text": &*request.prompt
        }));
        for image in &request.input_images {
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.media_type, image.data_base64)
                }
            }));
        }
        for video in &request.input_videos {
            let data_base64 = if video.data_base64.is_empty() {
                video
                    .raw_bytes
                    .as_ref()
                    .map(|bytes| {
                        Cow::Owned(base64::engine::general_purpose::STANDARD.encode(bytes))
                    })
                    .unwrap_or_default()
            } else {
                Cow::Borrowed(video.data_base64.as_str())
            };
            content.push(serde_json::json!({
                "type": "video_url",
                "video_url": {
                    "url": format!("data:{};base64,{}", video.media_type, data_base64)
                }
            }));
        }
        Value::Array(content)
    };
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    });
    if let Some(schema) = &request.guided_json {
        // vLLM ≥0.15 requires response_format (not extra_body.guided_json)
        let parsed = match serde_json::from_str::<Value>(schema) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "guided_json schema is invalid JSON, sending unconstrained");
                Value::Null
            }
        };
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "teacher_label",
                "schema": parsed
            }
        });
    }
    // Disable thinking/reasoning tokens for Qwen3.5 models.
    // This prevents chain-of-thought noise in the output and
    // significantly reduces latency on MoE models.
    if model.contains("Qwen3.5") {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    }
    body.to_string()
}

#[derive(serde::Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

/// OpenAI-compatible `usage` block (vLLM/SGLang). Absent on servers that don't
/// report it, in which case token counts stay zero.
#[derive(serde::Deserialize)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

#[derive(serde::Deserialize)]
struct CompletionChoice {
    finish_reason: Option<String>,
    message: Option<CompletionMessage>,
    text: Option<Value>,
}

#[derive(serde::Deserialize)]
struct CompletionMessage {
    content: Value,
}

/// Returns `(output_text, finish_reason, usage)`. `usage` is zeroed when the
/// server does not report a `usage` block.
fn parse_completion(raw: &str) -> Result<(String, Option<String>, TokenUsage), ProviderError> {
    let resp: CompletionResponse = serde_json::from_str(raw).map_err(|e| {
        ProviderError::InvalidResponse(format!("invalid json response: {e}").into())
    })?;

    let usage = resp
        .usage
        .map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            thinking_tokens: 0,
            total_tokens: u.total_tokens,
        })
        .unwrap_or_default();

    let first = resp
        .choices
        .into_iter()
        .next()
        .ok_or(ProviderError::InvalidResponse(
            "choices array is empty".into(),
        ))?;

    let finish_reason = first.finish_reason;

    let content =
        first
            .message
            .map(|m| m.content)
            .or(first.text)
            .ok_or(ProviderError::InvalidResponse(
                "missing choices[0].message.content".into(),
            ))?;

    parse_content_value(content).map(|text| (text, finish_reason, usage))
}

fn parse_content_value(value: Value) -> Result<String, ProviderError> {
    match value {
        Value::String(s) => Ok(s),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    Value::String(s) => out.push_str(&s),
                    Value::Object(map) => {
                        if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                            out.push_str(text);
                        }
                    }
                    _ => {}
                }
            }

            if out.is_empty() {
                Err(ProviderError::InvalidResponse(
                    "content array does not contain text".into(),
                ))
            } else {
                Ok(out)
            }
        }
        _ => Err(ProviderError::InvalidResponse(
            "unsupported content shape".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_payload, infer_with_endpoints, AdmittedProvider, HttpTransport, InferenceImage,
        InferenceProvider, InferenceRequest, InferenceResult, InferenceScheduling, InferenceVideo,
        ModelRoutingProvider, OpenAiCompatProvider, ProviderError, ProviderKind, ProviderRouter,
        TokenUsage, Transport,
    };
    use crate::admission::{AdmissionLimits, InferenceAdmission, LatencyClass, LimitClass};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    struct MockTransport {
        calls: AtomicUsize,
        response: Result<String, ProviderError>,
    }

    impl MockTransport {
        fn ok(payload: &str) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                response: Ok(payload.to_string()),
            }
        }

        fn err(error: ProviderError) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                response: Err(error),
            }
        }
    }

    impl Transport for MockTransport {
        fn call(
            &self,
            _endpoint: &str,
            _body: String,
            _timeout_ms: u64,
        ) -> Result<String, ProviderError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.response.clone()
        }
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            model: Arc::from("openbmb/MiniCPM-V-4.5"),
            prompt: Arc::from("hello"),
            input_images: Vec::new(),
            input_videos: Vec::new(),
            max_tokens: 64,
            temperature: 0.0,
            timeout_ms: 500,
            allow_fallback: true,
            guided_json: None,
            scheduling: Default::default(),
        }
    }

    fn completion_json(text: &str) -> String {
        format!(
            "{{\"id\":\"cmpl\",\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{text}\"}}}}]}}"
        )
    }

    #[test]
    fn normalizes_model_alias_before_call() {
        let provider = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("ok")),
            ProviderKind::Vllm,
        );
        let result = provider.infer(&request()).expect("inference");
        assert_eq!(&*result.model, "openbmb/MiniCPM-V-4_5");
        assert_eq!(result.output_text, "ok");
    }

    #[test]
    fn repeated_same_model_results_reuse_cached_arc() {
        let provider = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("ok")),
            ProviderKind::Vllm,
        );

        let first = provider.infer(&request()).expect("first inference");
        let second = provider.infer(&request()).expect("second inference");

        assert!(Arc::ptr_eq(&first.model, &second.model));
    }

    #[test]
    fn uses_fallback_on_retryable_error() {
        let primary = OpenAiCompatProvider::new(
            MockTransport::err(ProviderError::HttpStatus(503)),
            ProviderKind::Vllm,
        );
        let fallback = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("fallback")),
            ProviderKind::Sglang,
        );
        let router = ProviderRouter::new(primary, fallback);

        let result = router.infer(&request()).expect("fallback");
        assert!(result.fallback_used);
        assert_eq!(result.output_text, "fallback");
    }

    #[test]
    fn uses_fallback_on_http_timeout_statuses() {
        for code in [408, 504] {
            let primary = OpenAiCompatProvider::new(
                MockTransport::err(ProviderError::HttpStatus(code)),
                ProviderKind::Vllm,
            );
            let fallback = OpenAiCompatProvider::new(
                MockTransport::ok(&completion_json("fallback")),
                ProviderKind::Sglang,
            );
            let router = ProviderRouter::new(primary, fallback);

            let result = router.infer(&request()).expect("fallback");
            assert!(result.fallback_used, "status {code} should be retryable");
            assert_eq!(result.output_text, "fallback");
        }
    }

    #[test]
    fn does_not_fallback_on_non_retryable_4xx() {
        let primary = OpenAiCompatProvider::new(
            MockTransport::err(ProviderError::HttpStatus(400)),
            ProviderKind::Vllm,
        );
        let fallback = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("fallback")),
            ProviderKind::Sglang,
        );
        let router = ProviderRouter::new(primary, fallback);

        let err = router.infer(&request()).unwrap_err();
        assert_eq!(err, ProviderError::HttpStatus(400));
    }

    #[test]
    fn uses_fallback_on_transport_error() {
        let primary = OpenAiCompatProvider::new(
            MockTransport::err(ProviderError::Transport("connection reset".to_string())),
            ProviderKind::Vllm,
        );
        let fallback = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("fallback")),
            ProviderKind::Sglang,
        );
        let router = ProviderRouter::new(primary, fallback);

        let result = router.infer(&request()).expect("fallback");
        assert!(result.fallback_used);
        assert_eq!(result.output_text, "fallback");
    }

    #[test]
    fn rejects_unsupported_model() {
        let provider = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("ok")),
            ProviderKind::Vllm,
        );
        let mut req = request();
        req.model = Arc::from("unknown/model");
        let err = provider.infer(&req).unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedModel(_)));
    }

    #[test]
    fn parses_array_content_shape() {
        let provider = OpenAiCompatProvider::new(MockTransport::ok(
            "{\"choices\":[{\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"hello\"},{\"type\":\"text\",\"text\":\" world\"}]}}]}",
        ), ProviderKind::Vllm);
        let result = provider.infer(&request()).unwrap();
        assert_eq!(result.output_text, "hello world");
    }

    #[test]
    fn http_transport_roundtrip_and_router() {
        // Hold the crate-wide env lock for the duration of the real HTTP call.
        // `HttpTransport`'s reqwest client honors ambient proxy env vars, and
        // the proxy-env tests in `ingest::fetch` set HTTP_PROXY globally while
        // they hold this same lock. Without taking it here, one of those tests
        // could set a proxy mid-call and reroute this request away from the
        // local server, failing it. Serializing against them removes the race.
        let _env_guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // The lock stops a concurrent proxy-env test from hijacking this call,
        // but HttpTransport also captures reqwest's proxy config when its client
        // is built, so an HTTP_PROXY already exported into the environment (a CI
        // host, a contributor behind a corporate proxy) could still reroute this
        // request off the local listener. Bypass proxies for loopback while we
        // hold the lock. Restored below; if an assertion panics the test has
        // already failed, and a leaked NO_PROXY=127.0.0.1 only bypasses proxies
        // for loopback, which the proxy-env tests set explicitly anyway.
        let prev_no_proxy = std::env::var_os("NO_PROXY");
        let prev_no_proxy_lower = std::env::var_os("no_proxy");
        std::env::set_var("NO_PROXY", "127.0.0.1");
        std::env::set_var("no_proxy", "127.0.0.1");

        let body = completion_json("from-server");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let base = format!("http://{addr}");
        let router = ProviderRouter::new(
            OpenAiCompatProvider::new(HttpTransport::new(&base).unwrap(), ProviderKind::Vllm),
            OpenAiCompatProvider::new(HttpTransport::new(&base).unwrap(), ProviderKind::Sglang),
        );
        let result = infer_with_endpoints(&router, &request()).unwrap();
        assert_eq!(result.output_text, "from-server");
        assert_eq!(result.provider, ProviderKind::Vllm);
        server.join().unwrap();

        match prev_no_proxy {
            Some(v) => std::env::set_var("NO_PROXY", v),
            None => std::env::remove_var("NO_PROXY"),
        }
        match prev_no_proxy_lower {
            Some(v) => std::env::set_var("no_proxy", v),
            None => std::env::remove_var("no_proxy"),
        }
    }

    #[test]
    fn payload_includes_multimodal_content_when_images_exist() {
        let mut req = request();
        req.input_images = vec![InferenceImage {
            media_type: "image/jpeg",
            data_base64: "YWJj".to_string(),
        }];
        let body = build_payload("openbmb/MiniCPM-V-4_5", &req);
        let value: Value = serde_json::from_str(&body).unwrap();
        let content = value["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert_eq!(content[1]["type"].as_str(), Some("image_url"));
        assert_eq!(
            content[1]["image_url"]["url"].as_str(),
            Some("data:image/jpeg;base64,YWJj")
        );
    }

    #[test]
    fn payload_includes_video_content_parts_after_images() {
        let mut req = request();
        req.input_images = vec![InferenceImage {
            media_type: "image/jpeg",
            data_base64: "aW1n".to_string(),
        }];
        req.input_videos = vec![InferenceVideo {
            media_type: "video/mp4",
            raw_bytes: None,
            data_base64: "dmlk".to_string(),
        }];
        let body = build_payload("openbmb/MiniCPM-V-4_5", &req);
        let value: Value = serde_json::from_str(&body).unwrap();
        let content = value["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert_eq!(content[1]["type"].as_str(), Some("image_url"));
        assert_eq!(
            content[1]["image_url"]["url"].as_str(),
            Some("data:image/jpeg;base64,aW1n")
        );
        assert_eq!(content[2]["type"].as_str(), Some("video_url"));
        assert_eq!(
            content[2]["video_url"]["url"].as_str(),
            Some("data:video/mp4;base64,dmlk")
        );
    }

    #[test]
    fn payload_videos_only_produces_array_content() {
        let mut req = request();
        req.input_videos = vec![InferenceVideo {
            media_type: "video/mp4",
            raw_bytes: None,
            data_base64: "dmlk".to_string(),
        }];
        let body = build_payload("openbmb/MiniCPM-V-4_5", &req);
        let value: Value = serde_json::from_str(&body).unwrap();
        let content = value["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert_eq!(content[1]["type"].as_str(), Some("video_url"));
        assert_eq!(
            content[1]["video_url"]["url"].as_str(),
            Some("data:video/mp4;base64,dmlk")
        );
    }

    #[test]
    fn payload_lazily_encodes_raw_video_for_openai_compatibility() {
        let mut req = request();
        req.input_videos = vec![InferenceVideo {
            media_type: "video/mp4",
            raw_bytes: Some(Arc::from(&b"vid"[..])),
            data_base64: String::new(),
        }];

        let body = build_payload("openbmb/MiniCPM-V-4_5", &req);
        let value: Value = serde_json::from_str(&body).unwrap();
        let content = value["messages"][0]["content"].as_array().unwrap();
        assert_eq!(
            content[1]["video_url"]["url"].as_str(),
            Some("data:video/mp4;base64,dmlk")
        );
    }

    #[test]
    fn payload_includes_guided_json_when_set() {
        let mut req = request();
        req.guided_json = Some(Arc::from(
            r#"{"type":"object","properties":{"event_type":{"type":"string"}}}"#,
        ));
        let body = build_payload("openbmb/MiniCPM-V-4_5", &req);
        let value: Value = serde_json::from_str(&body).unwrap();
        let schema = &value["response_format"]["json_schema"]["schema"];
        assert_eq!(schema["type"].as_str(), Some("object"));
    }

    #[test]
    fn parses_finish_reason_from_response() {
        let json = r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#;
        let provider = OpenAiCompatProvider::new(MockTransport::ok(json), ProviderKind::Vllm);
        let result = provider.infer(&request()).unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        assert_eq!(result.output_text, "done");
    }

    #[test]
    fn finish_reason_is_none_when_absent() {
        let provider = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("ok")),
            ProviderKind::Vllm,
        );
        let result = provider.infer(&request()).unwrap();
        assert_eq!(result.finish_reason, None);
    }

    #[test]
    fn inference_latency_ms_is_non_negative() {
        let provider = OpenAiCompatProvider::new(
            MockTransport::ok(&completion_json("ok")),
            ProviderKind::Vllm,
        );
        let result = provider.infer(&request()).unwrap();
        // MockTransport returns instantly; just verify the field is present and >= 0.
        let _ = result.inference_latency_ms; // u64, always >= 0
    }

    // ── ModelRoutingProvider ─────────────────────────────────────────────────

    /// Records the model id of every request it receives instead of talking
    /// to a real transport, so tests can assert which provider a request
    /// landed on.
    struct RecordingModelProvider {
        kind: ProviderKind,
        seen_models: Mutex<Vec<String>>,
        seen_timeouts: Mutex<Vec<u64>>,
    }

    impl RecordingModelProvider {
        fn new(kind: ProviderKind) -> Self {
            Self {
                kind,
                seen_models: Mutex::new(Vec::new()),
                seen_timeouts: Mutex::new(Vec::new()),
            }
        }
    }

    impl InferenceProvider for RecordingModelProvider {
        fn kind(&self) -> ProviderKind {
            self.kind
        }

        fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
            self.seen_models
                .lock()
                .unwrap()
                .push(request.model.to_string());
            self.seen_timeouts.lock().unwrap().push(request.timeout_ms);
            Ok(InferenceResult {
                provider: self.kind,
                model: Arc::clone(&request.model),
                output_text: "ok".to_string(),
                fallback_used: false,
                finish_reason: None,
                inference_latency_ms: 0,
                usage: TokenUsage::default(),
            })
        }
    }

    #[test]
    fn model_routing_dispatches_to_mapped_provider_for_configured_model_id() {
        let routed = Arc::new(RecordingModelProvider::new(ProviderKind::Gemini));
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert("gemini-3.1-flash-lite".to_string(), routed.clone());
        let router = ModelRoutingProvider::new(routes, default.clone());

        let mut req = request();
        req.model = Arc::from("gemini-3.1-flash-lite");
        router.infer(&req).expect("routed inference");

        assert_eq!(
            routed.seen_models.lock().unwrap().as_slice(),
            ["gemini-3.1-flash-lite"]
        );
        assert!(default.seen_models.lock().unwrap().is_empty());
    }

    #[test]
    fn model_routing_falls_back_to_default_for_unmapped_model_id() {
        let routed = Arc::new(RecordingModelProvider::new(ProviderKind::Gemini));
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert("gemini-3.1-flash-lite".to_string(), routed.clone());
        let router = ModelRoutingProvider::new(routes, default.clone());

        let mut req = request();
        req.model = Arc::from("some/other-model");
        router.infer(&req).expect("default inference");

        assert!(routed.seen_models.lock().unwrap().is_empty());
        assert_eq!(
            default.seen_models.lock().unwrap().as_slice(),
            ["some/other-model"]
        );
    }

    #[test]
    fn routed_model_failure_is_single_target_and_never_touches_default() {
        // Contract: a routed model runs only on the backend it is pinned to.
        // When that backend errors, the router surfaces the error instead of
        // retrying the default chain, even with allow_fallback set. Falling
        // through could send the request to a backend that was not pinned for
        // that model.
        struct Failing;
        impl InferenceProvider for Failing {
            fn kind(&self) -> ProviderKind {
                ProviderKind::Gemini
            }
            fn infer(&self, _request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
                Err(ProviderError::HttpStatus(429))
            }
        }

        let routed = Arc::new(Failing);
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert("gemini-3.1-flash-lite".to_string(), routed);
        let router = ModelRoutingProvider::new(routes, default.clone());

        let mut req = request();
        req.model = Arc::from("gemini-3.1-flash-lite");
        req.allow_fallback = true; // must not spill a routed call to the default

        let err = router
            .infer(&req)
            .expect_err("routed failure should surface");
        assert_eq!(err, ProviderError::HttpStatus(429));
        assert!(
            default.seen_models.lock().unwrap().is_empty(),
            "the default backend must never be consulted for a routed model"
        );
    }

    #[test]
    fn model_routing_route_models_lists_configured_keys_sorted() {
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert(
            "zeta-model".to_string(),
            Arc::new(RecordingModelProvider::new(ProviderKind::Gemini)) as _,
        );
        routes.insert(
            "alpha-model".to_string(),
            Arc::new(RecordingModelProvider::new(ProviderKind::Gemini)) as _,
        );
        let router = ModelRoutingProvider::new(routes, default);

        assert_eq!(router.route_models(), vec!["alpha-model", "zeta-model"]);
    }

    #[test]
    fn model_routing_kind_reflects_default_provider() {
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Sglang));
        let router = ModelRoutingProvider::new(HashMap::new(), default);
        assert_eq!(router.kind(), ProviderKind::Sglang);
    }

    #[test]
    fn model_routing_kind_for_model_reports_the_serving_backend() {
        // kind_for_model must track infer's routing so a recorded error lands
        // on the backend that ran the model. A routed id reports its route's
        // kind; an unmapped id reports the default's, same as the router's
        // bare kind().
        let routed = Arc::new(RecordingModelProvider::new(ProviderKind::Gemini));
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert("gemini-3.1-flash-lite".to_string(), routed);
        let router = ModelRoutingProvider::new(routes, default);

        assert_eq!(
            router.kind_for_model("gemini-3.1-flash-lite"),
            ProviderKind::Gemini
        );
        assert_eq!(
            router.kind_for_model("some/local-model"),
            ProviderKind::Vllm
        );
        // The bare kind() still reflects the default, unchanged.
        assert_eq!(router.kind(), ProviderKind::Vllm);
    }

    #[test]
    fn kind_for_model_dispatches_through_arc_dyn_wrapper() {
        // Regression: production wires the router as Arc<dyn InferenceProvider +
        // Send + Sync> and calls kind_for_model on that Arc (clip.rs, workers.rs,
        // semantic_infer.rs). The Arc forwarding impl must reach the router's
        // override rather than fall through to the default kind, which would
        // report the default backend and silently defeat the fix.
        let routed = Arc::new(RecordingModelProvider::new(ProviderKind::Gemini));
        let default = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let mut routes: HashMap<String, Arc<dyn InferenceProvider + Send + Sync>> = HashMap::new();
        routes.insert("gemini-3.1-flash-lite".to_string(), routed);
        let router: Arc<dyn InferenceProvider + Send + Sync> =
            Arc::new(ModelRoutingProvider::new(routes, default));

        assert_eq!(
            router.kind_for_model("gemini-3.1-flash-lite"),
            ProviderKind::Gemini
        );
        assert_eq!(
            router.kind_for_model("some/local-model"),
            ProviderKind::Vllm
        );

        // WebRTC's worker generic can leave the provider as an Arc wrapping an
        // Arc<dyn>. The forwarding must survive that nesting too.
        let nested: Arc<dyn InferenceProvider + Send + Sync> = Arc::new(Arc::clone(&router));
        assert_eq!(
            nested.kind_for_model("gemini-3.1-flash-lite"),
            ProviderKind::Gemini
        );
        assert_eq!(
            nested.kind_for_model("some/local-model"),
            ProviderKind::Vllm
        );
    }

    #[test]
    fn leaf_provider_kind_for_model_is_just_its_kind() {
        // The trait default: a leaf serves everything itself, so every model
        // maps to its own kind regardless of the id.
        let leaf = RecordingModelProvider::new(ProviderKind::Sglang);
        assert_eq!(leaf.kind_for_model("anything"), ProviderKind::Sglang);
        assert_eq!(
            leaf.kind_for_model("gemini-3.1-flash-lite"),
            ProviderKind::Sglang
        );
    }

    #[test]
    fn admitted_provider_bounds_calls_and_releases_capacity() {
        let admission = Arc::new(
            InferenceAdmission::new(AdmissionLimits {
                global_in_flight: 1,
                per_principal_in_flight: 1,
                global_waiters: 1,
                wait_timeout: Duration::from_millis(5),
                max_in_flight_tokens: 1_000_000,
                max_in_flight_bytes: 1024 * 1024 * 1024,
            })
            .unwrap(),
        );
        let inner: Arc<dyn InferenceProvider + Send + Sync> =
            Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let provider = AdmittedProvider::new(inner, Arc::clone(&admission), "tenant-secret".into());
        let held = admission.acquire("tenant-secret").unwrap();

        assert_eq!(
            provider.infer(&request()).unwrap_err(),
            ProviderError::Saturated {
                retry_after_ms: 5,
                blocked_by: LimitClass::Principal,
            }
        );
        drop(held);
        provider.infer(&request()).expect("capacity was released");
        assert_eq!(admission.snapshot().active, 0);
    }

    #[test]
    fn admitted_provider_clamps_transport_timeout_to_absolute_deadline() {
        let admission = Arc::new(
            InferenceAdmission::new(AdmissionLimits {
                global_in_flight: 1,
                per_principal_in_flight: 1,
                global_waiters: 1,
                wait_timeout: Duration::from_secs(1),
                max_in_flight_tokens: 1_000,
                max_in_flight_bytes: 1_000,
            })
            .unwrap(),
        );
        let inner = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let provider = AdmittedProvider::new(
            Arc::clone(&inner) as Arc<dyn InferenceProvider + Send + Sync>,
            admission,
            "tenant".into(),
        );
        let mut req = request();
        req.timeout_ms = 5_000;
        req.scheduling =
            InferenceScheduling::new(Arc::from("camera-1"), LatencyClass::Live, 100, 1);

        provider.infer(&req).unwrap();
        let timeout = inner.seen_timeouts.lock().unwrap()[0];
        assert!((1..=100).contains(&timeout));
    }

    #[test]
    fn admitted_provider_rejects_expired_and_oversized_raw_video() {
        let admission = Arc::new(
            InferenceAdmission::new(AdmissionLimits {
                global_in_flight: 1,
                per_principal_in_flight: 1,
                global_waiters: 1,
                wait_timeout: Duration::from_secs(1),
                max_in_flight_tokens: 1_000,
                max_in_flight_bytes: 70,
            })
            .unwrap(),
        );
        let inner = Arc::new(RecordingModelProvider::new(ProviderKind::Vllm));
        let provider = AdmittedProvider::new(
            Arc::clone(&inner) as Arc<dyn InferenceProvider + Send + Sync>,
            admission,
            "tenant".into(),
        );
        let mut req = request();
        req.input_videos.push(InferenceVideo {
            media_type: "video/mp4",
            raw_bytes: Some(Arc::from(&b"123456"[..])),
            data_base64: String::new(),
        });
        req.scheduling =
            InferenceScheduling::new(Arc::from("camera-1"), LatencyClass::Live, 100, 1);
        assert!(matches!(
            provider.infer(&req),
            Err(ProviderError::RequestBudget)
        ));

        req.input_videos.clear();
        req.scheduling.deadline_at = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(matches!(
            provider.infer(&req),
            Err(ProviderError::DeadlineMissed)
        ));
        assert!(inner.seen_models.lock().unwrap().is_empty());
    }
}

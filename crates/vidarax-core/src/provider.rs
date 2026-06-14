use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use vidarax_contracts::models::normalize_model_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Vllm,
    Sglang,
    Gemini,
}

impl ProviderKind {
    /// Returns a stable lowercase string name suitable for logging and display.
    pub fn name(self) -> &'static str {
        match self {
            ProviderKind::Vllm => "vllm",
            ProviderKind::Sglang => "sglang",
            ProviderKind::Gemini => "gemini",
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
    pub data_base64: String,
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub provider: ProviderKind,
    pub model: Arc<str>,
    pub output_text: String,
    pub fallback_used: bool,
    pub finish_reason: Option<String>,
    pub inference_latency_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    UnsupportedModel(String),
    HttpStatus(u16),
    Transport(String),
    InvalidResponse(std::borrow::Cow<'static, str>),
}

impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::HttpStatus(code) => matches!(*code, 408 | 429 | 500..=599),
            ProviderError::Transport(_) => true,
            ProviderError::UnsupportedModel(_) | ProviderError::InvalidResponse(_) => false,
        }
    }
}

pub trait Transport: Send + Sync {
    fn call(&self, endpoint: &str, body: String, timeout_ms: u64) -> Result<String, ProviderError>;
}

pub trait InferenceProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError>;
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
}

impl<T: Transport> OpenAiCompatProvider<T> {
    pub fn new(transport: T, kind: ProviderKind) -> Self {
        Self { transport, kind }
    }
}

impl<T: Transport> InferenceProvider for OpenAiCompatProvider<T> {
    fn kind(&self) -> ProviderKind {
        self.kind
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let model = canonical_model(&request.model)?;
        let body = build_payload(model, request);
        let t0 = Instant::now();
        let response = self
            .transport
            .call("/v1/chat/completions", body, request.timeout_ms)?;
        let inference_latency_ms = t0.elapsed().as_millis() as u64;
        let (output_text, finish_reason) = parse_completion(&response)?;

        Ok(InferenceResult {
            provider: self.kind,
            model: Arc::from(model),
            output_text,
            fallback_used: false,
            finish_reason,
            inference_latency_ms,
        })
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
        match self.primary.infer(request) {
            Ok(result) => Ok(result),
            Err(err) if request.allow_fallback && err.is_retryable() => {
                let mut fallback_result = self.fallback.infer(request)?;
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
        // Mirror inherent method logic — cannot call `self.infer()` here as that
        // would recurse into this trait impl.
        match self.primary.infer(request) {
            Ok(result) => Ok(result),
            Err(err) if request.allow_fallback && err.is_retryable() => {
                let mut r = self.fallback.infer(request)?;
                r.fallback_used = true;
                Ok(r)
            }
            Err(err) => Err(err),
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

fn canonical_model(model: &str) -> Result<&'static str, ProviderError> {
    normalize_model_id(model).ok_or_else(|| ProviderError::UnsupportedModel(model.to_string()))
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
            content.push(serde_json::json!({
                "type": "video_url",
                "video_url": {
                    "url": format!("data:{};base64,{}", video.media_type, video.data_base64)
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

/// Returns `(output_text, finish_reason)`.
fn parse_completion(raw: &str) -> Result<(String, Option<String>), ProviderError> {
    let resp: CompletionResponse = serde_json::from_str(raw)
        .map_err(|e| ProviderError::InvalidResponse(format!("invalid json response: {e}").into()))?;

    let first = resp
        .choices
        .into_iter()
        .next()
        .ok_or(ProviderError::InvalidResponse("choices array is empty".into()))?;

    let finish_reason = first.finish_reason;

    let content = first
        .message
        .map(|m| m.content)
        .or(first.text)
        .ok_or(ProviderError::InvalidResponse("missing choices[0].message.content".into()))?;

    parse_content_value(content).map(|text| (text, finish_reason))
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
                Err(ProviderError::InvalidResponse("content array does not contain text".into()))
            } else {
                Ok(out)
            }
        }
        _ => Err(ProviderError::InvalidResponse("unsupported content shape".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_payload, infer_with_endpoints, HttpTransport, InferenceImage, InferenceProvider,
        InferenceRequest, InferenceVideo, OpenAiCompatProvider, ProviderError, ProviderKind,
        ProviderRouter, Transport,
    };
    use serde_json::Value;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
    use std::thread;

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
        }
    }

    fn completion_json(text: &str) -> String {
        format!(
            "{{\"id\":\"cmpl\",\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{text}\"}}}}]}}"
        )
    }

    #[test]
    fn normalizes_model_alias_before_call() {
        let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
        let result = provider.infer(&request()).expect("inference");
        assert_eq!(&*result.model, "openbmb/MiniCPM-V-4_5");
        assert_eq!(result.output_text, "ok");
    }

    #[test]
    fn uses_fallback_on_retryable_error() {
        let primary = OpenAiCompatProvider::new(MockTransport::err(ProviderError::HttpStatus(503)), ProviderKind::Vllm);
        let fallback = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("fallback")), ProviderKind::Sglang);
        let router = ProviderRouter::new(primary, fallback);

        let result = router.infer(&request()).expect("fallback");
        assert!(result.fallback_used);
        assert_eq!(result.output_text, "fallback");
    }

    #[test]
    fn uses_fallback_on_http_timeout_statuses() {
        for code in [408, 504] {
            let primary = OpenAiCompatProvider::new(MockTransport::err(ProviderError::HttpStatus(code)), ProviderKind::Vllm);
            let fallback = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("fallback")), ProviderKind::Sglang);
            let router = ProviderRouter::new(primary, fallback);

            let result = router.infer(&request()).expect("fallback");
            assert!(result.fallback_used, "status {code} should be retryable");
            assert_eq!(result.output_text, "fallback");
        }
    }

    #[test]
    fn does_not_fallback_on_non_retryable_4xx() {
        let primary = OpenAiCompatProvider::new(MockTransport::err(ProviderError::HttpStatus(400)), ProviderKind::Vllm);
        let fallback = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("fallback")), ProviderKind::Sglang);
        let router = ProviderRouter::new(primary, fallback);

        let err = router.infer(&request()).unwrap_err();
        assert_eq!(err, ProviderError::HttpStatus(400));
    }

    #[test]
    fn uses_fallback_on_transport_error() {
        let primary = OpenAiCompatProvider::new(MockTransport::err(ProviderError::Transport("connection reset".to_string())), ProviderKind::Vllm);
        let fallback = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("fallback")), ProviderKind::Sglang);
        let router = ProviderRouter::new(primary, fallback);

        let result = router.infer(&request()).expect("fallback");
        assert!(result.fallback_used);
        assert_eq!(result.output_text, "fallback");
    }

    #[test]
    fn rejects_unsupported_model() {
        let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
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
    fn payload_includes_guided_json_when_set() {
        let mut req = request();
        req.guided_json = Some(Arc::from(r#"{"type":"object","properties":{"event_type":{"type":"string"}}}"#));
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
        let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
        let result = provider.infer(&request()).unwrap();
        assert_eq!(result.finish_reason, None);
    }

    #[test]
    fn inference_latency_ms_is_non_negative() {
        let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
        let result = provider.infer(&request()).unwrap();
        // MockTransport returns instantly; just verify the field is present and >= 0.
        let _ = result.inference_latency_ms; // u64, always >= 0
    }
}

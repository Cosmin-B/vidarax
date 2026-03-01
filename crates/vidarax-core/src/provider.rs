use std::time::Duration;

use serde_json::Value;
use vidarax_contracts::errors::classify_status_code;
use vidarax_contracts::models::normalize_model_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Vllm,
    Sglang,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    pub input_images: Vec<InferenceImage>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_ms: u64,
    pub allow_fallback: bool,
}

#[derive(Debug, Clone)]
pub struct InferenceImage {
    pub media_type: String,
    pub data_base64: String,
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub provider: ProviderKind,
    pub model: String,
    pub output_text: String,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    UnsupportedModel(String),
    HttpStatus(u16),
    Transport(String),
    InvalidResponse(String),
}

impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::HttpStatus(code) => classify_status_code(*code).is_retryable(),
            ProviderError::Transport(_) => true,
            ProviderError::UnsupportedModel(_) | ProviderError::InvalidResponse(_) => false,
        }
    }
}

pub trait Transport: Send + Sync {
    fn call(&self, endpoint: &str, body: &str, timeout_ms: u64) -> Result<String, ProviderError>;
}

pub trait InferenceProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError>;
}

#[derive(Debug, Clone)]
pub struct ProviderEndpoints {
    pub vllm_base_url: String,
    pub sglang_base_url: String,
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
    fn call(&self, endpoint: &str, body: &str, timeout_ms: u64) -> Result<String, ProviderError> {
        let url = self.endpoint_url(endpoint);
        let response = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .timeout(Duration::from_millis(timeout_ms.max(1)))
            .body(body.to_string())
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

pub struct VllmProvider<T: Transport> {
    transport: T,
}

pub struct SglangProvider<T: Transport> {
    transport: T,
}

impl<T: Transport> VllmProvider<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}

impl<T: Transport> SglangProvider<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}

impl<T: Transport> InferenceProvider for VllmProvider<T> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vllm
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let model = canonical_model(&request.model)?;
        let body = build_payload(model, request);
        let response = self
            .transport
            .call("/v1/chat/completions", &body, request.timeout_ms)?;
        let output_text = parse_completion_text(&response)?;

        Ok(InferenceResult {
            provider: ProviderKind::Vllm,
            model: model.to_string(),
            output_text,
            fallback_used: false,
        })
    }
}

impl<T: Transport> InferenceProvider for SglangProvider<T> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Sglang
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let model = canonical_model(&request.model)?;
        let body = build_payload(model, request);
        let response = self
            .transport
            .call("/v1/chat/completions", &body, request.timeout_ms)?;
        let output_text = parse_completion_text(&response)?;

        Ok(InferenceResult {
            provider: ProviderKind::Sglang,
            model: model.to_string(),
            output_text,
            fallback_used: false,
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

pub fn infer_with_endpoints(
    endpoints: &ProviderEndpoints,
    primary: ProviderKind,
    request: &InferenceRequest,
) -> Result<InferenceResult, ProviderError> {
    let vllm = VllmProvider::new(HttpTransport::new(&endpoints.vllm_base_url)?);
    let sglang = SglangProvider::new(HttpTransport::new(&endpoints.sglang_base_url)?);

    match primary {
        ProviderKind::Vllm => ProviderRouter::new(vllm, sglang).infer(request),
        ProviderKind::Sglang => ProviderRouter::new(sglang, vllm).infer(request),
    }
}

fn canonical_model(model: &str) -> Result<&'static str, ProviderError> {
    normalize_model_id(model).ok_or_else(|| ProviderError::UnsupportedModel(model.to_string()))
}

fn build_payload(model: &str, request: &InferenceRequest) -> String {
    let user_content = if request.input_images.is_empty() {
        Value::String(request.prompt.clone())
    } else {
        let mut content = Vec::with_capacity(request.input_images.len() + 1);
        content.push(serde_json::json!({
            "type": "text",
            "text": request.prompt
        }));
        for image in &request.input_images {
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.media_type, image.data_base64)
                }
            }));
        }
        Value::Array(content)
    };
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    })
    .to_string()
}

fn parse_completion_text(raw: &str) -> Result<String, ProviderError> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|err| ProviderError::InvalidResponse(format!("invalid json: {err}")))?;

    let choices = value
        .get("choices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::InvalidResponse("missing choices array".to_string()))?;
    let first = choices
        .first()
        .ok_or_else(|| ProviderError::InvalidResponse("choices array is empty".to_string()))?;

    if let Some(content) = first
        .get("message")
        .and_then(|v| v.get("content"))
        .or_else(|| first.get("text"))
    {
        return parse_content_value(content);
    }

    Err(ProviderError::InvalidResponse(
        "missing choices[0].message.content".to_string(),
    ))
}

fn parse_content_value(value: &Value) -> Result<String, ProviderError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    Value::String(s) => out.push_str(s),
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
                    "content array does not contain text".to_string(),
                ))
            } else {
                Ok(out)
            }
        }
        _ => Err(ProviderError::InvalidResponse(
            "unsupported content shape".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_payload, infer_with_endpoints, InferenceImage, InferenceProvider, InferenceRequest,
        ProviderEndpoints, ProviderError, ProviderKind, ProviderRouter, SglangProvider, Transport,
        VllmProvider,
    };
    use serde_json::Value;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
            _body: &str,
            _timeout_ms: u64,
        ) -> Result<String, ProviderError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.response.clone()
        }
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            model: "openbmb/MiniCPM-V-4.5".to_string(),
            prompt: "hello".to_string(),
            input_images: Vec::new(),
            max_tokens: 64,
            temperature: 0.0,
            timeout_ms: 500,
            allow_fallback: true,
        }
    }

    fn completion_json(text: &str) -> String {
        format!(
            "{{\"id\":\"cmpl\",\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{text}\"}}}}]}}"
        )
    }

    #[test]
    fn normalizes_model_alias_before_call() {
        let provider = VllmProvider::new(MockTransport::ok(&completion_json("ok")));
        let result = provider.infer(&request()).expect("inference");
        assert_eq!(result.model, "openbmb/MiniCPM-V-4_5");
        assert_eq!(result.output_text, "ok");
    }

    #[test]
    fn uses_fallback_on_retryable_error() {
        let primary = VllmProvider::new(MockTransport::err(ProviderError::HttpStatus(503)));
        let fallback = SglangProvider::new(MockTransport::ok(&completion_json("fallback")));
        let router = ProviderRouter::new(primary, fallback);

        let result = router.infer(&request()).expect("fallback");
        assert!(result.fallback_used);
        assert_eq!(result.output_text, "fallback");
    }

    #[test]
    fn rejects_unsupported_model() {
        let provider = VllmProvider::new(MockTransport::ok(&completion_json("ok")));
        let mut req = request();
        req.model = "unknown/model".to_string();
        let err = provider.infer(&req).unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedModel(_)));
    }

    #[test]
    fn parses_array_content_shape() {
        let provider = VllmProvider::new(MockTransport::ok(
            "{\"choices\":[{\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"hello\"},{\"type\":\"text\",\"text\":\" world\"}]}}]}",
        ));
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

        let endpoints = ProviderEndpoints {
            vllm_base_url: format!("http://{addr}"),
            sglang_base_url: format!("http://{addr}"),
        };
        let result = infer_with_endpoints(&endpoints, ProviderKind::Vllm, &request()).unwrap();
        assert_eq!(result.output_text, "from-server");
        assert_eq!(result.provider, ProviderKind::Vllm);
        server.join().unwrap();
    }

    #[test]
    fn payload_includes_multimodal_content_when_images_exist() {
        let mut req = request();
        req.input_images = vec![InferenceImage {
            media_type: "image/jpeg".to_string(),
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
}

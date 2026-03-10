//! Gemini VLM provider using the native `generateContent` API.
//!
//! Supports both inline media (< 20 MB) and the File API for larger payloads.
//!
//! # Examples
//!
//! ```no_run
//! use vidarax_core::gemini::GeminiProvider;
//! use vidarax_core::provider::{InferenceProvider, InferenceRequest, ProviderKind};
//! use std::sync::Arc;
//!
//! let provider = GeminiProvider::new(
//!     "MY_API_KEY".to_string(),
//!     "gemini-2.5-flash-preview-05-20".to_string(),
//! ).unwrap();
//! assert_eq!(provider.kind(), ProviderKind::Gemini);
//! ```

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::Value;

use crate::provider::{
    InferenceProvider, InferenceRequest, InferenceResult, InferenceVideo, ProviderError,
    ProviderKind,
};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
const INLINE_SIZE_LIMIT: usize = 20 * 1024 * 1024; // 20 MB

// ── Static model-string interning cache ──────────────────────────────────────

/// Thread-safe cache that interns model name strings so that `InferenceResult`
/// can hold `&'static str` without leaking memory on every call.
///
/// The cache grows monotonically (one entry per unique model name seen).
/// In practice this is bounded by the number of distinct Gemini model IDs.
static MODEL_CACHE: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);

pub(crate) fn intern_model(s: &str) -> &'static str {
    let mut guard = MODEL_CACHE.lock().expect("model cache lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&cached) = map.get(s) {
        return cached;
    }
    // Leak exactly once per unique model string — bounded by the number of
    // distinct model IDs the process ever sees.
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    map.insert(s.to_string(), leaked);
    leaked
}

// ── Provider ─────────────────────────────────────────────────────────────────

pub struct GeminiProvider {
    api_key: String,
    default_model: String,
    client: reqwest::blocking::Client,
}

impl GeminiProvider {
    /// Create a new [`GeminiProvider`].
    ///
    /// `api_key` must be a Google AI Studio / Vertex AI API key.
    /// `default_model` is used when [`InferenceRequest::model`] is empty.
    pub fn new(api_key: String, default_model: String) -> Result<Self, ProviderError> {
        let client = reqwest::blocking::Client::builder()
            .pool_max_idle_per_host(4)
            .build()
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        Ok(Self {
            api_key,
            default_model,
            client,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Build the `generateContent` request body as a JSON string.
    pub(crate) fn build_payload(&self, request: &InferenceRequest) -> Result<String, ProviderError> {
        // Media parts first (Gemini best practice), text prompt last.
        let mut parts: Vec<Value> = Vec::new();

        // Inline images
        for img in &request.input_images {
            parts.push(serde_json::json!({
                "inlineData": {
                    "mimeType": img.media_type,
                    "data": img.data_base64
                }
            }));
        }

        // Videos — inline if small, File API if large
        for video in &request.input_videos {
            let approx_bytes = video.data_base64.len() * 3 / 4;
            if approx_bytes < INLINE_SIZE_LIMIT {
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": video.media_type,
                        "data": video.data_base64
                    }
                }));
            } else {
                let uri = self.upload_file(video)?;
                parts.push(serde_json::json!({
                    "fileData": {
                        "mimeType": video.media_type,
                        "fileUri": uri
                    }
                }));
            }
        }

        // Text prompt always last
        parts.push(serde_json::json!({"text": &*request.prompt}));

        let mut gen_config = serde_json::json!({
            "maxOutputTokens": request.max_tokens,
            "temperature": request.temperature
        });

        if let Some(schema_str) = &request.guided_json {
            let schema: Value = serde_json::from_str(schema_str).map_err(|e| {
                ProviderError::InvalidResponse(
                    format!("guided_json is not valid JSON: {e}").into(),
                )
            })?;
            gen_config["responseMimeType"] = Value::String("application/json".to_string());
            gen_config["responseSchema"] = schema;
        }

        let body = serde_json::json!({
            "contents": [{"parts": parts}],
            "generationConfig": gen_config
        });

        Ok(body.to_string())
    }

    /// Upload `video` via the Gemini File API (resumable upload) and return
    /// the file URI. Polls until the file reaches `ACTIVE` state.
    fn upload_file(&self, video: &InferenceVideo) -> Result<String, ProviderError> {
        let raw_bytes = base64::engine::general_purpose::STANDARD
            .decode(&video.data_base64)
            .map_err(|e| ProviderError::Transport(format!("base64 decode failed: {e}")))?;

        let byte_count = raw_bytes.len();

        // Step 1: initiate the resumable upload, get the upload URL.
        let init_url = format!(
            "{}/upload/v1beta/files?key={}",
            GEMINI_API_BASE, self.api_key
        );
        let init_resp = self
            .client
            .post(&init_url)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", byte_count.to_string())
            .header("X-Goog-Upload-Header-Content-Type", video.media_type)
            .header("content-type", "application/json")
            .body(r#"{"file":{"display_name":"vidarax_upload"}}"#)
            .send()
            .map_err(|e| ProviderError::Transport(format!("file API init failed: {e}")))?;

        let status = init_resp.status();
        if !status.is_success() {
            return Err(ProviderError::HttpStatus(status.as_u16()));
        }

        let upload_url = init_resp
            .headers()
            .get("x-goog-upload-url")
            .ok_or_else(|| {
                ProviderError::InvalidResponse(
                    "file API init response missing x-goog-upload-url header".into(),
                )
            })?
            .to_str()
            .map_err(|_| {
                ProviderError::InvalidResponse("x-goog-upload-url header is not valid UTF-8".into())
            })?
            .to_string();

        // Step 2: upload the raw bytes and finalize.
        let upload_resp = self
            .client
            .put(&upload_url)
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header("content-type", video.media_type)
            .body(raw_bytes)
            .send()
            .map_err(|e| ProviderError::Transport(format!("file API upload failed: {e}")))?;

        let up_status = upload_resp.status();
        if !up_status.is_success() {
            return Err(ProviderError::HttpStatus(up_status.as_u16()));
        }

        let upload_json: Value = upload_resp
            .json()
            .map_err(|e| ProviderError::Transport(format!("file API upload JSON parse: {e}")))?;

        let file_uri = upload_json["file"]["uri"]
            .as_str()
            .ok_or_else(|| {
                ProviderError::InvalidResponse("file API response missing file.uri".into())
            })?
            .to_string();

        // The name is the last path segment of the URI or the `file.name` field.
        let file_name = upload_json["file"]["name"]
            .as_str()
            .ok_or_else(|| {
                ProviderError::InvalidResponse("file API response missing file.name".into())
            })?
            .to_string();

        // Step 3: poll until ACTIVE (typically 1–5 s, timeout 60 s).
        self.poll_file_active(&file_name)?;

        Ok(file_uri)
    }

    /// Poll `GET /v1beta/files/{name}` until `state == "ACTIVE"`, with a 60 s timeout.
    fn poll_file_active(&self, file_name: &str) -> Result<(), ProviderError> {
        let poll_url = format!(
            "{}/v1beta/{}?key={}",
            GEMINI_API_BASE, file_name, self.api_key
        );
        let deadline = Instant::now() + Duration::from_secs(60);

        loop {
            let resp = self
                .client
                .get(&poll_url)
                .send()
                .map_err(|e| ProviderError::Transport(format!("file poll failed: {e}")))?;

            let st = resp.status();
            if !st.is_success() {
                return Err(ProviderError::HttpStatus(st.as_u16()));
            }

            let json: Value = resp
                .json()
                .map_err(|e| ProviderError::Transport(format!("file poll JSON: {e}")))?;

            let state = json["state"].as_str().unwrap_or("");
            match state {
                "ACTIVE" => return Ok(()),
                "FAILED" => {
                    return Err(ProviderError::InvalidResponse(
                        "File API upload entered FAILED state".into(),
                    ))
                }
                _ => {} // PROCESSING or unknown — keep polling
            }

            if Instant::now() >= deadline {
                return Err(ProviderError::Transport(
                    "Timed out waiting for file to become ACTIVE (60 s)".to_string(),
                ));
            }

            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Parse a raw `generateContent` JSON response into `(output_text, finish_reason)`.
    pub(crate) fn parse_response(
        &self,
        raw: &str,
        model: &str,
        start: Instant,
    ) -> Result<InferenceResult, ProviderError> {
        let json: Value = serde_json::from_str(raw).map_err(|e| {
            ProviderError::InvalidResponse(format!("invalid JSON from Gemini: {e}").into())
        })?;

        let text = json
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProviderError::InvalidResponse(
                    "Gemini response missing candidates[0].content.parts[0].text".into(),
                )
            })?
            .to_string();

        let finish_reason = json
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
            .map(|r| map_finish_reason(r).to_string());

        Ok(InferenceResult {
            provider: ProviderKind::Gemini,
            model: intern_model(model),
            output_text: text,
            fallback_used: false,
            finish_reason,
            inference_latency_ms: start.elapsed().as_millis() as u64,
        })
    }
}

impl InferenceProvider for GeminiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Gemini
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        let start = Instant::now();
        let model = if request.model.is_empty() {
            &self.default_model
        } else {
            &*request.model
        };

        let body = self.build_payload(request)?;
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            GEMINI_API_BASE, model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(Duration::from_millis(request.timeout_ms.max(1)))
            .body(body)
            .send()
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::HttpStatus(status.as_u16()));
        }

        let text = response
            .text()
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        self.parse_response(&text, model, start)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn map_finish_reason(raw: &str) -> &'static str {
    match raw {
        "STOP" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        other => {
            // For unknown variants we intern a lowercase copy.
            intern_model(&other.to_ascii_lowercase())
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{InferenceImage, InferenceRequest, InferenceVideo};
    use serde_json::Value;
    use std::sync::Arc;

    fn provider() -> GeminiProvider {
        GeminiProvider::new("test-key".to_string(), "gemini-2.5-flash-preview-05-20".to_string())
            .unwrap()
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            model: Arc::from(""),
            prompt: Arc::from("describe this"),
            input_images: Vec::new(),
            input_videos: Vec::new(),
            max_tokens: 160,
            temperature: 0.0,
            timeout_ms: 5000,
            allow_fallback: false,
            guided_json: None,
        }
    }

    // ── build_payload ─────────────────────────────────────────────────────────

    #[test]
    fn payload_text_only_has_single_text_part() {
        let p = provider();
        let body = p.build_payload(&request()).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let parts = v["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"].as_str(), Some("describe this"));
    }

    #[test]
    fn payload_image_comes_before_text_prompt() {
        let p = provider();
        let mut req = request();
        req.input_images = vec![InferenceImage {
            media_type: "image/jpeg",
            data_base64: "YWJj".to_string(),
        }];
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let parts = v["contents"][0]["parts"].as_array().unwrap();
        // Part 0 should be the image, part 1 should be the text.
        assert_eq!(parts.len(), 2);
        assert!(parts[0].get("inlineData").is_some(), "first part must be inlineData");
        assert!(parts[1].get("text").is_some(), "last part must be text");
    }

    #[test]
    fn payload_image_uses_inline_data_with_raw_base64() {
        let p = provider();
        let mut req = request();
        req.input_images = vec![InferenceImage {
            media_type: "image/png",
            data_base64: "aGVsbG8=".to_string(),
        }];
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let inline = &v["contents"][0]["parts"][0]["inlineData"];
        assert_eq!(inline["mimeType"].as_str(), Some("image/png"));
        assert_eq!(inline["data"].as_str(), Some("aGVsbG8="));
    }

    #[test]
    fn payload_small_video_uses_inline_data() {
        let p = provider();
        let mut req = request();
        // A tiny base64 string decodes well under 20 MB
        req.input_videos = vec![InferenceVideo {
            media_type: "video/mp4",
            data_base64: "dmlkZW8=".to_string(),
        }];
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let parts = v["contents"][0]["parts"].as_array().unwrap();
        let video_part = &parts[0];
        assert!(
            video_part.get("inlineData").is_some(),
            "small video must use inlineData, got: {video_part}"
        );
        assert_eq!(video_part["inlineData"]["mimeType"].as_str(), Some("video/mp4"));
    }

    #[test]
    fn payload_generation_config_present() {
        let p = provider();
        let mut req = request();
        req.max_tokens = 256;
        req.temperature = 0.5;
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let cfg = &v["generationConfig"];
        assert_eq!(cfg["maxOutputTokens"].as_u64(), Some(256));
        assert!((cfg["temperature"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn payload_guided_json_adds_response_mime_and_schema() {
        let p = provider();
        let mut req = request();
        req.guided_json = Some(Arc::from(
            r#"{"type":"object","properties":{"event_type":{"type":"string"}}}"#,
        ));
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let cfg = &v["generationConfig"];
        assert_eq!(cfg["responseMimeType"].as_str(), Some("application/json"));
        assert_eq!(cfg["responseSchema"]["type"].as_str(), Some("object"));
    }

    #[test]
    fn payload_invalid_guided_json_returns_error() {
        let p = provider();
        let mut req = request();
        req.guided_json = Some(Arc::from("not-json{{{"));
        assert!(p.build_payload(&req).is_err());
    }

    #[test]
    fn payload_multiple_images_all_inline() {
        let p = provider();
        let mut req = request();
        req.input_images = vec![
            InferenceImage { media_type: "image/jpeg", data_base64: "YWJj".to_string() },
            InferenceImage { media_type: "image/png", data_base64: "eHl6".to_string() },
        ];
        let body = p.build_payload(&req).unwrap();
        let v: Value = serde_json::from_str(&body).unwrap();
        let parts = v["contents"][0]["parts"].as_array().unwrap();
        // 2 images + 1 text = 3 parts
        assert_eq!(parts.len(), 3);
        assert!(parts[0].get("inlineData").is_some());
        assert!(parts[1].get("inlineData").is_some());
        assert!(parts[2].get("text").is_some());
    }

    // ── parse_response ────────────────────────────────────────────────────────

    fn make_gemini_response(text: &str, finish_reason: &str) -> String {
        format!(
            r#"{{"candidates":[{{"content":{{"parts":[{{"text":"{text}"}}]}},"finishReason":"{finish_reason}"}}]}}"#
        )
    }

    #[test]
    fn parse_response_extracts_text_and_stop_reason() {
        let p = provider();
        let raw = make_gemini_response("hello world", "STOP");
        let result = p
            .parse_response(&raw, "gemini-2.5-flash-preview-05-20", Instant::now())
            .unwrap();
        assert_eq!(result.output_text, "hello world");
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        assert_eq!(result.provider, ProviderKind::Gemini);
    }

    #[test]
    fn parse_response_maps_max_tokens_to_length() {
        let p = provider();
        let raw = make_gemini_response("truncated", "MAX_TOKENS");
        let result = p
            .parse_response(&raw, "gemini-2.0-flash", Instant::now())
            .unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn parse_response_maps_safety_to_content_filter() {
        let p = provider();
        let raw = make_gemini_response("", "SAFETY");
        let result = p
            .parse_response(&raw, "gemini-2.0-flash", Instant::now())
            .unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("content_filter"));
    }

    #[test]
    fn parse_response_unknown_finish_reason_lowercased() {
        let p = provider();
        let raw = make_gemini_response("text", "RECITATION");
        let result = p
            .parse_response(&raw, "gemini-2.0-flash", Instant::now())
            .unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("recitation"));
    }

    #[test]
    fn parse_response_model_is_interned() {
        let p = provider();
        let raw = make_gemini_response("hi", "STOP");
        let r1 = p
            .parse_response(&raw, "gemini-2.5-flash-preview-05-20", Instant::now())
            .unwrap();
        let r2 = p
            .parse_response(&raw, "gemini-2.5-flash-preview-05-20", Instant::now())
            .unwrap();
        // Same &'static str pointer for the same model name.
        assert!(std::ptr::eq(r1.model, r2.model));
    }

    #[test]
    fn parse_response_invalid_json_returns_error() {
        let p = provider();
        assert!(p.parse_response("{{{{", "gemini-2.0-flash", Instant::now()).is_err());
    }

    #[test]
    fn parse_response_missing_candidates_returns_error() {
        let p = provider();
        let raw = r#"{"candidates":[]}"#;
        assert!(p.parse_response(raw, "gemini-2.0-flash", Instant::now()).is_err());
    }

    // ── intern_model ──────────────────────────────────────────────────────────

    #[test]
    fn intern_model_returns_same_pointer_for_same_string() {
        let a = intern_model("gemini-test-model");
        let b = intern_model("gemini-test-model");
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn intern_model_distinct_strings_distinct_pointers() {
        let a = intern_model("model-alpha");
        let b = intern_model("model-beta");
        assert!(!std::ptr::eq(a, b));
    }

    // ── Live integration tests (require GEMINI_API_KEY) ─────────────────────

    #[test]
    #[ignore = "requires GEMINI_API_KEY env var"]
    fn live_text_only() {
        let key = std::env::var("GEMINI_API_KEY").unwrap();
        let p = GeminiProvider::new(key, "gemini-2.0-flash".to_string()).unwrap();
        let req = InferenceRequest {
            model: std::sync::Arc::from("gemini-2.0-flash"),
            prompt: std::sync::Arc::from("Say hello in exactly 3 words."),
            input_images: vec![],
            input_videos: vec![],
            max_tokens: 50,
            temperature: 0.0,
            timeout_ms: 30000,
            allow_fallback: false,
            guided_json: None,
        };
        let r = p.infer(&req).expect("text-only inference failed");
        assert_eq!(r.finish_reason.as_deref(), Some("stop"));
        assert!(!r.output_text.is_empty());
        println!("text-only: {:?} ({}ms)", r.output_text, r.inference_latency_ms);
    }

    #[test]
    #[ignore = "requires GEMINI_API_KEY env var + /tmp/test_frame.jpg"]
    fn live_image() {
        let key = std::env::var("GEMINI_API_KEY").unwrap();
        let p = GeminiProvider::new(key, "gemini-2.0-flash".to_string()).unwrap();
        let jpeg = std::fs::read("/tmp/test_frame.jpg").expect("need /tmp/test_frame.jpg");
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &jpeg);
        let req = InferenceRequest {
            model: std::sync::Arc::from("gemini-2.0-flash"),
            prompt: std::sync::Arc::from("What does this screenshot show? One sentence."),
            input_images: vec![crate::provider::InferenceImage { media_type: "image/jpeg", data_base64: b64 }],
            input_videos: vec![],
            max_tokens: 100,
            temperature: 0.0,
            timeout_ms: 30000,
            allow_fallback: false,
            guided_json: None,
        };
        let r = p.infer(&req).expect("image inference failed");
        assert!(!r.output_text.is_empty());
        println!("image: {:?} ({}ms)", r.output_text, r.inference_latency_ms);
    }

    #[test]
    #[ignore = "requires GEMINI_API_KEY env var + /tmp/test_clip.mp4"]
    fn live_inline_video() {
        let key = std::env::var("GEMINI_API_KEY").unwrap();
        let p = GeminiProvider::new(key, "gemini-2.0-flash".to_string()).unwrap();
        let mp4 = std::fs::read("/tmp/test_clip.mp4").expect("need /tmp/test_clip.mp4");
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &mp4);
        let req = InferenceRequest {
            model: std::sync::Arc::from("gemini-2.0-flash"),
            prompt: std::sync::Arc::from("Describe what happens in this video in one sentence."),
            input_images: vec![],
            input_videos: vec![crate::provider::InferenceVideo { media_type: "video/mp4", data_base64: b64 }],
            max_tokens: 100,
            temperature: 0.0,
            timeout_ms: 30000,
            allow_fallback: false,
            guided_json: None,
        };
        let r = p.infer(&req).expect("video inference failed");
        assert!(!r.output_text.is_empty());
        println!("video: {:?} ({}ms)", r.output_text, r.inference_latency_ms);
    }

    #[test]
    #[ignore = "requires GEMINI_API_KEY env var + /tmp/test_frame.jpg"]
    fn live_structured_json() {
        let key = std::env::var("GEMINI_API_KEY").unwrap();
        let p = GeminiProvider::new(key, "gemini-2.0-flash".to_string()).unwrap();
        let jpeg = std::fs::read("/tmp/test_frame.jpg").expect("need /tmp/test_frame.jpg");
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &jpeg);
        let schema = r#"{"type":"object","properties":{"title":{"type":"string"},"has_button":{"type":"boolean"}},"required":["title","has_button"]}"#;
        let req = InferenceRequest {
            model: std::sync::Arc::from("gemini-2.0-flash"),
            prompt: std::sync::Arc::from("Extract the page title and whether there is a visible button."),
            input_images: vec![crate::provider::InferenceImage { media_type: "image/jpeg", data_base64: b64 }],
            input_videos: vec![],
            max_tokens: 200,
            temperature: 0.0,
            timeout_ms: 30000,
            allow_fallback: false,
            guided_json: Some(std::sync::Arc::from(schema)),
        };
        let r = p.infer(&req).expect("structured json inference failed");
        let parsed: serde_json::Value = serde_json::from_str(&r.output_text).expect("output not valid JSON");
        assert!(parsed.get("title").is_some(), "missing 'title' field");
        assert!(parsed.get("has_button").is_some(), "missing 'has_button' field");
        println!("structured: {} ({}ms)", r.output_text, r.inference_latency_ms);
    }
}

//! Offline review of real 60 FPS footage; inference is entirely mocked.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};
use tower::ServiceExt;
use vidarax_api::{app_router, AppState};
use vidarax_core::provider::{
    InferenceProvider, InferenceRequest, InferenceResult, MediaTransport, ProviderError,
    ProviderKind, TokenUsage,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const PROMPT: &str = "Inspect the short white pulse and return caller-defined measurements; do not substitute a generic description.";

fn path(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vidarax-dense-{}-{}-{suffix}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let file = path("source.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=96x64:rate=60:duration=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=3",
                "-vf",
                "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,0.9,1.1)'",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-g",
                "120",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-y",
            ])
            .arg(&file)
            .status()
            .unwrap();
        assert!(status.success());
        Self(file)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Clone)]
struct Call {
    model: String,
    prompt: String,
    images: usize,
    fps: Option<f32>,
    video_frames: Option<u64>,
    white_frames: usize,
    schema: Value,
    fallback: bool,
}
struct Recorder {
    calls: Mutex<Vec<Call>>,
    image_limit: usize,
    native: bool,
}
impl Recorder {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            image_limit: 256,
            native: true,
        }
    }
}
impl InferenceProvider for Recorder {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Gemini
    }
    fn media_transport_for_model(&self, _: &str) -> MediaTransport {
        if self.native {
            MediaTransport::BinaryFile
        } else {
            MediaTransport::JsonDataUrl
        }
    }
    fn max_input_images_for_model(&self, _: &str) -> usize {
        self.image_limit
    }
    fn max_video_fps_for_model(&self, _: &str) -> Option<f32> {
        self.native.then_some(24.0)
    }
    fn infer(&self, req: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        assert!(req.prompt.contains(PROMPT));
        let mut white_frames = 0;
        let video_frames = req.input_videos.first().map(|video| {
            let file = path("clip.mp4");
            fs::write(&file, video.raw_bytes.as_ref().unwrap()).unwrap();
            let output = Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                ])
                .arg(&file)
                .output()
                .unwrap();
            let pixels = Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(&file)
                .args(["-an", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
                .output()
                .unwrap();
            assert!(pixels.status.success());
            white_frames = pixels
                .stdout
                .chunks_exact(96 * 64)
                .filter(|frame| frame.iter().all(|pixel| *pixel > 240))
                .count();
            let _ = fs::remove_file(file);
            assert!(output.status.success());
            let probe: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(probe["streams"][0]["r_frame_rate"], "60/1");
            probe["streams"][0]["nb_read_frames"]
                .as_str()
                .unwrap()
                .parse()
                .unwrap()
        });
        let schema = serde_json::from_str(req.guided_json.as_ref().unwrap()).unwrap();
        self.calls.lock().unwrap().push(Call {
            model: req.model.to_string(),
            prompt: req.prompt.to_string(),
            images: req.input_images.len(),
            fps: req.input_videos.first().and_then(|v| v.sampling_fps),
            video_frames,
            white_frames,
            schema,
            fallback: req.allow_fallback,
        });
        Ok(InferenceResult {
            provider: ProviderKind::Gemini,
            model: Arc::clone(&req.model),
            output_text: json!({"caller_measurement": "pulse", "confidence": 0.1}).to_string(),
            fallback_used: false,
            finish_reason: Some("stop".into()),
            inference_latency_ms: 1,
            usage: TokenUsage::default(),
        })
    }
}

fn request(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
async fn body(response: axum::response::Response) -> Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}
fn schema() -> Value {
    json!({"type":"object","properties":{"caller_measurement":{"type":"string"},"confidence":{"type":"number"}},"required":["caller_measurement","confidence"]})
}

async fn run(
    fixture: &Fixture,
    provider: Arc<Recorder>,
    extra: Value,
) -> (StatusCode, Value, Value) {
    let router = app_router(AppState::with_wal_for_tests_and_endpoints(
        path("wal"),
        Some(provider),
    ));
    let created = body(
        router
            .clone()
            .oneshot(request("/v1/runs", json!({"model":"gemini-3.8-flash"})))
            .await
            .unwrap(),
    )
    .await;
    let id = created["run_id"].as_str().unwrap();
    let mut payload = json!({
        "source_uri": fixture.0.to_string_lossy(), "model":"gemini-3.8-flash",
        "sampling_policy":"fixed", "fixed_fps":60, "max_frames":200,
        "semantic_prompt":PROMPT, "output_schema":schema(),
        "semantic_timeout_ms":10000, "vlm_concurrency":1,
        "include_frame_metadata":true
    });
    payload
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let response = router
        .clone()
        .oneshot(request(&format!("/v1/runs/{id}/reason"), payload))
        .await
        .unwrap();
    let status = response.status();
    let result = body(response).await;
    let events = body(
        router
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/runs/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
    )
    .await;
    (status, result, events)
}

#[tokio::test]
async fn dense_frames_preserve_custom_prompt_schema_order_and_targeted_timestamps_across_tiers() {
    let fixture = Fixture::new();
    let provider = Arc::new(Recorder::new());
    let (status, result, events) = run(
        &fixture,
        provider.clone(),
        json!({
            "chunk_size":30, "semantic_frames_per_chunk":42, "semantic_context_frames":6,
            "source_start_ms":500, "source_end_ms":2500,
            "first_pass_model":"gemini-3.8-flash", "second_pass_model":"gemini-3.6-flash"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    let calls = provider.calls.lock().unwrap();
    assert_eq!(calls.len(), 8); // Four owned chunks, each escalated to second pass.
    for pair in calls.chunks(2) {
        assert_eq!(pair[0].model, "gemini-3.8-flash");
        assert_eq!(pair[1].model, "gemini-3.6-flash");
        assert_eq!(pair[0].prompt, pair[1].prompt);
        assert_eq!(pair[0].schema, schema());
        assert_eq!(pair[1].schema, schema());
        assert!(!pair[0].fallback);
        assert!(pair[0].images >= 36 && pair[0].images <= 42);
        let timestamps: Value = serde_json::from_str(
            pair[0]
                .prompt
                .split("ordered_image_timestamps=")
                .nth(1)
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
        )
        .unwrap();
        let times = timestamps.as_array().unwrap();
        assert_eq!(times.len(), pair[0].images);
        for image in times {
            let frame = image["frame_index"].as_u64().unwrap();
            let pts = image["pts_ms"].as_u64().unwrap();
            assert!((500..2500).contains(&pts));
            assert!((pts as i64 - (frame as f64 / 60.0 * 1000.0).round() as i64).abs() <= 1);
        }
        assert!(times
            .windows(2)
            .all(|w| w[0]["frame_index"].as_u64() < w[1]["frame_index"].as_u64()));
    }
    let generated = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "analysis_generated")
        .unwrap();
    assert_eq!(generated["payload"]["frames"], 120); // Context does not duplicate metadata.
    let chunks: Vec<_> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == "semantic_chunk_inferred")
        .collect();
    assert_eq!(chunks.len(), 4);
    assert!(chunks
        .iter()
        .all(|chunk| chunk["payload"]["input_image_timestamps"]
            .as_array()
            .unwrap()
            .len()
            >= 36));
}

#[tokio::test]
async fn native_video_overlap_preserves_full_rate_action_boundaries_and_provider_fps() {
    let fixture = Fixture::new();
    for mode in ["video", "audio_video"] {
        let provider = Arc::new(Recorder::new());
        let (status, result, events) = run(&fixture, provider.clone(), json!({
        "source_start_ms":250,"source_end_ms":2750,
        "media":{"mode":mode,"window_ms":1000,"overlap_ms":500,"video_fps":24,"resolution":"high"},
        "first_pass_model":"gemini-3.8-flash","second_pass_model":"gemini-3.6-flash"
    })).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 8);
        for (index, pair) in calls.chunks(2).enumerate() {
            assert_eq!(pair[0].prompt, pair[1].prompt);
            assert_eq!(pair[1].schema, schema());
            let start = 250 + index * 500;
            assert!(pair[0]
                .prompt
                .contains(&format!("chunk_pts_start_ms={start}")));
            assert!(pair[0]
                .prompt
                .contains(&format!("chunk_pts_end_ms={}", start + 1000)));
            assert_eq!(pair[0].video_frames, Some(60));
            assert_eq!(pair[0].fps, Some(24.0));
            assert_eq!(pair[1].fps, Some(24.0));
            assert!(!pair[0].fallback);
        }
        // 0.9..1.1s action is complete in both neighboring review windows.
        assert!(calls[0].prompt.contains("chunk_pts_start_ms=250"));
        assert!(calls[2].prompt.contains("chunk_pts_start_ms=750"));
        assert!(calls[0].white_frames >= 12);
        assert_eq!(calls[0].white_frames, calls[2].white_frames);
        let generated = events["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["kind"] == "analysis_generated")
            .unwrap();
        assert_eq!(generated["payload"]["frames"], 150);
        let chunks: Vec<_> = events["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "semantic_chunk_inferred")
            .collect();
        assert!(chunks
            .iter()
            .all(|chunk| chunk["payload"]["provider_sampling_interval_ms"] == 42));
        assert!(chunks
            .iter()
            .all(|chunk| chunk["payload"]["provider_sampling_status"] == "requested_unverified"));
        assert!(chunks
            .iter()
            .all(|chunk| chunk["payload"]["timestamp_resolution_ms"] == 1000));
    }
}

#[tokio::test]
async fn explicit_coverage_rejects_unsupported_controls_and_insufficient_budgets() {
    let fixture = Fixture::new();
    for extra in [
        json!({"semantic_frames_per_chunk":257}),
        json!({"chunk_size":1,"semantic_frames_per_chunk":256,"semantic_context_frames":128}),
        json!({"semantic_context_frames":129}),
        json!({"semantic_frames_per_chunk":10,"max_frames":20}),
        json!({"source_start_ms":2500,"source_end_ms":2000}),
        json!({"source_end_ms":4000}),
        json!({"media":{"mode":"video","window_ms":1000,"overlap_ms":900}}),
        json!({"media":{"mode":"frames","video_fps":10}}),
        json!({"media":{"mode":"video","video_fps":60}}),
    ] {
        let provider = Arc::new(Recorder::new());
        let (status, _, _) = run(&fixture, provider.clone(), extra).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(provider.calls.lock().unwrap().is_empty());
    }
    let provider = Arc::new(Recorder {
        image_limit: 5,
        native: false,
        calls: Mutex::new(Vec::new()),
    });
    for extra in [
        json!({"semantic_frames_per_chunk":30}),
        json!({"media":{"mode":"video","video_fps":24}}),
    ] {
        let (status, _, _) = run(&fixture, provider.clone(), extra).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }
}

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::Value;
use vidarax_core::gate::FrameSignal;
use vidarax_core::ingest::InputSource;
use vidarax_core::loop_detector::LoopDetector;
use vidarax_core::provider::{
    InferenceProvider, InferenceRequest, OpenAiCompatProvider, ProviderError, ProviderKind,
    ProviderRouter, Transport,
};
use vidarax_core::webrtc::clip::{ClipAccumulator, ClipConfig};
use vidarax_core::webrtc::workers::StreamFrame;

// ── Test infrastructure ───────────────────────────────────────────────────────

/// Returns JSON that will be parsed as a successful completion.
fn completion_json(text: &str) -> String {
    format!(
        r#"{{"choices":[{{"message":{{"role":"assistant","content":"{text}"}}}}]}}"#
    )
}

/// Returns a completion JSON with an explicit finish_reason field.
fn completion_with_reason(text: &str, reason: &str) -> String {
    format!(
        r#"{{"choices":[{{"finish_reason":"{reason}","message":{{"role":"assistant","content":"{text}"}}}}]}}"#
    )
}

/// Minimal valid request used as a base for most tests.
fn base_request() -> InferenceRequest {
    InferenceRequest {
        model: Arc::from("openbmb/MiniCPM-V-4.5"),
        prompt: Arc::from("describe"),
        input_images: Vec::new(),
        input_videos: Vec::new(),
        max_tokens: 64,
        temperature: 0.0,
        timeout_ms: 1_000,
        allow_fallback: false,
        guided_json: None,
    }
}

/// Mock transport returning a fixed response and optionally capturing the request body.
struct MockTransport {
    response: Result<String, ProviderError>,
    captured_body: Arc<Mutex<Option<String>>>,
}

impl MockTransport {
    fn ok(payload: &str) -> Self {
        Self {
            response: Ok(payload.to_string()),
            captured_body: Arc::new(Mutex::new(None)),
        }
    }

    fn err(error: ProviderError) -> Self {
        Self {
            response: Err(error),
            captured_body: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns a mock that records each request body; the Arc can be cloned
    /// before constructing the provider so the caller can inspect it afterward.
    fn capturing(payload: &str) -> (Self, Arc<Mutex<Option<String>>>) {
        let captured_body = Arc::new(Mutex::new(None));
        let transport = Self {
            response: Ok(payload.to_string()),
            captured_body: Arc::clone(&captured_body),
        };
        (transport, captured_body)
    }
}

impl Transport for MockTransport {
    fn call(&self, _endpoint: &str, body: String, _timeout_ms: u64) -> Result<String, ProviderError> {
        *self.captured_body.lock().unwrap() = Some(body);
        self.response.clone()
    }
}

/// Transport that sleeps before returning so callers can observe latency.
struct DelayTransport {
    delay_ms: u64,
    response: String,
}

impl Transport for DelayTransport {
    fn call(&self, _endpoint: &str, _body: String, _timeout_ms: u64) -> Result<String, ProviderError> {
        thread::sleep(Duration::from_millis(self.delay_ms));
        Ok(self.response.clone())
    }
}

/// Build a minimal `StreamFrame` with a valid JPEG stub.
fn make_stream_frame(seq: u64, pts_ms: u64) -> StreamFrame {
    StreamFrame {
        signal: FrameSignal {
            frame_index: seq,
            pts_ms,
            perceptual_hash: seq.wrapping_mul(0xDEAD_BEEF),
            luma_mean: 0.5,
            flicker_score: 0.0,
            ghosting_score: 0.0,
            noise_variance_score: 0.0,
        },
        jpeg: Some([0xff_u8, 0xd8, 0xaa, 0xbb, 0xff, 0xd9].into()),
        pts_ms,
        seq,
    }
}

// ── 1. Structured output schema in payload ────────────────────────────────────
//
// The field `guided_json: Option<String>` carries the schema as a JSON string.
// The provider serializes it into `response_format.json_schema.schema` for
// vLLM ≥0.15 constrained-decoding compatibility.

#[test]
fn output_schema_adds_guided_json_to_payload() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "event_type": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["event_type", "confidence"]
    });

    let (transport, captured) = MockTransport::capturing(&completion_json("ok"));
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);
    let mut req = base_request();
    req.guided_json = Some(Arc::from(schema.to_string()));

    provider.infer(&req).unwrap();

    let raw = captured.lock().unwrap().clone().expect("body captured");
    let value: Value = serde_json::from_str(&raw).unwrap();
    // Provider writes the schema under response_format.json_schema.schema.
    let guided = &value["response_format"]["json_schema"]["schema"];
    assert_eq!(guided["type"].as_str(), Some("object"), "schema.type should be object");
    assert!(
        guided["properties"]["event_type"].is_object(),
        "event_type property should be present"
    );
    assert!(
        guided["properties"]["confidence"].is_object(),
        "confidence property should be present"
    );
}

#[test]
fn no_output_schema_omits_response_format_from_payload() {
    let (transport, captured) = MockTransport::capturing(&completion_json("ok"));
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);

    provider.infer(&base_request()).unwrap();

    let raw = captured.lock().unwrap().clone().expect("body captured");
    let value: Value = serde_json::from_str(&raw).unwrap();
    assert!(
        value.get("response_format").is_none(),
        "response_format must be absent when guided_json is None"
    );
}

#[test]
fn output_schema_is_forwarded_verbatim_to_payload() {
    let schema = serde_json::json!({"type": "array", "items": {"type": "string"}});
    let (transport, captured) = MockTransport::capturing(&completion_json("ok"));
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Sglang);
    let mut req = base_request();
    req.guided_json = Some(Arc::from(schema.to_string()));

    provider.infer(&req).unwrap();

    let raw = captured.lock().unwrap().clone().expect("body captured");
    let value: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(value["response_format"]["json_schema"]["schema"], schema);
}

#[test]
fn output_schema_with_nested_properties_round_trips() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "scene": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        }
    });

    let (transport, captured) = MockTransport::capturing(&completion_json("ok"));
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);
    let mut req = base_request();
    req.guided_json = Some(Arc::from(schema.to_string()));

    provider.infer(&req).unwrap();

    let raw = captured.lock().unwrap().clone().expect("body captured");
    let value: Value = serde_json::from_str(&raw).unwrap();
    let guided = &value["response_format"]["json_schema"]["schema"];
    assert_eq!(guided["properties"]["scene"]["type"].as_str(), Some("object"));
}

// ── 2. finish_reason parsing ──────────────────────────────────────────────────

#[test]
fn finish_reason_stop_is_parsed() {
    let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_with_reason("done", "stop")), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    assert_eq!(result.output_text, "done");
}

#[test]
fn finish_reason_length_is_parsed() {
    let provider =
        OpenAiCompatProvider::new(MockTransport::ok(&completion_with_reason("partial", "length")), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason.as_deref(), Some("length"));
}

#[test]
fn finish_reason_absent_yields_none() {
    let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason, None);
}

#[test]
fn finish_reason_null_yields_none() {
    let json =
        r#"{"choices":[{"finish_reason":null,"message":{"role":"assistant","content":"ok"}}]}"#;
    let provider = OpenAiCompatProvider::new(MockTransport::ok(json), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason, None);
}

#[test]
fn finish_reason_custom_value_is_preserved() {
    let provider =
        OpenAiCompatProvider::new(MockTransport::ok(&completion_with_reason("text", "content_filter")), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason.as_deref(), Some("content_filter"));
}

#[test]
fn sglang_provider_also_parses_finish_reason() {
    let provider =
        OpenAiCompatProvider::new(MockTransport::ok(&completion_with_reason("sglang-out", "stop")), ProviderKind::Sglang);
    let result = provider.infer(&base_request()).unwrap();
    assert_eq!(result.finish_reason.as_deref(), Some("stop"));
}

// ── 3. Per-result latency ─────────────────────────────────────────────────────

#[test]
fn inference_latency_reflects_transport_duration() {
    let delay_ms = 30u64;
    let transport = DelayTransport {
        delay_ms,
        response: completion_with_reason("response", "stop"),
    };
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert!(
        result.inference_latency_ms >= delay_ms,
        "latency {}ms should be >= delay {}ms",
        result.inference_latency_ms,
        delay_ms
    );
}

#[test]
fn fast_transport_yields_non_negative_latency() {
    let provider = OpenAiCompatProvider::new(MockTransport::ok(&completion_json("ok")), ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    // u64 is always non-negative; this test documents the contract
    let _: u64 = result.inference_latency_ms;
}

#[test]
fn latency_is_independent_of_output_content() {
    let delay_ms = 20u64;
    let long_response = completion_with_reason(&"x".repeat(4096), "stop");
    let transport = DelayTransport { delay_ms, response: long_response };
    let provider = OpenAiCompatProvider::new(transport, ProviderKind::Vllm);
    let result = provider.infer(&base_request()).unwrap();
    assert!(
        result.inference_latency_ms >= delay_ms,
        "latency should be >= delay regardless of response length"
    );
}

#[test]
fn fallback_result_also_carries_latency() {
    let primary = OpenAiCompatProvider::new(MockTransport::err(ProviderError::HttpStatus(503)), ProviderKind::Vllm);
    let fallback_transport = DelayTransport {
        delay_ms: 15,
        response: completion_with_reason("fallback", "stop"),
    };
    let fallback = OpenAiCompatProvider::new(fallback_transport, ProviderKind::Sglang);
    let router = ProviderRouter::new(primary, fallback);

    let mut req = base_request();
    req.allow_fallback = true;
    let result = router.infer(&req).unwrap();

    assert!(result.fallback_used);
    assert!(
        result.inference_latency_ms >= 15,
        "fallback latency should be measured"
    );
}

// ── 4. Clip accumulator batching ──────────────────────────────────────────────

#[test]
fn accumulator_emits_clip_after_window_fills() {
    let cfg = ClipConfig {
        target_fps: 10,
        clip_length_seconds: 0.5,
        delay_seconds: 0.0,
    };
    let mut acc =
        ClipAccumulator::new(cfg, "run-a".into(), "sess-a".into(), "describe".into());

    // Feed frames at 100ms intervals; 500ms window requires 6 frames (pts 0..500).
    let mut emitted = None;
    for i in 0..8u64 {
        emitted = acc.push(make_stream_frame(i, i * 100));
        if emitted.is_some() {
            break;
        }
    }

    let clip = emitted.expect("a clip should have been emitted");
    assert!(!clip.frames.is_empty(), "clip must contain at least one frame");
    assert!(clip.pts_end >= clip.pts_start, "pts must be ordered");
    assert_eq!(&*clip.run_id, "run-a");
    assert_eq!(&*clip.session_id, "sess-a");
    assert_eq!(&*clip.prompt, "describe");
}

#[test]
fn accumulator_does_not_emit_before_window_is_full() {
    let cfg = ClipConfig {
        target_fps: 10,
        clip_length_seconds: 2.0, // 2 seconds required
        delay_seconds: 0.0,
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    // Send 10 frames at 100ms each — only 1 second elapsed, below 2s window.
    for i in 0..10u64 {
        let result = acc.push(make_stream_frame(i, i * 100));
        assert!(result.is_none(), "frame {i}: should not emit before window fills");
    }
}

#[test]
fn accumulator_rate_limits_to_target_fps() {
    // target_fps=2 → accept one frame every 500ms
    let cfg = ClipConfig {
        target_fps: 2,
        clip_length_seconds: 2.0, // 2fps * 2s = 4 frames minimum
        delay_seconds: 0.0,
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    // Push 20 frames at 100ms intervals (0, 100, 200, …, 1900ms).
    // At 500ms interval only frames at 0, 500, 1000, 1500, 2000ms are accepted.
    let mut clip = None;
    for i in 0..25u64 {
        clip = acc.push(make_stream_frame(i, i * 100));
        if clip.is_some() {
            break;
        }
    }

    let clip = clip.expect("clip should emit after 2s window at 2fps");
    // Accepted set: 0, 500, 1000, 1500, 2000ms → 5 frames, but window triggers
    // once elapsed_pts >= 2000ms, so we expect at least 4 frames.
    assert!(
        clip.frames.len() >= 4,
        "expected >= 4 frames at 2fps over 2s, got {}",
        clip.frames.len()
    );
}

#[test]
fn accumulator_drops_frames_with_no_jpeg() {
    let cfg = ClipConfig {
        target_fps: 30,
        clip_length_seconds: 0.1,
        delay_seconds: 0.0,
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    let mut no_jpeg = make_stream_frame(0, 0);
    no_jpeg.jpeg = None;
    let result = acc.push(no_jpeg);
    assert!(result.is_none(), "frames without JPEG data must be dropped");
}

#[test]
fn clip_config_min_frame_constraint_rejects_too_short_window() {
    // target_fps=1, clip_length=0.1 → 0.1 frames < 3 minimum
    let cfg = ClipConfig {
        target_fps: 1,
        clip_length_seconds: 0.1,
        delay_seconds: 0.0,
    };
    assert!(
        cfg.validate().is_err(),
        "1 fps * 0.1s = 0.1 frames, which is below the 3-frame minimum"
    );
}

#[test]
fn clip_config_min_frame_constraint_accepts_exact_boundary() {
    // target_fps=3, clip_length=1.0 → 3 frames, exactly at minimum
    let cfg = ClipConfig {
        target_fps: 3,
        clip_length_seconds: 1.0,
        delay_seconds: 0.0,
    };
    assert!(cfg.validate().is_ok(), "3 fps * 1s = 3 frames, exactly at minimum");
}

#[test]
fn clip_accumulator_delay_suppresses_rapid_second_emission() {
    // delay_seconds=60 means a second clip cannot emit within 60s wall-clock.
    // After the first clip we push another full window immediately (no sleep),
    // the delay guard must prevent a second emission.
    let cfg = ClipConfig {
        target_fps: 10,
        clip_length_seconds: 0.5,
        delay_seconds: 60.0, // impossible to satisfy in test time
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    // Fill first window (pts 0–600ms) and trigger emission.
    let mut first = None;
    for i in 0..8u64 {
        first = acc.push(make_stream_frame(i, i * 100));
        if first.is_some() {
            break;
        }
    }
    assert!(first.is_some(), "first clip must emit");

    // Push another complete window immediately after (pts 700–1300ms).
    // The delay guard should prevent a second emission.
    let mut second = None;
    for i in 8..16u64 {
        second = acc.push(make_stream_frame(i, i * 100));
        if second.is_some() {
            break;
        }
    }
    assert!(
        second.is_none(),
        "second clip must be suppressed while inter-emission delay has not elapsed"
    );
}

#[test]
fn accumulator_buffer_cleared_after_emission() {
    let cfg = ClipConfig {
        target_fps: 10,
        clip_length_seconds: 0.5,
        delay_seconds: 0.0,
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    // Trigger first emission.
    let mut first_clip = None;
    for i in 0..8u64 {
        first_clip = acc.push(make_stream_frame(i, i * 100));
        if first_clip.is_some() {
            break;
        }
    }
    assert!(first_clip.is_some(), "first clip should emit");

    // After emission the buffer is cleared. Sending a single frame should NOT
    // emit immediately — the window must fill again.
    let result = acc.push(make_stream_frame(99, 9999));
    assert!(
        result.is_none(),
        "single frame after emission should not trigger another clip"
    );
}

#[test]
fn clip_work_pts_span_covers_window() {
    let cfg = ClipConfig {
        target_fps: 10,
        clip_length_seconds: 0.5,
        delay_seconds: 0.0,
    };
    let mut acc = ClipAccumulator::new(cfg, "r".into(), "s".into(), "".into());

    let mut clip = None;
    for i in 0..8u64 {
        clip = acc.push(make_stream_frame(i, i * 100));
        if clip.is_some() {
            break;
        }
    }

    let clip = clip.unwrap();
    let span_ms = clip.pts_end.saturating_sub(clip.pts_start);
    assert!(
        span_ms >= 500,
        "pts span {}ms should be >= 500ms (clip_length_seconds=0.5)",
        span_ms
    );
}

// ── 5. HLS URL validation ─────────────────────────────────────────────────────

#[test]
fn https_m3u8_url_is_accepted_as_hls_stream() {
    let allowed = vec![std::env::temp_dir()];
    let result =
        InputSource::parse_and_validate("https://example.com/live/feed.m3u8", &allowed);
    assert!(result.is_ok(), "valid https .m3u8 should be accepted: {result:?}");
    assert!(
        matches!(result.unwrap(), InputSource::HlsStream(_)),
        "should resolve to HlsStream"
    );
}

#[test]
fn http_m3u8_url_is_accepted_as_hls_stream() {
    let allowed = vec![std::env::temp_dir()];
    let result =
        InputSource::parse_and_validate("http://example.com/live/stream.m3u8", &allowed);
    assert!(result.is_ok(), "valid http .m3u8 should be accepted: {result:?}");
    assert!(matches!(result.unwrap(), InputSource::HlsStream(_)));
}

#[test]
fn non_m3u8_https_url_is_not_hls_stream() {
    let allowed = vec![std::env::temp_dir()];
    let result =
        InputSource::parse_and_validate("https://example.com/clip.mp4", &allowed).unwrap();
    assert!(
        !matches!(result, InputSource::HlsStream(_)),
        "non-.m3u8 URL should not be HlsStream"
    );
}

#[test]
fn hls_url_with_embedded_credentials_is_rejected() {
    let allowed = vec![std::env::temp_dir()];
    assert!(
        InputSource::parse_and_validate(
            "https://user:secret@example.com/feed.m3u8",
            &allowed
        )
        .is_err(),
        "credentials in HLS URL must be rejected"
    );
}

#[test]
fn hls_url_targeting_private_ip_is_rejected() {
    let allowed = vec![std::env::temp_dir()];
    for ip in &["192.168.0.1", "10.0.0.1", "172.16.0.1"] {
        let url = format!("https://{ip}/stream.m3u8");
        assert!(
            InputSource::parse_and_validate(&url, &allowed).is_err(),
            "private IP {ip} in HLS URL should be rejected"
        );
    }
}

#[test]
fn hls_url_targeting_localhost_is_rejected() {
    let allowed = vec![std::env::temp_dir()];
    assert!(
        InputSource::parse_and_validate("https://localhost/stream.m3u8", &allowed).is_err(),
        "localhost in HLS URL must be rejected"
    );
}

#[test]
fn hls_url_targeting_metadata_endpoint_is_rejected() {
    let allowed = vec![std::env::temp_dir()];
    assert!(
        InputSource::parse_and_validate(
            "http://169.254.169.254/latest/meta-data/stream.m3u8",
            &allowed
        )
        .is_err(),
        "link-local metadata endpoint must be rejected"
    );
}

#[test]
fn hls_scheme_url_is_accepted_as_hls_stream() {
    let allowed = vec![std::env::temp_dir()];
    // hls:// is a native scheme that ffmpeg supports directly.
    let result =
        InputSource::parse_and_validate("hls://example.com/live.m3u8", &allowed);
    assert!(result.is_ok(), "hls:// URL should be accepted: {result:?}");
    assert!(
        matches!(result.unwrap(), InputSource::HlsStream(_)),
        "hls:// URL should resolve to HlsStream variant"
    );
}

#[test]
fn hls_ffmpeg_input_returns_original_url() {
    let url = "https://example.com/live.m3u8".to_string();
    let src = InputSource::HlsStream(url.clone());
    assert_eq!(src.as_ffmpeg_input(), url.as_str());
}

// ── 6. Loop detector behavior ─────────────────────────────────────────────────

#[test]
fn loop_not_triggered_on_first_occurrence() {
    let mut d = LoopDetector::new(6, 3);
    assert!(!d.check(0xDEAD_BEEF_CAFE_BABE), "first occurrence must not trigger");
}

#[test]
fn loop_triggered_after_repeat_trigger_count() {
    let mut d = LoopDetector::new(6, 3);
    let hash = 0xAAAA_AAAA_AAAA_AAAA;
    assert!(!d.check(hash));
    assert!(!d.check(hash));
    assert!(!d.check(hash));
    // Fourth identical hash: 3 slots now match → triggers.
    assert!(d.check(hash), "should trigger after repeat_trigger identical hashes");
}

#[test]
fn similar_hashes_within_threshold_trigger_loop() {
    let mut d = LoopDetector::new(6, 3);
    let base = 0xDEAD_BEEF_CAFE_BABE;
    // Hamming distance of 1 per flip — all within threshold of 6.
    assert!(!d.check(base));
    assert!(!d.check(base ^ 0x01));
    assert!(!d.check(base ^ 0x02));
    assert!(d.check(base ^ 0x04), "near-duplicate hashes should trigger loop");
}

#[test]
fn distinct_hashes_do_not_trigger_loop() {
    let mut d = LoopDetector::new(6, 3);
    // Each hash differs by > 6 bits from all others.
    assert!(!d.check(0x0000_0000_0000_00FF));
    assert!(!d.check(0x0000_0000_00FF_0000));
    assert!(!d.check(0x0000_00FF_0000_0000));
    assert!(!d.check(0xFF00_0000_0000_0000));
}

#[test]
fn loop_detector_reset_clears_state() {
    let mut d = LoopDetector::new(6, 3);
    let hash = 0xBEEF_CAFE_DEAD_BABE;
    d.check(hash);
    d.check(hash);
    d.check(hash);
    // Loop was building up; now reset.
    d.reset();
    // After reset a single occurrence should not trigger.
    assert!(!d.check(hash), "reset should clear accumulated state");
}

#[test]
fn loop_detector_evicts_old_hashes_from_ring_buffer() {
    let mut d = LoopDetector::new(6, 3);
    let loop_hash = 0x1234_5678_9ABC_DEF0;
    // Populate 3 matching hashes.
    d.check(loop_hash);
    d.check(loop_hash);
    d.check(loop_hash);
    // Overwrite the ring buffer (8 slots) with distinct hashes.
    for i in 0..8u64 {
        d.check(i.wrapping_mul(0x1111_1111_1111_1111));
    }
    // Old hashes are gone; a fresh occurrence should not trigger.
    assert!(!d.check(loop_hash), "evicted hashes must not count toward trigger");
}

#[test]
fn loop_detector_high_threshold_allows_more_variation() {
    // threshold=32 means up to 31 differing bits are considered "same".
    let mut d = LoopDetector::new(32, 3);
    let base = 0x0000_0000_0000_0000u64;
    // Flip 16 bits — within threshold of 32.
    let _similar = 0x0000_0000_FFFF_FFFFu64; // 32 bits different — exactly 32 is not < threshold
    // Actually (base ^ similar).count_ones() = 32, and threshold check is `< threshold`, so 32 < 32 is false.
    // Use a variant with 16 bit flips to stay safely within threshold.
    let within = 0x0000_FFFF_0000_0000u64; // 16 bits different
    assert!(!d.check(base));
    assert!(!d.check(within));
    assert!(!d.check(within ^ 0x01));
    assert!(d.check(within ^ 0x02), "high threshold should match across moderate variation");
}

#[test]
fn loop_detector_strict_threshold_ignores_slight_variation() {
    // threshold=1 means only exact matches count (0 differing bits < 1).
    let mut d = LoopDetector::new(1, 3);
    let base = 0xDEAD_BEEF_CAFE_BABEu64;
    d.check(base);
    d.check(base);
    d.check(base);
    // Flip one bit — hamming distance = 1, not < 1 → not a match.
    assert!(
        !d.check(base ^ 0x01),
        "strict threshold=1 should not match 1-bit variants"
    );
}

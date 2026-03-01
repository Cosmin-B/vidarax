use std::sync::Arc;

use vidarax_core::tiered_vlm::TieredVlmConfig;

#[test]
fn default_config_uses_same_model_for_both_passes() {
    let config = TieredVlmConfig::default();
    assert_eq!(config.first_pass_model, config.second_pass_model);
    assert!(config.second_pass_threshold > 0.0);
    assert!(config.second_pass_threshold <= 1.0);
}

#[test]
fn tiered_config_detects_when_second_pass_needed() {
    let config = TieredVlmConfig {
        first_pass_model: "Qwen/Qwen3-VL-2B-Instruct".to_string(),
        second_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
        second_pass_threshold: 0.7,
        second_pass_max_tokens: 256,
    };
    assert!(config.needs_second_pass(0.5));   // below threshold
    assert!(!config.needs_second_pass(0.8));  // above threshold
    assert!(!config.needs_second_pass(0.7));  // at threshold = no second pass
}

#[test]
fn single_model_config_never_needs_second_pass() {
    let config = TieredVlmConfig::single_model("Qwen/Qwen3-VL-8B-Instruct");
    assert!(!config.needs_second_pass(0.1));
    assert!(!config.needs_second_pass(0.0));
}

#[test]
fn is_tiered_returns_true_only_when_models_differ() {
    let tiered = TieredVlmConfig {
        first_pass_model: "Qwen/Qwen3-VL-2B-Instruct".to_string(),
        second_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
        second_pass_threshold: 0.7,
        second_pass_max_tokens: 256,
    };
    assert!(tiered.is_tiered());

    let single = TieredVlmConfig::single_model("Qwen/Qwen3-VL-8B-Instruct");
    assert!(!single.is_tiered());
}

#[test]
fn keyframe_work_has_prompt_field() {
    use vidarax_core::webrtc::workers::KeyframeWork;

    let kw = KeyframeWork {
        run_id: "r".into(),
        session_id: "s".into(),
        frame_index: 0,
        pts_ms: 0,
        event_type: "scene_cut".into(),
        confidence: 0.9,
        jpeg_bytes: Arc::from([] as [u8; 0]),
        prompt: "Describe this frame.".into(),
    };
    assert_eq!(kw.prompt, "Describe this frame.");
}

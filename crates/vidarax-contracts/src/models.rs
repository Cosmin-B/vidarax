pub const REQUIRED_MEDIUM_MODELS: &[&str] = &["Qwen/Qwen3-VL-8B-Instruct", "allenai/Molmo2-8B"];

pub const REQUIRED_SMALL_MODELS: &[&str] = &[
    "Qwen/Qwen3-VL-4B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "openbmb/MiniCPM-V-4_5",
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
];

pub const REQUIRED_MODELS: &[&str] = &[
    "Qwen/Qwen3-VL-8B-Instruct",
    "allenai/Molmo2-8B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "openbmb/MiniCPM-V-4_5",
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
];

pub fn normalize_model_id(input: &str) -> Option<&'static str> {
    match input.to_ascii_lowercase().as_str() {
        "qwen/qwen3-vl-8b-instruct" => Some("Qwen/Qwen3-VL-8B-Instruct"),
        "allenai/molmo2-8b" => Some("allenai/Molmo2-8B"),
        "qwen/qwen3-vl-4b-instruct" => Some("Qwen/Qwen3-VL-4B-Instruct"),
        "opengvlab/internvl3_5-4b" | "opengvlab/internvl3.5-4b" => Some("OpenGVLab/InternVL3_5-4B"),
        "qwen/qwen3-vl-2b-instruct" => Some("Qwen/Qwen3-VL-2B-Instruct"),
        "openbmb/minicpm-v-4_5" | "openbmb/minicpm-v-4.5" => Some("openbmb/MiniCPM-V-4_5"),
        "liquidai/lfm2-vl-450m" => Some("LiquidAI/LFM2-VL-450M"),
        "liquidai/lfm2.5-vl-1.6b" => Some("LiquidAI/LFM2.5-VL-1.6B"),
        _ => None,
    }
}

pub fn is_required_model(input: &str) -> bool {
    normalize_model_id(input).is_some()
}

pub fn fallback_candidates(requested: &str) -> Vec<&'static str> {
    let Some(canonical) = normalize_model_id(requested) else {
        return REQUIRED_MODELS.to_vec();
    };

    match canonical {
        "Qwen/Qwen3-VL-8B-Instruct" | "allenai/Molmo2-8B" => REQUIRED_MEDIUM_MODELS.to_vec(),
        _ => REQUIRED_SMALL_MODELS.to_vec(),
    }
}

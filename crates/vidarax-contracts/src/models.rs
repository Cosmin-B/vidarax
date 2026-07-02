pub const REQUIRED_MEDIUM_MODELS: &[&str] = &[
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "allenai/Molmo2-8B",
];

pub const REQUIRED_SMALL_MODELS: &[&str] = &[
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "openbmb/MiniCPM-V-4_5",
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
];

pub const REQUIRED_MODELS: &[&str] = &[
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "allenai/Molmo2-8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "openbmb/MiniCPM-V-4_5",
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
];

/// Canonical Gemini model IDs recognised by Vidarax.
pub const GEMINI_MODELS: &[&str] = &[
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
];

pub fn normalize_model_id(input: &str) -> Option<&'static str> {
    if input.len() > 64 {
        return None;
    }
    let mut buf = [0u8; 64];
    let bytes = input.as_bytes();
    buf[..bytes.len()].copy_from_slice(bytes);
    let lower = &mut buf[..bytes.len()];
    lower.make_ascii_lowercase();
    // make_ascii_lowercase only modifies ASCII bytes, so UTF-8 validity is preserved.
    match std::str::from_utf8(lower).unwrap_or("") {
        "qwen/qwen3.5-35b-a3b-fp8" => Some("Qwen/Qwen3.5-35B-A3B-FP8"),
        "qwen/qwen3.5-9b" => Some("Qwen/Qwen3.5-9B"),
        "qwen/qwen3-vl-8b-instruct" => Some("Qwen/Qwen3-VL-8B-Instruct"),
        "allenai/molmo2-8b" => Some("allenai/Molmo2-8B"),
        "qwen/qwen3.5-4b" => Some("Qwen/Qwen3.5-4B"),
        "qwen/qwen3-vl-4b-instruct" => Some("Qwen/Qwen3-VL-4B-Instruct"),
        "opengvlab/internvl3_5-4b" | "opengvlab/internvl3.5-4b" => Some("OpenGVLab/InternVL3_5-4B"),
        "qwen/qwen3-vl-2b-instruct" => Some("Qwen/Qwen3-VL-2B-Instruct"),
        "openbmb/minicpm-v-4_5" | "openbmb/minicpm-v-4.5" => Some("openbmb/MiniCPM-V-4_5"),
        "liquidai/lfm2-vl-450m" => Some("LiquidAI/LFM2-VL-450M"),
        "liquidai/lfm2.5-vl-1.6b" | "lfm2.5-vl-1.6b-q4_0.gguf" => Some("LiquidAI/LFM2.5-VL-1.6B"),
        // Gemini cloud models — short aliases map to their canonical preview IDs.
        "gemini-2.5-flash" | "gemini-2.5-flash-preview-05-20" => {
            Some("gemini-2.5-flash-preview-05-20")
        }
        "gemini-2.5-pro" | "gemini-2.5-pro-preview-05-06" => Some("gemini-2.5-pro-preview-05-06"),
        "gemini-2.0-flash" => Some("gemini-2.0-flash"),
        "gemini-2.0-flash-lite" => Some("gemini-2.0-flash-lite"),
        "gemini-3.1-flash-lite-preview" | "gemini-3.1-flash-lite" => {
            Some("gemini-3.1-flash-lite-preview")
        }
        "gemini-3-flash-preview" | "gemini-3-flash" => Some("gemini-3-flash-preview"),
        _ => None,
    }
}

pub fn is_required_model(input: &str) -> bool {
    normalize_model_id(input).is_some()
}

pub fn fallback_candidates(requested: &str) -> &'static [&'static str] {
    let Some(canonical) = normalize_model_id(requested) else {
        return REQUIRED_MODELS;
    };

    match canonical {
        "Qwen/Qwen3.5-35B-A3B-FP8"
        | "Qwen/Qwen3.5-9B"
        | "Qwen/Qwen3-VL-8B-Instruct"
        | "allenai/Molmo2-8B" => REQUIRED_MEDIUM_MODELS,
        _ => REQUIRED_SMALL_MODELS,
    }
}

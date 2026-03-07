use vidarax_contracts::models::normalize_model_id;
use vidarax_contracts::processing::ProcessingMode;

pub fn normalize_mode(mode: Option<String>) -> Result<&'static str, String> {
    let mode = mode.unwrap_or_else(|| "balanced".to_string());
    ProcessingMode::parse(&mode)
        .map(ProcessingMode::as_str)
        .ok_or_else(|| "mode must be one of: balanced, detailed, efficiency, custom".to_string())
}

pub fn normalize_model(model: Option<String>) -> Result<Option<&'static str>, String> {
    match model {
        Some(model) => normalize_model_id(&model)
            .map(Some)
            .ok_or_else(|| "model is not in the supported model contract".to_string()),
        None => Ok(None),
    }
}

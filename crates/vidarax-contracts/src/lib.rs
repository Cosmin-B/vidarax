#![forbid(unsafe_code)]

pub mod errors;
pub mod lifecycle;
pub mod models;
pub mod processing;
pub mod triggers;

#[cfg(test)]
mod tests {
    use crate::errors::classify_status_code;
    use crate::lifecycle::StreamState;
    use crate::models::{is_required_model, normalize_model_id};
    use crate::processing::{
        defaults_for_mode, validate_processing_config, ProcessingConfig, ProcessingMode,
    };

    #[test]
    fn normalizes_aliases() {
        assert_eq!(
            normalize_model_id("OpenGVLab/InternVL3.5-4B"),
            Some("OpenGVLab/InternVL3_5-4B")
        );
        assert!(is_required_model("openbmb/MiniCPM-V-4.5"));
    }

    #[test]
    fn stream_state_terminal() {
        assert!(!StreamState::Processing.is_terminal());
        assert!(StreamState::Completed.is_terminal());
    }

    #[test]
    fn retry_classification() {
        assert!(classify_status_code(503).is_retryable());
        assert!(!classify_status_code(422).is_retryable());
    }

    #[test]
    fn required_models_is_medium_plus_small() {
        use crate::models::{REQUIRED_MEDIUM_MODELS, REQUIRED_MODELS, REQUIRED_SMALL_MODELS};
        let combined: Vec<&str> = REQUIRED_MEDIUM_MODELS
            .iter()
            .chain(REQUIRED_SMALL_MODELS.iter())
            .copied()
            .collect();
        assert_eq!(REQUIRED_MODELS, combined.as_slice());
    }

    #[test]
    fn gemini_models_resolve_through_catalog() {
        use crate::models::{normalize_model_id, GEMINI_MODELS};
        for id in GEMINI_MODELS {
            assert_eq!(normalize_model_id(id), Some(*id));
        }
        assert_eq!(
            normalize_model_id("gemini-flash-lite-latest"),
            Some("gemini-3.5-flash-lite")
        );
        assert_eq!(
            normalize_model_id("gemini-3.1-flash-lite-preview"),
            Some("gemini-3.1-flash-lite")
        );
    }

    #[test]
    fn mode_defaults_validate() {
        let cfg = defaults_for_mode(ProcessingMode::Balanced);
        assert!(validate_processing_config(&cfg).is_ok());
    }

    #[test]
    fn rejects_out_of_range_processing_values() {
        let cfg = ProcessingConfig {
            fps: 10.0,
            ..defaults_for_mode(ProcessingMode::Custom)
        };
        assert_eq!(
            validate_processing_config(&cfg).unwrap_err(),
            "fps must be in [0.2, 4.0]"
        );
    }
}

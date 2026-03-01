#![forbid(unsafe_code)]

pub mod errors;
pub mod lifecycle;
pub mod models;
pub mod processing;

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

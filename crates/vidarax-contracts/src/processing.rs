#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    Balanced,
    Detailed,
    Efficiency,
    Custom,
}

impl ProcessingMode {
    pub fn parse(input: &str) -> Option<Self> {
        match input.to_ascii_lowercase().as_str() {
            "balanced" => Some(Self::Balanced),
            "detailed" => Some(Self::Detailed),
            "efficiency" => Some(Self::Efficiency),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Detailed => "detailed",
            Self::Efficiency => "efficiency",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessingConfig {
    pub mode: ProcessingMode,
    pub fps: f32,
    pub segment_length_secs: u32,
    pub max_summary_length: u32,
    pub min_confidence: f32,
    pub sampling_rate: u32,
    pub allow_fallback: bool,
    pub keep_alive_minutes: u32,
}

pub fn defaults_for_mode(mode: ProcessingMode) -> ProcessingConfig {
    match mode {
        ProcessingMode::Balanced => ProcessingConfig {
            mode,
            fps: 1.0,
            segment_length_secs: 60,
            max_summary_length: 250,
            min_confidence: 0.35,
            sampling_rate: 4,
            allow_fallback: true,
            keep_alive_minutes: 35,
        },
        ProcessingMode::Detailed => ProcessingConfig {
            mode,
            fps: 2.5,
            segment_length_secs: 45,
            max_summary_length: 400,
            min_confidence: 0.3,
            sampling_rate: 8,
            allow_fallback: true,
            keep_alive_minutes: 35,
        },
        ProcessingMode::Efficiency => ProcessingConfig {
            mode,
            fps: 0.5,
            segment_length_secs: 90,
            max_summary_length: 150,
            min_confidence: 0.45,
            sampling_rate: 2,
            allow_fallback: false,
            keep_alive_minutes: 35,
        },
        ProcessingMode::Custom => ProcessingConfig {
            mode,
            fps: 1.0,
            segment_length_secs: 60,
            max_summary_length: 250,
            min_confidence: 0.35,
            sampling_rate: 4,
            allow_fallback: false,
            keep_alive_minutes: 35,
        },
    }
}

pub fn validate_processing_config(config: &ProcessingConfig) -> Result<(), &'static str> {
    if !config.fps.is_finite() {
        return Err("fps must be finite");
    }
    if config.fps < 0.2 || config.fps > 4.0 {
        return Err("fps must be in [0.2, 4.0]");
    }
    if config.segment_length_secs < 5 || config.segment_length_secs > 600 {
        return Err("segment_length_secs must be in [5, 600]");
    }
    if config.max_summary_length < 10 || config.max_summary_length > 500 {
        return Err("max_summary_length must be in [10, 500]");
    }
    if !config.min_confidence.is_finite() {
        return Err("min_confidence must be finite");
    }
    if config.min_confidence < 0.0 || config.min_confidence > 1.0 {
        return Err("min_confidence must be in [0.0, 1.0]");
    }
    if config.sampling_rate < 1 || config.sampling_rate > 30 {
        return Err("sampling_rate must be in [1, 30]");
    }
    if config.keep_alive_minutes < 30 || config.keep_alive_minutes > 45 {
        return Err("keep_alive_minutes must be in [30, 45]");
    }
    Ok(())
}

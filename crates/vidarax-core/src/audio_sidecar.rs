//! Binary client for local audio perception and speech generation.
//!
//! Encoded media never enters JSON. Analysis requests carry a bounded mono WAV
//! payload, responses carry MessagePack metadata, and synthesized audio remains
//! a separate binary WAV payload.

use std::fmt::{Display, Formatter};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub const MAX_AUDIO_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_FEEDBACK_TEXT_BYTES: usize = 16 * 1024;
pub const MAX_AUDIO_METADATA_BYTES: usize = 512 * 1024;
pub const MAX_SYNTHESIZED_AUDIO_BYTES: usize = 16 * 1024 * 1024;

const PROTOCOL_VERSION: u8 = 1;
const REQUEST_MAGIC: [u8; 4] = *b"VXAU";
const RESPONSE_MAGIC: [u8; 4] = *b"VXAR";
const REQUEST_HEADER_BYTES: usize = 32;
const RESPONSE_HEADER_BYTES: usize = 16;
const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum AudioProfile {
    General = 0,
    Gameplay = 1,
    ScreenRecording = 2,
    PhysicalWorld = 3,
}

impl AudioProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Gameplay => "gameplay",
            Self::ScreenRecording => "screen_recording",
            Self::PhysicalWorld => "physical_world",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SpeechEngine {
    None = 0,
    Auto = 1,
    SenseVoice = 2,
    Moonshine = 3,
    Qwen3Asr = 4,
    Lfm25Audio = 5,
}

impl SpeechEngine {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::SenseVoice => "sensevoice",
            Self::Moonshine => "moonshine",
            Self::Qwen3Asr => "qwen3_asr",
            Self::Lfm25Audio => "lfm2_5_audio",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AudioObservation {
    pub start_offset_ms: u64,
    pub end_offset_ms: u64,
    pub kind: String,
    pub label: String,
    pub confidence: f32,
    pub model: String,
    #[serde(default)]
    pub transcript: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub emotion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AudioAnalysis {
    pub profile: AudioProfile,
    pub speech_engine: SpeechEngine,
    pub models: Vec<String>,
    pub observations: Vec<AudioObservation>,
    #[serde(default)]
    pub audio_duration_ms: u64,
    #[serde(default)]
    pub audio_bytes: u64,
    #[serde(default)]
    pub queue_wait_ms: u64,
    #[serde(default)]
    pub stages: AudioStageTimings,
    #[serde(default)]
    pub capacity: SidecarCapacity,
    pub processing_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AudioStageTimings {
    #[serde(default)]
    pub decode_ms: u64,
    #[serde(default)]
    pub vad_ms: u64,
    #[serde(default)]
    pub classifier_ms: u64,
    #[serde(default)]
    pub asr_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SidecarCapacity {
    #[serde(default)]
    pub active: u64,
    #[serde(default)]
    pub queued: u64,
    #[serde(default)]
    pub limit: u64,
}

#[derive(Debug, Clone)]
pub struct AudioAnalysisRequest<'a> {
    pub profile: AudioProfile,
    pub speech_engine: SpeechEngine,
    pub source_start_ms: u64,
    pub min_confidence: f32,
    pub max_events: u16,
    pub wav: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    pub bytes: Vec<u8>,
    pub media_type: &'static str,
    pub sample_rate_hz: u32,
    pub processing_ms: u64,
    pub model: String,
    pub capacity: SidecarCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFailureReason {
    Decode,
    Timeout,
    Overloaded,
    ModelLoad,
    MalformedResponse,
    Reconnect,
    Inference,
}

impl AudioFailureReason {
    pub const ALL: [Self; 7] = [
        Self::Decode,
        Self::Timeout,
        Self::Overloaded,
        Self::ModelLoad,
        Self::MalformedResponse,
        Self::Reconnect,
        Self::Inference,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Timeout => "timeout",
            Self::Overloaded => "overloaded",
            Self::ModelLoad => "model_load",
            Self::MalformedResponse => "malformed_response",
            Self::Reconnect => "reconnect",
            Self::Inference => "inference",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::Decode => 0,
            Self::Timeout => 1,
            Self::Overloaded => 2,
            Self::ModelLoad => 3,
            Self::MalformedResponse => 4,
            Self::Reconnect => 5,
            Self::Inference => 6,
        }
    }
}

#[derive(Debug)]
pub enum AudioSidecarError {
    InvalidAddress(String),
    AudioTooLarge(usize),
    TextTooLarge(usize),
    BackingOff,
    ReconnectFailed(std::io::Error),
    Io(std::io::Error),
    Protocol(&'static str),
    Metadata(String),
    SidecarStatus {
        status: u8,
        reason: AudioFailureReason,
        message: String,
        capacity: SidecarCapacity,
    },
}

impl AudioSidecarError {
    pub fn reason(&self) -> AudioFailureReason {
        match self {
            Self::AudioTooLarge(_) | Self::TextTooLarge(_) | Self::InvalidAddress(_) => {
                AudioFailureReason::MalformedResponse
            }
            Self::BackingOff | Self::ReconnectFailed(_) => AudioFailureReason::Reconnect,
            Self::Io(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                AudioFailureReason::Timeout
            }
            Self::Io(_) => AudioFailureReason::Reconnect,
            Self::Protocol(_) | Self::Metadata(_) => AudioFailureReason::MalformedResponse,
            Self::SidecarStatus { reason, .. } => *reason,
        }
    }

    pub fn capacity(&self) -> Option<SidecarCapacity> {
        match self {
            Self::SidecarStatus { capacity, .. } => Some(*capacity),
            _ => None,
        }
    }
}

impl Display for AudioSidecarError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAddress(address) => {
                write!(f, "audio sidecar address has no TCP endpoint: {address}")
            }
            Self::AudioTooLarge(bytes) => {
                write!(
                    f,
                    "audio payload is {bytes} bytes; maximum is {MAX_AUDIO_BYTES}"
                )
            }
            Self::TextTooLarge(bytes) => write!(
                f,
                "feedback text is {bytes} bytes; maximum is {MAX_FEEDBACK_TEXT_BYTES}"
            ),
            Self::BackingOff => f.write_str("audio sidecar reconnect backoff is active"),
            Self::ReconnectFailed(error) => write!(f, "audio sidecar reconnect failed: {error}"),
            Self::Io(error) => write!(f, "audio sidecar I/O: {error}"),
            Self::Protocol(message) => write!(f, "audio sidecar protocol: {message}"),
            Self::Metadata(message) => write!(f, "audio sidecar metadata: {message}"),
            Self::SidecarStatus {
                status, message, ..
            } => {
                write!(f, "audio sidecar returned status {status}: {message}")
            }
        }
    }
}

impl std::error::Error for AudioSidecarError {}

impl From<std::io::Error> for AudioSidecarError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Operation {
    Analyze = 1,
    Synthesize = 2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SynthesisMetadata {
    model: String,
    sample_rate_hz: u32,
    processing_ms: u64,
    #[serde(default)]
    capacity: SidecarCapacity,
}

#[derive(Debug, Deserialize)]
struct ErrorMetadata {
    reason: String,
    message: String,
    #[serde(default)]
    capacity: SidecarCapacity,
}

/// Persistent connection with bounded payloads and reconnect backoff.
pub struct AudioSidecarClient {
    address: SocketAddr,
    timeout: Duration,
    stream: Option<TcpStream>,
    consecutive_failures: u32,
    retry_after: Option<Instant>,
}

impl AudioSidecarClient {
    pub fn new(address: &str, timeout_ms: u64) -> Result<Self, AudioSidecarError> {
        let address = address.strip_prefix("tcp://").unwrap_or(address);
        let socket = address
            .to_socket_addrs()
            .map_err(AudioSidecarError::Io)?
            .next()
            .ok_or_else(|| AudioSidecarError::InvalidAddress(address.to_string()))?;
        Ok(Self {
            address: socket,
            timeout: Duration::from_millis(timeout_ms.max(1)),
            stream: None,
            consecutive_failures: 0,
            retry_after: None,
        })
    }

    pub fn analyze(
        &mut self,
        request: AudioAnalysisRequest<'_>,
    ) -> Result<AudioAnalysis, AudioSidecarError> {
        if request.wav.is_empty() || request.wav.len() > MAX_AUDIO_BYTES {
            return Err(AudioSidecarError::AudioTooLarge(request.wav.len()));
        }
        let confidence = request.min_confidence.clamp(0.0, 1.0);
        let threshold = (confidence * 10_000.0).round() as u16;
        let (metadata, audio) = self.exchange(
            Operation::Analyze,
            request.profile,
            request.speech_engine,
            0,
            request.source_start_ms,
            threshold,
            request.max_events.max(1),
            request.wav,
            &[],
        )?;
        if !audio.is_empty() {
            return Err(AudioSidecarError::Protocol(
                "analysis response included unexpected audio",
            ));
        }
        let analysis: AudioAnalysis = rmp_serde::from_slice(&metadata)
            .map_err(|error| AudioSidecarError::Metadata(error.to_string()))?;
        validate_analysis(&analysis)?;
        Ok(analysis)
    }

    pub fn synthesize(&mut self, text: &str) -> Result<SynthesizedAudio, AudioSidecarError> {
        let text = text.trim();
        if text.is_empty() || text.len() > MAX_FEEDBACK_TEXT_BYTES {
            return Err(AudioSidecarError::TextTooLarge(text.len()));
        }
        let (metadata, audio) = self.exchange(
            Operation::Synthesize,
            AudioProfile::ScreenRecording,
            SpeechEngine::Lfm25Audio,
            0,
            0,
            0,
            1,
            &[],
            text.as_bytes(),
        )?;
        if audio.is_empty() || audio.len() > MAX_SYNTHESIZED_AUDIO_BYTES {
            return Err(AudioSidecarError::Protocol(
                "synthesis response has invalid audio length",
            ));
        }
        let metadata: SynthesisMetadata = rmp_serde::from_slice(&metadata)
            .map_err(|error| AudioSidecarError::Metadata(error.to_string()))?;
        Ok(SynthesizedAudio {
            bytes: audio,
            media_type: "audio/wav",
            sample_rate_hz: metadata.sample_rate_hz,
            processing_ms: metadata.processing_ms,
            model: metadata.model,
            capacity: metadata.capacity,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn exchange(
        &mut self,
        operation: Operation,
        profile: AudioProfile,
        speech_engine: SpeechEngine,
        flags: u16,
        source_start_ms: u64,
        threshold: u16,
        max_events: u16,
        audio: &[u8],
        text: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), AudioSidecarError> {
        if self
            .retry_after
            .is_some_and(|deadline| Instant::now() < deadline)
        {
            return Err(AudioSidecarError::BackingOff);
        }
        let outcome = self.exchange_inner(
            operation,
            profile,
            speech_engine,
            flags,
            source_start_ms,
            threshold,
            max_events,
            audio,
            text,
        );
        match outcome {
            Ok(response) => {
                self.consecutive_failures = 0;
                self.retry_after = None;
                Ok(response)
            }
            Err(error) => {
                self.stream = None;
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                let shift = self.consecutive_failures.saturating_sub(1).min(6);
                let backoff_ms = INITIAL_BACKOFF_MS
                    .saturating_mul(1_u64 << shift)
                    .min(MAX_BACKOFF_MS);
                self.retry_after = Some(Instant::now() + Duration::from_millis(backoff_ms));
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exchange_inner(
        &mut self,
        operation: Operation,
        profile: AudioProfile,
        speech_engine: SpeechEngine,
        flags: u16,
        source_start_ms: u64,
        threshold: u16,
        max_events: u16,
        audio: &[u8],
        text: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), AudioSidecarError> {
        let audio_len = u32::try_from(audio.len())
            .map_err(|_| AudioSidecarError::AudioTooLarge(audio.len()))?;
        let text_len =
            u32::try_from(text.len()).map_err(|_| AudioSidecarError::TextTooLarge(text.len()))?;
        let stream = self.connection()?;
        let mut header = [0_u8; REQUEST_HEADER_BYTES];
        header[..4].copy_from_slice(&REQUEST_MAGIC);
        header[4] = PROTOCOL_VERSION;
        header[5] = operation as u8;
        header[6] = profile as u8;
        header[7] = speech_engine as u8;
        header[8..10].copy_from_slice(&flags.to_be_bytes());
        header[12..20].copy_from_slice(&source_start_ms.to_be_bytes());
        header[20..24].copy_from_slice(&audio_len.to_be_bytes());
        header[24..28].copy_from_slice(&text_len.to_be_bytes());
        header[28..30].copy_from_slice(&max_events.to_be_bytes());
        header[30..32].copy_from_slice(&threshold.to_be_bytes());
        stream.write_all(&header)?;
        stream.write_all(audio)?;
        stream.write_all(text)?;

        let mut response = [0_u8; RESPONSE_HEADER_BYTES];
        stream.read_exact(&mut response)?;
        if response[..4] != RESPONSE_MAGIC {
            return Err(AudioSidecarError::Protocol("bad response magic"));
        }
        if response[4] != PROTOCOL_VERSION {
            return Err(AudioSidecarError::Protocol("unsupported version"));
        }
        let status = response[5];
        let metadata_len =
            u32::from_be_bytes(response[8..12].try_into().expect("4 bytes")) as usize;
        let response_audio_len =
            u32::from_be_bytes(response[12..16].try_into().expect("4 bytes")) as usize;
        if metadata_len > MAX_AUDIO_METADATA_BYTES {
            return Err(AudioSidecarError::Protocol("metadata exceeds limit"));
        }
        if response_audio_len > MAX_SYNTHESIZED_AUDIO_BYTES {
            return Err(AudioSidecarError::Protocol("audio response exceeds limit"));
        }
        let mut metadata = vec![0_u8; metadata_len];
        stream.read_exact(&mut metadata)?;
        let mut response_audio = vec![0_u8; response_audio_len];
        stream.read_exact(&mut response_audio)?;
        if status != 0 {
            let decoded = rmp_serde::from_slice::<ErrorMetadata>(&metadata).ok();
            let message = decoded
                .as_ref()
                .map(|error| error.message.clone())
                .unwrap_or_else(|| String::from_utf8_lossy(&metadata).into_owned());
            let reason = decoded
                .as_ref()
                .map(|error| parse_failure_reason(&error.reason))
                .unwrap_or_else(|| {
                    if status == 3 {
                        AudioFailureReason::Overloaded
                    } else {
                        AudioFailureReason::Inference
                    }
                });
            let capacity = decoded.map_or_else(SidecarCapacity::default, |error| error.capacity);
            return Err(AudioSidecarError::SidecarStatus {
                status,
                reason,
                message,
                capacity,
            });
        }
        Ok((metadata, response_audio))
    }

    fn connection(&mut self) -> Result<&mut TcpStream, AudioSidecarError> {
        if self.stream.is_none() {
            let reconnecting = self.consecutive_failures > 0;
            let stream =
                TcpStream::connect_timeout(&self.address, self.timeout).map_err(|error| {
                    if reconnecting {
                        AudioSidecarError::ReconnectFailed(error)
                    } else {
                        AudioSidecarError::Io(error)
                    }
                })?;
            stream.set_read_timeout(Some(self.timeout))?;
            stream.set_write_timeout(Some(self.timeout))?;
            stream.set_nodelay(true)?;
            self.stream = Some(stream);
        }
        self.stream
            .as_mut()
            .ok_or(AudioSidecarError::Protocol("missing connection"))
    }
}

fn parse_failure_reason(value: &str) -> AudioFailureReason {
    match value {
        "decode" => AudioFailureReason::Decode,
        "timeout" => AudioFailureReason::Timeout,
        "overloaded" => AudioFailureReason::Overloaded,
        "model_load" => AudioFailureReason::ModelLoad,
        "malformed_response" => AudioFailureReason::MalformedResponse,
        "reconnect" => AudioFailureReason::Reconnect,
        _ => AudioFailureReason::Inference,
    }
}

fn validate_analysis(analysis: &AudioAnalysis) -> Result<(), AudioSidecarError> {
    if analysis.models.len() > 16 || analysis.observations.len() > 64 {
        return Err(AudioSidecarError::Metadata(
            "collection exceeds protocol limit".to_string(),
        ));
    }
    for model in &analysis.models {
        validate_metadata_text(model, 256, "model")?;
    }
    for observation in &analysis.observations {
        if observation.end_offset_ms < observation.start_offset_ms
            || observation.end_offset_ms > 60_000
            || !observation.confidence.is_finite()
            || !(0.0..=1.0).contains(&observation.confidence)
        {
            return Err(AudioSidecarError::Metadata(
                "observation has invalid time or confidence".to_string(),
            ));
        }
        validate_metadata_text(&observation.kind, 128, "observation kind")?;
        validate_metadata_text(&observation.label, 128, "observation label")?;
        validate_metadata_text(&observation.model, 256, "observation model")?;
        if let Some(transcript) = &observation.transcript {
            validate_metadata_text(transcript, MAX_FEEDBACK_TEXT_BYTES, "transcript")?;
        }
        if let Some(language) = &observation.language {
            validate_metadata_text(language, 64, "language")?;
        }
        if let Some(emotion) = &observation.emotion {
            validate_metadata_text(emotion, 64, "emotion")?;
        }
    }
    Ok(())
}

fn validate_metadata_text(
    value: &str,
    max_bytes: usize,
    field: &str,
) -> Result<(), AudioSidecarError> {
    if value.is_empty() || value.len() > max_bytes {
        return Err(AudioSidecarError::Metadata(format!(
            "{field} must be 1..={max_bytes} bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        validate_analysis, AudioAnalysis, AudioFailureReason, AudioObservation, AudioProfile,
        AudioSidecarError, SidecarCapacity, SpeechEngine,
    };

    #[test]
    fn wire_enums_have_stable_values() {
        assert_eq!(AudioProfile::Gameplay as u8, 1);
        assert_eq!(AudioProfile::ScreenRecording.as_str(), "screen_recording");
        assert_eq!(SpeechEngine::SenseVoice as u8, 2);
        assert_eq!(SpeechEngine::Lfm25Audio.as_str(), "lfm2_5_audio");
    }

    #[test]
    fn rejects_untrusted_non_finite_metadata() {
        let analysis = AudioAnalysis {
            profile: AudioProfile::Gameplay,
            speech_engine: SpeechEngine::None,
            models: vec!["efficientat/mn10_as".to_string()],
            observations: vec![AudioObservation {
                start_offset_ms: 0,
                end_offset_ms: 1_000,
                kind: "audio_event".to_string(),
                label: "explosion".to_string(),
                confidence: f32::NAN,
                model: "efficientat/mn10_as".to_string(),
                transcript: None,
                language: None,
                emotion: None,
            }],
            audio_duration_ms: 1_000,
            audio_bytes: 32_044,
            queue_wait_ms: 0,
            stages: Default::default(),
            capacity: Default::default(),
            processing_ms: 1,
        };
        assert!(matches!(
            validate_analysis(&analysis),
            Err(AudioSidecarError::Metadata(_))
        ));
    }

    #[test]
    fn sidecar_errors_keep_fixed_reason_and_capacity() {
        let error = AudioSidecarError::SidecarStatus {
            status: 3,
            reason: AudioFailureReason::Overloaded,
            message: "full".to_string(),
            capacity: SidecarCapacity {
                active: 2,
                queued: 8,
                limit: 2,
            },
        };
        assert_eq!(error.reason(), AudioFailureReason::Overloaded);
        assert_eq!(error.capacity().expect("capacity").queued, 8);
    }
}

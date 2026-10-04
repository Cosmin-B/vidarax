//! Bounded local analysis for inbound WebRTC Opus audio.
//!
//! Encoded access units stay binary from RTP ingress through Ogg framing.
//! ffmpeg converts each bounded window to mono PCM WAV, then the local audio
//! sidecar produces timestamped observations. Only event metadata reaches the
//! WAL.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use tokio::sync::mpsc;
use tracing::Instrument;
use vidarax_core::audio_sidecar::{
    AudioAnalysis, AudioAnalysisRequest, AudioFailureReason, AudioSidecarClient,
};
use vidarax_core::metrics::PipelineMetrics;
use vidarax_core::webrtc::session::{LiveAudioConfig, LiveAudioFrame};

use crate::state::AppState;

const LIVE_AUDIO_WORK_QUEUE_CAPACITY: usize = 2;
const MIN_LIVE_AUDIO_WINDOW_MS: u64 = 250;
const OPUS_PRE_SKIP: u16 = 312;
const OPUS_CLOCK_RATE: u32 = 48_000;
// RFC 7587 assigns one Opus packet to each RTP payload. The minimum packet
// duration is 2.5 ms, which advances the 48 kHz clock by 120 ticks. A window
// therefore retains at most ceil(window_ms * 48 / 120) + 1 packets: 1601 at
// four seconds. Each access unit also has the ingress byte limit.
const OPUS_MIN_TIMESTAMP_ADVANCE: u32 = OPUS_CLOCK_RATE / 400;

struct TrackAccumulator {
    anchor_timestamp: u32,
    window_start_timestamp: u32,
    clock_rate: u32,
    last_sequence_number: Option<u16>,
    last_timestamp: u32,
    frames: Vec<LiveAudioFrame>,
}

impl TrackAccumulator {
    fn new(frame: LiveAudioFrame) -> Self {
        let timestamp = frame.rtp_timestamp;
        let clock_rate = frame.clock_rate;
        Self {
            anchor_timestamp: timestamp,
            window_start_timestamp: timestamp,
            clock_rate,
            last_sequence_number: frame.sequence_number,
            last_timestamp: timestamp,
            frames: vec![frame],
        }
    }

    fn push(&mut self, frame: LiveAudioFrame) {
        if self.frames.is_empty() {
            self.window_start_timestamp = frame.rtp_timestamp;
        }
        self.last_sequence_number = frame.sequence_number;
        self.last_timestamp = frame.rtp_timestamp;
        self.frames.push(frame);
    }

    fn accepts(&self, frame: &LiveAudioFrame) -> bool {
        let advance = frame.rtp_timestamp.wrapping_sub(self.last_timestamp);
        self.clock_rate == frame.clock_rate
            && (OPUS_MIN_TIMESTAMP_ADVANCE..=i32::MAX as u32).contains(&advance)
            && match (self.last_sequence_number, frame.sequence_number) {
                (Some(previous), Some(next)) => previous.wrapping_add(1) == next,
                _ => true,
            }
    }

    fn duration_ms(&self) -> u64 {
        let Some(last) = self.frames.last() else {
            return 0;
        };
        if self.clock_rate == 0 {
            return 0;
        }
        u64::from(last.rtp_timestamp.wrapping_sub(self.window_start_timestamp))
            .saturating_mul(1_000)
            / u64::from(self.clock_rate)
    }

    fn source_start_ms(&self) -> u64 {
        if self.clock_rate == 0 {
            return 0;
        }
        u64::from(
            self.window_start_timestamp
                .wrapping_sub(self.anchor_timestamp),
        )
        .saturating_mul(1_000)
            / u64::from(self.clock_rate)
    }

    fn take_window(&mut self) -> LiveAudioWindow {
        let frames = std::mem::take(&mut self.frames);
        let source_start_ms = self.source_start_ms();
        let track_id = frames.first().map_or(0, |frame| frame.track_id);
        LiveAudioWindow {
            track_id,
            source_start_ms,
            frames,
        }
    }
}

struct LiveAudioWindow {
    track_id: u64,
    source_start_ms: u64,
    frames: Vec<LiveAudioFrame>,
}

pub(crate) struct LiveAudioPipeline {
    pub state: AppState,
    pub run_id: Arc<str>,
    pub session_id: Arc<str>,
    pub analysis: LiveAudioConfig,
    pub sidecar_addr: Arc<str>,
    pub sidecar_timeout_ms: u64,
    pub metrics: Arc<PipelineMetrics>,
}

pub(crate) fn spawn_live_audio_pipeline(
    pipeline: LiveAudioPipeline,
    mut audio_rx: mpsc::Receiver<LiveAudioFrame>,
) {
    let pipeline = Arc::new(pipeline);
    let (work_tx, mut work_rx) = mpsc::channel::<LiveAudioWindow>(LIVE_AUDIO_WORK_QUEUE_CAPACITY);
    let aggregation_metrics = Arc::clone(&pipeline.metrics);
    let aggregation_config = pipeline.analysis;
    tokio::spawn(async move {
        let mut tracks: HashMap<u64, TrackAccumulator> = HashMap::new();
        while let Some(frame) = audio_rx.recv().await {
            if frame.clock_rate != OPUS_CLOCK_RATE || frame.data.is_empty() {
                aggregation_metrics.inc_webrtc_audio_queue_drop();
                continue;
            }
            let track_id = frame.track_id;
            let Some(accumulator) = tracks.get_mut(&track_id) else {
                tracks.insert(track_id, TrackAccumulator::new(frame));
                continue;
            };
            if !accumulator.accepts(&frame) {
                let stale = accumulator.take_window();
                if !stale.frames.is_empty()
                    && (stale.frames.len() == 1 || work_tx.try_send(stale).is_err())
                {
                    aggregation_metrics.inc_webrtc_audio_queue_drop();
                }
                *accumulator = TrackAccumulator::new(frame);
                continue;
            }
            accumulator.push(frame);
            if accumulator.duration_ms() >= aggregation_config.window_ms {
                let window = accumulator.take_window();
                if work_tx.try_send(window).is_err() {
                    aggregation_metrics.inc_webrtc_audio_queue_drop();
                }
            }
        }
        for accumulator in tracks.values_mut() {
            if accumulator.duration_ms() >= MIN_LIVE_AUDIO_WINDOW_MS
                && work_tx.try_send(accumulator.take_window()).is_err()
            {
                aggregation_metrics.inc_webrtc_audio_queue_drop();
            }
        }
    });

    tokio::spawn(async move {
        let mut chunk_index = 0usize;
        while let Some(window) = work_rx.recv().await {
            let track_id = window.track_id;
            let request_id = format!("live-audio:{}", pipeline.session_id);
            let span = tracing::info_span!(
                "live_audio_chunk",
                run_id = %pipeline.run_id,
                request_id = %request_id,
                session_id = %pipeline.session_id,
                chunk_index,
                track_id,
            );
            let result = process_window(Arc::clone(&pipeline), window, chunk_index)
                .instrument(span)
                .await;
            if let Err(error) = result {
                tracing::warn!(%error, chunk_index, "live audio window failed");
            }
            chunk_index = chunk_index.saturating_add(1);
        }
    });
}

async fn process_window(
    pipeline: Arc<LiveAudioPipeline>,
    window: LiveAudioWindow,
    chunk_index: usize,
) -> Result<(), String> {
    let source_start_ms = window.source_start_ms;
    let track_id = window.track_id;
    let frame_count = window.frames.len();
    let decode_started = Instant::now();
    let wav = match tokio::task::spawn_blocking(move || opus_window_to_wav(&window.frames)).await {
        Ok(Ok(wav)) => wav,
        Ok(Err(error)) => {
            pipeline
                .metrics
                .record_local_audio_failure_reason(AudioFailureReason::Decode);
            return Err(error);
        }
        Err(error) => {
            pipeline
                .metrics
                .record_local_audio_failure_reason(AudioFailureReason::Decode);
            return Err(format!("live audio decode worker failed: {error}"));
        }
    };
    let decode_ms = decode_started.elapsed().as_millis() as u64;
    let wav_bytes = wav.len() as u64;

    let request_started = Instant::now();
    let metrics_for_request = Arc::clone(&pipeline.metrics);
    let sidecar_addr = Arc::clone(&pipeline.sidecar_addr);
    let sidecar_timeout_ms = pipeline.sidecar_timeout_ms;
    let config = pipeline.analysis;
    let analysis = match tokio::task::spawn_blocking(move || {
        let _active = metrics_for_request.begin_local_audio_sidecar_request();
        let mut client =
            AudioSidecarClient::new(&sidecar_addr, sidecar_timeout_ms).map_err(|error| {
                metrics_for_request.record_local_audio_failure(&error);
                error.to_string()
            })?;
        client
            .analyze(AudioAnalysisRequest {
                profile: config.profile,
                speech_engine: config.speech_engine,
                source_start_ms,
                min_confidence: config.min_confidence,
                max_events: config.max_events,
                wav: &wav,
            })
            .map_err(|error| {
                metrics_for_request.record_local_audio_failure(&error);
                error.to_string()
            })
    })
    .await
    {
        Ok(Ok(analysis)) => analysis,
        Ok(Err(error)) => return Err(error),
        Err(error) => {
            pipeline
                .metrics
                .record_local_audio_failure_reason(AudioFailureReason::Inference);
            return Err(format!("live audio sidecar worker failed: {error}"));
        }
    };
    let round_trip_ms = request_started.elapsed().as_millis() as u64;
    pipeline.metrics.record_local_audio_extraction(
        wav_bytes,
        analysis.audio_duration_ms,
        decode_ms,
    );
    pipeline
        .metrics
        .record_local_audio_analysis(&analysis, decode_ms.saturating_add(round_trip_ms));

    append_live_audio_events(
        &pipeline.state,
        &pipeline.run_id,
        &pipeline.session_id,
        track_id,
        chunk_index,
        source_start_ms,
        frame_count,
        decode_ms,
        round_trip_ms,
        &analysis,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn append_live_audio_events(
    state: &AppState,
    run_id: &str,
    session_id: &str,
    track_id: u64,
    chunk_index: usize,
    source_start_ms: u64,
    frame_count: usize,
    decode_ms: u64,
    round_trip_ms: u64,
    analysis: &AudioAnalysis,
) -> Result<(), String> {
    state
        .append_run_event_async(
            run_id,
            "semantic_chunk_inferred",
            json!({
                "request_id": format!("live-audio:{session_id}"),
                "stream_id": "default",
                "session_id": session_id,
                "chunk_index": chunk_index,
                "track_id": track_id,
                "provider": "local_audio",
                "media_mode": "live_audio",
                "pts_start_ms": source_start_ms,
                "pts_end_ms": source_start_ms.saturating_add(analysis.audio_duration_ms),
                "timestamp_resolution_ms": 1,
                "rtp_access_units": frame_count,
                "decode_ms": decode_ms,
                "round_trip_ms": round_trip_ms,
                "local_audio": analysis,
            }),
        )
        .await?;

    let mut moment_count = 0u64;
    for (moment_index, observation) in analysis
        .observations
        .iter()
        .filter(|observation| {
            observation.kind != "voice_activity"
                && observation.end_offset_ms > observation.start_offset_ms
        })
        .enumerate()
    {
        let kind = match (observation.kind.as_str(), observation.label.as_str()) {
            ("speech", _) => "speech",
            (_, "music") => "music",
            (
                _,
                "silence"
                | "inside_small_room"
                | "outside_urban_or_manmade"
                | "outside_rural_or_natural",
            ) => "ambient",
            (_, "engine" | "engine_starting" | "idling" | "mechanisms" | "tools" | "vehicle") => {
                "mechanical"
            }
            _ => "sound_effect",
        };
        let description = observation
            .transcript
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .map_or_else(
                || observation.label.clone(),
                |text| format!("Speaker: {}", text.trim()),
            );
        state
            .append_run_event_async(
                run_id,
                "multimodal_moment",
                json!({
                    "moment_id": format!("live-audio:{session_id}:{chunk_index}:{moment_index}"),
                    "request_id": format!("live-audio:{session_id}"),
                    "stream_id": "default",
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "track_id": track_id,
                    "start_offset_ms": observation.start_offset_ms,
                    "end_offset_ms": observation.end_offset_ms,
                    "start_pts_ms": source_start_ms.saturating_add(observation.start_offset_ms),
                    "end_pts_ms": source_start_ms.saturating_add(observation.end_offset_ms),
                    "timestamp_resolution_ms": 1,
                    "modalities": ["audio"],
                    "kind": kind,
                    "description": description,
                    "intent": null,
                    "audio_visual_relation": null,
                    "confidence": observation.confidence,
                    "provider": "local_audio",
                    "evidence": null,
                }),
            )
            .await?;
        moment_count = moment_count.saturating_add(1);
    }
    state
        .pipeline_metrics()
        .add_multimodal_moments(moment_count);
    Ok(())
}

fn opus_window_to_wav(frames: &[LiveAudioFrame]) -> Result<Vec<u8>, String> {
    if frames.is_empty() {
        return Err("live audio window is empty".to_string());
    }
    if frames
        .iter()
        .any(|frame| frame.clock_rate != OPUS_CLOCK_RATE)
    {
        return Err("live audio codec clock must be 48000 Hz Opus".to_string());
    }
    let ogg = build_ogg_opus(frames)?;
    let wav = vidarax_core::ingest::decode_ogg_opus_wav(ogg)?;
    if wav.len() < 44 {
        return Err("ffmpeg returned an empty live audio WAV".to_string());
    }
    Ok(wav)
}

fn build_ogg_opus(frames: &[LiveAudioFrame]) -> Result<Vec<u8>, String> {
    let serial = 0x5649_4458;
    let mut output = Vec::new();
    let mut sequence = 0u32;
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1);
    head.push(2);
    head.extend_from_slice(&OPUS_PRE_SKIP.to_le_bytes());
    head.extend_from_slice(&OPUS_CLOCK_RATE.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(0);
    append_ogg_page(&mut output, &head, 0, serial, sequence, 0x02)?;
    sequence = sequence.saturating_add(1);

    let vendor = b"vidarax";
    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes());
    append_ogg_page(&mut output, &tags, 0, serial, sequence, 0)?;
    sequence = sequence.saturating_add(1);

    let first_timestamp = frames[0].rtp_timestamp;
    for (index, frame) in frames.iter().enumerate() {
        if frame.data.is_empty() {
            continue;
        }
        let next_duration = frames
            .get(index + 1)
            .map(|next| next.rtp_timestamp.wrapping_sub(frame.rtp_timestamp))
            .or_else(|| {
                index.checked_sub(1).map(|previous| {
                    frame
                        .rtp_timestamp
                        .wrapping_sub(frames[previous].rtp_timestamp)
                })
            })
            .unwrap_or(960)
            .clamp(120, 5_760);
        let granule = u64::from(OPUS_PRE_SKIP)
            .saturating_add(u64::from(frame.rtp_timestamp.wrapping_sub(first_timestamp)))
            .saturating_add(u64::from(next_duration));
        let flags = if index + 1 == frames.len() { 0x04 } else { 0 };
        append_ogg_page(&mut output, &frame.data, granule, serial, sequence, flags)?;
        sequence = sequence.saturating_add(1);
    }
    Ok(output)
}

fn append_ogg_page(
    output: &mut Vec<u8>,
    packet: &[u8],
    granule: u64,
    serial: u32,
    sequence: u32,
    flags: u8,
) -> Result<(), String> {
    let full_segments = packet.len() / 255;
    let remainder = packet.len() % 255;
    let segment_count = full_segments.saturating_add(1);
    if segment_count > 255 {
        return Err("Opus packet is too large for one Ogg page".to_string());
    }
    let page_start = output.len();
    output.extend_from_slice(b"OggS");
    output.push(0);
    output.push(flags);
    output.extend_from_slice(&granule.to_le_bytes());
    output.extend_from_slice(&serial.to_le_bytes());
    output.extend_from_slice(&sequence.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    output.push(segment_count as u8);
    output.extend(std::iter::repeat_n(255u8, full_segments));
    output.push(remainder as u8);
    output.extend_from_slice(packet);
    let checksum = ogg_crc(&output[page_start..]);
    output[page_start + 22..page_start + 26].copy_from_slice(&checksum.to_le_bytes());
    Ok(())
}

fn ogg_crc(bytes: &[u8]) -> u32 {
    let mut crc = 0u32;
    for byte in bytes {
        crc ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(timestamp: u32, data: &[u8]) -> LiveAudioFrame {
        LiveAudioFrame {
            track_id: 0,
            rtp_timestamp: timestamp,
            clock_rate: OPUS_CLOCK_RATE,
            sequence_number: Some(1),
            data: data.to_vec(),
        }
    }

    #[test]
    fn ogg_opus_has_headers_and_terminal_page() {
        let frames = [frame(0, &[0xf8, 0xff, 0xfe])];
        let ogg = build_ogg_opus(&frames).unwrap();
        assert!(ogg.starts_with(b"OggS"));
        assert!(ogg.windows(8).any(|window| window == b"OpusHead"));
        assert!(ogg.windows(8).any(|window| window == b"OpusTags"));
        assert!(ogg.iter().filter(|byte| **byte == b'O').count() >= 3);
        let wav = opus_window_to_wav(&frames).unwrap();
        assert!(wav.starts_with(b"RIFF"));
    }

    #[test]
    fn accumulator_reports_relative_window_time() {
        let mut accumulator = TrackAccumulator::new(frame(48_000, &[1]));
        accumulator.push(frame(96_000, &[2]));
        assert_eq!(accumulator.duration_ms(), 1_000);
        let first = accumulator.take_window();
        assert_eq!(first.source_start_ms, 0);
        accumulator.push(frame(144_000, &[3]));
        assert_eq!(accumulator.source_start_ms(), 2_000);
    }

    #[test]
    fn accumulator_rejects_repeated_and_short_timestamps() {
        let accumulator = TrackAccumulator::new(frame(48_000, &[1]));
        for timestamp in [48_000, 48_001, 48_119, 47_880] {
            let mut next = frame(timestamp, &[2]);
            next.sequence_number = Some(2);
            assert!(!accumulator.accepts(&next));
        }
        let mut next = frame(48_120, &[2]);
        next.sequence_number = Some(2);
        assert!(accumulator.accepts(&next));
        next.clock_rate = 8_000;
        assert!(!accumulator.accepts(&next));
    }

    #[test]
    fn accumulator_preserves_timestamp_and_sequence_wrap() {
        let mut first = frame(u32::MAX - 59, &[1]);
        first.sequence_number = Some(u16::MAX);
        let mut accumulator = TrackAccumulator::new(first);
        let mut next = frame(60, &[2]);
        next.sequence_number = Some(0);
        assert!(accumulator.accepts(&next));
        accumulator.push(next);
        assert_eq!(accumulator.duration_ms(), 2);
        accumulator.take_window();

        let mut repeated = frame(60, &[3]);
        repeated.sequence_number = Some(1);
        assert!(!accumulator.accepts(&repeated));
        let mut next = frame(180, &[3]);
        next.sequence_number = Some(1);
        assert!(accumulator.accepts(&next));
        next.sequence_number = Some(2);
        assert!(!accumulator.accepts(&next));
    }

    #[test]
    fn accumulator_minimum_opus_packets_reach_four_second_window_bound() {
        let start = u32::MAX - 59;
        let mut accumulator = TrackAccumulator::new(frame(start, &[1]));
        for index in 1..=1600u32 {
            let mut next = frame(start.wrapping_add(index * 120), &[1]);
            next.sequence_number = Some((index + 1) as u16);
            assert!(accumulator.accepts(&next));
            accumulator.push(next);
            if index < 1600 {
                assert!(accumulator.duration_ms() < 4_000);
            }
        }
        assert_eq!(accumulator.duration_ms(), 4_000);
        assert_eq!(accumulator.take_window().frames.len(), 1601);
    }
}

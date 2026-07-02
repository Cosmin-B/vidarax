use std::net::SocketAddr;

use rustrtc::media::depacketizer::{DefaultDepacketizerFactory, Depacketizer, DepacketizerFactory};
use rustrtc::media::frame::{MediaKind, MediaSample, VideoFrame, VideoPixelFormat};
use rustrtc::media::MediaResult;
use rustrtc::rtp::RtpPacket;

use crate::webrtc::decode::VideoCodec;

#[derive(Debug)]
pub struct VidaraxDepacketizerFactory {
    codec: VideoCodec,
}

impl VidaraxDepacketizerFactory {
    pub fn new(codec: VideoCodec) -> Self {
        Self { codec }
    }
}

impl DepacketizerFactory for VidaraxDepacketizerFactory {
    fn create(&self, kind: MediaKind) -> Box<dyn Depacketizer> {
        match (kind, self.codec) {
            (MediaKind::Video, VideoCodec::Vp8) => Box::new(Vp8Depacketizer::new()),
            _ => DefaultDepacketizerFactory.create(kind),
        }
    }
}

pub struct Vp8Depacketizer {
    frame_buf: Vec<u8>,
    frame_timestamp: u32,
    last_seq: Option<u16>,
    dropping_until_start: bool,
}

impl Default for Vp8Depacketizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp8Depacketizer {
    pub fn new() -> Self {
        Self {
            frame_buf: Vec::new(),
            frame_timestamp: 0,
            last_seq: None,
            dropping_until_start: false,
        }
    }

    fn reset_frame(&mut self) {
        self.frame_buf.clear();
        self.last_seq = None;
    }

    fn parse_descriptor(payload: &[u8]) -> Option<Vp8PayloadDescriptor> {
        let first = *payload.first()?;
        let mut offset = 1usize;

        if first & 0x80 != 0 {
            let ext = *payload.get(offset)?;
            offset += 1;

            if ext & 0x80 != 0 {
                let picture_id = *payload.get(offset)?;
                offset += 1;
                if picture_id & 0x80 != 0 {
                    payload.get(offset)?;
                    offset += 1;
                }
            }

            if ext & 0x40 != 0 {
                payload.get(offset)?;
                offset += 1;
            }

            if ext & 0x30 != 0 {
                payload.get(offset)?;
                offset += 1;
            }
        }

        Some(Vp8PayloadDescriptor {
            is_start: first & 0x10 != 0,
            partition_id: first & 0x07,
            payload_offset: offset,
        })
    }

    fn video_sample(&self, data: Vec<u8>, packet: &RtpPacket, addr: SocketAddr) -> MediaSample {
        MediaSample::Video(VideoFrame {
            rtp_timestamp: self.frame_timestamp,
            width: 0,
            height: 0,
            format: VideoPixelFormat::Unspecified,
            rotation_deg: 0,
            is_last_packet: true,
            data: data.into(),
            header_extension: packet.header.extension.clone(),
            csrcs: packet.header.csrcs.clone(),
            sequence_number: Some(packet.header.sequence_number),
            payload_type: Some(packet.header.payload_type),
            source_addr: Some(addr),
            raw_packet: Some(packet.clone()),
        })
    }
}

struct Vp8PayloadDescriptor {
    is_start: bool,
    partition_id: u8,
    payload_offset: usize,
}

impl Depacketizer for Vp8Depacketizer {
    fn push(
        &mut self,
        packet: RtpPacket,
        clock_rate: u32,
        addr: SocketAddr,
        kind: MediaKind,
    ) -> MediaResult<Vec<MediaSample>> {
        if kind == MediaKind::Audio {
            return Ok(vec![MediaSample::from_rtp_packet(
                packet, kind, clock_rate, addr,
            )]);
        }

        let marker = packet.header.marker;
        let seq = packet.header.sequence_number;
        let timestamp = packet.header.timestamp;
        let payload = &packet.payload;
        if payload.is_empty() {
            return Ok(vec![]);
        }

        let descriptor = match Self::parse_descriptor(payload) {
            Some(descriptor) => descriptor,
            None => {
                self.reset_frame();
                self.dropping_until_start = true;
                return Ok(vec![]);
            }
        };

        let is_frame_start = descriptor.is_start && descriptor.partition_id == 0;
        let partition_payload = &payload[descriptor.payload_offset..];

        if is_frame_start {
            if !self.frame_buf.is_empty() && timestamp == self.frame_timestamp {
                self.frame_buf.clear();
                self.last_seq = Some(seq);
                self.dropping_until_start = true;
                return Ok(vec![]);
            }

            self.frame_buf.clear();
            self.frame_timestamp = timestamp;
            self.dropping_until_start = false;
        } else {
            let is_continuous = self
                .last_seq
                .is_some_and(|last_seq| seq == last_seq.wrapping_add(1));
            let same_timestamp = timestamp == self.frame_timestamp;

            if self.dropping_until_start || !is_continuous || !same_timestamp {
                self.frame_buf.clear();
                self.last_seq = Some(seq);
                self.dropping_until_start = true;
                return Ok(vec![]);
            }
        }

        self.last_seq = Some(seq);
        self.frame_buf.extend_from_slice(partition_payload);

        if marker {
            if self.frame_buf.is_empty() || self.dropping_until_start {
                self.reset_frame();
                return Ok(vec![]);
            }

            let data = std::mem::take(&mut self.frame_buf);
            let sample = self.video_sample(data, &packet, addr);
            self.reset_frame();
            return Ok(vec![sample]);
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::{VidaraxDepacketizerFactory, Vp8Depacketizer};
    use crate::webrtc::decode::VideoCodec;
    use rustrtc::media::depacketizer::{Depacketizer, DepacketizerFactory};
    use rustrtc::media::frame::{MediaKind, MediaSample};
    use rustrtc::rtp::{RtpHeader, RtpPacket};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn packet(payload: Vec<u8>, sequence_number: u16, timestamp: u32, marker: bool) -> RtpPacket {
        let mut header = RtpHeader::new(96, sequence_number, timestamp, 12345);
        header.marker = marker;
        RtpPacket::new(header, payload)
    }

    fn addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 1234)
    }

    fn video_data(samples: Vec<MediaSample>) -> Vec<u8> {
        assert_eq!(samples.len(), 1);
        match samples.into_iter().next().unwrap() {
            MediaSample::Video(frame) => frame.data.to_vec(),
            MediaSample::Audio(_) => panic!("expected video sample"),
        }
    }

    #[test]
    fn strips_minimal_vp8_descriptor() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x10, 0xaa, 0xbb], 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), vec![0xaa, 0xbb]);
    }

    #[test]
    fn strips_one_byte_picture_id_descriptor() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x90, 0x80, 0x7f, 0xcc, 0xdd], 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), vec![0xcc, 0xdd]);
    }

    #[test]
    fn strips_extended_descriptor_with_two_byte_picture_id_tl0picidx_and_tid_keyidx() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(
                    vec![0x90, 0xf0, 0x80, 0x01, 0x22, 0x33, 0xde, 0xad],
                    1,
                    100,
                    true,
                ),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), vec![0xde, 0xad]);
    }

    #[test]
    fn emits_single_packet_frame() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x10, 0x30, 0x01], 9, 45_000, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(samples.len(), 1);
        match &samples[0] {
            MediaSample::Video(frame) => {
                assert_eq!(frame.data.as_ref(), &[0x30, 0x01]);
                assert_eq!(frame.rtp_timestamp, 45_000);
                assert!(frame.is_last_packet);
                assert_eq!(frame.sequence_number, Some(9));
                assert_eq!(frame.payload_type, Some(96));
            }
            MediaSample::Audio(_) => panic!("expected video sample"),
        }
    }

    #[test]
    fn reassembles_multi_packet_frame() {
        let mut depacketizer = Vp8Depacketizer::new();
        let first = depacketizer
            .push(
                packet(vec![0x10, 0x30, 0x01], 10, 90_000, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let second = depacketizer
            .push(
                packet(vec![0x00, 0x02, 0x03], 11, 90_000, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(second), vec![0x30, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn reassembles_partition_start_pid_nonzero() {
        let mut depacketizer = Vp8Depacketizer::new();
        let first = depacketizer
            .push(
                packet(vec![0x10, 0x30, 0x01], 10, 90_000, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let second = depacketizer
            .push(
                packet(vec![0x11, 0x02, 0x03], 11, 90_000, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(second), vec![0x30, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn audio_sample_passes_through_vp8_depacketizer() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x11, 0x22], 1, 100, true),
                48_000,
                addr(),
                MediaKind::Audio,
            )
            .unwrap();

        assert_eq!(samples.len(), 1);
        match &samples[0] {
            MediaSample::Audio(frame) => {
                assert_eq!(frame.data.as_ref(), &[0x11, 0x22]);
                assert_eq!(frame.rtp_timestamp, 100);
                assert_eq!(frame.clock_rate, 48_000);
                assert_eq!(frame.sequence_number, Some(1));
                assert_eq!(frame.payload_type, Some(96));
                assert!(frame.marker);
            }
            MediaSample::Video(_) => panic!("expected audio sample"),
        }

        let video = depacketizer
            .push(
                packet(vec![0x10, 0x33], 2, 200, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(video), vec![0x33]);
    }

    #[test]
    fn drops_malformed_duplicate_frame_start_same_timestamp() {
        let mut depacketizer = Vp8Depacketizer::new();
        let first = depacketizer
            .push(
                packet(vec![0x10, 0xaa], 10, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let duplicate = depacketizer
            .push(
                packet(vec![0x10, 0xbb], 11, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(duplicate.is_empty());

        let recovered = depacketizer
            .push(
                packet(vec![0x10, 0xcc], 12, 200, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(recovered), vec![0xcc]);
    }

    #[test]
    fn drops_gap_frame_and_recovers_on_next_start() {
        let mut depacketizer = Vp8Depacketizer::new();
        let first = depacketizer
            .push(
                packet(vec![0x10, 0x30], 20, 1000, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let gap = depacketizer
            .push(
                packet(vec![0x00, 0x31], 22, 1000, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(gap.is_empty());

        let recovered = depacketizer
            .push(
                packet(vec![0x10, 0x40], 23, 2000, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(recovered), vec![0x40]);
    }

    #[test]
    fn drops_truncated_extended_descriptor_without_emitting() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x90], 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert!(samples.is_empty());
    }

    #[test]
    fn drops_truncated_m_bit_picture_id() {
        let mut depacketizer = Vp8Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(vec![0x90, 0x80, 0x80], 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert!(samples.is_empty());
    }

    #[test]
    fn factory_selects_vp8_h264_and_audio_depacketizers() {
        let mut vp8 = VidaraxDepacketizerFactory::new(VideoCodec::Vp8).create(MediaKind::Video);
        let vp8_start = vp8
            .push(
                packet(vec![0x10, 0x30], 1, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(vp8_start.is_empty());

        let mut h264 = VidaraxDepacketizerFactory::new(VideoCodec::H264).create(MediaKind::Video);
        let h264_start = h264
            .push(
                packet(vec![0x7c, 0x85, 0xaa], 1, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(h264_start.is_empty());

        let mut audio = VidaraxDepacketizerFactory::new(VideoCodec::Vp8).create(MediaKind::Audio);
        let audio_samples = audio
            .push(
                packet(vec![0x11, 0x22], 1, 100, true),
                48_000,
                addr(),
                MediaKind::Audio,
            )
            .unwrap();
        assert_eq!(audio_samples.len(), 1);
    }
}

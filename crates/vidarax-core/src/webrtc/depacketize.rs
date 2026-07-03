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
            (MediaKind::Video, VideoCodec::H265) => Box::new(H265Depacketizer::new()),
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

const H265_NAL_TYPE_AP: u8 = 48;
const H265_NAL_TYPE_FU: u8 = 49;
const H265_NAL_TYPE_PACI: u8 = 50;
const ANNEX_B_START_CODE: &[u8; 4] = &[0x00, 0x00, 0x00, 0x01];

pub struct H265Depacketizer {
    access_unit_buf: Vec<u8>,
    fragment_buf: Vec<u8>,
    frame_timestamp: u32,
    last_seq: Option<u16>,
    dropping_until_new_ts: bool,
}

impl Default for H265Depacketizer {
    fn default() -> Self {
        Self::new()
    }
}

impl H265Depacketizer {
    pub fn new() -> Self {
        Self {
            access_unit_buf: Vec::new(),
            fragment_buf: Vec::new(),
            frame_timestamp: 0,
            last_seq: None,
            dropping_until_new_ts: false,
        }
    }

    fn reset_access_unit(&mut self) {
        self.access_unit_buf.clear();
        self.fragment_buf.clear();
    }

    fn append_nal(&mut self, nal: &[u8]) -> bool {
        if nal.len() < 2 {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        }

        if !self.access_unit_buf.is_empty() {
            self.access_unit_buf.extend_from_slice(ANNEX_B_START_CODE);
        }
        self.access_unit_buf.extend_from_slice(nal);
        true
    }

    fn append_fragment_nal(&mut self) {
        if self.access_unit_buf.is_empty() {
            self.access_unit_buf = std::mem::take(&mut self.fragment_buf);
            return;
        }

        self.access_unit_buf.extend_from_slice(ANNEX_B_START_CODE);
        self.access_unit_buf.extend_from_slice(&self.fragment_buf);
        self.fragment_buf.clear();
    }

    fn push_aggregation_packet(&mut self, payload: &[u8]) -> bool {
        let mut offset = 2usize;
        let mut saw_nal = false;

        // sprop-max-don-diff is assumed to be 0, so RFC 7798 DONL/DOND fields
        // are not present and each aggregation unit starts with NALUSize.
        while offset < payload.len() {
            let Some(size_end) = offset.checked_add(2) else {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                return false;
            };
            let Some(&[size0, size1]) = payload.get(offset..size_end) else {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                return false;
            };
            let nalu_size = u16::from_be_bytes([size0, size1]) as usize;
            offset += 2;

            let Some(nal_end) = offset.checked_add(nalu_size) else {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                return false;
            };
            let Some(nal) = payload.get(offset..nal_end) else {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                return false;
            };

            if !self.append_nal(nal) {
                return false;
            }
            saw_nal = true;
            offset += nalu_size;
        }

        if !saw_nal {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        }

        true
    }

    fn push_fragmentation_unit(&mut self, payload: &[u8]) -> bool {
        let Some(&[payload_header0, payload_header1]) = payload.get(..2) else {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        };
        let Some(fu_header) = payload.get(2).copied() else {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        };
        let Some(fu_payload) = payload.get(3..) else {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        };

        let is_start = fu_header & 0x80 != 0;
        let is_end = fu_header & 0x40 != 0;
        let fu_type = fu_header & 0x3f;

        if is_start {
            if !self.fragment_buf.is_empty() {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                return false;
            }
            let reconstructed_header0 = (payload_header0 & 0x81) | (fu_type << 1);
            self.fragment_buf.push(reconstructed_header0);
            self.fragment_buf.push(payload_header1);
            self.fragment_buf.extend_from_slice(fu_payload);
        } else if self.fragment_buf.is_empty() {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return false;
        } else {
            self.fragment_buf.extend_from_slice(fu_payload);
        }

        if is_end {
            self.append_fragment_nal();
        }

        true
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

impl Depacketizer for H265Depacketizer {
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

        let new_au = self.last_seq.is_none() || timestamp != self.frame_timestamp;
        let is_continuous = self
            .last_seq
            .is_some_and(|last| seq == last.wrapping_add(1));
        if new_au {
            self.reset_access_unit();
            self.frame_timestamp = timestamp;
            self.dropping_until_new_ts = false;
        } else if self.dropping_until_new_ts {
            self.last_seq = Some(seq);
            return Ok(vec![]);
        } else if !is_continuous {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            self.last_seq = Some(seq);
            return Ok(vec![]);
        }
        self.last_seq = Some(seq);

        let Some(&[payload_header0, _payload_header1]) = payload.get(..2) else {
            self.reset_access_unit();
            self.dropping_until_new_ts = true;
            return Ok(vec![]);
        };

        let nal_type = (payload_header0 >> 1) & 0x3f;
        let parsed = match nal_type {
            0..=47 => self.append_nal(payload),
            H265_NAL_TYPE_AP => self.push_aggregation_packet(payload),
            H265_NAL_TYPE_FU => self.push_fragmentation_unit(payload),
            H265_NAL_TYPE_PACI | 51..=63 => {
                // PACI and reserved/unspecified RTP payload structures are out of scope.
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                false
            }
            _ => {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                false
            }
        };

        if !parsed {
            return Ok(vec![]);
        }

        if marker {
            if self.access_unit_buf.is_empty() || !self.fragment_buf.is_empty() {
                self.reset_access_unit();
                self.dropping_until_new_ts = true;
                self.last_seq = Some(seq);
                return Ok(vec![]);
            }

            let data = std::mem::take(&mut self.access_unit_buf);
            let sample = self.video_sample(data, &packet, addr);
            self.reset_access_unit();
            self.last_seq = Some(seq);
            self.dropping_until_new_ts = true;
            return Ok(vec![sample]);
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::{H265Depacketizer, VidaraxDepacketizerFactory, Vp8Depacketizer};
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

    fn h265_header(nal_type: u8) -> [u8; 2] {
        [nal_type << 1, 0x01]
    }

    fn h265_nal(nal_type: u8, rbsp: &[u8]) -> Vec<u8> {
        let mut nal = h265_header(nal_type).to_vec();
        nal.extend_from_slice(rbsp);
        nal
    }

    fn h265_payload_header(nal_type: u8) -> Vec<u8> {
        h265_header(nal_type).to_vec()
    }

    fn h265_ap_payload(nals: &[&[u8]]) -> Vec<u8> {
        let mut payload = h265_payload_header(48);
        for nal in nals {
            payload.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            payload.extend_from_slice(nal);
        }
        payload
    }

    fn with_start_code_between(first: &[u8], second: &[u8]) -> Vec<u8> {
        let mut data = first.to_vec();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        data.extend_from_slice(second);
        data
    }

    fn assert_empty(samples: rustrtc::media::MediaResult<Vec<MediaSample>>) {
        assert!(matches!(samples, Ok(samples) if samples.is_empty()));
    }

    fn assert_h265_recovery(depacketizer: &mut H265Depacketizer, sequence_number: u16) {
        let clean = h265_nal(32, &[0xfa]);
        let recovered = depacketizer
            .push(
                packet(clean.clone(), sequence_number, 200, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(recovered), clean);
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
    fn h265_single_nal_emits_exact_nal_only_on_marker() {
        let nal = h265_nal(32, &[0xaa, 0xbb]);
        let mut pending = H265Depacketizer::new();
        let unmarked = pending
            .push(
                packet(nal.clone(), 1, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(unmarked.is_empty());

        let mut depacketizer = H265Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(nal.clone(), 2, 200, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), nal);
    }

    #[test]
    fn h265_aggregation_packet_joins_nals_with_internal_start_code() {
        let nal1 = h265_nal(32, &[0xaa]);
        let nal2 = h265_nal(33, &[0xbb, 0xcc]);
        let payload = h265_ap_payload(&[&nal1, &nal2]);
        let mut depacketizer = H265Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(payload, 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), with_start_code_between(&nal1, &nal2));
    }

    #[test]
    fn h265_fragmentation_unit_reconstructs_original_nal_header_and_payload() {
        let mut depacketizer = H265Depacketizer::new();
        let mut start = h265_payload_header(49);
        start.extend_from_slice(&[0x80 | 19, 0xaa, 0xbb]);
        let mut middle = h265_payload_header(49);
        middle.extend_from_slice(&[19, 0xcc]);
        let mut end = h265_payload_header(49);
        end.extend_from_slice(&[0x40 | 19, 0xdd, 0xee]);

        let first = depacketizer
            .push(
                packet(start, 1, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let second = depacketizer
            .push(
                packet(middle, 2, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(second.is_empty());

        let samples = depacketizer
            .push(packet(end, 3, 100, true), 90_000, addr(), MediaKind::Video)
            .unwrap();

        assert_eq!(
            video_data(samples),
            vec![0x26, 0x01, 0xaa, 0xbb, 0xcc, 0xdd, 0xee]
        );
    }

    #[test]
    fn h265_multiple_single_nals_in_one_access_unit_use_four_byte_separator() {
        let nal1 = h265_nal(32, &[0xaa]);
        let nal2 = h265_nal(34, &[0xbb]);
        let mut depacketizer = H265Depacketizer::new();
        let first = depacketizer
            .push(
                packet(nal1.clone(), 1, 100, false),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert!(first.is_empty());

        let samples = depacketizer
            .push(
                packet(nal2.clone(), 2, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();

        assert_eq!(video_data(samples), with_start_code_between(&nal1, &nal2));
    }

    #[test]
    fn h265_drops_duplicate_single_packet_access_unit() {
        let nal = h265_nal(32, &[0xaa]);
        let mut depacketizer = H265Depacketizer::new();
        let samples = depacketizer
            .push(
                packet(nal.clone(), 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(samples), nal);

        assert_empty(depacketizer.push(
            packet(h265_nal(32, &[0xaa]), 1, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 2);
    }

    #[test]
    fn h265_drops_duplicate_marker_packet_of_multi_packet_access_unit() {
        let nal1 = h265_nal(32, &[0xaa]);
        let nal2 = h265_nal(34, &[0xbb]);
        let mut depacketizer = H265Depacketizer::new();

        assert_empty(depacketizer.push(
            packet(nal1.clone(), 1, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        let samples = depacketizer
            .push(
                packet(nal2.clone(), 2, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(samples), with_start_code_between(&nal1, &nal2));

        assert_empty(depacketizer.push(
            packet(h265_nal(34, &[0xbb]), 2, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 3);
    }

    #[test]
    fn h265_malformed_packets_drop_without_panic_and_recover_on_clean_access_unit() {
        let clean = h265_nal(32, &[0xaa]);
        let mut ap_overrun = h265_payload_header(48);
        ap_overrun.extend_from_slice(&5u16.to_be_bytes());
        ap_overrun.extend_from_slice(&[0xbb]);

        let mut fu_without_start = h265_payload_header(49);
        fu_without_start.extend_from_slice(&[19, 0xcc]);

        let malformed_payloads = vec![
            ap_overrun,
            fu_without_start,
            vec![0x00],
            h265_payload_header(50),
            h265_payload_header(63),
        ];

        for malformed in malformed_payloads {
            let mut depacketizer = H265Depacketizer::new();
            let dropped = depacketizer
                .push(
                    packet(malformed, 1, 100, true),
                    90_000,
                    addr(),
                    MediaKind::Video,
                )
                .unwrap();
            assert!(dropped.is_empty());

            let recovered = depacketizer
                .push(
                    packet(clean.clone(), 2, 200, true),
                    90_000,
                    addr(),
                    MediaKind::Video,
                )
                .unwrap();
            assert_eq!(video_data(recovered), clean);
        }
    }

    #[test]
    fn h265_drops_fragmented_access_unit_after_middle_packet_loss_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let mut start = h265_payload_header(49);
        start.extend_from_slice(&[0x80 | 19, 0xaa]);
        let mut end = h265_payload_header(49);
        end.extend_from_slice(&[0x40 | 19, 0xcc]);

        assert_empty(depacketizer.push(
            packet(start, 10, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(end, 12, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 13);
    }

    #[test]
    fn h265_drops_multi_packet_single_nal_access_unit_after_packet_loss_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let nal1 = h265_nal(32, &[0xaa]);
        let nal2 = h265_nal(34, &[0xbb]);

        assert_empty(depacketizer.push(
            packet(nal1, 10, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(nal2, 12, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 13);
    }

    #[test]
    fn h265_drops_access_unit_after_duplicate_fu_start_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let nal = h265_nal(32, &[0xaa]);
        let mut first_start = h265_payload_header(49);
        first_start.extend_from_slice(&[0x80 | 19, 0xbb]);
        let mut second_start = h265_payload_header(49);
        second_start.extend_from_slice(&[0x80 | 19, 0xcc]);
        let mut end = h265_payload_header(49);
        end.extend_from_slice(&[0x40 | 19, 0xdd]);

        assert_empty(depacketizer.push(
            packet(nal, 1, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(first_start, 2, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(second_start, 3, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(end, 4, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 5);
    }

    #[test]
    fn h265_drops_header_only_ap_terminator_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let nal = h265_nal(32, &[0xaa]);

        assert_empty(depacketizer.push(
            packet(nal, 1, 100, false),
            90_000,
            addr(),
            MediaKind::Video,
        ));
        assert_empty(depacketizer.push(
            packet(h265_payload_header(48), 2, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 3);
    }

    #[test]
    fn h265_drops_ap_with_trailing_one_byte_length_field_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let mut payload = h265_payload_header(48);
        payload.push(0x00);

        assert_empty(depacketizer.push(
            packet(payload, 1, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 2);
    }

    #[test]
    fn h265_drops_ap_with_zero_length_nalu_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();
        let mut payload = h265_payload_header(48);
        payload.extend_from_slice(&0u16.to_be_bytes());

        assert_empty(depacketizer.push(
            packet(payload, 1, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 2);
    }

    #[test]
    fn h265_drops_fu_packet_too_short_for_fu_header_and_recovers() {
        let mut depacketizer = H265Depacketizer::new();

        assert_empty(depacketizer.push(
            packet(h265_payload_header(49), 1, 100, true),
            90_000,
            addr(),
            MediaKind::Video,
        ));

        assert_h265_recovery(&mut depacketizer, 2);
    }

    #[test]
    fn audio_sample_passes_through_h265_depacketizer() {
        let mut depacketizer = H265Depacketizer::new();
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

        let mut h265 = VidaraxDepacketizerFactory::new(VideoCodec::H265).create(MediaKind::Video);
        let h265_nal = h265_nal(32, &[0xaa]);
        let h265_samples = h265
            .push(
                packet(h265_nal.clone(), 1, 100, true),
                90_000,
                addr(),
                MediaKind::Video,
            )
            .unwrap();
        assert_eq!(video_data(h265_samples), h265_nal);
    }
}

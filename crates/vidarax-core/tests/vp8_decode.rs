#![cfg(feature = "vp8")]

use vidarax_core::webrtc::decode::{DecodeError, Decoder, DecoderConfig, VideoCodec};

#[test]
fn libvpx_decodes_keyframe_and_inter_frame_from_ivf() {
    let ivf = IvfFile::parse(include_bytes!("fixtures/vp8_sample.ivf"));
    assert_eq!(ivf.width, 64);
    assert_eq!(ivf.height, 48);
    assert_eq!(ivf.frames.len(), 10);

    let config = DecoderConfig {
        gpu_available: false,
        codec: VideoCodec::Vp8,
        width: ivf.width,
        height: ivf.height,
        output_pool_slots: 1,
    };
    let mut decoder = Decoder::new(&config);
    let mut decoded_inter_frame = false;

    for (index, payload) in ivf.frames.iter().enumerate() {
        let frame = match decoder.decode(payload) {
            Ok(frame) => frame,
            Err(DecodeError::UnsupportedCodec(codec)) => {
                panic!("VP8 decode returned UnsupportedCodec({codec:?})")
            }
            Err(err) => panic!("VP8 fixture frame {index} failed to decode: {err}"),
        };

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.y.len(), 64 * 48);
        assert_eq!(frame.u.len(), 32 * 24);
        assert_eq!(frame.v.len(), 32 * 24);

        if index > 0 && payload[0] & 1 == 1 {
            decoded_inter_frame = true;
        }
    }

    assert!(decoded_inter_frame, "no VP8 inter-frame was exercised");
}

struct IvfFile<'a> {
    width: u32,
    height: u32,
    frames: Vec<&'a [u8]>,
}

impl<'a> IvfFile<'a> {
    fn parse(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() >= 32, "IVF header missing");
        assert_eq!(&bytes[0..4], b"DKIF");
        let header_len = u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize;
        assert!(header_len >= 32, "IVF header too short");
        assert!(bytes.len() >= header_len, "truncated IVF header");

        let width = u16::from_le_bytes(bytes[12..14].try_into().unwrap()) as u32;
        let height = u16::from_le_bytes(bytes[14..16].try_into().unwrap()) as u32;
        let mut offset = header_len;
        let mut frames = Vec::new();
        while offset < bytes.len() {
            assert!(bytes.len() - offset >= 12, "truncated IVF frame header");
            let frame_size =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            let payload_start = offset + 12;
            let payload_end = payload_start + frame_size;
            assert!(payload_end <= bytes.len(), "truncated IVF frame payload");
            frames.push(&bytes[payload_start..payload_end]);
            offset = payload_end;
        }

        Self {
            width,
            height,
            frames,
        }
    }
}

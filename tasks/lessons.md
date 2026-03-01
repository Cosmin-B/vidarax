# Lessons Learned

## Task 0: rustrtc spike (2026-02-28)

### Verdict: PASS (with toolchain constraint)

rustrtc successfully exposes H.264 NAL bytes to application code. The API is clean and
well-factored. Proceed with rustrtc for the full WebRTC ingestion epic.

---

### API surface confirmed

| Concern | API |
|---|---|
| Create PeerConnection | `PeerConnection::new(RtcConfiguration::default())` |
| Apply browser offer | `pc.set_remote_description(SessionDescription::parse(SdpType::Offer, &sdp)?)` |
| Produce answer | `pc.create_answer().await?` then `pc.set_local_description(answer)` |
| Get ICE candidates | `pc.subscribe_ice_candidates()` — watch channel, wait for at least one |
| Track arrival event | `pc.recv().await` → `PeerConnectionEvent::Track(transceiver)` |
| Get track handle | `transceiver.receiver()?.track()` → `Arc<dyn MediaStreamTrack>` |
| Receive media data | `track.recv().await` → `MediaSample::Video(VideoFrame)` |
| H.264 bytes | `VideoFrame.data: bytes::Bytes` — depacketized NAL payload |
| Frame boundary | `VideoFrame.is_last_packet: bool` — mirrors RTP marker bit |
| RTP payload type | `VideoFrame.payload_type: Option<u8>` — typically `Some(96)` for H.264 |

### What arrives in `VideoFrame.data`

The library runs `H264Depacketizer` (RFC 6184) internally before delivering to the
application. This means:

- **FU-A** (fragmented NAL): fragments are reassembled into a single complete NAL unit
  before `recv()` returns. Application never sees partial NALs.
- **STAP-A** (aggregated NALs): split into one `VideoFrame` per NAL unit.
- **Single NAL**: delivered as-is.

The `data` bytes are raw NAL payload without the 4-byte Annex B start code
(`0x00 0x00 0x00 0x01`). If openh264 or NVDEC requires Annex B format, prepend the
start code before passing to the decoder.

### Toolchain constraint

rustrtc v0.3.25 uses Rust edition 2024 and depends on features stabilised in Rust 1.88:
- `let … && let` chains (`E0658` on older toolchains)
- `usize::is_multiple_of` (`E0658` on older toolchains)

The project's `rust-toolchain.toml` pins to **1.85.0**, which is incompatible.
Two options:

**Option A (recommended):** Bump `rust-toolchain.toml` to `1.88.0`.
```toml
[toolchain]
channel = "1.88.0"
components = ["rustfmt", "clippy"]
```
1.88 has been released since May 2025 and is stable. No breaking changes affect this
codebase.

**Option B:** Pin rustrtc to the last commit before edition-2024 features were introduced.
Fragile; not recommended.

### No pivot needed

The plan's fallback to str0m 0.16 is not required. rustrtc's media API is clean and
directly exposes raw NAL bytes. Proceed with Tasks 1-9 after bumping the toolchain.

### Next step

Update `rust-toolchain.toml` as the first action before Task 1. The spike example
(`rustrtc_spike.rs`) already passes `cargo check` under `RUSTUP_TOOLCHAIN=1.88.0`.

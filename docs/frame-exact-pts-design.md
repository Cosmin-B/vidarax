# Frame-exact PTS Findings

## Current state

The WebRTC input path receives `RtpFrame` values with access unit bytes,
`pts_ms`, `seq`, and `codec`. The decode worker calls `Decoder::decode` and
then builds a `StreamFrame` with the returned pixels and a timestamp label.

The in-process openh264 H.264 path is already frame-exact in this model.
`Decoder::decode` passes one access unit to openh264 synchronously. If openh264
returns a frame, the worker attaches the `seq` and `pts_ms` from that same
access unit. Parameter-set-only input returns `Buffered` and emits no frame.

The ffmpeg sidecar path is different. It writes raw encoded bytes to stdin and
reads raw YUV420 frames from stdout. Rawvideo stdout has no PTS, frame index, or
packet metadata. For GPU H.264, the worker labels any returned frame with the
current RTP access unit as a best-effort approximation.

## Rejected ffmpeg container approach

Feeding ffmpeg a timestamp-carrying container over stdin was investigated as a
way to carry per-frame PTS through the sidecar. For VP8, the practical
PTS-carrying input container is IVF.

That approach is not viable for this live pipeline. The input pipe is
long-lived and never closed during a session. IVF over that pipe buffers many
frames before decoded output appears, and the resulting latency is unacceptable
for real-time analysis. Demuxer and AVIO low-latency flags did not remove that
buffering.

A container sidecar also still needs a reliable way to pair raw YUV stdout
frames with output PTS metadata. Without that, it only moves the correlation
problem from input to the subprocess boundary.

## Decision

VP8 is unsupported for live decode in the current zero-dependency design.
Constructing a VP8 decoder creates an unsupported decoder state, and the first
decode call returns an unsupported codec error. The decode worker logs that
clear error once and stops instead of silently dropping every frame.

The honest path to VP8 support, and to frame-exact PTS for non-openh264 codecs,
is an in-process decoder. That decoder must accept packets with timestamps and
return decoded frames with their own timestamps, without relying on subprocess
demuxer buffering or a separate metadata side channel. This is a future
dependency-bearing change and is outside the current scope.

Until then:

- H.264 software decode via openh264 is frame-exact.
- H.264 GPU decode via the ffmpeg sidecar uses best-effort current-RTP labels.
- VP8 fails fast as an unsupported configuration.

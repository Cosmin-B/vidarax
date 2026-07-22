# Frame-exact PTS Findings

## Current state

The WebRTC input path receives `RtpFrame` values with access unit bytes,
`pts_ms`, `seq`, and `codec`. The decode worker calls `Decoder::decode` and
then builds a `StreamFrame` with the returned pixels and a timestamp label.

Live H.264 and H.265 now use the ffmpeg process boundary on both CPU and GPU
paths. The sidecar writes raw encoded bytes to stdin and
reads raw YUV420 frames from stdout. Rawvideo stdout has no PTS, frame index, or
packet metadata. The worker labels any returned frame with the
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

Native crash containment wins over frame-exact labels for the default H.264
path. Both CPU and GPU H.264 use the supervised ffmpeg child, and their labels
remain best-effort. A broken sidecar faults the whole session generation. It is
not restarted underneath temporal workers.

The optional `vp8` feature provides an in-process libvpx decoder and can retain
the synchronous packet-to-frame association, but it is an explicit native
in-process exception without sidecar crash containment. Builds without that
feature reject VP8 during codec selection.

True frame-exact labels across a process boundary require an output protocol
that carries decoded-frame PTS alongside pixels without the long-lived
container buffering observed here. Rawvideo stdout does not provide it.

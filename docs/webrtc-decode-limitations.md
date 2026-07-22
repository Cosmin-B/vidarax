# WebRTC Decode Limitations

## Supported live decode

H.264 and H.265 use a long-lived ffmpeg sidecar on both CPU and GPU paths. The
GPU path adds `-hwaccel auto`. The CPU path leaves acceleration unset. Both
feeds raw Annex B H.264 into ffmpeg stdin and reads raw YUV420 frames from
stdout. That raw pipe carries no output PTS or frame index, and inputs do not
map 1:1 to output frames. Parameter sets, pre-sync input, undecodable
inter-frames, and decoder reorder can all shift when a decoded frame appears.

For both H.264 paths, decoded frames are labeled by the decode worker with the
current RTP access unit's `seq` and `pts_ms` as a best-effort approximation. The
pixels and perceptual signals are computed from the decoded output. Only the
timestamp/index label is approximate.

## Optional VP8 live decode

VP8 is not supported in the default build. With the optional `vp8` feature,
Vidarax uses libvpx in process and retains the synchronous packet-to-frame
association that the raw ffmpeg pipe cannot provide. That path is an explicit
native in-process exception: a libvpx crash is not contained by a child-process
boundary.

ffmpeg has no live-usable raw VP8 demuxer for `-f vp8 -i pipe:0`. Its
PTS-carrying IVF path buffers many frames on a never-closed stdin pipe, which
violates the latency target for WebRTC ingest.

Without the `vp8` feature, VP8-negotiated sessions fail fast with a clear
unsupported codec error. The client should offer H.264 or H.265 to the default
build.

The feature exists for deployments that accept the native in-process fault
boundary. Selection must be explicit.

## ffmpeg YUV reader behavior

The ffmpeg YUV reader handoff is bounded and uses blocking sends. Each
`decode()` call drains all currently-ready reader output before writing more
encoded input to ffmpeg stdin. That ordering makes room for output already
produced and narrows the coupled-pipe deadlock window. The implementation has
no measured or enforced maximum decoded-output burst per input write. After the write, the decoder returns the freshest
pending decoded YUV frame. If several decoded frames were already waiting, the
older decoded YUV frames are shed, their pooled Y/U/V buffers are recycled, and
`vidarax_pipeline_frames_dropped_total` is incremented for the shed count.

This is a real-time freshness policy: under sustained overload the engine sheds
the oldest decoded output, not encoded input, so codec state is preserved while
analysis stays as close as possible to the current RTP label.

Boundedness still comes from the existing end-to-end backpressure. When
analysis is slow, the decode worker blocks on the bounded downstream
`frame_tx`, stops calling `decode()`, stops pulling RTP, and stops feeding
ffmpeg input. ffmpeg then stops producing decoded YUV beyond its normal
pipeline depth. A one-time diagnostic warns and increments a metric if the
decoder-local pending FIFO exceeds a generous sanity bound.

## YUV output pool sizing

The YUV output pool is sized per decode backend:

- `NvDec` and `FfmpegSw` use the ffmpeg reader path: 16 reader handoff slots, 4
  decoder-pending allowance slots, 1 reader-constructing frame, and 1
  decode-consumer frame, for 22 YUV420 slots. At 1920x1080 this is about
  65,318,400 bytes, or 62.3 MiB per session.
- `Unsupported` allocates no decode output pool work because it never produces
  YUV frames.

The direct openh264 decoder remains in the core module for targeted use, but
live backend selection does not choose it. This is deliberate process
isolation: a native H.264 crash kills one ffmpeg child, after which the session
supervisor faults and closes that generation. The API process remains alive.

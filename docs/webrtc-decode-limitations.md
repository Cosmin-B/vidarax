# WebRTC Decode Limitations

## Supported live decode

H.264 and H.265 use a long-lived ffmpeg sidecar on both CPU and GPU paths. The
GPU path adds `-hwaccel auto`; the CPU path leaves acceleration unset. Both
feeds raw Annex B H.264 into ffmpeg stdin and reads raw YUV420 frames from
stdout. That raw pipe carries no output PTS or frame index, and inputs do not
map 1:1 to output frames. Parameter sets, pre-sync input, undecodable
inter-frames, and decoder reorder can all shift when a decoded frame appears.

For both H.264 paths, decoded frames are labeled by the decode worker with the
current RTP access unit's `seq` and `pts_ms` as a best-effort approximation. The
pixels and perceptual signals are computed from the decoded output. Only the
timestamp/index label is approximate.

## Unsupported VP8 live decode

VP8 is not supported for live decode in the current zero-dependency design.

ffmpeg does not provide a live-usable raw VP8 demuxer for `-f vp8 -i pipe:0`.
The available PTS-carrying container path is IVF, but IVF over a never-closed
stdin pipe buffers many frames before producing output. That startup and
steady-state lag is not acceptable for the real-time WebRTC pipeline.

Because of that, VP8-negotiated sessions fail fast with a clear unsupported
codec error. The client should offer H.264 for live decode.

Adding VP8 support requires an in-process decoder that can accept packets with
timestamps and return decoded frames without subprocess demuxer buffering. That
would add a native dependency, so it is outside the current design.

## ffmpeg YUV reader behavior

The ffmpeg YUV reader handoff is bounded and uses blocking sends. Each
`decode()` call drains all currently-ready reader output before writing more
encoded input to ffmpeg stdin, so the reader's blocking send cannot deadlock
against a `decode()` call blocked on stdin. After the write, the decoder returns
the freshest pending decoded YUV frame. If several decoded frames were already
waiting, the older decoded YUV frames are shed, their pooled Y/U/V buffers are
recycled, and `vidarax_pipeline_frames_dropped_total` is incremented for the
shed count.

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
supervisor faults and closes that generation instead of aborting the API.

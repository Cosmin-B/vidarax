# WebRTC Decode Limitations

## ffmpeg raw-pipe frame labels

The `FfmpegSw` and `NvDec` paths feed raw Annex-B H.264 or raw VP8 into
ffmpeg stdin and read raw YUV420 frames from stdout. That raw pipe does not
carry output PTS or a frame index, and inputs do not map 1:1 to output frames:
pre-sync input, parameter sets, undecodable inter-frames, and decoder reorder
can all shift when a decoded frame appears.

For these ffmpeg paths, decoded frames are therefore labeled by the decode
worker with the current RTP access unit's `seq` and `pts_ms` as a best-effort
approximation. The pixels and perceptual signals are exact for the decoded
output; only the timestamp/index label can be off, typically by about 1-2
frames of decoder latency. This is negligible for low-latency WebRTC streams
without B-frames.

The ffmpeg YUV reader handoff is bounded and uses blocking sends. Each
`decode()` call still drains all currently-ready reader output before writing
more encoded input to ffmpeg stdin, so the reader's blocking send cannot
deadlock against a `decode()` call blocked on stdin. After the write, the
decoder returns the freshest pending decoded YUV frame. If several decoded
frames were already waiting, the older decoded YUV frames are shed, their
pooled Y/U/V buffers are recycled, and `vidarax_pipeline_frames_dropped_total`
is incremented for the shed count.

This is intentionally a real-time freshness policy: under sustained overload
the engine sheds the oldest decoded output, not encoded input, so codec state
is preserved while analysis stays as close as possible to the current RTP
`seq`/`pts_ms` label. In steady state there is usually one ready decoded frame,
so no decoded output is shed and labels remain the existing best-effort
current-RTP approximation (typically within about 1-2 frames).

Boundedness still comes from the existing end-to-end backpressure. When
analysis is slow, the decode worker blocks on the bounded downstream
`frame_tx`, stops calling `decode()`, stops pulling RTP, and stops feeding
ffmpeg input. ffmpeg then stops producing decoded YUV beyond its normal
pipeline depth. A one-time diagnostic warns and increments a metric if the
decoder-local pending FIFO exceeds a generous sanity bound.

## YUV output pool sizing

The YUV output pool is sized per decode backend:

- `NvDec` and `FfmpegSw` use the ffmpeg reader path: 16 reader handoff slots,
  4 decoder-pending allowance slots, 1 reader-constructing frame, and 1
  decode-consumer frame, for 22 YUV420 slots. At 1920x1080 this is about
  65,318,400 bytes, or 62.3 MiB per session.
- `Software` uses the synchronous openh264 path: 2 YUV420 slots, enough for
  one caller-held output and the next decoded output. At 1920x1080 this is
  about 6,220,800 bytes, or 5.9 MiB per session.

Frame-exact labels on the ffmpeg paths require a future decoder mode that feeds
ffmpeg container-framed input with PTS passthrough and reads per-frame output
PTS back from decoder output. The synchronous openh264 `Software` H.264 path is
already frame-exact because it returns output directly from the access unit
passed to `decode()`.

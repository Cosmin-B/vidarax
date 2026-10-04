# WebRTC decode limitations

## Supported live decode

H.264 and H.265 use a long-lived ffmpeg sidecar on both CPU and GPU paths. The
GPU path adds `-hwaccel auto`. The CPU path leaves acceleration unset. Both
feed raw Annex B H.264 or H.265 into ffmpeg stdin and read raw YUV420 frames from
stdout. That raw pipe carries no output PTS or frame index, and inputs do not
map 1:1 to output frames. Parameter sets, pre-sync input, undecodable
inter-frames, and decoder reorder can all shift when a decoded frame appears.

For both codecs on the CPU and GPU paths, decoded frames are labeled by the decode worker with the
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

The rejected timestamp-carrying container design is recorded in [Frame-exact
PTS findings](frame-exact-pts-design.md).

## ffmpeg YUV reader behavior

The ffmpeg YUV reader handoff holds at most 16 frames and uses blocking
sends. Each `decode()` call receives at most 16 ready frames before writing
encoded input. The decoder retains the newest received frame and recycles each
older frame as its replacement arrives. It counts those replacements in
`vidarax_pipeline_frames_dropped_total`.

This bounds the drain's work even when the reader refills the channel. The
reader handoff remains lossless. The decoder sheds decoded output to keep
analysis close to the current RTP label. It always writes encoded input so the
codec retains its state.

The drain makes room for output that ffmpeg has already produced. The
implementation has no measured or enforced maximum decoded-output burst per
input write, so this ordering does not establish that the coupled pipes cannot
deadlock. Source-time labels remain an approximation because raw YUV output
has no timestamp channel.

Decoder destruction disconnects the reader channel before killing and reaping
the ffmpeg child. Disconnection releases a reader blocked on a full channel.
The decoder then joins that reader before releasing its resources.

## YUV output pool sizing

The YUV output pool is sized per decode backend:

- `NvDec` and `FfmpegSw` use 16 reader handoff positions, one newest
  decoded frame, one received replacement, one constructing frame, and one
  consumer frame. The pool has 20 positions. At 1920x1080, the luma capacity
  rounds to 2 MiB and each chroma capacity is 512 KiB. The plane capacities
  total 60 MiB per session. This is a source-derived pool capacity, not RSS.
- `Unsupported` allocates no decode output pool work because it never produces
  YUV frames.

The direct openh264 decoder remains in the core module for targeted use, but
live backend selection does not choose it. This is deliberate process
isolation: a native H.264 crash kills one ffmpeg child, after which the session
supervisor faults and closes that generation. The API process remains alive.

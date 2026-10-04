# Resources through completion

Accepted work keeps its resource owner until the work finishes. Replaceable
live work releases stale buffers as soon as a newer item replaces it.

| Path | Before | After |
|---|---|---|
| Recorded inference | A dropped HTTP waiter could drop the provider result before its WAL append. | The admitted task owns its permit and result through the append. A dropped waiter does not undo accepted work. |
| Parallel semantic chunks | Dropping the dispatcher could lose active completions. | Active calls retain their media and journal sender. Cancellation stops later chunks. A missing chunk does not discard completed later chunks. |
| Media subprocess | Some calls waited before draining pipes, retained unrestricted output, or left cleanup to an error path. | One helper drains both pipes, caps retained bytes, enforces a deadline, kills on failure, reaps the child and joins its readers. The frame limit also reaches ffmpeg. |
| Live decoded frames | A pending FIFO retained stale frames, and a producer could keep a drain loop running. | One pending frame retains the newest output. Each call drains at most 16 items and recycles replacements. Drop disconnects the reader before joining it. |
| JPEG encoding | A byte check after encoding allowed the temporary buffer to grow past its limit. | The encoder writer rejects output before it exceeds 2 MiB. Normal encoded bytes stay identical. |
| Sidecar exchange | Partial socket progress could restart a timeout. A failed embedding batch could strand queued callers. | One deadline covers framing, payload and response. A failed batch releases its bytes and wakes its callers; the worker continues. |
| Structured Gemini response | Hidden thinking could exhaust the output budget after starting JSON, leaving a partial answer without a retry. | A length cutoff with thinking tokens and incomplete requested JSON uses the existing single headroom retry. Complete JSON and free-form text keep their prior policy. |
| Loop history | Unused slots contained an all-ones hash and counted as prior frames. | Only populated slots count. The first all-ones frame no longer starts a false loop, including after reset. |
| RTP reassembly | Compressed payload limits applied after fragment assembly. Queue buffers could grow beyond their reservation when Annex B framing was added. | Fragment append checks the existing 2 MiB limit. Queue-buffer growth is capped and its reservation includes the four-byte prefix. Overflow drops incomplete media and recovers on clean input. |
| Semantic failure | Literal and numeric codes became owned strings before their result was retained. Media-failure accounting inspected a string prefix. | Results retain enum variants and numeric details. Event serialization preserves the wire text; accounting matches the failure kind. |
| Frame metadata | A reusable gate-event batch sat between two independent phases. | Both phases finish per frame and retain only the metadata output. |
| Decoder FFI | Enabling VP8 allowed unsafe operations throughout the decoder module. | Only the private libvpx module receives that exception. The parent decoder remains under the crate-wide restriction. |
| Browser and SDK | A completed response header could end the timeout before the body. Replacement could leave a reader or late media callback alive. | Body reads share the request deadline. Stream exit cancels and releases its reader. Generation ownership closes late tracks and rejects stale callbacks. |

```mermaid
flowchart LR
  subgraph Before
    A[HTTP waiter] --> B[Provider call]
    B --> C[WAL append]
    X[Waiter cancelled] --> Y[Completion owner dropped]
  end
  subgraph After
    D[Admission permit] --> E[Owned provider task]
    E --> F[WAL append]
    F --> G[Release permit]
    H[HTTP waiter] -. awaits .-> E
  end
```

## Capacity and cost

The decoder pool covers 20 retained positions, including the old pending frame
and its replacement. Rounded YUV plane capacities reserve 30 MiB at 720p and
60 MiB at 1080p. These are source capacity calculations, not process RSS.
The previous 22-position allowance reserved 33 MiB and 66 MiB respectively
when using the same rounded plane sizes.

A local comparison called `decode_mp4_to_frame_signals` with 60 FPS sampling
and `max_frames = 3`. Both versions returned identical frame fields, including
PTS and hashes. Each arm used one warm-up and seven timed calls. Timing covered
the full function, including probing, both decode passes, parsing, subprocess
startup and cleanup.

| Synthetic input | Baseline median (range), ms | Current median (range), ms |
|---|---:|---:|
| One second, 60 frames | 57.531 (56.620–58.044) | 99.656 (89.289–115.367) |
| Thirty seconds, 1,800 frames | 233.592 (231.423–235.208) | 99.969 (99.111–100.619) |

Setup: baseline `442543456969326724f78dab132d50f5aea5331e`, Rust 1.89,
unoptimized dev profile, macOS 26.4 arm64, ffmpeg 8.1.1, 256×144 H.264 input.
The long input repeats the one-second fixture without re-encoding. The fixture
uses generated gray UI bars and a red block for four frames. No remote model
is involved. Arms ran sequentially on one host.

The frame limit avoids decoding the rest of the long input. The bounded helper
adds pipe-reader threads and a 20 ms child-status polling interval; short calls
cost more in this run. These dev-profile samples do not establish release
latency, service throughput or a workload-wide speedup.

## Frame-analysis storage and cost

The frame-analysis comparison removed the intermediate gate-event buffer.
Both versions returned identical values for every metadata field, checked over
960,000 frames per run. Each run included eight pipeline constructions, gate
classification, window scoring, metadata output and a caller checksum.
Input preparation and process startup were outside the timed region.

| Batch size | Commit policy | Before median (range), ns/frame | After median (range), ns/frame | Allocation calls per pipeline |
|---|---|---:|---:|---:|
| 1 | Immediate | 22.089 (22.049–22.540) | 21.205 (20.917–21.785) | 3 → 2 |
| 1 | Deferred | 22.491 (22.457–22.613) | 21.740 (21.286–21.986) | 3 → 2 |
| 32 | Immediate | 34.319 (34.262–34.364) | 30.221 (30.177–30.298) | 9 → 5 |
| 32 | Deferred | 33.210 (33.128–33.348) | 30.195 (30.155–30.353) | 9 → 5 |
| 256 | Immediate | 34.741 (34.649–34.801) | 31.107 (31.046–31.207) | 15 → 8 |
| 256 | Deferred | 33.762 (33.737–33.869) | 31.105 (31.077–31.192) | 15 → 8 |

Setup: baseline `a238545f0fb56b310ed48f298816d5fc71f60b06`, Rust 1.89,
`rustc --edition=2021 -O`, macOS 26.4 arm64, a 16-frame context window and
the default gate configuration. Both arms used the same prepared 120,000-frame
signal array for each pipeline. Deferred runs committed the batch after
consuming its metadata. One warm-up preceded seven runs per arm, alternating
the order of the arms. Allocation counts include buffer growth.

The medians were 3–12% lower on this fixture. The removed buffer saves 96 bytes
of allocated storage per pipeline at batch size 1, 1,440 bytes of allocation
traffic at size 32 and 12,192 bytes at size 256. These byte differences include
growth allocations, not peak process memory. The comparison does not measure
decode, inference or network latency.

## Remaining limits

The live ffmpeg path still couples synchronous input writes to bounded output
queues. No maximum output burst per input access unit is established. The
drain limit alone cannot prove that every input avoids a coupled-pipe stall.

Generation memory admission covers the declared video pools and workers. It
does not establish an aggregate process budget for every live audio track,
sidecar connection, allocator or codec allocation. Process boundaries do not
provide an OS security sandbox.

See [Runtime contracts](native-systems-profile.md) for measurement limits and
the [Core ownership map](core-ownership-map.md) for module boundaries.

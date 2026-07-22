//! Zero-alloc pipeline metrics.
//!
//! Every counter is backed by an `AtomicU64` and uses `Relaxed` ordering.
//! These are observability counters, not synchronisation primitives; strict
//! ordering guarantees across different counters are intentionally not
//! provided.
//!
//! Latency histograms follow the same zero-alloc pattern: fixed-size
//! bucket arrays, one `AtomicU64` per bucket.  The `render_prometheus`
//! method emits cumulative bucket counts compatible with the Prometheus
//! histogram exposition format.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::webrtc::runtime::{PipelineFault, PipelineFaultReason, PipelineShutdown, PipelineStage};

const PIPELINE_STAGE_COUNT: usize = 5;
const PIPELINE_FAULT_REASON_COUNT: usize = 4;

pub const DECODE_LATENCY_US_BUCKETS: [u64; 8] =
    [100, 250, 500, 1_000, 2_000, 5_000, 10_000, 50_000];
pub const GATE_LATENCY_US_BUCKETS: [u64; 8] = [1, 5, 10, 50, 100, 500, 1_000, 5_000];
pub const VLM_LATENCY_MS_BUCKETS: [u64; 8] = [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000];
pub const EMBEDDING_LATENCY_MS_BUCKETS: [u64; 8] = [1, 2, 5, 10, 25, 50, 100, 500];
pub const STDB_EMIT_LATENCY_MS_BUCKETS: [u64; 8] = [1, 5, 10, 25, 50, 100, 250, 1_000];

/// Zero-alloc histogram with 8 fixed upper-bound buckets.
///
/// Bucket semantics match the Prometheus histogram exposition format:
/// each bucket counts observations **strictly less than or equal to** its
/// upper bound.  The `+Inf` bucket is always equal to `count`.
///
/// `record` places each observation into exactly one bucket (the first
/// bucket whose bound is `>= value`), mirroring how `InferenceMetrics`
/// handles its own latency buckets.
pub struct LatencyHistogram {
    bounds: [u64; 8],
    buckets: [AtomicU64; 8],
    sum: AtomicU64,
    count: AtomicU64,
}

impl LatencyHistogram {
    /// Construct a histogram with the given fixed upper bounds.
    ///
    /// `bounds` must be in ascending order.  The last element acts as the
    /// largest explicit bucket; any observation above it falls only into
    /// the implicit `+Inf` bucket (i.e. it is counted in `count` and `sum`
    /// but not in any named bucket, which is the correct Prometheus
    /// behaviour).
    pub fn new(bounds: [u64; 8]) -> Self {
        Self {
            bounds,
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a single observation.
    #[inline]
    pub fn record(&self, value: u64) {
        self.record_many(value, 1);
    }

    /// Record `count` observations with the same value using one atomic update
    /// per field. Batch file decoders use this to preserve per-frame histogram
    /// semantics without looping over hundreds of thousands of samples.
    #[inline]
    pub fn record_many(&self, value: u64, count: u64) {
        if count == 0 {
            return;
        }
        self.sum
            .fetch_add(value.saturating_mul(count), Ordering::Relaxed);
        self.count.fetch_add(count, Ordering::Relaxed);
        for (idx, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.buckets[idx].fetch_add(count, Ordering::Relaxed);
                return;
            }
        }
    }

    pub fn render_prometheus(&self, metric_name: &str, _unit: &str) -> String {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);

        let mut out = String::with_capacity(256);
        use std::fmt::Write as _;
        let mut cumulative: u64 = 0;
        for (idx, &bound) in self.bounds.iter().enumerate() {
            cumulative += self.buckets[idx].load(Ordering::Relaxed);
            let _ = writeln!(out, "{metric_name}_bucket{{le=\"{bound}\"}} {cumulative}");
        }
        let _ = writeln!(out, "{metric_name}_bucket{{le=\"+Inf\"}} {count}");
        let _ = writeln!(out, "{metric_name}_sum {sum}");
        let _ = writeln!(out, "{metric_name}_count {count}");
        out
    }
}

/// Zero-alloc pipeline metrics backed by `AtomicU64` counters and latency histograms.
pub struct PipelineMetrics {
    /// RTP frames received by decode workers.
    rtp_frames_received_total: AtomicU64,
    /// Video frames decoded successfully.
    frames_decoded_total: AtomicU64,
    /// Decoded video frames shed by real-time decode freshness policy.
    frames_dropped_total: AtomicU64,
    /// Frames evaluated by the gate across live and recorded-file pipelines.
    gate_frames_analyzed_total: AtomicU64,
    /// Frames selected as keyframes by the gate, before downstream queueing.
    gate_keyframes_selected_total: AtomicU64,
    /// ffmpeg YUV pending FIFO exceeded the backpressure sanity bound.
    decode_pending_sanity_violations_total: AtomicU64,
    /// Keyframes forwarded from analysis workers to VLM workers.
    keyframes_total: AtomicU64,
    /// Keyframes dropped because the VLM queue was full.
    keyframes_dropped_total: AtomicU64,
    /// Keyframe storage payloads dropped because the sink JPEG backlog is full.
    sink_keyframes_dropped_total: AtomicU64,
    /// New content-addressed keyframe blobs committed to local storage.
    keyframe_blobs_written_total: AtomicU64,
    /// Keyframes that reused an already-present content-addressed blob.
    keyframe_blobs_reused_total: AtomicU64,
    /// Local keyframe blob writes that failed before event metadata append.
    keyframe_blob_failures_total: AtomicU64,
    /// Raw JPEG bytes committed for newly-created local blobs.
    keyframe_blob_bytes_total: AtomicU64,
    /// Restricted-zone assertions durably committed with their evidence.
    restricted_zone_assertions_total: AtomicU64,
    /// Restricted-zone evidence that failed before the assertion commit.
    restricted_zone_evidence_failures_total: AtomicU64,
    /// Restricted-zone assertions dropped at the bounded writer queue.
    restricted_zone_queue_dropped_total: AtomicU64,
    /// Generic trigger assertions durably committed with keyframe evidence.
    trigger_assertions_total: AtomicU64,
    /// Trigger binary sidecar writes that failed before event commit.
    trigger_binary_write_failures_total: AtomicU64,
    /// Trigger assertions dropped at the bounded writer queue.
    trigger_queue_dropped_total: AtomicU64,
    /// Live trigger evaluations that referenced an unavailable signal.
    trigger_missing_signal_total: AtomicU64,
    /// Trigger metadata datagrams delivered to the configured local output.
    trigger_local_outputs_total: AtomicU64,
    /// Trigger local-output datagrams that could not be delivered.
    trigger_local_output_failures_total: AtomicU64,
    /// VLM inference calls dispatched.
    vlm_inferences_total: AtomicU64,
    /// T2 candidates evaluated with a semantic embedding.
    novelty_evaluated_total: AtomicU64,
    /// T2 candidates reused without a VLM call.
    novelty_reused_total: AtomicU64,
    /// Reuse decisions overridden by the TTL or drift limit.
    novelty_forced_refresh_total: AtomicU64,
    /// Embedding requests that failed or timed out.
    novelty_embedding_unavailable_total: AtomicU64,
    /// Reuse decisions sampled through the VLM for calibration.
    novelty_shadow_sampled_total: AtomicU64,
    /// Shadow probes that returned a usable non-empty description.
    novelty_shadow_completed_total: AtomicU64,
    /// Shadow samples whose descriptions differed materially from the anchor.
    novelty_shadow_changed_total: AtomicU64,
    /// Loop detection events emitted by analysis workers.
    loop_detected_total: AtomicU64,
    /// WHIP sessions created.
    sessions_created_total: AtomicU64,
    /// WHIP sessions removed (terminated).
    sessions_removed_total: AtomicU64,
    /// Live, fully-started media-pipeline generations.
    pipeline_generations_active: AtomicU64,
    pipeline_generations_started_total: AtomicU64,
    pipeline_generation_clean_shutdown_total: AtomicU64,
    pipeline_generation_faulted_shutdown_total: AtomicU64,
    pipeline_generation_forced_shutdown_total: AtomicU64,
    pipeline_detached_workers_total: AtomicU64,
    pipeline_last_fault_timestamp_seconds: AtomicU64,
    /// Fixed stage × reason matrix; labels are rendered only at scrape time.
    pipeline_worker_faults_total: [AtomicU64; PIPELINE_STAGE_COUNT * PIPELINE_FAULT_REASON_COUNT],

    /// Per-frame H.264 decode latency in microseconds.
    pub decode_latency_us: LatencyHistogram,
    /// Gate engine processing latency in microseconds.
    pub gate_latency_us: LatencyHistogram,
    /// VLM inference round-trip latency in milliseconds.
    pub vlm_latency_ms: LatencyHistogram,
    /// Raw-binary embedding sidecar round-trip latency in milliseconds.
    pub novelty_embedding_latency_ms: LatencyHistogram,
    /// Content-address/hash/write latency for local keyframe blobs.
    pub keyframe_blob_latency_ms: LatencyHistogram,
    /// SpacetimeDB HTTP POST latency in milliseconds.
    pub stdb_emit_latency_ms: LatencyHistogram,
}

impl PipelineMetrics {
    /// Create a new zero-initialised set of counters and histograms.
    pub fn new() -> Self {
        Self {
            rtp_frames_received_total: AtomicU64::new(0),
            frames_decoded_total: AtomicU64::new(0),
            frames_dropped_total: AtomicU64::new(0),
            gate_frames_analyzed_total: AtomicU64::new(0),
            gate_keyframes_selected_total: AtomicU64::new(0),
            decode_pending_sanity_violations_total: AtomicU64::new(0),
            keyframes_total: AtomicU64::new(0),
            keyframes_dropped_total: AtomicU64::new(0),
            sink_keyframes_dropped_total: AtomicU64::new(0),
            keyframe_blobs_written_total: AtomicU64::new(0),
            keyframe_blobs_reused_total: AtomicU64::new(0),
            keyframe_blob_failures_total: AtomicU64::new(0),
            keyframe_blob_bytes_total: AtomicU64::new(0),
            restricted_zone_assertions_total: AtomicU64::new(0),
            restricted_zone_evidence_failures_total: AtomicU64::new(0),
            restricted_zone_queue_dropped_total: AtomicU64::new(0),
            trigger_assertions_total: AtomicU64::new(0),
            trigger_binary_write_failures_total: AtomicU64::new(0),
            trigger_queue_dropped_total: AtomicU64::new(0),
            trigger_missing_signal_total: AtomicU64::new(0),
            trigger_local_outputs_total: AtomicU64::new(0),
            trigger_local_output_failures_total: AtomicU64::new(0),
            vlm_inferences_total: AtomicU64::new(0),
            novelty_evaluated_total: AtomicU64::new(0),
            novelty_reused_total: AtomicU64::new(0),
            novelty_forced_refresh_total: AtomicU64::new(0),
            novelty_embedding_unavailable_total: AtomicU64::new(0),
            novelty_shadow_sampled_total: AtomicU64::new(0),
            novelty_shadow_completed_total: AtomicU64::new(0),
            novelty_shadow_changed_total: AtomicU64::new(0),
            loop_detected_total: AtomicU64::new(0),
            sessions_created_total: AtomicU64::new(0),
            sessions_removed_total: AtomicU64::new(0),
            pipeline_generations_active: AtomicU64::new(0),
            pipeline_generations_started_total: AtomicU64::new(0),
            pipeline_generation_clean_shutdown_total: AtomicU64::new(0),
            pipeline_generation_faulted_shutdown_total: AtomicU64::new(0),
            pipeline_generation_forced_shutdown_total: AtomicU64::new(0),
            pipeline_detached_workers_total: AtomicU64::new(0),
            pipeline_last_fault_timestamp_seconds: AtomicU64::new(0),
            pipeline_worker_faults_total: std::array::from_fn(|_| AtomicU64::new(0)),

            decode_latency_us: LatencyHistogram::new(DECODE_LATENCY_US_BUCKETS),
            gate_latency_us: LatencyHistogram::new(GATE_LATENCY_US_BUCKETS),
            vlm_latency_ms: LatencyHistogram::new(VLM_LATENCY_MS_BUCKETS),
            novelty_embedding_latency_ms: LatencyHistogram::new(EMBEDDING_LATENCY_MS_BUCKETS),
            keyframe_blob_latency_ms: LatencyHistogram::new(STDB_EMIT_LATENCY_MS_BUCKETS),
            stdb_emit_latency_ms: LatencyHistogram::new(STDB_EMIT_LATENCY_MS_BUCKETS),
        }
    }

    #[inline]
    pub fn inc_rtp_received(&self) {
        self.rtp_frames_received_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_frames_decoded(&self) {
        self.frames_decoded_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch decoder result while retaining per-frame latency
    /// semantics in the exported histogram.
    #[inline]
    pub fn record_decoded_batch(&self, frames: u64, elapsed_us: u64) {
        if frames == 0 {
            return;
        }
        self.frames_decoded_total
            .fetch_add(frames, Ordering::Relaxed);
        let per_frame_us = elapsed_us.div_ceil(frames).max(1);
        self.decode_latency_us.record_many(per_frame_us, frames);
    }

    #[inline]
    pub fn inc_gate_frame_analyzed(&self) {
        self.gate_frames_analyzed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_gate_keyframe_selected(&self) {
        self.gate_keyframes_selected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record one batch gate pass while retaining per-frame latency semantics.
    #[inline]
    pub fn record_gate_batch(&self, analyzed: u64, selected: u64, elapsed_us: u64) {
        if analyzed == 0 {
            return;
        }
        self.gate_frames_analyzed_total
            .fetch_add(analyzed, Ordering::Relaxed);
        self.gate_keyframes_selected_total
            .fetch_add(selected.min(analyzed), Ordering::Relaxed);
        let per_frame_us = elapsed_us.div_ceil(analyzed).max(1);
        self.gate_latency_us.record_many(per_frame_us, analyzed);
    }

    #[inline]
    pub fn inc_frames_dropped(&self) {
        self.frames_dropped_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_frames_dropped_by(&self, count: u64) {
        self.frames_dropped_total
            .fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn frames_dropped_total(&self) -> u64 {
        self.frames_dropped_total.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn inc_decode_pending_sanity_violations(&self) {
        self.decode_pending_sanity_violations_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn decode_pending_sanity_violations_total(&self) -> u64 {
        self.decode_pending_sanity_violations_total
            .load(Ordering::Relaxed)
    }

    #[inline]
    pub fn inc_keyframes(&self) {
        self.keyframes_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_keyframes_dropped(&self) {
        self.keyframes_dropped_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_sink_keyframes_dropped(&self) {
        self.sink_keyframes_dropped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn sink_keyframes_dropped_total(&self) -> u64 {
        self.sink_keyframes_dropped_total.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn record_keyframe_blob_written(&self, bytes: u64) {
        self.keyframe_blobs_written_total
            .fetch_add(1, Ordering::Relaxed);
        self.keyframe_blob_bytes_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_keyframe_blob_reused(&self) {
        self.keyframe_blobs_reused_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_keyframe_blob_failure(&self) {
        self.keyframe_blob_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_restricted_zone_assertion(&self) {
        self.restricted_zone_assertions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_restricted_zone_evidence_failure(&self) {
        self.restricted_zone_evidence_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_restricted_zone_queue_dropped(&self) {
        self.restricted_zone_queue_dropped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_assertion(&self) {
        self.trigger_assertions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_binary_write_failure(&self) {
        self.trigger_binary_write_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_queue_dropped(&self) {
        self.trigger_queue_dropped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_missing_signal(&self) {
        self.trigger_missing_signal_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_local_output(&self) {
        self.trigger_local_outputs_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_trigger_local_output_failure(&self) {
        self.trigger_local_output_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_vlm_inferences(&self) {
        self.vlm_inferences_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_evaluated(&self) {
        self.novelty_evaluated_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_reused(&self) {
        self.novelty_reused_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_forced_refresh(&self) {
        self.novelty_forced_refresh_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_embedding_unavailable(&self) {
        self.novelty_embedding_unavailable_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_shadow_sampled(&self) {
        self.novelty_shadow_sampled_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_shadow_completed(&self) {
        self.novelty_shadow_completed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_novelty_shadow_changed(&self) {
        self.novelty_shadow_changed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_loop_detected(&self) {
        self.loop_detected_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_sessions_created(&self) {
        self.sessions_created_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_sessions_removed(&self) {
        self.sessions_removed_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pipeline_generation_started(&self) {
        self.pipeline_generations_active
            .fetch_add(1, Ordering::Relaxed);
        self.pipeline_generations_started_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_pipeline_start_failure(
        &self,
        fault: PipelineFault,
        join_deadline: Option<PipelineFault>,
        detached: u32,
    ) {
        self.record_pipeline_fault(fault);
        if let Some(overrun) = join_deadline {
            self.record_pipeline_fault(overrun);
            self.pipeline_generation_forced_shutdown_total
                .fetch_add(1, Ordering::Relaxed);
        }
        self.pipeline_detached_workers_total
            .fetch_add(u64::from(detached), Ordering::Relaxed);
        self.pipeline_generation_faulted_shutdown_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn pipeline_generation_stopped(&self, outcome: PipelineShutdown) {
        // One stop is recorded for every successful start. Avoid wrapping if a
        // future caller violates that invariant; observability must not create
        // a nonsensical u64 gauge.
        let _ = self.pipeline_generations_active.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |active| active.checked_sub(1),
        );
        match outcome {
            PipelineShutdown::Clean => {
                self.pipeline_generation_clean_shutdown_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineShutdown::Faulted(fault) => {
                self.record_pipeline_fault(fault);
                self.pipeline_generation_faulted_shutdown_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineShutdown::JoinDeadline {
                fault,
                overrun,
                detached,
            } => {
                if let Some(fault) = fault {
                    self.record_pipeline_fault(fault);
                }
                self.record_pipeline_fault(overrun);
                self.pipeline_generation_forced_shutdown_total
                    .fetch_add(1, Ordering::Relaxed);
                // The active gauge goes down because the generation is over,
                // but these threads are still running. Track them separately
                // so a forced shutdown is not mistaken for a full cleanup.
                self.pipeline_detached_workers_total
                    .fetch_add(u64::from(detached), Ordering::Relaxed);
            }
        }
    }

    fn record_pipeline_fault(&self, fault: PipelineFault) {
        let index = fault.stage.index() * PIPELINE_FAULT_REASON_COUNT + fault.reason.index();
        self.pipeline_worker_faults_total[index].fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        self.pipeline_last_fault_timestamp_seconds
            .store(timestamp, Ordering::Relaxed);
    }

    /// Render all counters and latency histograms as Prometheus-compatible text.
    pub fn render_prometheus(&self) -> String {
        let rtp = self.rtp_frames_received_total.load(Ordering::Relaxed);
        let decoded = self.frames_decoded_total.load(Ordering::Relaxed);
        let dropped = self.frames_dropped_total.load(Ordering::Relaxed);
        let gate_analyzed = self.gate_frames_analyzed_total.load(Ordering::Relaxed);
        let gate_selected = self.gate_keyframes_selected_total.load(Ordering::Relaxed);
        let pending_sanity = self
            .decode_pending_sanity_violations_total
            .load(Ordering::Relaxed);
        let kf = self.keyframes_total.load(Ordering::Relaxed);
        let kf_drop = self.keyframes_dropped_total.load(Ordering::Relaxed);
        let sink_kf_drop = self.sink_keyframes_dropped_total.load(Ordering::Relaxed);
        let keyframe_blobs_written = self.keyframe_blobs_written_total.load(Ordering::Relaxed);
        let keyframe_blobs_reused = self.keyframe_blobs_reused_total.load(Ordering::Relaxed);
        let keyframe_blob_failures = self.keyframe_blob_failures_total.load(Ordering::Relaxed);
        let keyframe_blob_bytes = self.keyframe_blob_bytes_total.load(Ordering::Relaxed);
        let restricted_zone_assertions = self
            .restricted_zone_assertions_total
            .load(Ordering::Relaxed);
        let restricted_zone_evidence_failures = self
            .restricted_zone_evidence_failures_total
            .load(Ordering::Relaxed);
        let restricted_zone_queue_dropped = self
            .restricted_zone_queue_dropped_total
            .load(Ordering::Relaxed);
        let trigger_assertions = self.trigger_assertions_total.load(Ordering::Relaxed);
        let trigger_binary_write_failures = self
            .trigger_binary_write_failures_total
            .load(Ordering::Relaxed);
        let trigger_queue_dropped = self.trigger_queue_dropped_total.load(Ordering::Relaxed);
        let trigger_missing_signal = self.trigger_missing_signal_total.load(Ordering::Relaxed);
        let trigger_local_outputs = self.trigger_local_outputs_total.load(Ordering::Relaxed);
        let trigger_local_output_failures = self
            .trigger_local_output_failures_total
            .load(Ordering::Relaxed);
        let vlm = self.vlm_inferences_total.load(Ordering::Relaxed);
        let novelty_evaluated = self.novelty_evaluated_total.load(Ordering::Relaxed);
        let novelty_reused = self.novelty_reused_total.load(Ordering::Relaxed);
        let novelty_forced_refresh = self.novelty_forced_refresh_total.load(Ordering::Relaxed);
        let novelty_embedding_unavailable = self
            .novelty_embedding_unavailable_total
            .load(Ordering::Relaxed);
        let novelty_shadow_sampled = self.novelty_shadow_sampled_total.load(Ordering::Relaxed);
        let novelty_shadow_completed = self.novelty_shadow_completed_total.load(Ordering::Relaxed);
        let novelty_shadow_changed = self.novelty_shadow_changed_total.load(Ordering::Relaxed);
        let novelty_reuse_ratio = if novelty_evaluated == 0 {
            0.0
        } else {
            novelty_reused as f64 / novelty_evaluated as f64
        };
        let novelty_shadow_change_ratio = if novelty_shadow_completed == 0 {
            0.0
        } else {
            novelty_shadow_changed as f64 / novelty_shadow_completed as f64
        };
        let loops = self.loop_detected_total.load(Ordering::Relaxed);
        let sess_created = self.sessions_created_total.load(Ordering::Relaxed);
        let sess_removed = self.sessions_removed_total.load(Ordering::Relaxed);
        let generations_active = self.pipeline_generations_active.load(Ordering::Relaxed);
        let generations_started = self
            .pipeline_generations_started_total
            .load(Ordering::Relaxed);
        let generations_clean = self
            .pipeline_generation_clean_shutdown_total
            .load(Ordering::Relaxed);
        let generations_faulted = self
            .pipeline_generation_faulted_shutdown_total
            .load(Ordering::Relaxed);
        let detached_workers = self.pipeline_detached_workers_total.load(Ordering::Relaxed);
        let generations_forced = self
            .pipeline_generation_forced_shutdown_total
            .load(Ordering::Relaxed);
        let last_fault_timestamp = self
            .pipeline_last_fault_timestamp_seconds
            .load(Ordering::Relaxed);

        let mut out = format!(
            "vidarax_pipeline_rtp_frames_received_total {rtp}\n\
             vidarax_pipeline_frames_decoded_total {decoded}\n\
             vidarax_pipeline_frames_dropped_total {dropped}\n\
             vidarax_pipeline_gate_frames_analyzed_total {gate_analyzed}\n\
             vidarax_pipeline_gate_keyframes_selected_total {gate_selected}\n\
             vidarax_pipeline_decode_pending_sanity_violations_total {pending_sanity}\n\
             vidarax_pipeline_keyframes_total {kf}\n\
             vidarax_pipeline_keyframes_dropped_total {kf_drop}\n\
             vidarax_pipeline_sink_keyframes_dropped_total {sink_kf_drop}\n\
             vidarax_pipeline_keyframe_blobs_written_total {keyframe_blobs_written}\n\
             vidarax_pipeline_keyframe_blobs_reused_total {keyframe_blobs_reused}\n\
             vidarax_pipeline_keyframe_blob_failures_total {keyframe_blob_failures}\n\
             vidarax_pipeline_keyframe_blob_bytes_total {keyframe_blob_bytes}\n\
             vidarax_pipeline_restricted_zone_assertions_total {restricted_zone_assertions}\n\
             vidarax_pipeline_restricted_zone_evidence_failures_total {restricted_zone_evidence_failures}\n\
             vidarax_pipeline_restricted_zone_queue_dropped_total {restricted_zone_queue_dropped}\n\
             vidarax_pipeline_trigger_assertions_total {trigger_assertions}\n\
             vidarax_pipeline_trigger_binary_write_failures_total {trigger_binary_write_failures}\n\
             vidarax_pipeline_trigger_queue_dropped_total {trigger_queue_dropped}\n\
             vidarax_pipeline_trigger_missing_signal_total {trigger_missing_signal}\n\
             vidarax_pipeline_trigger_local_outputs_total {trigger_local_outputs}\n\
             vidarax_pipeline_trigger_local_output_failures_total {trigger_local_output_failures}\n\
             vidarax_pipeline_vlm_inferences_total {vlm}\n\
             vidarax_pipeline_novelty_evaluated_total {novelty_evaluated}\n\
             vidarax_pipeline_novelty_reused_total {novelty_reused}\n\
             vidarax_pipeline_novelty_forced_refresh_total {novelty_forced_refresh}\n\
             vidarax_pipeline_novelty_embedding_unavailable_total {novelty_embedding_unavailable}\n\
             vidarax_pipeline_novelty_shadow_sampled_total {novelty_shadow_sampled}\n\
             vidarax_pipeline_novelty_shadow_completed_total {novelty_shadow_completed}\n\
             vidarax_pipeline_novelty_shadow_changed_total {novelty_shadow_changed}\n\
             vidarax_pipeline_novelty_reuse_ratio {novelty_reuse_ratio}\n\
             vidarax_pipeline_novelty_shadow_change_ratio {novelty_shadow_change_ratio}\n\
             vidarax_pipeline_loop_detected_total {loops}\n\
             vidarax_pipeline_sessions_created_total {sess_created}\n\
             vidarax_pipeline_sessions_removed_total {sess_removed}\n\
             vidarax_pipeline_generations_active {generations_active}\n\
             vidarax_pipeline_generations_started_total {generations_started}\n\
             vidarax_pipeline_generation_shutdown_total{{outcome=\"clean\"}} {generations_clean}\n\
             vidarax_pipeline_generation_shutdown_total{{outcome=\"faulted\"}} {generations_faulted}\n\
             vidarax_pipeline_generation_shutdown_total{{outcome=\"forced\"}} {generations_forced}\n\
             vidarax_pipeline_generation_shutdown_clean_total {generations_clean}\n\
             vidarax_pipeline_generation_shutdown_faulted_total {generations_faulted}\n\
             vidarax_pipeline_generation_shutdown_forced_total {generations_forced}\n\
             vidarax_pipeline_detached_workers_total {detached_workers}\n\
             vidarax_pipeline_last_fault_timestamp_seconds {last_fault_timestamp}\n"
        );

        use std::fmt::Write as _;
        for stage in PipelineStage::ALL {
            for reason in PipelineFaultReason::ALL {
                let index = stage.index() * PIPELINE_FAULT_REASON_COUNT + reason.index();
                let count = self.pipeline_worker_faults_total[index].load(Ordering::Relaxed);
                let _ = writeln!(
                    out,
                    "vidarax_pipeline_worker_faults_total{{stage=\"{}\",reason=\"{}\"}} {count}",
                    stage.as_str(),
                    reason.as_str(),
                );
            }
        }
        let worker_faults_total = self
            .pipeline_worker_faults_total
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .sum::<u64>();
        let _ = writeln!(
            out,
            "vidarax_pipeline_worker_faults_total_all {worker_faults_total}"
        );

        out.push_str(
            &self
                .decode_latency_us
                .render_prometheus("vidarax_pipeline_decode_latency_us", "us"),
        );
        out.push_str(
            &self
                .gate_latency_us
                .render_prometheus("vidarax_pipeline_gate_latency_us", "us"),
        );
        out.push_str(
            &self
                .vlm_latency_ms
                .render_prometheus("vidarax_pipeline_vlm_latency_ms", "ms"),
        );
        out.push_str(
            &self
                .novelty_embedding_latency_ms
                .render_prometheus("vidarax_pipeline_novelty_embedding_latency_ms", "ms"),
        );
        out.push_str(
            &self
                .keyframe_blob_latency_ms
                .render_prometheus("vidarax_pipeline_keyframe_blob_latency_ms", "ms"),
        );
        out.push_str(
            &self
                .stdb_emit_latency_ms
                .render_prometheus("vidarax_pipeline_stdb_emit_latency_ms", "ms"),
        );

        out
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{LatencyHistogram, PipelineMetrics};
    use crate::webrtc::runtime::{
        PipelineFault, PipelineFaultReason, PipelineShutdown, PipelineStage,
    };

    #[test]
    fn counters_increment_and_render() {
        let m = PipelineMetrics::new();
        m.inc_rtp_received();
        m.inc_rtp_received();
        m.inc_frames_decoded();
        m.inc_frames_dropped();
        m.inc_keyframes();
        m.inc_vlm_inferences();
        m.inc_sessions_created();
        m.inc_restricted_zone_assertion();
        m.inc_restricted_zone_evidence_failure();
        m.inc_restricted_zone_queue_dropped();
        m.inc_trigger_assertion();
        m.inc_trigger_binary_write_failure();
        m.inc_trigger_queue_dropped();
        m.inc_trigger_missing_signal();
        m.inc_trigger_local_output();
        m.inc_trigger_local_output_failure();

        let text = m.render_prometheus();
        assert!(text.contains("vidarax_pipeline_rtp_frames_received_total 2"));
        assert!(text.contains("vidarax_pipeline_frames_decoded_total 1"));
        assert!(text.contains("vidarax_pipeline_frames_dropped_total 1"));
        assert!(text.contains("vidarax_pipeline_keyframes_total 1"));
        assert!(text.contains("vidarax_pipeline_vlm_inferences_total 1"));
        assert!(text.contains("vidarax_pipeline_sessions_created_total 1"));
        assert!(text.contains("vidarax_pipeline_keyframes_dropped_total 0"));
        assert!(text.contains("vidarax_pipeline_sink_keyframes_dropped_total 0"));
        assert!(text.contains("vidarax_pipeline_restricted_zone_assertions_total 1"));
        assert!(text.contains("vidarax_pipeline_restricted_zone_evidence_failures_total 1"));
        assert!(text.contains("vidarax_pipeline_restricted_zone_queue_dropped_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_assertions_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_binary_write_failures_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_queue_dropped_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_missing_signal_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_local_outputs_total 1"));
        assert!(text.contains("vidarax_pipeline_trigger_local_output_failures_total 1"));
    }

    #[test]
    fn histogram_records_into_correct_bucket() {
        let h = LatencyHistogram::new([100, 250, 500, 1_000, 2_000, 5_000, 10_000, 50_000]);
        h.record(80); // → bucket[0] (≤100)
        h.record(200); // → bucket[1] (≤250)
        h.record(200); // → bucket[1] (≤250)
        h.record(60_000); // → +Inf (above 50 000)

        let text = h.render_prometheus("vidarax_pipeline_decode_latency_us", "us");

        // Cumulative counts: bucket 0 = 1, bucket 1 = 1+2 = 3, rest = 3 until +Inf = 4.
        assert!(text.contains("bucket{le=\"100\"} 1"));
        assert!(text.contains("bucket{le=\"250\"} 3"));
        assert!(text.contains("bucket{le=\"500\"} 3"));
        assert!(text.contains("bucket{le=\"+Inf\"} 4"));
        assert!(text.contains("_sum 60480")); // 80 + 200 + 200 + 60000
        assert!(text.contains("_count 4"));
    }

    #[test]
    fn histogram_renders_in_pipeline_metrics() {
        let m = PipelineMetrics::new();
        m.decode_latency_us.record(300); // → bucket ≤500 µs
        m.gate_latency_us.record(8); // → bucket ≤10 µs
        m.vlm_latency_ms.record(180); // → bucket ≤250 ms
        m.stdb_emit_latency_ms.record(3); // → bucket ≤5 ms

        let text = m.render_prometheus();
        assert!(text.contains("vidarax_pipeline_decode_latency_us_count 1"));
        assert!(text.contains("vidarax_pipeline_gate_latency_us_count 1"));
        assert!(text.contains("vidarax_pipeline_vlm_latency_ms_count 1"));
        assert!(text.contains("vidarax_pipeline_stdb_emit_latency_ms_count 1"));
    }

    #[test]
    fn generation_lifecycle_uses_fixed_stage_and_reason_labels() {
        let m = PipelineMetrics::new();
        m.pipeline_generation_started();
        m.pipeline_generation_stopped(PipelineShutdown::Faulted(PipelineFault {
            stage: PipelineStage::Decode,
            reason: PipelineFaultReason::Panic,
        }));

        let text = m.render_prometheus();
        assert!(text.contains("vidarax_pipeline_generations_active 0\n"));
        assert!(text.contains("vidarax_pipeline_generations_started_total 1\n"));
        assert!(text.contains("vidarax_pipeline_generation_shutdown_total{outcome=\"faulted\"} 1"));
        assert!(text
            .contains("vidarax_pipeline_worker_faults_total{stage=\"decode\",reason=\"panic\"} 1"));
        assert!(!text.contains("session_id="));
    }
}

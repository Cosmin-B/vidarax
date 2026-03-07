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
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//! use vidarax_core::metrics::PipelineMetrics;
//!
//! let m = Arc::new(PipelineMetrics::new());
//! m.inc_rtp_received();
//! m.inc_frames_decoded();
//! m.decode_latency_us.record(420);
//! let text = m.render_prometheus();
//! assert!(text.contains("vidarax_pipeline_rtp_frames_received_total 1"));
//! assert!(text.contains("vidarax_pipeline_decode_latency_us_count 1"));
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

// ─── LatencyHistogram ─────────────────────────────────────────────────────────

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
    /// Upper bounds for the 8 buckets (values in the same unit as observations).
    bounds: [u64; 8],
    /// Per-bucket non-cumulative observation counts.
    buckets: [AtomicU64; 8],
    /// Running sum of all observed values.
    sum: AtomicU64,
    /// Total number of observations.
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
    ///
    /// Finds the first bucket whose upper bound is `>= value` and increments
    /// it; also increments `sum` and `count`.  O(8) — branchless inner loop.
    #[inline]
    pub fn record(&self, value: u64) {
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (idx, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.buckets[idx].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Value exceeds the largest explicit bound — counts only in +Inf.
    }

    /// Emit Prometheus histogram lines for this histogram.
    ///
    /// `metric_name` is the base name (e.g. `"vidarax_pipeline_decode_latency_us"`).
    /// `unit` is the label suffix used in `le` values (the bounds are raw
    /// numbers, so the caller controls how they are presented).
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

// ─── PipelineMetrics ──────────────────────────────────────────────────────────

/// Zero-alloc pipeline metrics backed by `AtomicU64` counters and latency
/// histograms.
///
/// Histogram bucket boundaries per stage:
///
/// | Histogram            | Bounds (µs)                                     |
/// |----------------------|-------------------------------------------------|
/// | `decode_latency_us`  | 100, 250, 500, 1000, 2000, 5000, 10000, 50000  |
/// | `gate_latency_us`    | 1, 5, 10, 50, 100, 500, 1000, 5000             |
/// | `vlm_latency_ms`     | 50, 100, 250, 500, 1000, 2500, 5000, 10000     |
/// | `stdb_emit_latency_ms` | 1, 5, 10, 25, 50, 100, 250, 1000            |
pub struct PipelineMetrics {
    /// RTP frames received by decode workers.
    rtp_frames_received_total: AtomicU64,
    /// Video frames decoded successfully.
    frames_decoded_total: AtomicU64,
    /// Keyframes forwarded from analysis workers to VLM workers.
    keyframes_total: AtomicU64,
    /// Keyframes dropped because the VLM queue was full.
    keyframes_dropped_total: AtomicU64,
    /// VLM inference calls dispatched.
    vlm_inferences_total: AtomicU64,
    /// Loop detection events emitted by analysis workers.
    loop_detected_total: AtomicU64,
    /// WHIP sessions created.
    sessions_created_total: AtomicU64,
    /// WHIP sessions removed (terminated).
    sessions_removed_total: AtomicU64,

    // ── Latency histograms ────────────────────────────────────────────────
    /// Per-frame H.264 decode latency in microseconds.
    ///
    /// Bounds: [100, 250, 500, 1000, 2000, 5000, 10000, 50000] µs
    pub decode_latency_us: LatencyHistogram,
    /// Gate engine processing latency in microseconds.
    ///
    /// Bounds: [1, 5, 10, 50, 100, 500, 1000, 5000] µs
    pub gate_latency_us: LatencyHistogram,
    /// VLM inference round-trip latency in milliseconds.
    ///
    /// Bounds: [50, 100, 250, 500, 1000, 2500, 5000, 10000] ms
    pub vlm_latency_ms: LatencyHistogram,
    /// SpacetimeDB HTTP POST latency in milliseconds.
    ///
    /// Bounds: [1, 5, 10, 25, 50, 100, 250, 1000] ms
    pub stdb_emit_latency_ms: LatencyHistogram,
}

impl PipelineMetrics {
    /// Create a new zero-initialised set of counters and histograms.
    pub fn new() -> Self {
        Self {
            rtp_frames_received_total: AtomicU64::new(0),
            frames_decoded_total: AtomicU64::new(0),
            keyframes_total: AtomicU64::new(0),
            keyframes_dropped_total: AtomicU64::new(0),
            vlm_inferences_total: AtomicU64::new(0),
            loop_detected_total: AtomicU64::new(0),
            sessions_created_total: AtomicU64::new(0),
            sessions_removed_total: AtomicU64::new(0),

            decode_latency_us: LatencyHistogram::new([
                100, 250, 500, 1_000, 2_000, 5_000, 10_000, 50_000,
            ]),
            gate_latency_us: LatencyHistogram::new([1, 5, 10, 50, 100, 500, 1_000, 5_000]),
            vlm_latency_ms: LatencyHistogram::new([
                50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000,
            ]),
            stdb_emit_latency_ms: LatencyHistogram::new([1, 5, 10, 25, 50, 100, 250, 1_000]),
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

    #[inline]
    pub fn inc_keyframes(&self) {
        self.keyframes_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_keyframes_dropped(&self) {
        self.keyframes_dropped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_vlm_inferences(&self) {
        self.vlm_inferences_total.fetch_add(1, Ordering::Relaxed);
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

    /// Render all counters and latency histograms as Prometheus-compatible text.
    pub fn render_prometheus(&self) -> String {
        let rtp = self
            .rtp_frames_received_total
            .load(Ordering::Relaxed);
        let decoded = self.frames_decoded_total.load(Ordering::Relaxed);
        let kf = self.keyframes_total.load(Ordering::Relaxed);
        let kf_drop = self.keyframes_dropped_total.load(Ordering::Relaxed);
        let vlm = self.vlm_inferences_total.load(Ordering::Relaxed);
        let loops = self.loop_detected_total.load(Ordering::Relaxed);
        let sess_created = self.sessions_created_total.load(Ordering::Relaxed);
        let sess_removed = self.sessions_removed_total.load(Ordering::Relaxed);

        let mut out = format!(
            "vidarax_pipeline_rtp_frames_received_total {rtp}\n\
             vidarax_pipeline_frames_decoded_total {decoded}\n\
             vidarax_pipeline_keyframes_total {kf}\n\
             vidarax_pipeline_keyframes_dropped_total {kf_drop}\n\
             vidarax_pipeline_vlm_inferences_total {vlm}\n\
             vidarax_pipeline_loop_detected_total {loops}\n\
             vidarax_pipeline_sessions_created_total {sess_created}\n\
             vidarax_pipeline_sessions_removed_total {sess_removed}\n"
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

    #[test]
    fn counters_increment_and_render() {
        let m = PipelineMetrics::new();
        m.inc_rtp_received();
        m.inc_rtp_received();
        m.inc_frames_decoded();
        m.inc_keyframes();
        m.inc_vlm_inferences();
        m.inc_sessions_created();

        let text = m.render_prometheus();
        assert!(text.contains("vidarax_pipeline_rtp_frames_received_total 2"));
        assert!(text.contains("vidarax_pipeline_frames_decoded_total 1"));
        assert!(text.contains("vidarax_pipeline_keyframes_total 1"));
        assert!(text.contains("vidarax_pipeline_vlm_inferences_total 1"));
        assert!(text.contains("vidarax_pipeline_sessions_created_total 1"));
        assert!(text.contains("vidarax_pipeline_keyframes_dropped_total 0"));
    }

    #[test]
    fn histogram_records_into_correct_bucket() {
        let h = LatencyHistogram::new([100, 250, 500, 1_000, 2_000, 5_000, 10_000, 50_000]);
        h.record(80);   // → bucket[0] (≤100)
        h.record(200);  // → bucket[1] (≤250)
        h.record(200);  // → bucket[1] (≤250)
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
        m.gate_latency_us.record(8);     // → bucket ≤10 µs
        m.vlm_latency_ms.record(180);    // → bucket ≤250 ms
        m.stdb_emit_latency_ms.record(3); // → bucket ≤5 ms

        let text = m.render_prometheus();
        assert!(text.contains("vidarax_pipeline_decode_latency_us_count 1"));
        assert!(text.contains("vidarax_pipeline_gate_latency_us_count 1"));
        assert!(text.contains("vidarax_pipeline_vlm_latency_ms_count 1"));
        assert!(text.contains("vidarax_pipeline_stdb_emit_latency_ms_count 1"));
    }
}

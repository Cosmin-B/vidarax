//! Zero-alloc pipeline metrics.
//!
//! Every counter is backed by an `AtomicU64` and uses `Relaxed` ordering.
//! These are observability counters, not synchronisation primitives; strict
//! ordering guarantees across different counters are intentionally not
//! provided.
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
//! let text = m.render_prometheus();
//! assert!(text.contains("vidarax_pipeline_rtp_frames_received_total 1"));
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

/// Zero-alloc pipeline metrics backed by `AtomicU64` counters.
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
}

impl PipelineMetrics {
    /// Create a new zero-initialised set of counters.
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

    /// Render all counters as Prometheus-compatible text.
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

        format!(
            "vidarax_pipeline_rtp_frames_received_total {rtp}\n\
             vidarax_pipeline_frames_decoded_total {decoded}\n\
             vidarax_pipeline_keyframes_total {kf}\n\
             vidarax_pipeline_keyframes_dropped_total {kf_drop}\n\
             vidarax_pipeline_vlm_inferences_total {vlm}\n\
             vidarax_pipeline_loop_detected_total {loops}\n\
             vidarax_pipeline_sessions_created_total {sess_created}\n\
             vidarax_pipeline_sessions_removed_total {sess_removed}\n"
        )
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineMetrics;

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
}

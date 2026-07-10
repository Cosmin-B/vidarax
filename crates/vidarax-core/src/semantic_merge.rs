//! T3 note-merge — collapse a stream of per-chunk VLM descriptions into a small
//! set of *activity intervals*, and (separately) skip VLM calls on frozen frames.
//!
//! # Why this exists
//!
//! The single expensive resource is the VLM. On dense, chrome-heavy footage
//! (an editor screenshare, a dashboard) two things hold, the second measured:
//!
//! 1. **No cheap upstream pixel signal can safely skip a call while the screen
//!    is *active*.** Any active frame might carry a short moment that is gone
//!    before the next scheduled call, so a call-saver only fires when the cheap
//!    signal reads the screen as *frozen*. That frozen test is a calibrated
//!    heuristic, not a proof: a coarse hash or embedding can read two genuinely
//!    different frames as identical, so a reuse keeps a small residual chance of
//!    dropping such a moment. `max_hold` bounds how long that chance can run
//!    before a fresh call is forced; it does not remove it.
//! 2. **The redundancy leaves the output, not the bill.** The VLM re-describes
//!    the same activity many times over; a semantic embedding of its own text
//!    collapses those repeats about 7.5x on the one screen-capture fixture
//!    measured here (150 chunks to 20 activities at cosine 0.75, guarded against
//!    quantisation drift) without merging genuinely different activities.
//!
//! This module is library-only for now: nothing in the live capture pipeline
//! calls these gates yet, so the call savings described here are design targets
//! rather than measured live results.
//!
//! This module implements both, sharing one fast primitive:
//!
//! * [`SemanticMerge`] — the downstream activity state machine. Per analysed
//!   chunk the decision is **binary**: *extend* the open activity or *start* a
//!   new one. There is no third "escalate" branch and no second model. A large
//!   boundary jump is recorded as [`ActivityNote::boundary_novelty`] so a
//!   downstream reader can *flag* important moments without the merge itself
//!   ever making a three-way call.
//! * [`PhashFrameGate`] — the perceptual-hash upstream call-saver. Keyed on the
//!   64-bit perceptual hash the frame pipeline already computes, so it needs no
//!   embedding, no inference server, and no ONNX dependency on the per-frame
//!   path. Same admit-on-doubt contract as below.
//! * [`StaticFrameGate`] — the embedding-based variant of the same idea, kept
//!   for callers that already have a frame embedding in hand. **Admit-on-doubt**:
//!   it only returns [`GateDecision::Reuse`] when the frame is near-identical to
//!   the last *analysed* frame, so a reuse is unlikely to drop a moment, though
//!   a coarse embedding match leaves a small residual risk. On active
//!   content it correctly almost never fires; its wins come from idle / paused
//!   stretches, and it generalises to any video that has static spans.
//!
//! # Cost
//!
//! The per-decision scoring path is an int8 dot product
//! ([`crate::novelty::dot_i8`]) over the quantised anchor, the same path the T2
//! gate uses, and it allocates nothing. Starting a new activity does allocate:
//! it quantises a fresh anchor buffer and the emitted [`ActivityNote`] owns its
//! representative description (one `String`). Notes are produced at activity
//! cadence, not per frame; how many that is over any span depends entirely on
//! the footage.

use serde::{Deserialize, Serialize};

use crate::novelty::{dot_i8, quantize_unit_into};
use crate::timeline::TimelineEvent;

/// [`TimelineEvent::kind`] written for a collapsed activity interval.
pub const ACTIVITY_EVENT_KIND: &str = "activity";

/// Default cosine at or above which two consecutive descriptions are the *same*
/// activity. Measured sweet spot on real Gemini output: keeps distinct
/// activities apart while collapsing paraphrase (0.55 over-merges, 0.85
/// over-splits; 0.75 landed 150 chunks → 20 activities).
pub const DEFAULT_MERGE_THRESHOLD: f32 = 0.75;

/// Default cosine at or above which a frame is treated as *frozen* vs the last
/// analysed frame. Deliberately high: reuse must never cost a moment, so we only
/// skip when the pixels barely moved.
pub const DEFAULT_STATIC_THRESHOLD: f32 = 0.996;

/// Default cap on consecutive reuses before a fresh call is forced. Bounds
/// staleness a second way, independent of the similarity threshold.
pub const DEFAULT_MAX_HOLD: u32 = 32;

/// Configuration for [`SemanticMerge`].
#[derive(Debug, Clone, Copy)]
pub struct MergeConfig {
    /// Cosine `>=` this ⇒ extend the open activity; below ⇒ start a new one.
    pub merge_threshold: f32,
    /// Dimensionality of the semantic text embedding (e.g. 384 for MiniLM-L6).
    pub embed_dim: usize,
}

impl MergeConfig {
    /// Config for a given embedding width using the measured default threshold.
    pub fn new(embed_dim: usize) -> Self {
        Self {
            merge_threshold: DEFAULT_MERGE_THRESHOLD,
            embed_dim,
        }
    }
}

/// One collapsed activity interval — the unit written to the journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityNote {
    /// Presentation timestamp of the chunk that opened this activity.
    pub start_pts_ms: u64,
    /// Presentation timestamp of the most recent chunk still in this activity.
    pub end_pts_ms: u64,
    /// Number of analysed chunks that collapsed into this activity (`>= 1`).
    pub chunk_count: u32,
    /// The description of the chunk that opened the activity — its representative.
    pub description: String,
    /// `1 − cosine` between this activity's anchor and the *previous* activity's
    /// anchor, in `[0, 1]`. High ⇒ a sharp change worth flagging. The first
    /// activity of a stream is `1.0` (everything is new).
    pub boundary_novelty: f32,
}

impl ActivityNote {
    /// Encode this activity as an append-only journal event.
    ///
    /// The note becomes the event `payload` as compact JSON (`kind =
    /// "activity"`), timestamped at the activity's start. `boundary_novelty` is
    /// clamped to `[0,1]`, so the JSON never carries a non-finite float — safe
    /// for the tab-delimited WAL, which only escapes tabs and newlines.
    pub fn to_timeline_event(
        &self,
        seq: u64,
        run_id: impl Into<String>,
        stream_id: impl Into<String>,
    ) -> TimelineEvent {
        // Infallible for this flat, finite struct; empty payload on the
        // impossible error keeps this non-panicking on the emit path.
        let payload = serde_json::to_string(self).unwrap_or_default();
        debug_assert!(!payload.is_empty(), "activity note must serialise");
        TimelineEvent {
            seq,
            run_id: run_id.into(),
            stream_id: stream_id.into(),
            pts_ms: self.start_pts_ms,
            kind: ACTIVITY_EVENT_KIND.to_owned(),
            payload,
        }
    }
}

/// The open (not-yet-closed) activity, held between [`SemanticMerge::observe`]
/// calls. Split out so the anchor's quantised form lives next to its metadata.
struct OpenActivity {
    start_pts_ms: u64,
    end_pts_ms: u64,
    chunk_count: u32,
    description: String,
    boundary_novelty: f32,
    /// Quantised, L2-normalised anchor embedding (`embed_dim` int8 lanes).
    anchor_q: Vec<i8>,
    /// Scale such that `anchor_q[i] · anchor_scale ≈ unit_anchor[i]`.
    anchor_scale: f32,
}

impl OpenActivity {
    fn into_note(self) -> ActivityNote {
        ActivityNote {
            start_pts_ms: self.start_pts_ms,
            end_pts_ms: self.end_pts_ms,
            chunk_count: self.chunk_count,
            description: self.description,
            boundary_novelty: self.boundary_novelty,
        }
    }
}

/// Streaming activity merge. One instance per stream; not `Sync`.
///
/// Feed analysed chunks in presentation order with [`observe`]; each call
/// returns the just-*closed* activity when the chunk began a new one, and `None`
/// when it extended the open one. Call [`flush`] at end-of-stream to emit the
/// final open activity.
///
/// [`observe`]: SemanticMerge::observe
/// [`flush`]: SemanticMerge::flush
pub struct SemanticMerge {
    cfg: MergeConfig,
    open: Option<OpenActivity>,
    /// Scratch for the incoming embedding's quantised form; reused every call.
    scratch_q: Vec<i8>,
}

impl SemanticMerge {
    /// Build a merge for the given config, pre-allocating the scratch lane.
    pub fn new(cfg: MergeConfig) -> Self {
        let dim = cfg.embed_dim.max(1);
        Self {
            cfg: MergeConfig {
                embed_dim: dim,
                ..cfg
            },
            open: None,
            scratch_q: vec![0i8; dim],
        }
    }

    /// Merge with the measured default threshold for `embed_dim`.
    pub fn with_dim(embed_dim: usize) -> Self {
        Self::new(MergeConfig::new(embed_dim))
    }

    /// Number of chunks currently collapsed into the open (unclosed) activity;
    /// `0` when no activity is open.
    pub fn open_chunk_count(&self) -> u32 {
        self.open.as_ref().map_or(0, |a| a.chunk_count)
    }

    /// Feed one analysed chunk.
    ///
    /// Returns `Some(closed_note)` when `embedding` was different enough from the
    /// open activity's anchor to begin a new activity — the returned note is the
    /// activity that just closed. Returns `None` when the chunk extended the open
    /// activity (its end-time is bumped and its chunk count incremented).
    ///
    /// A wrong-length embedding is treated as maximally novel (forces a new
    /// activity) — the safe direction, never a false merge.
    pub fn observe(
        &mut self,
        pts_ms: u64,
        embedding: &[f32],
        description: &str,
    ) -> Option<ActivityNote> {
        let dim = self.cfg.embed_dim;

        // Quantise the incoming embedding once; `scale == 0.0` marks "unusable"
        // (wrong length or zero vector) → never merges.
        let inc_scale = if embedding.len() == dim {
            quantize_unit_into(embedding, &mut self.scratch_q)
        } else {
            0.0
        };

        // Cosine to the open anchor, or -1 (maximally novel) if usable-ness fails
        // or there is no open activity yet.
        let sim = match (&self.open, inc_scale) {
            (Some(open), s) if s != 0.0 => {
                let dot = dot_i8(&self.scratch_q, &open.anchor_q) as f32;
                dot * inc_scale * open.anchor_scale
            }
            _ => -1.0,
        };

        // Extend: same activity. Anchor stays put (paraphrase drift must not
        // walk the anchor away from the activity's opening meaning). The
        // `if let` also makes this panic-free for any threshold a caller sets.
        if sim >= self.cfg.merge_threshold {
            if let Some(open) = self.open.as_mut() {
                open.end_pts_ms = pts_ms;
                open.chunk_count += 1;
                return None;
            }
        }

        // Start a new activity. Its boundary novelty is measured against the
        // activity we are about to close (1.0 for the very first activity).
        let boundary_novelty = if sim <= -1.0 && self.open.is_none() {
            1.0
        } else {
            (1.0 - sim).clamp(0.0, 1.0)
        };

        let mut anchor_q = vec![0i8; dim];
        let anchor_scale = if inc_scale != 0.0 {
            anchor_q.copy_from_slice(&self.scratch_q);
            inc_scale
        } else {
            // Unusable embedding: a zero anchor's cosine with anything is 0, so
            // the next chunk is compared as "novel" and this degrades to
            // never-merge rather than wrongly-merge.
            1.0
        };

        let closed = self.open.replace(OpenActivity {
            start_pts_ms: pts_ms,
            end_pts_ms: pts_ms,
            chunk_count: 1,
            description: description.to_owned(),
            boundary_novelty,
            anchor_q,
            anchor_scale,
        });

        closed.map(OpenActivity::into_note)
    }

    /// Close and return the final open activity, if any. Leaves the merge empty.
    pub fn flush(&mut self) -> Option<ActivityNote> {
        self.open.take().map(OpenActivity::into_note)
    }
}

/// Configuration for [`StaticFrameGate`].
#[derive(Debug, Clone, Copy)]
pub struct StaticGateConfig {
    /// Cosine `>=` this vs the last analysed frame ⇒ frozen ⇒ reuse is safe.
    pub static_threshold: f32,
    /// Force a fresh call after this many consecutive reuses (drift guard).
    pub max_hold: u32,
    /// Dimensionality of the frame embedding (e.g. 768 for the SigLIP encoder).
    pub embed_dim: usize,
}

impl StaticGateConfig {
    /// Config for a given embedding width using the measured-safe defaults.
    pub fn new(embed_dim: usize) -> Self {
        Self {
            static_threshold: DEFAULT_STATIC_THRESHOLD,
            max_hold: DEFAULT_MAX_HOLD,
            embed_dim,
        }
    }
}

/// Whether a frame needs a fresh VLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Call the VLM: the frame moved, or the hold cap was reached.
    Run,
    /// Skip the VLM and carry the last activity forward: the frame is frozen.
    Reuse,
}

/// Upstream near-identity reuse gate. One instance per stream; not `Sync`.
///
/// **Admit-on-doubt.** The anchor is the last frame we actually *analysed*, not
/// the last frame we *saw* — so slow drift is measured against a fixed reference
/// and cannot creep past the threshold unnoticed. Combined with `max_hold`, a
/// reuse run is bounded both by similarity and by count. In the common case the
/// only error is a redundant call (overspend), but an embedding that scores two
/// genuinely different frames as near-identical can still reuse across a real
/// change, so the false-reuse risk is small rather than zero.
pub struct StaticFrameGate {
    cfg: StaticGateConfig,
    anchor_q: Vec<i8>,
    anchor_scale: f32,
    has_anchor: bool,
    held: u32,
    scratch_q: Vec<i8>,
}

impl StaticFrameGate {
    /// Build a gate for the given config, pre-allocating anchor + scratch lanes.
    pub fn new(cfg: StaticGateConfig) -> Self {
        let dim = cfg.embed_dim.max(1);
        Self {
            cfg: StaticGateConfig {
                embed_dim: dim,
                ..cfg
            },
            anchor_q: vec![0i8; dim],
            anchor_scale: 1.0,
            has_anchor: false,
            held: 0,
            scratch_q: vec![0i8; dim],
        }
    }

    /// Gate with measured-safe defaults for `embed_dim`.
    pub fn with_dim(embed_dim: usize) -> Self {
        Self::new(StaticGateConfig::new(embed_dim))
    }

    /// Consecutive reuses since the last fresh call.
    pub fn held(&self) -> u32 {
        self.held
    }

    /// Decide whether `frame_embedding` needs a fresh VLM call.
    ///
    /// A wrong-length or zero embedding always yields [`GateDecision::Run`]
    /// without disturbing the anchor — we never reuse on a signal we can't trust.
    pub fn decide(&mut self, frame_embedding: &[f32]) -> GateDecision {
        if frame_embedding.len() != self.cfg.embed_dim {
            return GateDecision::Run;
        }
        let inc_scale = quantize_unit_into(frame_embedding, &mut self.scratch_q);
        if inc_scale == 0.0 {
            return GateDecision::Run;
        }

        if self.has_anchor {
            let dot = dot_i8(&self.scratch_q, &self.anchor_q) as f32;
            let sim = dot * inc_scale * self.anchor_scale;
            if sim >= self.cfg.static_threshold && self.held < self.cfg.max_hold {
                self.held += 1;
                return GateDecision::Reuse;
            }
        }

        // Fresh call: this frame becomes the new anchor and resets the hold.
        self.anchor_q.copy_from_slice(&self.scratch_q);
        self.anchor_scale = inc_scale;
        self.has_anchor = true;
        self.held = 0;
        GateDecision::Run
    }

    /// Forget the anchor so the next frame is always analysed (e.g. after a
    /// stream discontinuity or seek).
    pub fn reset(&mut self) {
        self.has_anchor = false;
        self.held = 0;
    }
}

/// Default max Hamming distance (of 64) at which a frame counts as *frozen* vs
/// the last analysed frame. The 8×8 average-hash barely moves on a paused screen
/// (codec dither flips ≤1–2 bits) but flips many bits at once on real motion — a
/// cursor move, a scroll, a panel repaint all cross ≥4. So any threshold in the
/// wide 1–8 valley behaves identically; 2 sits in it with margin either side.
pub const DEFAULT_FROZEN_HAMMING: u32 = 2;

/// Configuration for [`PhashFrameGate`].
#[derive(Debug, Clone, Copy)]
pub struct PhashGateConfig {
    /// Hamming distance `<=` this vs the last analysed frame ⇒ frozen ⇒ reuse.
    pub frozen_hamming: u32,
    /// Force a fresh call after this many consecutive reuses (drift guard).
    pub max_hold: u32,
}

impl PhashGateConfig {
    /// Measured-safe defaults.
    pub fn new() -> Self {
        Self {
            frozen_hamming: DEFAULT_FROZEN_HAMMING,
            max_hold: DEFAULT_MAX_HOLD,
        }
    }
}

impl Default for PhashGateConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Upstream frozen-frame reuse gate keyed on the 64-bit perceptual hash the
/// frame pipeline already computes ([`crate::webrtc::signals`]), so it costs no
/// extra pixel work and needs no learned embedder, a low-cost call-saver
/// for idle/paused screens.
///
/// Same **admit-on-doubt** contract as [`StaticFrameGate`]: the anchor is the
/// last frame we actually *analysed*, drift is measured against that fixed
/// reference, and `max_hold` bounds a reuse run by count as well as by
/// hash distance. A 64-bit average hash is lossy, so two genuinely different
/// frames can share it or differ by only a bit or two. A reuse therefore
/// carries a real if small chance of skipping a distinct frame, and `max_hold`
/// caps how long that can persist rather than removing it.
///
/// Zero-allocation: the whole state is three words. Prefer this over the
/// embedding-based [`StaticFrameGate`] for the per-frame path — no per-frame
/// embedding means no inference server round-trip and no ONNX dependency.
pub struct PhashFrameGate {
    cfg: PhashGateConfig,
    anchor: u64,
    has_anchor: bool,
    held: u32,
}

impl PhashFrameGate {
    /// Build a gate from an explicit config.
    pub fn new(cfg: PhashGateConfig) -> Self {
        Self {
            cfg,
            anchor: 0,
            has_anchor: false,
            held: 0,
        }
    }

    /// Consecutive reuses since the last fresh call.
    pub fn held(&self) -> u32 {
        self.held
    }

    /// Decide whether the frame with perceptual hash `phash` needs a fresh VLM
    /// call. Reuse only when the hash is within `frozen_hamming` of the last
    /// analysed frame *and* the hold cap has not been reached.
    pub fn decide(&mut self, phash: u64) -> GateDecision {
        if self.has_anchor {
            let hamming = (phash ^ self.anchor).count_ones();
            if hamming <= self.cfg.frozen_hamming && self.held < self.cfg.max_hold {
                self.held += 1;
                return GateDecision::Reuse;
            }
        }

        // Fresh call: this frame becomes the new anchor and resets the hold.
        self.anchor = phash;
        self.has_anchor = true;
        self.held = 0;
        GateDecision::Run
    }

    /// Forget the anchor so the next frame is always analysed (e.g. after a
    /// stream discontinuity or seek).
    pub fn reset(&mut self) {
        self.has_anchor = false;
        self.held = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 8;

    /// A unit vector at angle `theta` in the first two lanes; cosine between two
    /// such vectors is `cos(theta_a − theta_b)` — a clean way to hit a target
    /// similarity in tests.
    fn planar(theta: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[0] = theta.cos();
        v[1] = theta.sin();
        v
    }

    #[test]
    fn identical_chunks_collapse_to_one_activity() {
        let mut m = SemanticMerge::with_dim(DIM);
        let e = planar(0.3);
        for i in 0..5 {
            assert!(m.observe(i * 100, &e, "editing a material graph").is_none());
        }
        assert_eq!(m.open_chunk_count(), 5);
        let note = m.flush().expect("one open activity");
        assert_eq!(note.chunk_count, 5);
        assert_eq!(note.start_pts_ms, 0);
        assert_eq!(note.end_pts_ms, 400);
        assert_eq!(note.description, "editing a material graph");
        assert!(
            (note.boundary_novelty - 1.0).abs() < 1e-6,
            "first activity is all-new"
        );
    }

    #[test]
    fn orthogonal_chunks_each_start_a_new_activity() {
        let mut m = SemanticMerge::with_dim(DIM);
        let a = planar(0.0);
        let b = planar(std::f32::consts::FRAC_PI_2); // 90° ⇒ cosine 0

        assert!(
            m.observe(0, &a, "A").is_none(),
            "first opens, nothing to close"
        );
        let closed = m
            .observe(100, &b, "B")
            .expect("orthogonal ⇒ new activity closes A");
        assert_eq!(closed.description, "A");
        assert_eq!(closed.chunk_count, 1);

        let closed = m.observe(200, &a, "A again").expect("switch back closes B");
        assert_eq!(closed.description, "B");
        // B vs A were orthogonal ⇒ boundary novelty ≈ 1.
        assert!(closed.boundary_novelty > 0.9);
    }

    #[test]
    fn threshold_decides_merge_vs_split() {
        // Just-similar pair (cosine ≈ 0.99) merges; well-separated pair
        // (cosine ≈ 0.5) splits, straddling the 0.75 default.
        let mut m = SemanticMerge::with_dim(DIM);
        let base = planar(0.0);
        let close = planar(0.14); // cos(0.14) ≈ 0.990
        let far = planar(std::f32::consts::FRAC_PI_3); // cos(60°) = 0.5

        assert!(m.observe(0, &base, "base").is_none());
        assert!(
            m.observe(100, &close, "still base").is_none(),
            "0.99 ≥ 0.75 ⇒ extend"
        );
        assert_eq!(m.open_chunk_count(), 2);
        assert!(
            m.observe(200, &far, "different").is_some(),
            "0.5 < 0.75 ⇒ split"
        );
    }

    #[test]
    fn flush_on_empty_is_none() {
        let mut m = SemanticMerge::with_dim(DIM);
        assert!(m.flush().is_none());
    }

    #[test]
    fn note_round_trips_through_the_journal() {
        let mut m = SemanticMerge::with_dim(DIM);
        assert!(m.observe(0, &planar(0.0), "compiling shaders").is_none());
        assert!(m.observe(500, &planar(0.05), "compiling shaders").is_none());
        let note = m.flush().expect("open activity");

        let event = note.to_timeline_event(7, "run-abc", "stream-0");
        assert_eq!(event.kind, ACTIVITY_EVENT_KIND);
        assert_eq!(event.seq, 7);
        assert_eq!(event.pts_ms, note.start_pts_ms);

        // Payload survives the tab-delimited WAL and decodes back to the note.
        let dir = std::env::temp_dir().join(format!("vidarax_merge_wal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("activity.wal");
        crate::timeline::append_event(&path, &event).unwrap();
        let read = crate::timeline::read_all_events(&path).unwrap();
        assert_eq!(read.len(), 1);
        let back: ActivityNote = serde_json::from_str(&read[0].payload).unwrap();
        assert_eq!(back, note);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wrong_length_embedding_forces_split_never_false_merge() {
        let mut m = SemanticMerge::with_dim(DIM);
        assert!(m.observe(0, &planar(0.0), "ok").is_none());
        // A malformed embedding must not merge into the open activity.
        let closed = m
            .observe(100, &[1.0, 2.0], "bad dims")
            .expect("forces a new activity");
        assert_eq!(closed.description, "ok");
    }

    #[test]
    fn realistic_run_collapses_many_chunks() {
        // Three well-separated activities, four paraphrase chunks each (small
        // in-activity jitter well inside threshold). Expect exactly three notes
        // out regardless of the intra-activity repetition.
        let mut m = SemanticMerge::with_dim(DIM);
        let bases = [0.0f32, 1.2, 2.6];
        let mut notes = Vec::new();
        let mut pts = 0u64;
        for base in bases {
            for j in 0..4 {
                let jittered = planar(base + 0.02 * j as f32); // < 0.001 cosine drift
                if let Some(n) = m.observe(pts, &jittered, "desc") {
                    notes.push(n);
                }
                pts += 50;
            }
        }
        if let Some(n) = m.flush() {
            notes.push(n);
        }
        assert_eq!(notes.len(), 3, "3 activities in ⇒ 3 notes out");
        assert!(notes.iter().all(|n| n.chunk_count == 4));
    }

    // ---- StaticFrameGate ----

    #[test]
    fn gate_reuses_frozen_frames() {
        let mut g = StaticFrameGate::with_dim(DIM);
        let f = planar(0.5);
        assert_eq!(g.decide(&f), GateDecision::Run, "first frame always runs");
        assert_eq!(g.decide(&f), GateDecision::Reuse, "identical ⇒ reuse");
        assert_eq!(g.decide(&f), GateDecision::Reuse);
        assert_eq!(g.held(), 2);
    }

    #[test]
    fn gate_runs_on_real_motion() {
        let mut g = StaticFrameGate::with_dim(DIM);
        assert_eq!(g.decide(&planar(0.0)), GateDecision::Run);
        // A clearly different frame (cosine 0.5 ≪ 0.996) must run.
        assert_eq!(
            g.decide(&planar(std::f32::consts::FRAC_PI_3)),
            GateDecision::Run
        );
    }

    #[test]
    fn gate_max_hold_forces_a_fresh_call() {
        let cfg = StaticGateConfig {
            static_threshold: DEFAULT_STATIC_THRESHOLD,
            max_hold: 3,
            embed_dim: DIM,
        };
        let mut g = StaticFrameGate::new(cfg);
        let f = planar(0.5);
        assert_eq!(g.decide(&f), GateDecision::Run); // establishes anchor
        assert_eq!(g.decide(&f), GateDecision::Reuse);
        assert_eq!(g.decide(&f), GateDecision::Reuse);
        assert_eq!(g.decide(&f), GateDecision::Reuse);
        assert_eq!(g.decide(&f), GateDecision::Run, "hold cap forces a call");
        assert_eq!(g.held(), 0, "and resets the hold");
    }

    #[test]
    fn gate_wrong_length_always_runs() {
        let mut g = StaticFrameGate::with_dim(DIM);
        assert_eq!(g.decide(&[1.0, 0.0]), GateDecision::Run);
    }

    #[test]
    fn gate_reset_reanalyses_next_frame() {
        let mut g = StaticFrameGate::with_dim(DIM);
        let f = planar(0.5);
        assert_eq!(g.decide(&f), GateDecision::Run);
        assert_eq!(g.decide(&f), GateDecision::Reuse);
        g.reset();
        assert_eq!(g.decide(&f), GateDecision::Run, "after reset, re-analyse");
    }

    // ---- PhashFrameGate ----

    /// Flip `n` low bits of `h` — models `n` cells of the 8×8 average-hash
    /// crossing the mean, i.e. a Hamming distance of exactly `n`.
    fn flip_low(h: u64, n: u32) -> u64 {
        let mut out = h;
        for i in 0..n {
            out ^= 1u64 << i;
        }
        out
    }

    #[test]
    fn phash_gate_reuses_frozen_frames() {
        let mut g = PhashFrameGate::new(PhashGateConfig::new());
        let h = 0xABCD_1234_5678_9F0Eu64;
        assert_eq!(g.decide(h), GateDecision::Run, "first frame always runs");
        assert_eq!(g.decide(h), GateDecision::Reuse, "identical hash ⇒ reuse");
        // One or two flipped bits (codec dither) stays under the frozen bar.
        assert_eq!(g.decide(flip_low(h, 1)), GateDecision::Reuse);
        assert_eq!(g.decide(flip_low(h, 2)), GateDecision::Reuse);
        assert_eq!(g.held(), 3);
    }

    #[test]
    fn phash_gate_runs_on_real_motion() {
        let mut g = PhashFrameGate::new(PhashGateConfig::new());
        let h = 0xFFFF_0000_FFFF_0000u64;
        assert_eq!(g.decide(h), GateDecision::Run);
        // Real motion flips many bits at once — well past frozen_hamming = 2.
        assert_eq!(g.decide(flip_low(h, 8)), GateDecision::Run, "motion ⇒ run");
    }

    #[test]
    fn phash_gate_anchor_is_last_analysed_not_last_seen() {
        // Drift of 1 bit per frame must not creep past the bar: each reuse is
        // scored against the fixed anchor, so accumulated drift is caught.
        let mut g = PhashFrameGate::new(PhashGateConfig::new());
        let h = 0u64;
        assert_eq!(g.decide(h), GateDecision::Run); // anchor = 0
        assert_eq!(g.decide(flip_low(h, 2)), GateDecision::Reuse); // 2 vs anchor
                                                                   // 4 bits differ from the anchor now — over the bar, so a fresh call,
                                                                   // even though it's only 2 bits from the *previous* frame.
        assert_eq!(g.decide(flip_low(h, 4)), GateDecision::Run, "drift caught");
    }

    #[test]
    fn phash_gate_max_hold_forces_a_fresh_call() {
        let cfg = PhashGateConfig {
            frozen_hamming: DEFAULT_FROZEN_HAMMING,
            max_hold: 3,
        };
        let mut g = PhashFrameGate::new(cfg);
        let h = 0x00FF_00FF_00FF_00FFu64;
        assert_eq!(g.decide(h), GateDecision::Run); // establishes anchor
        assert_eq!(g.decide(h), GateDecision::Reuse);
        assert_eq!(g.decide(h), GateDecision::Reuse);
        assert_eq!(g.decide(h), GateDecision::Reuse);
        assert_eq!(g.decide(h), GateDecision::Run, "hold cap forces a call");
        assert_eq!(g.held(), 0, "and resets the hold");
    }

    #[test]
    fn phash_gate_reset_reanalyses_next_frame() {
        let mut g = PhashFrameGate::new(PhashGateConfig::default());
        let h = 0x1234_5678_9ABC_DEF0u64;
        assert_eq!(g.decide(h), GateDecision::Run);
        assert_eq!(g.decide(h), GateDecision::Reuse);
        g.reset();
        assert_eq!(g.decide(h), GateDecision::Run, "after reset, re-analyse");
    }
}

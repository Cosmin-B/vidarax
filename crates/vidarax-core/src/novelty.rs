//! Pre-VLM semantic novelty policies.
//!
//! [`NoveltyGate`] is the generic three-signal gate. [`LiveNoveltyGate`] is the
//! embedding-only live policy with bounded reuse time and cumulative drift.

use crate::embedding_sidecar::EMBEDDING_DIM;

/// FNV-1a 64-bit constants (public-domain algorithm).
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Number of MinHash slots. 64 × `u32` = 256 bytes per signature and a Jaccard
/// estimate with standard error ≈ `1/sqrt(64)` ≈ 0.125 — enough to separate
/// "same screen" from "new screen" without a heavyweight set representation.
pub const MINHASH_SLOTS: usize = 64;

/// Default weight on the OCR text-distance signal.
pub const DEFAULT_W_TEXT: f32 = 0.50;
/// Default weight on the embedding-distance signal.
pub const DEFAULT_W_EMBED: f32 = 0.40;
/// Default weight on the perceptual-hash-distance signal.
pub const DEFAULT_W_PHASH: f32 = 0.10;
/// Default admit threshold: at or above this, run the full VLM.
pub const DEFAULT_TAU_HI: f32 = 0.45;
/// Default drop threshold: at or below this, spend nothing.
pub const DEFAULT_TAU_LO: f32 = 0.12;
/// Default rolling-window size (kept chunks retained for comparison).
pub const DEFAULT_WINDOW: usize = 8;

/// Longest time live capture may reuse its last semantic anchor.
pub const LIVE_MAX_REUSE_MS: u64 = 2_000;
/// Deadline for a live embedding request. Failure is admit-on-doubt.
pub const LIVE_EMBEDDING_TIMEOUT_MS: u64 = 2_000;
/// Sum of individually-reusable scores allowed before a safety refresh.
pub const LIVE_MAX_CUMULATIVE_DRIFT: f32 = 0.50;
/// Default invisible sampling rate for online false-drop evidence.
pub const LIVE_SHADOW_SAMPLE_RATE: f32 = 0.01;
/// Conservative live reuse threshold selected from preliminary labelled calibration.
///
/// This is intentionally separate from [`DEFAULT_TAU_LO`]: the generic gate
/// fuses OCR, embedding, and perceptual-hash signals, while live capture makes
/// its decision from embedding distance alone. Deployments should still
/// calibrate this value on representative labelled streams.
pub const LIVE_REUSE_THRESHOLD: f32 = 0.01;

/// splitmix64 — a fast, well-distributed integer mixer. Used both to derive the
/// per-slot MinHash seeds and to spread a token's base hash across the slots.
#[inline]
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-slot MinHash seeds, derived once at compile time.
const SEEDS: [u64; MINHASH_SLOTS] = {
    let mut s = [0u64; MINHASH_SLOTS];
    let mut i = 0;
    while i < MINHASH_SLOTS {
        s[i] = splitmix64(i as u64 + 1);
        i += 1;
    }
    s
};

/// FNV-1a over `bytes`, folding ASCII case so "Health" and "health" hash alike.
#[inline]
fn fnv1a64_lower(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b.to_ascii_lowercase());
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// A MinHash signature of an OCR token set.
///
/// Two signatures' fraction of matching slots is an unbiased estimator of the
/// Jaccard similarity of the underlying token sets, computable in `O(SLOTS)`
/// without materialising either set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinHashSig {
    slots: [u32; MINHASH_SLOTS],
}

impl MinHashSig {
    /// An empty signature (every slot at the `u32::MAX` sentinel). Two empty
    /// signatures compare as Jaccard 1.0 — i.e. "no text vs no text" is treated
    /// as *not* a text-novelty signal, leaving the embedding and phash to decide.
    #[inline]
    pub fn empty() -> Self {
        Self {
            slots: [u32::MAX; MINHASH_SLOTS],
        }
    }

    /// Fold a single already-delimited token into the signature.
    #[inline]
    pub fn add_token(&mut self, token: &str) {
        if token.is_empty() {
            return;
        }
        let h = fnv1a64_lower(token.as_bytes());
        for (slot, &seed) in self.slots.iter_mut().zip(SEEDS.iter()) {
            let v = splitmix64(h ^ seed) as u32;
            if v < *slot {
                *slot = v;
            }
        }
    }

    /// Build a signature from raw OCR text, splitting on any non-alphanumeric
    /// character. Allocation-free: the splits borrow from `text`.
    pub fn from_ocr_text(text: &str) -> Self {
        let mut sig = Self::empty();
        for token in text.split(|c: char| !c.is_alphanumeric()) {
            sig.add_token(token);
        }
        sig
    }

    /// Build a signature from an iterator of pre-tokenised words.
    pub fn from_tokens<'a, I: IntoIterator<Item = &'a str>>(tokens: I) -> Self {
        let mut sig = Self::empty();
        for token in tokens {
            sig.add_token(token);
        }
        sig
    }

    /// Estimated Jaccard similarity in `[0,1]`: the fraction of slots that agree.
    #[inline]
    pub fn jaccard(&self, other: &Self) -> f32 {
        let mut eq = 0u32;
        for i in 0..MINHASH_SLOTS {
            eq += u32::from(self.slots[i] == other.slots[i]);
        }
        eq as f32 / MINHASH_SLOTS as f32
    }
}

impl Default for MinHashSig {
    fn default() -> Self {
        Self::empty()
    }
}

/// The gate's ruling on a candidate chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoveltyDecision {
    /// Clearly redundant (`n ≤ τ_lo`). Spend nothing; do not commit.
    Drop,
    /// Ambiguous (`τ_lo < n < τ_hi`). The caller should run a cheap confirm gate.
    Escalate {
        /// The fused novelty score that landed in the ambiguous band.
        novelty: f32,
    },
    /// Clearly new (`n ≥ τ_hi`). Run the full VLM, then commit.
    Admit {
        /// The fused novelty score that cleared the admit threshold.
        novelty: f32,
    },
}

impl NoveltyDecision {
    /// The fused novelty score behind this decision (`0.0` for [`Drop`]).
    ///
    /// [`Drop`]: NoveltyDecision::Drop
    #[inline]
    pub fn novelty(&self) -> f32 {
        match self {
            NoveltyDecision::Drop => 0.0,
            NoveltyDecision::Escalate { novelty } | NoveltyDecision::Admit { novelty } => *novelty,
        }
    }
}

/// Tunable weights and thresholds for the fused novelty decision.
///
/// The three weights are applied to the three distance signals; the two
/// thresholds split the fused score into drop / escalate / admit bands.
#[derive(Debug, Clone, Copy)]
pub struct NoveltyConfig {
    /// Weight on the OCR text-distance signal `n_t`.
    pub w_text: f32,
    /// Weight on the embedding-distance signal `n_e`.
    pub w_embed: f32,
    /// Weight on the perceptual-hash-distance signal `n_v`.
    pub w_phash: f32,
    /// Admit threshold `τ_hi`: `n ≥ τ_hi` ⇒ [`NoveltyDecision::Admit`].
    pub tau_hi: f32,
    /// Drop threshold `τ_lo`: `n ≤ τ_lo` ⇒ [`NoveltyDecision::Drop`].
    pub tau_lo: f32,
    /// Rolling-window size: how many kept chunks to compare against.
    pub window: usize,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            w_text: DEFAULT_W_TEXT,
            w_embed: DEFAULT_W_EMBED,
            w_phash: DEFAULT_W_PHASH,
            tau_hi: DEFAULT_TAU_HI,
            tau_lo: DEFAULT_TAU_LO,
            window: DEFAULT_WINDOW,
        }
    }
}

/// Live semantic-novelty settings. No sidecar address means disabled.
#[derive(Debug, Clone)]
pub struct LiveNoveltyConfig {
    pub embedding_sidecar_addr: Option<String>,
    pub max_reuse_ms: u64,
    pub max_cumulative_drift: f32,
    pub shadow_sample_rate: f32,
    pub embedding_timeout_ms: u64,
    pub reuse_threshold: f32,
}

impl Default for LiveNoveltyConfig {
    fn default() -> Self {
        Self {
            embedding_sidecar_addr: None,
            max_reuse_ms: LIVE_MAX_REUSE_MS,
            max_cumulative_drift: LIVE_MAX_CUMULATIVE_DRIFT,
            shadow_sample_rate: LIVE_SHADOW_SAMPLE_RATE,
            embedding_timeout_ms: LIVE_EMBEDDING_TIMEOUT_MS,
            reuse_threshold: LIVE_REUSE_THRESHOLD,
        }
    }
}

/// Why a [`NoveltyConfig`] failed [`NoveltyConfig::validate`].
///
/// A gate built from an unchecked config can silently break its own contract:
/// a non-finite weight makes the fused novelty non-finite, so the score handed
/// back in [`NoveltyDecision::Escalate`]/[`NoveltyDecision::Admit`] is no longer
/// the documented value in `[0,1]`, and thresholds that are out of order or out
/// of range collapse or invert the drop/escalate/admit bands. Validating up
/// front keeps those states unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoveltyConfigError {
    /// A weight was NaN, infinite, or negative.
    WeightNotFiniteNonNegative,
    /// The three weights summed to zero, so no signal drives the decision.
    WeightSumNotPositive,
    /// A threshold was NaN or fell outside `[0,1]`.
    ThresholdOutOfRange,
    /// The bands did not satisfy `tau_lo < tau_hi`.
    ThresholdsNotOrdered,
    /// The rolling window was empty.
    WindowEmpty,
}

impl core::fmt::Display for NoveltyConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::WeightNotFiniteNonNegative => "novelty weights must be finite and non-negative",
            Self::WeightSumNotPositive => "novelty weights must sum to a positive value",
            Self::ThresholdOutOfRange => "novelty thresholds must lie within [0, 1]",
            Self::ThresholdsNotOrdered => "novelty thresholds must satisfy tau_lo < tau_hi",
            Self::WindowEmpty => "novelty window must hold at least one chunk",
        })
    }
}

impl std::error::Error for NoveltyConfigError {}

impl NoveltyConfig {
    /// Check that the weights and thresholds can produce a decision that honours
    /// the documented `[0,1]` novelty contract and three non-degenerate bands.
    ///
    /// Requires finite non-negative weights whose sum is positive, thresholds in
    /// `[0,1]` with `tau_lo < tau_hi`, and a window of at least one chunk. The
    /// [`Default`] calibration passes.
    pub fn validate(&self) -> Result<(), NoveltyConfigError> {
        for w in [self.w_text, self.w_embed, self.w_phash] {
            if !w.is_finite() || w < 0.0 {
                return Err(NoveltyConfigError::WeightNotFiniteNonNegative);
            }
        }
        if self.w_text + self.w_embed + self.w_phash <= 0.0 {
            return Err(NoveltyConfigError::WeightSumNotPositive);
        }
        for t in [self.tau_lo, self.tau_hi] {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err(NoveltyConfigError::ThresholdOutOfRange);
            }
        }
        // Both thresholds are finite by the check above, so a direct comparison
        // is well-defined and catches equal-or-inverted bands.
        if self.tau_lo >= self.tau_hi {
            return Err(NoveltyConfigError::ThresholdsNotOrdered);
        }
        if self.window == 0 {
            return Err(NoveltyConfigError::WindowEmpty);
        }
        Ok(())
    }
}

/// The individual signal values behind a single decision, exposed for
/// telemetry and calibration. All fields are in `[0,1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoveltyBreakdown {
    /// OCR text distance, `1 − max Jaccard` over the window.
    pub n_text: f32,
    /// Embedding distance, `1 − max cosine` over the window.
    pub n_embed: f32,
    /// Perceptual-hash distance, `min Hamming / 64` over the window.
    pub n_phash: f32,
    /// The weighted fusion of the three, clamped to `[0,1]`.
    pub fused: f32,
}

/// Fixed-capacity ring of kept-chunk signatures, allocated once.
///
/// The three signature kinds are stored in parallel arrays so scoring walks
/// contiguous memory. Writes overwrite the slot at `head` in place; nothing is
/// allocated or freed after construction.
struct KeptRing {
    dim: usize,
    cap: usize,
    len: usize,
    head: usize,
    minhash: Vec<MinHashSig>,
    embed_q: Vec<i8>,      // cap * dim, row-major
    embed_scale: Vec<f32>, // cap
    phash: Vec<u64>,       // cap
}

impl KeptRing {
    fn new(cap: usize, dim: usize) -> Self {
        let cap = cap.max(1);
        Self {
            dim,
            cap,
            len: 0,
            head: 0,
            minhash: vec![MinHashSig::empty(); cap],
            embed_q: vec![0i8; cap * dim],
            embed_scale: vec![0.0f32; cap],
            phash: vec![0u64; cap],
        }
    }

    #[inline]
    fn embed_row(&self, slot: usize) -> &[i8] {
        &self.embed_q[slot * self.dim..slot * self.dim + self.dim]
    }

    /// Overwrite the slot at `head` with a freshly quantised candidate.
    fn push(&mut self, sig: &MinHashSig, q: &[i8], scale: f32, phash: u64) {
        let slot = self.head;
        self.minhash[slot] = *sig;
        self.embed_q[slot * self.dim..slot * self.dim + self.dim].copy_from_slice(q);
        self.embed_scale[slot] = scale;
        self.phash[slot] = phash;
        self.head = (self.head + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.head = 0;
    }
}

/// Integer dot product of two int8 rows of equal length.
///
/// Both operands come from unit-normalised vectors, so
/// `dot(a,b) · scale_a · scale_b ≈ cosine(a,b)`. `127·127·dim` stays well
/// inside `i32` for realistic embedding dimensions.
///
#[inline]
pub(crate) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = 0i32;
    for i in 0..a.len() {
        acc += i32::from(a[i]) * i32::from(b[i]);
    }
    acc
}

/// L2-normalise `src`, then int8-quantise into `dst`. Returns the per-vector
/// scale such that `dst[i] · scale ≈ src[i]/‖src‖`.
///
/// A zero vector yields an all-zero row and scale `1.0` (its cosine with
/// anything is `0`, i.e. maximally novel on the embedding axis).
///
/// `src` and `dst` are expected to share the configured embedding dimension.
/// A mismatch is handled defensively rather than by panicking, because `dst`
/// is a reused scratch lane: only the shared prefix is quantised, every `dst`
/// lane the source cannot reach is zeroed so no earlier candidate can bleed
/// through, and only that same prefix of `src` feeds the norm so a
/// longer-than-configured vector cannot skew the scale.
///
pub(crate) fn quantize_unit_into(src: &[f32], dst: &mut [i8]) -> f32 {
    let n = src.len().min(dst.len());
    // Zero any dst lane the source cannot fill so a shorter candidate never
    // leaves a stale lane behind from whoever wrote this scratch last.
    for d in dst[n..].iter_mut() {
        *d = 0;
    }
    let used = &src[..n];
    let norm_sq: f32 = used.iter().map(|&x| x * x).sum();
    if norm_sq <= f32::MIN_POSITIVE {
        for d in dst[..n].iter_mut() {
            *d = 0;
        }
        return 1.0;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    // Largest magnitude after normalisation sets the quantisation scale.
    let mut max_abs = 0.0f32;
    for &x in used {
        let u = (x * inv_norm).abs();
        if u > max_abs {
            max_abs = u;
        }
    }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;
    for (d, &x) in dst[..n].iter_mut().zip(used.iter()) {
        let q = (x * inv_norm * inv_scale).round();
        *d = q.clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Generic semantic-novelty gate. One instance per stream; not `Sync`.
pub struct NoveltyGate {
    cfg: NoveltyConfig,
    ring: KeptRing,
    /// Reused quantisation scratch for the candidate under evaluation. Written
    /// and consumed within a single `evaluate` call; never read across calls.
    scratch_q: Vec<i8>,
    evaluated: u64,
    dropped: u64,
    escalated: u64,
    admitted: u64,
    committed: u64,
}

impl NoveltyGate {
    /// Create a gate for embeddings of dimension `embed_dim`, using `cfg`.
    ///
    /// All backing storage is allocated here; steady-state operation is
    /// allocation-free.
    pub fn new(cfg: NoveltyConfig, embed_dim: usize) -> Self {
        Self {
            ring: KeptRing::new(cfg.window, embed_dim),
            scratch_q: vec![0i8; embed_dim],
            cfg,
            evaluated: 0,
            dropped: 0,
            escalated: 0,
            admitted: 0,
            committed: 0,
        }
    }

    /// Create a gate after checking `cfg`, returning an error instead of
    /// building a gate whose config would violate the `[0,1]` novelty contract
    /// or collapse the decision bands. See [`NoveltyConfig::validate`].
    pub fn try_new(cfg: NoveltyConfig, embed_dim: usize) -> Result<Self, NoveltyConfigError> {
        cfg.validate()?;
        Ok(Self::new(cfg, embed_dim))
    }

    /// Create a gate with the default calibration.
    pub fn with_dim(embed_dim: usize) -> Self {
        Self::new(NoveltyConfig::default(), embed_dim)
    }

    /// Score `embedding`, `sig`, and `phash` against the rolling window and
    /// return both the decision and the raw per-signal breakdown. Does **not**
    /// commit.
    ///
    /// `embedding` is expected to carry the gate's `embed_dim`. A wrong-length
    /// vector does not panic and cannot corrupt state: it is quantised over the
    /// shared prefix into a fully-rewritten scratch lane (see
    /// [`quantize_unit_into`]), so the reading it produces is confined to this
    /// call and never leaks into the next candidate.
    pub fn evaluate_detailed(
        &mut self,
        sig: &MinHashSig,
        embedding: &[f32],
        phash: u64,
    ) -> (NoveltyDecision, NoveltyBreakdown) {
        self.evaluated += 1;

        let cand_scale = quantize_unit_into(embedding, &mut self.scratch_q);

        let breakdown = if self.ring.len == 0 {
            // Empty window: nothing to be redundant against ⇒ maximally novel.
            NoveltyBreakdown {
                n_text: 1.0,
                n_embed: 1.0,
                n_phash: 1.0,
                fused: (self.cfg.w_text + self.cfg.w_embed + self.cfg.w_phash).clamp(0.0, 1.0),
            }
        } else {
            let mut max_jaccard = 0.0f32;
            let mut max_cosine = -1.0f32;
            let mut min_hamming = u32::MAX;
            for slot in 0..self.ring.len {
                let j = sig.jaccard(&self.ring.minhash[slot]);
                if j > max_jaccard {
                    max_jaccard = j;
                }
                let dot = dot_i8(&self.scratch_q, self.ring.embed_row(slot));
                let cosine = dot as f32 * cand_scale * self.ring.embed_scale[slot];
                if cosine > max_cosine {
                    max_cosine = cosine;
                }
                let hd = (phash ^ self.ring.phash[slot]).count_ones();
                if hd < min_hamming {
                    min_hamming = hd;
                }
            }
            let n_text = (1.0 - max_jaccard).clamp(0.0, 1.0);
            let n_embed = (1.0 - max_cosine).clamp(0.0, 1.0);
            let n_phash = (min_hamming as f32 / 64.0).clamp(0.0, 1.0);
            let fused = (self.cfg.w_text * n_text
                + self.cfg.w_embed * n_embed
                + self.cfg.w_phash * n_phash)
                .clamp(0.0, 1.0);
            NoveltyBreakdown {
                n_text,
                n_embed,
                n_phash,
                fused,
            }
        };

        let n = breakdown.fused;
        let decision = if n >= self.cfg.tau_hi {
            self.admitted += 1;
            NoveltyDecision::Admit { novelty: n }
        } else if n <= self.cfg.tau_lo {
            self.dropped += 1;
            NoveltyDecision::Drop
        } else {
            self.escalated += 1;
            NoveltyDecision::Escalate { novelty: n }
        };
        (decision, breakdown)
    }

    /// Convenience wrapper over [`evaluate_detailed`](Self::evaluate_detailed)
    /// returning only the decision.
    #[inline]
    pub fn evaluate(&mut self, sig: &MinHashSig, embedding: &[f32], phash: u64) -> NoveltyDecision {
        self.evaluate_detailed(sig, embedding, phash).0
    }

    /// Record a kept chunk into the rolling window. Call this for every chunk
    /// the caller decides to keep (all `Admit`s, plus confirmed `Escalate`s).
    /// Overwrites the oldest slot; performs no allocation.
    ///
    /// `embedding` is expected to carry the gate's `embed_dim`. A wrong-length
    /// vector does not panic: it is quantised over the shared prefix, with the
    /// remaining lanes zeroed, so the committed row stays self-consistent (see
    /// [`quantize_unit_into`]).
    pub fn commit(&mut self, sig: &MinHashSig, embedding: &[f32], phash: u64) {
        let scale = quantize_unit_into(embedding, &mut self.scratch_q);
        // `ring` and `scratch_q` are disjoint fields, so the ring can copy the
        // freshly-quantised candidate straight out of the scratch buffer.
        self.ring.push(sig, &self.scratch_q, scale, phash);
        self.committed += 1;
    }

    /// Clear the window and all counters; the next chunk is treated as first.
    pub fn reset(&mut self) {
        self.ring.clear();
        self.evaluated = 0;
        self.dropped = 0;
        self.escalated = 0;
        self.admitted = 0;
        self.committed = 0;
    }

    /// Number of kept chunks currently in the window.
    #[inline]
    pub fn window_len(&self) -> usize {
        self.ring.len
    }

    /// Maximum window capacity (`K`).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.ring.cap
    }

    /// Total chunks evaluated since construction or last [`reset`](Self::reset).
    #[inline]
    pub fn evaluated_count(&self) -> u64 {
        self.evaluated
    }

    /// Chunks dropped as redundant.
    #[inline]
    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Chunks sent to the cheap escalate gate.
    #[inline]
    pub fn escalated_count(&self) -> u64 {
        self.escalated
    }

    /// Chunks admitted straight to the full VLM.
    #[inline]
    pub fn admitted_count(&self) -> u64 {
        self.admitted
    }

    /// Chunks committed into the window.
    #[inline]
    pub fn committed_count(&self) -> u64 {
        self.committed
    }

    /// Live admit rate (`admitted / evaluated`), or `0.0` before any input.
    /// Watch this for drift against the rate seen during calibration.
    #[inline]
    pub fn admit_rate(&self) -> f32 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.admitted as f32 / self.evaluated as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveNoveltyOutcome {
    Reuse,
    Run,
    ForcedRefresh,
}

/// Production reuse policy shared by live capture and calibration.
pub struct LiveNoveltyGate {
    gate: NoveltyGate,
    max_reuse_ms: u64,
    max_cumulative_drift: f32,
    last_commit_pts_ms: Option<u64>,
    cumulative_drift: f32,
}

impl LiveNoveltyGate {
    pub fn try_new(config: &LiveNoveltyConfig) -> Result<Self, NoveltyConfigError> {
        let gate = NoveltyConfig {
            w_text: 0.0,
            w_embed: 1.0,
            w_phash: 0.0,
            tau_hi: (config.reuse_threshold + 1.0) * 0.5,
            tau_lo: config.reuse_threshold,
            window: 1,
        };
        Ok(Self {
            gate: NoveltyGate::try_new(gate, EMBEDDING_DIM)?,
            max_reuse_ms: config.max_reuse_ms,
            max_cumulative_drift: config.max_cumulative_drift,
            last_commit_pts_ms: None,
            cumulative_drift: 0.0,
        })
    }

    pub fn evaluate(
        &mut self,
        embedding: &[f32; EMBEDDING_DIM],
        pts_ms: u64,
    ) -> LiveNoveltyOutcome {
        let (decision, breakdown) = self
            .gate
            .evaluate_detailed(&MinHashSig::empty(), embedding, 0);
        if !matches!(decision, NoveltyDecision::Drop) {
            return LiveNoveltyOutcome::Run;
        }

        self.cumulative_drift = (self.cumulative_drift + breakdown.fused).min(f32::MAX);
        let refresh_due = self
            .last_commit_pts_ms
            .is_some_and(|last| pts_ms < last || pts_ms - last >= self.max_reuse_ms)
            || self.cumulative_drift >= self.max_cumulative_drift;
        if refresh_due {
            LiveNoveltyOutcome::ForcedRefresh
        } else {
            LiveNoveltyOutcome::Reuse
        }
    }

    /// Commit after the VLM returns a usable description.
    pub fn commit(&mut self, embedding: &[f32; EMBEDDING_DIM], pts_ms: u64) {
        self.gate.commit(&MinHashSig::empty(), embedding, 0);
        self.last_commit_pts_ms = Some(pts_ms);
        self.cumulative_drift = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;

    #[test]
    fn live_default_uses_calibrated_conservative_threshold() {
        assert_eq!(LiveNoveltyConfig::default().reuse_threshold, 0.01);
    }

    /// A unit-ish embedding pointing mostly along axis `axis`, with a little
    /// energy spread so quantisation has something to represent.
    fn embed(axis: usize, mag: f32) -> Vec<f32> {
        let mut v = vec![0.01f32; DIM];
        v[axis % DIM] = mag;
        v
    }

    #[test]
    fn minhash_identical_sets_are_jaccard_one() {
        let a = MinHashSig::from_ocr_text("health 100 mana 50 quest log");
        let b = MinHashSig::from_ocr_text("health 100 mana 50 quest log");
        assert_eq!(a.jaccard(&b), 1.0);
    }

    #[test]
    fn minhash_disjoint_sets_are_near_zero() {
        let a = MinHashSig::from_ocr_text("alpha bravo charlie delta echo foxtrot");
        let b = MinHashSig::from_ocr_text("uniform victor whiskey xray yankee zulu");
        // Disjoint token sets: Jaccard estimate should be small.
        assert!(a.jaccard(&b) < 0.2, "jaccard was {}", a.jaccard(&b));
    }

    #[test]
    fn minhash_partial_overlap_is_between() {
        let a = MinHashSig::from_ocr_text("one two three four five six seven eight");
        let b = MinHashSig::from_ocr_text("one two three four nine ten eleven twelve");
        let j = a.jaccard(&b);
        // True Jaccard = 4/12 ≈ 0.33; estimator should land in a loose band.
        assert!(j > 0.1 && j < 0.6, "jaccard was {j}");
    }

    #[test]
    fn empty_signatures_compare_identical() {
        let a = MinHashSig::empty();
        let b = MinHashSig::from_ocr_text("   !!!  ");
        assert_eq!(a.jaccard(&b), 1.0);
    }

    #[test]
    fn case_folding_makes_tokens_equal() {
        let a = MinHashSig::from_ocr_text("Health Mana Quest");
        let b = MinHashSig::from_ocr_text("health mana quest");
        assert_eq!(a.jaccard(&b), 1.0);
    }

    #[test]
    fn first_chunk_is_always_admitted() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::from_ocr_text("boot screen loading");
        let d = g.evaluate(&sig, &embed(0, 1.0), 0);
        assert!(matches!(d, NoveltyDecision::Admit { .. }), "got {d:?}");
        assert_eq!(g.window_len(), 0, "evaluate must not commit");
    }

    #[test]
    fn identical_chunk_after_commit_is_dropped() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::from_ocr_text("inventory sword shield potion");
        let e = embed(1, 1.0);
        let phash = 0xABCD_1234_5678_9F01;

        assert!(matches!(
            g.evaluate(&sig, &e, phash),
            NoveltyDecision::Admit { .. }
        ));
        g.commit(&sig, &e, phash);
        assert_eq!(g.window_len(), 1);

        // Exactly the same chunk should now read as redundant.
        let d = g.evaluate(&sig, &e, phash);
        assert!(matches!(d, NoveltyDecision::Drop), "got {d:?}");
    }

    #[test]
    fn changed_text_raises_novelty_above_drop() {
        let mut g = NoveltyGate::with_dim(DIM);
        let s1 = MinHashSig::from_ocr_text("main menu play options quit settings audio");
        let e = embed(2, 1.0);
        g.commit(&s1, &e, 0);

        // Same embedding + phash, completely different on-screen text.
        let s2 = MinHashSig::from_ocr_text("boss fight phase two enrage timer adds");
        let (d, b) = g.evaluate_detailed(&s2, &e, 0);
        assert!(b.n_text > 0.5, "text distance too low: {}", b.n_text);
        assert!(
            !matches!(d, NoveltyDecision::Drop),
            "text change was dropped: {d:?}"
        );
    }

    #[test]
    fn changed_embedding_raises_novelty() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::empty();
        g.commit(&sig, &embed(0, 1.0), 0);

        // Orthogonal embedding ⇒ cosine ≈ 0 ⇒ n_embed ≈ 1.
        let (_d, b) = g.evaluate_detailed(&sig, &embed(8, 1.0), 0);
        assert!(b.n_embed > 0.8, "embed distance too low: {}", b.n_embed);
    }

    #[test]
    fn similar_embedding_reads_low_novelty() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::empty();
        let a = embed(3, 1.0);
        g.commit(&sig, &a, 0);
        // Nearly the same vector ⇒ cosine ≈ 1 ⇒ n_embed ≈ 0.
        let mut a2 = a.clone();
        a2[3] = 0.98;
        let (_d, b) = g.evaluate_detailed(&sig, &a2, 0);
        assert!(b.n_embed < 0.2, "embed distance too high: {}", b.n_embed);
    }

    #[test]
    fn phash_distance_contributes() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::empty();
        let e = embed(4, 1.0);
        g.commit(&sig, &e, 0);
        // All 64 bits differ ⇒ n_phash = 1.0.
        let (_d, b) = g.evaluate_detailed(&sig, &e, u64::MAX);
        assert_eq!(b.n_phash, 1.0);
    }

    #[test]
    fn escalate_band_triggers_between_thresholds() {
        // Weight everything on phash so we can dial novelty precisely.
        let cfg = NoveltyConfig {
            w_text: 0.0,
            w_embed: 0.0,
            w_phash: 1.0,
            tau_hi: 0.45,
            tau_lo: 0.12,
            window: 4,
        };
        let mut g = NoveltyGate::new(cfg, DIM);
        let sig = MinHashSig::empty();
        let e = embed(5, 1.0);
        g.commit(&sig, &e, 0);

        // 20 differing bits / 64 = 0.3125 → between τ_lo and τ_hi.
        let phash = (1u64 << 20) - 1; // low 20 bits set
        assert_eq!(phash.count_ones(), 20);
        let (d, b) = g.evaluate_detailed(&sig, &e, phash);
        assert!((b.n_phash - 20.0 / 64.0).abs() < 1e-6);
        assert!(matches!(d, NoveltyDecision::Escalate { .. }), "got {d:?}");
    }

    #[test]
    fn ring_evicts_oldest_beyond_capacity() {
        let cfg = NoveltyConfig {
            window: 2,
            ..NoveltyConfig::default()
        };
        let mut g = NoveltyGate::new(cfg, DIM);

        let sa = MinHashSig::from_ocr_text("scene alpha alpha alpha");
        let sb = MinHashSig::from_ocr_text("scene bravo bravo bravo");
        let sc = MinHashSig::from_ocr_text("scene charlie charlie charlie");
        let (ea, eb, ec) = (embed(0, 1.0), embed(5, 1.0), embed(10, 1.0));

        g.commit(&sa, &ea, 0x0000_0000_0000_0000);
        g.commit(&sb, &eb, 0x0000_0000_0000_00FF);
        // Window now [A, B]. Commit C → evicts A.
        g.commit(&sc, &ec, 0x0000_0000_0000_FF00);
        assert_eq!(g.window_len(), 2);

        // A is gone from the window, so a chunk identical to A is novel again.
        let d = g.evaluate(&sa, &ea, 0x0000_0000_0000_0000);
        assert!(
            !matches!(d, NoveltyDecision::Drop),
            "evicted chunk still suppressed: {d:?}"
        );
    }

    #[test]
    fn telemetry_counts_add_up() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::from_ocr_text("hud overlay");
        let e = embed(0, 1.0);

        // 1st: admit (empty window).
        g.evaluate(&sig, &e, 0);
        g.commit(&sig, &e, 0);
        // 2nd: identical ⇒ drop.
        g.evaluate(&sig, &e, 0);
        // 3rd: totally different ⇒ admit.
        let s2 = MinHashSig::from_ocr_text("victory screen rewards xp gold loot");
        g.evaluate(&s2, &embed(9, 1.0), u64::MAX);

        assert_eq!(g.evaluated_count(), 3);
        assert_eq!(
            g.dropped_count() + g.escalated_count() + g.admitted_count(),
            g.evaluated_count()
        );
        assert_eq!(g.admitted_count(), 2);
        assert_eq!(g.dropped_count(), 1);
        assert!((g.admit_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_window_and_counters() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::from_ocr_text("pause menu");
        let e = embed(0, 1.0);
        g.evaluate(&sig, &e, 0);
        g.commit(&sig, &e, 0);
        assert_eq!(g.window_len(), 1);

        g.reset();
        assert_eq!(g.window_len(), 0);
        assert_eq!(g.evaluated_count(), 0);
        // First chunk after reset is admitted again.
        assert!(matches!(
            g.evaluate(&sig, &e, 0),
            NoveltyDecision::Admit { .. }
        ));
    }

    #[test]
    fn zero_embedding_is_maximally_novel_on_embed_axis() {
        let mut g = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::empty();
        g.commit(&sig, &embed(0, 1.0), 0);
        let zero = vec![0.0f32; DIM];
        let (_d, b) = g.evaluate_detailed(&sig, &zero, 0);
        // cosine with a zero vector is 0 ⇒ n_embed = 1.
        assert_eq!(b.n_embed, 1.0);
    }

    #[test]
    fn config_validate_rejects_non_finite_and_unordered() {
        let nan_weight = NoveltyConfig {
            w_text: f32::NAN,
            ..NoveltyConfig::default()
        };
        assert_eq!(
            nan_weight.validate(),
            Err(NoveltyConfigError::WeightNotFiniteNonNegative)
        );

        let swapped = NoveltyConfig {
            tau_lo: 0.8,
            tau_hi: 0.2,
            ..NoveltyConfig::default()
        };
        assert_eq!(
            swapped.validate(),
            Err(NoveltyConfigError::ThresholdsNotOrdered)
        );

        // The shipped default passes, and try_new mirrors validate.
        assert!(NoveltyConfig::default().validate().is_ok());
        assert!(NoveltyGate::try_new(NoveltyConfig::default(), DIM).is_ok());
        assert!(NoveltyGate::try_new(nan_weight, DIM).is_err());
    }

    #[test]
    fn wrong_length_embedding_does_not_panic_or_corrupt_state() {
        let mut gate = NoveltyGate::with_dim(DIM);
        let sig = MinHashSig::from_ocr_text("alpha bravo charlie");
        gate.commit(&sig, &embed(0, 1.0), 0);

        // A shorter and a longer embedding must not panic, and must not leave a
        // stale scratch lane that changes a later, correctly-sized decision.
        let _ = gate.evaluate_detailed(&sig, &[0.5f32; DIM / 2], 0);
        let _ = gate.evaluate_detailed(&sig, &[0.5f32; DIM * 2], 0);

        let (_d, b) = gate.evaluate_detailed(&sig, &embed(0, 1.0), 0);
        assert!(b.fused.is_finite() && (0.0..=1.0).contains(&b.fused));
        assert!(b.n_embed.is_finite() && (0.0..=1.0).contains(&b.n_embed));
    }
}

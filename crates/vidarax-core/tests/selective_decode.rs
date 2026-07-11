use vidarax_core::ingest::{build_select_expr, compute_semantic_frame_indices};

#[test]
fn indices_for_single_chunk() {
    // 25 frames, chunk_size=25 (1 chunk), 2 frames per chunk
    // select_semantic_images picks frame 0 and frame 24 (evenly spaced)
    let indices = compute_semantic_frame_indices(25, 25, 2);
    assert_eq!(indices, vec![0, 24]);
}

#[test]
fn indices_for_multiple_chunks() {
    // 50 frames, chunk_size=25 (2 chunks), 2 frames per chunk
    // chunk 0: frames 0-24 → selects 0, 24
    // chunk 1: frames 25-49 → selects 25, 49
    let indices = compute_semantic_frame_indices(50, 25, 2);
    assert_eq!(indices, vec![0, 24, 25, 49]);
}

#[test]
fn single_frame_per_chunk_picks_middle() {
    // 30 frames, chunk_size=10, 1 frame per chunk
    // chunk 0: frames 0-9 → middle = 5
    // chunk 1: frames 10-19 → middle = 15
    // chunk 2: frames 20-29 → middle = 25
    let indices = compute_semantic_frame_indices(30, 10, 1);
    assert_eq!(indices, vec![5, 15, 25]);
}

#[test]
fn partial_last_chunk() {
    // 27 frames, chunk_size=25, 2 frames per chunk
    // chunk 0: frames 0-24 → selects 0, 24
    // chunk 1: frames 25-26 (only 2 frames) → selects 25, 26
    let indices = compute_semantic_frame_indices(27, 25, 2);
    assert_eq!(indices, vec![0, 24, 25, 26]);
}

#[test]
fn zero_frames_returns_empty() {
    assert!(compute_semantic_frame_indices(0, 25, 2).is_empty());
    assert!(compute_semantic_frame_indices(100, 25, 0).is_empty());
}

#[test]
fn frames_per_chunk_exceeds_chunk_size() {
    // 10 frames, chunk_size=5, 8 frames per chunk (more than chunk has)
    // chunk 0: frames 0-4 → all 5 selected
    // chunk 1: frames 5-9 → all 5 selected
    let indices = compute_semantic_frame_indices(10, 5, 8);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn select_expr_single_index() {
    assert_eq!(build_select_expr(&[42]), "select='eq(n\\,42)'");
}

#[test]
fn select_expr_multiple_indices() {
    assert_eq!(
        build_select_expr(&[12, 37, 74]),
        "select='eq(n\\,12)+eq(n\\,37)+eq(n\\,74)'"
    );
}

#[test]
fn select_expr_empty() {
    assert_eq!(build_select_expr(&[]), "select='0'");
}

#[test]
fn select_expr_strided_singletons_is_constant_size() {
    // 1 frame/chunk midpoints: 2, 7, 12, ... — an arithmetic progression.
    // Must collapse to an O(1) expression, not one eq() term per frame, or
    // ffmpeg's filter parser runs out of memory on dense cadences.
    let indices: Vec<u64> = (0..167).map(|k| 2 + 5 * k).collect();
    assert_eq!(
        build_select_expr(&indices),
        "select='between(n\\,2\\,832)*eq(mod(n-2\\,5)\\,0)'"
    );
}

#[test]
fn select_expr_regular_clusters_is_constant_size() {
    // N frames/chunk: pairs 0,1 / 5,6 / 10,11 ... — equal-length runs, equal
    // stride. Collapses to (n-first) mod S < L.
    let indices = vec![0u64, 1, 5, 6, 10, 11, 15, 16];
    assert_eq!(
        build_select_expr(&indices),
        "select='between(n\\,0\\,16)*lt(mod(n\\,5)\\,2)'"
    );
}

#[test]
fn select_expr_contiguous_run_uses_between() {
    // A single contiguous block coalesces to one between().
    let indices: Vec<u64> = (0..111).collect();
    assert_eq!(build_select_expr(&indices), "select='between(n\\,0\\,110)'");
}

#[test]
fn select_expr_irregular_falls_back_to_terms() {
    // Non-strided, non-clustered singletons keep the per-term fallback so
    // sparse selections are unchanged.
    assert_eq!(
        build_select_expr(&[3, 4, 100]),
        "select='between(n\\,3\\,4)+eq(n\\,100)'"
    );
}

#[test]
fn select_expr_two_singletons_stay_literal() {
    // A 2-run block is below the collapse threshold (3), so it stays as two
    // literal eq() terms — shorter than the mod form for so few frames and,
    // more to the point, still bounded.
    assert_eq!(
        build_select_expr(&[10, 25]),
        "select='eq(n\\,10)+eq(n\\,25)'"
    );
}

#[test]
fn select_expr_is_order_and_dedup_independent() {
    // ffmpeg's select filter emits frames in ascending stream order, once per
    // distinct index, so an unsorted or duplicated caller list yields the same
    // filter. That invariant is exactly why the selective decoders restamp the
    // emitted frames against the sorted-unique index order rather than the
    // caller's order; stamping against the caller's order would swap identities
    // (e.g. [10, 2] emits 2 then 10 but would label them 10 then 2).
    let canonical = build_select_expr(&[2, 10]);
    assert_eq!(build_select_expr(&[10, 2]), canonical);
    assert_eq!(build_select_expr(&[2, 2, 10]), canonical);
    assert_eq!(build_select_expr(&[10, 2, 10, 2]), canonical);
}

#[test]
fn select_expr_partial_final_chunk_keeps_bulk_collapsed() {
    // THE production case: 833 frames, chunk_size=5, 1 frame/chunk. The 166
    // full chunks give evenly-strided midpoints 2,7,..,827; the trailing
    // 3-frame chunk's midpoint lands at 831 (stride 4, not 5). The dense bulk
    // must still collapse to one term — only the stray tail frame spills to an
    // eq() — or ffmpeg's parser OOMs on the ~167-term sum (bead vidarax-lvi).
    let mut indices: Vec<u64> = (0..166).map(|k| 2 + 5 * k).collect();
    indices.push(831);
    assert_eq!(
        build_select_expr(&indices),
        "select='between(n\\,2\\,827)*eq(mod(n-2\\,5)\\,0)+eq(n\\,831)'"
    );
}

// ---------------------------------------------------------------------------
// Semantic property tests: check the *frame set* the expression selects, not
// the emitted string. A reference evaluator expands `select='...'` the way
// ffmpeg would; the compact form must select exactly the distinct input
// indices. This catches bound/residue errors a pinned-string test can't.
// ---------------------------------------------------------------------------

/// Frames in `0..=max_n` that the emitted expression selects.
fn selected_frames(expr: &str, max_n: u64) -> Vec<u64> {
    let inner = expr
        .strip_prefix("select='")
        .and_then(|s| s.strip_suffix('\''))
        .expect("select='...' wrapper");
    if inner == "0" {
        return Vec::new();
    }
    // Undo ffmpeg comma-escaping so the tiny parser below stays readable.
    let unescaped = inner.replace("\\,", ",");
    let terms: Vec<&str> = unescaped.split('+').collect();
    (0..=max_n)
        .filter(|&n| terms.iter().any(|t| eval_term(t, n)))
        .collect()
}

fn eval_term(term: &str, n: u64) -> bool {
    match term.split_once('*') {
        Some((l, r)) => eval_atom(l, n) && eval_atom(r, n),
        None => eval_atom(term, n),
    }
}

fn eval_atom(atom: &str, n: u64) -> bool {
    if let Some(rest) = atom.strip_prefix("eq(n,") {
        let x: u64 = rest.strip_suffix(')').unwrap().parse().unwrap();
        return n == x;
    }
    if let Some(rest) = atom.strip_prefix("between(n,") {
        let (a, b) = rest.strip_suffix(')').unwrap().split_once(',').unwrap();
        return n >= a.parse::<u64>().unwrap() && n <= b.parse::<u64>().unwrap();
    }
    // eq(mod(M,S),0)  or  lt(mod(M,S),L), where M is `n` or `n-first`.
    let (is_eq, rest) = match (atom.strip_prefix("eq(mod("), atom.strip_prefix("lt(mod(")) {
        (Some(r), _) => (true, r),
        (_, Some(r)) => (false, r),
        _ => panic!("unrecognized select atom: {atom}"),
    };
    let (modinner, tail) = rest.split_once(')').unwrap(); // "M,S" , ",CMP)"
    let (m, s) = modinner.split_once(',').unwrap();
    let s: u64 = s.parse().unwrap();
    let cmp: u64 = tail.trim_matches(|c| c == ',' || c == ')').parse().unwrap();
    // `between(...)` gates this atom (S > L guarantees n >= first), so `n - first`
    // never underflows for any n that reaches here.
    let mval = if m == "n" {
        n
    } else {
        n - m.strip_prefix("n-").unwrap().parse::<u64>().unwrap()
    };
    let r = mval % s;
    if is_eq {
        r == cmp
    } else {
        r < cmp
    }
}

#[test]
fn select_expr_property_matches_input_set() {
    // Many pseudo-random sorted-unique sets: the emitted expression must select
    // exactly the input. Deterministic LCG keeps the fuzz reproducible.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }
    for _ in 0..3000 {
        let max = 1 + next(&mut state) % 64; // domain 1..=64
        let mut set = Vec::new();
        for i in 0..max {
            if next(&mut state) % 3 == 0 {
                set.push(i);
            }
        }
        let expr = build_select_expr(&set);
        assert_eq!(selected_frames(&expr, max - 1), set, "expr={expr}");
    }
}

#[test]
fn select_expr_property_matches_structured_blocks() {
    // Directly exercise the collapse path: 6 clusters of length L at stride S,
    // both from-zero (modarg `n`) and offset (modarg `n-first`). L < S so the
    // clusters never merge — the S > L invariant the collapse relies on.
    for s in 2u64..7 {
        for l in 1..s {
            for first in [0u64, 3] {
                let mut set = Vec::new();
                for k in 0..6u64 {
                    for o in 0..l {
                        set.push(first + k * s + o);
                    }
                }
                let expr = build_select_expr(&set);
                let max_n = *set.last().unwrap();
                assert_eq!(
                    selected_frames(&expr, max_n),
                    set,
                    "s={s} l={l} first={first} expr={expr}"
                );
            }
        }
    }
}

#[test]
fn select_expr_normalizes_unsorted_input() {
    // Regression: unsorted input used to silently select only the
    // first index. It now normalizes, so selection is order-independent.
    let expr = build_select_expr(&[10, 1, 2, 3]);
    assert_eq!(selected_frames(&expr, 10), vec![1, 2, 3, 10]);
}

#[test]
fn select_expr_dedups_duplicates() {
    let expr = build_select_expr(&[5, 5, 5]);
    assert_eq!(selected_frames(&expr, 6), vec![5]);
}

#[test]
fn select_expr_duplicate_u64_max_does_not_panic() {
    // Regression: `*hi + 1` overflowed (debug panic) on duplicate
    // MAX before normalization collapsed it to a single index.
    let expr = build_select_expr(&[u64::MAX, u64::MAX]);
    assert_eq!(expr, format!("select='eq(n\\,{})'", u64::MAX));
}

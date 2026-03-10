#!/usr/bin/env python3
"""Vidarax real-time streaming benchmark.

Measures per-frame detection latency by running /reason in a background thread
and polling /events every 200ms in the main thread. Shows detections in
real-time as they arrive from the pipeline WAL, then prints a full summary
with timeline, latency stats, and accuracy vs ground truth.

Usage:
    python benchmarks/bench_realtime.py
    python benchmarks/bench_realtime.py --api http://localhost:8080
    python benchmarks/bench_realtime.py --api http://localhost:8080 --fps 2 --chunk-size 10

# ─────────────────────────────────────────────────────────────────────────────
# WHIP STREAMING ANALYSIS (from crates/vidarax-api/src/whip.rs)
# ─────────────────────────────────────────────────────────────────────────────
#
# The WHIP endpoint (POST /v1/stream/whip) implements RFC 9725 WebRTC-HTTP
# Ingestion Protocol:
#   1. Client POSTs an SDP offer (Content-Type: application/sdp)
#   2. Server returns 201 + SDP answer + Location: /v1/stream/whip/{sess_id}
#   3. Client sends trickle ICE candidates via PATCH /v1/stream/whip/{sess_id}
#   4. Client DELETEs /v1/stream/whip/{sess_id} to terminate
#
# VERDICT: Full WebRTC via WHIP is NOT suitable for a simple benchmark harness.
# Reasons:
#   - Requires a real WebRTC stack (ICE negotiation, DTLS-SRTP, RTP packetizing)
#   - The drain task in whip.rs (line 178-189) currently DISCARDS all incoming
#     H.264 NAL units — the decode pipeline (x02.3) is not yet wired in.
#     Sending frames via WHIP would produce no detections at all right now.
#   - There is no simpler "chunked HTTP frame upload" path on the WHIP handler;
#     it only speaks full WebRTC signalling.
#   - The prompt-update endpoint (PATCH .../prompt) is interesting for live
#     reconfiguration but is irrelevant until the decode pipeline lands.
#
# CHOSEN APPROACH: Approach 3 — /reason + event polling
#   - POST to /v1/runs/{id}/reason with source_uri=file:///tmp/...
#   - Poll /v1/runs/{id}/events every 200ms
#   - Capture semantic_chunk_inferred events with non-empty raw_output
#   - This gives genuine streaming semantics because the pipeline writes WAL
#     events incrementally as each chunk is processed.
# ─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from difflib import SequenceMatcher
from pathlib import Path

# ─── ANSI colours ─────────────────────────────────────────────────────────────
BOLD    = "\033[1m"
RESET   = "\033[0m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
DIM     = "\033[2m"

# ─── API defaults ─────────────────────────────────────────────────────────────
DEFAULT_API   = os.environ.get("VIDARAX_API", "http://localhost:8080")
DEFAULT_VIDEO = "file:///tmp/vidarax-uploads/Wiki.mp4"
DEFAULT_GT    = str(Path(__file__).parent / "wiki_ground_truth.json")

POLL_INTERVAL_S = 0.2   # 200 ms
REASON_TIMEOUT  = 300   # seconds — video is ~157 s, give plenty of headroom


# ─── HTTP helpers ──────────────────────────────────────────────────────────────

def http_post(url: str, body: dict, timeout: int = REASON_TIMEOUT) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ─── String helpers ────────────────────────────────────────────────────────────

def fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Case-insensitive fuzzy string match (same logic as bench.py)."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def fmt_ms(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_wall(secs: float) -> str:
    """Format wall-clock elapsed seconds as +HH:MM:SS.s or +SS.s."""
    if secs < 60:
        return f"+{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"+{m:02d}:{s:02d}s"


# ─── Ground-truth matching ─────────────────────────────────────────────────────

def match_interactions(detected: list[dict], ground_truth: list[dict]):
    """Match detected interactions to GT using element fuzzy match.

    Returns (matches, false_positives, false_negatives).
    matches is a list of (gt_idx, det_idx, score) tuples.
    """
    gt_matched  = set()
    det_matched = set()
    matches     = []

    for gt_idx, gt in enumerate(ground_truth):
        gt_el = gt.get("element", "")
        gt_ac = gt.get("action", "")
        best_score = 0.0
        best_det   = None

        for det_idx, det in enumerate(detected):
            if det_idx in det_matched:
                continue
            det_el = det.get("element", "")
            det_ac = det.get("action", "")
            if not fuzzy_match(gt_el, det_el):
                continue
            score = SequenceMatcher(None, gt_el.lower(), det_el.lower()).ratio()
            action_ok = (
                gt_ac == det_ac
                or fuzzy_match(gt_ac, det_ac)
                or gt_ac in ("click", "navigate") and det_ac in (
                    "click", "navigate", "link_click",
                    "sidebar_selection", "page_change"
                )
            )
            if action_ok:
                score += 0.2
            if score > best_score:
                best_score = score
                best_det   = det_idx

        if best_det is not None and best_score > 0.5:
            gt_matched.add(gt_idx)
            det_matched.add(best_det)
            matches.append((gt_idx, best_det, round(best_score, 2)))

    false_negatives = [i for i in range(len(ground_truth)) if i not in gt_matched]
    false_positives = [i for i in range(len(detected))     if i not in det_matched]
    return matches, false_positives, false_negatives


# ─── Background reason thread ──────────────────────────────────────────────────

def _reason_worker(api: str, run_id: str, config: dict, holder: dict) -> None:
    """POST /reason and store the result in *holder*. Called in a thread."""
    try:
        resp = http_post(f"{api}/v1/runs/{run_id}/reason", config, timeout=REASON_TIMEOUT)
        holder["response"] = resp
    except Exception as exc:
        holder["error"] = str(exc)
    finally:
        holder["done"] = True


# ─── Main benchmark ────────────────────────────────────────────────────────────

def run_realtime_bench(
    api: str,
    source_uri: str,
    gt_path: str,
    config_overrides: dict | None = None,
) -> None:

    # ── Load ground truth ──────────────────────────────────────────────────────
    with open(gt_path) as f:
        gt_data = json.load(f)
    ground_truth: list[dict] = gt_data["interactions"]

    # ── Pipeline config ────────────────────────────────────────────────────────
    config = {
        "source_uri": source_uri,
        "mode": "detailed",
        "model": "Qwen/Qwen3.5-4B",
        "sampling_policy": "fixed",
        "fixed_fps": 2,
        "max_frames": 512,
        "window_size": 8,
        "segment_ms": 250,
        "chunk_size": 10,
        "semantic_inference": True,
        "semantic_frames_per_chunk": 1,
        "semantic_timeout_ms": 30000,
        "semantic_prompt": (
            "This frame is from a screen recording of someone browsing documentation. "
            "Look at the UI state carefully. Identify any evidence of user interaction: "
            "clicked buttons/links (highlighted, active state), newly loaded pages, "
            "expanded menus, cursor on clickable elements. "
            "Use chunk_pts_start_ms and chunk_pts_end_ms from context for timing. "
            "Return a JSON array of interactions found. Return [] if no interaction evidence."
        ),
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp":        {"type": "string"},
                    "element":          {"type": "string"},
                    "action":           {"type": "string"},
                    "duration_seconds": {"type": "number"},
                },
                "required": ["timestamp", "element", "action", "duration_seconds"],
            },
        },
    }
    if config_overrides:
        config.update(config_overrides)

    # ── Header ─────────────────────────────────────────────────────────────────
    W = 72
    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  VIDARAX REAL-TIME STREAMING BENCHMARK{RESET}")
    print(f"{BOLD}{'═' * W}{RESET}")
    print(f"  Source:  {source_uri}")
    print(f"  GT:      {gt_path}  ({len(ground_truth)} interactions)")
    print(f"  API:     {api}")
    print(f"  Config:  fps={config['fixed_fps']}  chunk={config['chunk_size']}  "
          f"frames/chunk={config['semantic_frames_per_chunk']}")
    print()

    # ── Create run ─────────────────────────────────────────────────────────────
    run_id = http_post(f"{api}/v1/runs", {"mode": "detailed"})["run_id"]
    print(f"  Run ID:  {CYAN}{run_id}{RESET}")
    print()

    # ── Launch /reason in background ───────────────────────────────────────────
    reason_holder: dict = {"done": False}
    t_start = time.time()

    bg = threading.Thread(
        target=_reason_worker,
        args=(api, run_id, config, reason_holder),
        daemon=True,
    )
    bg.start()

    # ── Live-stream header ─────────────────────────────────────────────────────
    COL_W  = 26
    TS_W   = 7
    WALL_W = 8
    LAT_W  = 9
    print(f"{BOLD}{'WALL':>{WALL_W}}  {'VIDEO':>{TS_W}}  {'LAT':>{LAT_W}}  "
          f"{'CHK':>4}  {'ACTION':>10}  {'ELEMENT':<{COL_W}}  GT{RESET}")
    print(f"{'─' * W}")

    # ── Polling state ──────────────────────────────────────────────────────────
    seen_chunks:     set[int]  = set()   # chunk_index values already printed
    live_detections: list[dict] = []     # (element, action, video_ts, wall_s, lat_s)
    chunk_process_ms: list[int] = []     # from semantic_chunk_generated events
    last_event_count = 0
    first_detection_wall: float | None = None

    # ── Poll loop ──────────────────────────────────────────────────────────────
    while not reason_holder.get("done"):
        time.sleep(POLL_INTERVAL_S)
        elapsed = time.time() - t_start

        try:
            events: list[dict] = http_get(f"{api}/v1/runs/{run_id}/events").get("events", [])
        except Exception:
            continue

        if len(events) == last_event_count:
            continue
        last_event_count = len(events)

        for evt in events:
            kind    = evt.get("kind", "")
            payload = evt.get("payload", {})

            # Collect chunk processing times for throughput stats
            if kind == "semantic_chunk_generated":
                ms = payload.get("process_ms")
                if isinstance(ms, int) and ms > 0:
                    chunk_process_ms.append(ms)
                continue

            if kind != "semantic_chunk_inferred":
                continue

            chunk_idx: int | None = payload.get("chunk_index")
            if chunk_idx in seen_chunks:
                continue
            seen_chunks.add(chunk_idx)

            # Skip errored or empty chunks
            if payload.get("semantic_error"):
                continue
            raw = payload.get("raw_output")
            if not raw or not isinstance(raw, list):
                continue

            chunk_pts_start_ms: int = payload.get("chunk_pts_start_ms", 0)
            chunk_pts_end_ms:   int = payload.get("chunk_pts_end_ms", 0)

            for item in raw:
                if not isinstance(item, dict):
                    continue

                element   = item.get("element", "?")
                action    = item.get("action",  "?")
                video_ts  = item.get("timestamp", fmt_ms(chunk_pts_start_ms))

                wall_s    = time.time() - t_start
                if first_detection_wall is None:
                    first_detection_wall = wall_s

                # Detection latency = wall-clock time at detection minus
                # the video timestamp of the chunk's start.
                # This measures how long after the depicted moment the system
                # produced the detection.
                video_ms = chunk_pts_start_ms
                lat_s    = wall_s - (video_ms / 1000.0)

                # GT lookup
                gt_label = ""
                gt_color = YELLOW
                for gt in ground_truth:
                    if fuzzy_match(element, gt["element"]):
                        gt_label = f"= {gt['element']} [{gt['timestamp']}]"
                        gt_color = GREEN
                        break
                if not gt_label:
                    gt_label = "(no GT match)"

                # Record for summary
                live_detections.append({
                    "element":     element,
                    "action":      action,
                    "video_ts":    video_ts,
                    "chunk_idx":   chunk_idx,
                    "chunk_start": chunk_pts_start_ms,
                    "wall_s":      wall_s,
                    "lat_s":       lat_s,
                    "gt_match":    bool(gt_label.startswith("=")),
                })

                # Live print row
                wall_fmt = fmt_wall(wall_s)
                lat_fmt  = f"{lat_s:+.1f}s"
                el_trunc = element[:COL_W - 1] if len(element) >= COL_W else element
                print(
                    f"{wall_fmt:>{WALL_W}}  "
                    f"{video_ts:>{TS_W}}  "
                    f"{lat_fmt:>{LAT_W}}  "
                    f"{chunk_idx:>4}  "
                    f"{action:>10}  "
                    f"{el_trunc:<{COL_W}}  "
                    f"{gt_color}{gt_label}{RESET}"
                )
                sys.stdout.flush()

    # Drain any final events that arrived after done flag set
    try:
        final_events: list[dict] = http_get(f"{api}/v1/runs/{run_id}/events").get("events", [])
        for evt in final_events:
            kind    = evt.get("kind", "")
            payload = evt.get("payload", {})
            if kind == "semantic_chunk_generated":
                ms = payload.get("process_ms")
                if isinstance(ms, int) and ms > 0:
                    chunk_process_ms.append(ms)
            if kind != "semantic_chunk_inferred":
                continue
            chunk_idx = payload.get("chunk_index")
            if chunk_idx in seen_chunks:
                continue
            seen_chunks.add(chunk_idx)
            if payload.get("semantic_error"):
                continue
            raw = payload.get("raw_output")
            if not raw or not isinstance(raw, list):
                continue
            chunk_pts_start_ms = payload.get("chunk_pts_start_ms", 0)
            for item in raw:
                if not isinstance(item, dict):
                    continue
                element  = item.get("element", "?")
                action   = item.get("action",  "?")
                video_ts = item.get("timestamp", fmt_ms(chunk_pts_start_ms))
                wall_s   = time.time() - t_start
                lat_s    = wall_s - (chunk_pts_start_ms / 1000.0)
                gt_label = ""
                gt_color = YELLOW
                for gt in ground_truth:
                    if fuzzy_match(element, gt["element"]):
                        gt_label = f"= {gt['element']} [{gt['timestamp']}]"
                        gt_color = GREEN
                        break
                if not gt_label:
                    gt_label = "(no GT match)"
                live_detections.append({
                    "element":     element,
                    "action":      action,
                    "video_ts":    video_ts,
                    "chunk_idx":   chunk_idx,
                    "chunk_start": chunk_pts_start_ms,
                    "wall_s":      wall_s,
                    "lat_s":       lat_s,
                    "gt_match":    bool(gt_label.startswith("=")),
                })
                wall_fmt = fmt_wall(wall_s)
                lat_fmt  = f"{lat_s:+.1f}s"
                el_trunc = element[:COL_W - 1] if len(element) >= COL_W else element
                print(
                    f"{wall_fmt:>{WALL_W}}  "
                    f"{video_ts:>{TS_W}}  "
                    f"{lat_fmt:>{LAT_W}}  "
                    f"{chunk_idx:>4}  "
                    f"{action:>10}  "
                    f"{el_trunc:<{COL_W}}  "
                    f"{gt_color}{gt_label}{RESET}"
                )
                sys.stdout.flush()
    except Exception:
        pass

    bg.join()
    t_total = time.time() - t_start

    print(f"\n{'─' * W}")

    # ── Fetch final interactions from API ──────────────────────────────────────
    try:
        final_interactions: list[dict] = (
            http_get(f"{api}/v1/runs/{run_id}/interactions").get("interactions", [])
        )
    except Exception:
        final_interactions = []

    reason_resp = reason_holder.get("response", {})
    reason_err  = reason_holder.get("error")

    # ── Accuracy ───────────────────────────────────────────────────────────────
    matches, fps_idx, fns_idx = match_interactions(final_interactions, ground_truth)
    n_det = len(final_interactions)
    n_gt  = len(ground_truth)
    n_tp  = len(matches)
    n_fp  = len(fps_idx)
    n_fn  = len(fns_idx)

    precision = n_tp / n_det if n_det else 0.0
    recall    = n_tp / n_gt  if n_gt  else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Latency stats ──────────────────────────────────────────────────────────
    all_lats   = [d["lat_s"]  for d in live_detections]
    all_walls  = [d["wall_s"] for d in live_detections]

    avg_lat   = sum(all_lats)  / len(all_lats)  if all_lats  else 0.0
    min_lat   = min(all_lats)                   if all_lats  else 0.0
    max_lat   = max(all_lats)                   if all_lats  else 0.0
    first_det = min(all_walls)                  if all_walls else 0.0

    # ── Throughput ─────────────────────────────────────────────────────────────
    decoded_frames = reason_resp.get("decoded_frames", 0) or 0
    throughput_fps = decoded_frames / t_total if t_total > 0 else 0.0

    avg_chunk_ms = (sum(chunk_process_ms) / len(chunk_process_ms)
                    if chunk_process_ms else 0.0)

    # ── Event counters ─────────────────────────────────────────────────────────
    try:
        all_events: list[dict] = http_get(f"{api}/v1/runs/{run_id}/events").get("events", [])
    except Exception:
        all_events = []

    vlm_ok    = sum(1 for e in all_events
                    if e.get("kind") == "semantic_chunk_inferred"
                    and not e.get("payload", {}).get("semantic_error"))
    vlm_err   = sum(1 for e in all_events
                    if e.get("kind") == "semantic_chunk_inferred"
                    and e.get("payload", {}).get("semantic_error"))

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  SUMMARY REPORT{RESET}")
    print(f"{BOLD}{'═' * W}{RESET}\n")

    # --- Timing ---------------------------------------------------------------
    print(f"{BOLD}  TIMING{RESET}")
    print(f"  {'─' * 40}")
    print(f"  {'Total wall time:':<28} {t_total:.2f}s")
    print(f"  {'Frames decoded:':<28} {decoded_frames}")
    print(f"  {'Throughput:':<28} {throughput_fps:.2f} frames/s")
    print(f"  {'VLM chunks (ok / err):':<28} {vlm_ok} / {vlm_err}")
    if chunk_process_ms:
        print(f"  {'Avg chunk process time:':<28} {avg_chunk_ms:.0f}ms")
        print(f"  {'Min/Max chunk time:':<28} {min(chunk_process_ms)}ms / {max(chunk_process_ms)}ms")
    print()

    # --- Latency --------------------------------------------------------------
    print(f"{BOLD}  DETECTION LATENCY{RESET}")
    print(f"  {'─' * 40}")
    if all_lats:
        print(f"  {'First detection (wall):':<28} {fmt_wall(first_det)}")
        print(f"  {'Avg detection latency:':<28} {avg_lat:+.2f}s")
        print(f"  {'Min detection latency:':<28} {min_lat:+.2f}s")
        print(f"  {'Max detection latency:':<28} {max_lat:+.2f}s")
        print()
        print(f"  {DIM}Latency = wall-clock time of detection minus video-time of chunk start.")
        print(f"  Negative values mean we detected an event before that video second elapsed.{RESET}")
    else:
        print(f"  {YELLOW}No detections recorded.{RESET}")
    print()

    # --- Accuracy -------------------------------------------------------------
    print(f"{BOLD}  ACCURACY vs GROUND TRUTH{RESET}")
    print(f"  {'─' * 40}")
    print(f"  {'Ground truth interactions:':<28} {n_gt}")
    print(f"  {'Detected interactions:':<28} {n_det}")
    print(f"  {'True positives:':<28} {n_tp}")
    print(f"  {'False positives:':<28} {n_fp}")
    print(f"  {'False negatives:':<28} {n_fn}")
    print()

    p_color = GREEN if precision >= 0.7 else (YELLOW if precision >= 0.4 else RED)
    r_color = GREEN if recall    >= 0.7 else (YELLOW if recall    >= 0.4 else RED)
    f_color = GREEN if f1        >= 0.7 else (YELLOW if f1        >= 0.4 else RED)
    print(f"  {'Precision:':<28} {p_color}{precision:.1%}{RESET}")
    print(f"  {'Recall:':<28}    {r_color}{recall:.1%}{RESET}")
    print(f"  {'F1 Score:':<28}  {f_color}{f1:.1%}{RESET}")
    print()

    # --- Timeline -------------------------------------------------------------
    print(f"{BOLD}  DETECTION TIMELINE{RESET}")
    print(f"  {'─' * 40}")
    if live_detections:
        # Sort by wall-clock time
        for d in sorted(live_detections, key=lambda x: x["wall_s"]):
            marker = f"{GREEN}[TP]{RESET}" if d["gt_match"] else f"{YELLOW}[FP]{RESET}"
            lat_str = f"{d['lat_s']:+.1f}s"
            el_trunc = d["element"][:28]
            print(f"  {fmt_wall(d['wall_s']):>8}  vid={d['video_ts']:>6}  lat={lat_str:>6}  "
                  f"chk={d['chunk_idx']:>3}  {marker}  {el_trunc}")
    else:
        print(f"  {YELLOW}No interactions detected during streaming.{RESET}")
    print()

    # --- Matched / missed / false positives -----------------------------------
    if matches:
        print(f"{BOLD}  MATCHED INTERACTIONS ({n_tp}){RESET}")
        print(f"  {'─' * 40}")
        for gt_idx, det_idx, score in matches:
            gt  = ground_truth[gt_idx]
            det = final_interactions[det_idx]
            print(f"  [{gt.get('timestamp','?')}]  {GREEN}{gt['element']:<24}{RESET}"
                  f"  ->  {det.get('element','?'):<24}  score={score}")
        print()

    if fns_idx:
        print(f"{BOLD}  MISSED ({n_fn}){RESET}")
        print(f"  {'─' * 40}")
        for idx in fns_idx:
            gt = ground_truth[idx]
            print(f"  [{gt.get('timestamp','?')}]  {RED}{gt['element']}{RESET}  ({gt['action']})")
        print()

    if fps_idx:
        print(f"{BOLD}  FALSE POSITIVES ({n_fp}){RESET}")
        print(f"  {'─' * 40}")
        for idx in fps_idx[:10]:
            det = final_interactions[idx]
            print(f"  chunk {det.get('chunk_index','?'):>3}:  {YELLOW}{det.get('element','?')}{RESET}"
                  f"  ({det.get('action','?')})")
        if n_fp > 10:
            print(f"  ... and {n_fp - 10} more")
        print()

    # --- Error ----------------------------------------------------------------
    if reason_err:
        print(f"{RED}  Reason thread error: {reason_err[:200]}{RESET}\n")

    # --- One-line summary -----------------------------------------------------
    print(f"{BOLD}{'═' * W}{RESET}")
    summary_color = GREEN if f1 >= 0.7 else (YELLOW if f1 >= 0.4 else RED)
    print(
        f"{BOLD}  RESULT: {summary_color}{f1:.0%} F1{RESET}{BOLD}  |  "
        f"{n_tp}/{n_gt} matched  |  "
        f"{t_total:.1f}s total  |  "
        f"{throughput_fps:.1f} fps  |  "
        f"first det {fmt_wall(first_det) if all_walls else 'N/A'}{RESET}"
    )
    print(f"{BOLD}{'═' * W}{RESET}\n")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vidarax real-time streaming benchmark (Approach 3: /reason + event polling)"
    )
    parser.add_argument("--api",         default=DEFAULT_API,   help="API base URL")
    parser.add_argument("--video",       default=DEFAULT_VIDEO, help="source_uri for the video")
    parser.add_argument("--gt",          default=DEFAULT_GT,    help="Path to ground-truth JSON")
    parser.add_argument("--fps",         type=int, default=2,   help="fixed_fps for sampling")
    parser.add_argument("--chunk-size",  type=int, default=10,  help="chunk_size (frames/chunk)")
    parser.add_argument("--frames-per-chunk", type=int, default=1,
                        help="semantic_frames_per_chunk")
    parser.add_argument("--temporal-chain", action="store_true",
                        help="Enable sequential temporal chaining (slower, more accurate)")
    parser.add_argument("--visual-diff", action="store_true",
                        help="Send previous frame as second image (implies temporal-chain)")
    parser.add_argument("--model", default=None,
                        help="Override model name")
    args = parser.parse_args()

    overrides = {
        "fixed_fps":                args.fps,
        "chunk_size":               args.chunk_size,
        "semantic_frames_per_chunk": args.frames_per_chunk,
        "temporal_chain":           args.temporal_chain or args.visual_diff,
        "visual_diff":              args.visual_diff,
    }
    if args.model:
        overrides["model"] = args.model

    run_realtime_bench(
        api=args.api,
        source_uri=args.video,
        gt_path=args.gt,
        config_overrides=overrides,
    )

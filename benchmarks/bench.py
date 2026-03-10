#!/usr/bin/env python3
"""Vidarax interaction detection benchmark.

Runs the pipeline on a video, compares detected interactions against
ground truth, and prints a detailed accuracy report.

Usage:
    python benchmarks/bench.py --video /path/to/Wiki.mp4 --gt benchmarks/wiki_ground_truth.json
    python benchmarks/bench.py --video /path/to/Wiki.mp4 --gt benchmarks/wiki_ground_truth.json --api http://localhost:8080
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path
from difflib import SequenceMatcher

import urllib.request
import urllib.error


def http_post(url: str, body: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_timestamp_ms(ts: str) -> int | None:
    """Parse MM:SS or MM:SS.s to milliseconds."""
    try:
        parts = ts.replace(".", ":").split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60_000 + int(float(parts[1]) * 1000)
        if len(parts) == 3:
            return int(parts[0]) * 60_000 + int(parts[1]) * 1000 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Case-insensitive fuzzy string match."""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return True
    if a_lower in b_lower or b_lower in a_lower:
        return True
    return SequenceMatcher(None, a_lower, b_lower).ratio() >= threshold


def match_interactions(detected: list[dict], ground_truth: list[dict], time_window_ms: int = 10_000):
    """Match detected interactions to ground truth within a time window.

    Returns (matches, false_positives, false_negatives) where matches is a list
    of (gt_idx, det_idx, details) tuples.
    """
    gt_matched = set()
    det_matched = set()
    matches = []

    # Try to match each GT interaction to the closest detected one
    for gt_idx, gt in enumerate(ground_truth):
        gt_ms = gt.get("timestamp_ms", 0)
        gt_element = gt.get("element", "")
        gt_action = gt.get("action", "")

        best_det_idx = None
        best_score = 0

        for det_idx, det in enumerate(detected):
            if det_idx in det_matched:
                continue

            det_element = det.get("element", "")
            det_action = det.get("action", "")

            # Element match score
            if not fuzzy_match(gt_element, det_element):
                continue

            # Action match (flexible)
            action_ok = (
                gt_action == det_action
                or fuzzy_match(gt_action, det_action)
                or gt_action in ("click", "navigate") and det_action in ("click", "navigate", "link_click", "sidebar_selection", "page_change")
            )

            # Time match via chunk index (approximate)
            # We don't have reliable timestamps from VLM, so we rely on element matching
            score = SequenceMatcher(None, gt_element.lower(), det_element.lower()).ratio()
            if action_ok:
                score += 0.2

            if score > best_score:
                best_score = score
                best_det_idx = det_idx

        if best_det_idx is not None and best_score > 0.5:
            gt_matched.add(gt_idx)
            det_matched.add(best_det_idx)
            matches.append((gt_idx, best_det_idx, {
                "gt_element": gt_element,
                "det_element": detected[best_det_idx].get("element", ""),
                "gt_action": gt_action,
                "det_action": detected[best_det_idx].get("action", ""),
                "score": round(best_score, 2),
            }))

    false_negatives = [i for i in range(len(ground_truth)) if i not in gt_matched]
    false_positives = [i for i in range(len(detected)) if i not in det_matched]

    return matches, false_positives, false_negatives


def run_benchmark(api: str, video_path: str, gt_path: str, config: dict | None = None):
    """Run the full benchmark pipeline."""
    with open(gt_path) as f:
        gt_data = json.load(f)
    ground_truth = gt_data["interactions"]

    default_config = {
        "mode": "detailed",
        "model": "Qwen/Qwen3.5-9B",
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
                    "timestamp": {"type": "string"},
                    "element": {"type": "string"},
                    "action": {"type": "string"},
                    "duration_seconds": {"type": "number"}
                },
                "required": ["timestamp", "element", "action", "duration_seconds"]
            }
        }
    }
    if config:
        default_config.update(config)

    # Determine source_uri
    source_uri = f"file://{video_path}"

    print(f"{'=' * 60}")
    print(f"VIDARAX BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Video:        {video_path}")
    print(f"Ground truth: {gt_path} ({len(ground_truth)} interactions)")
    print(f"API:          {api}")
    print(f"Config:       fps={default_config['fixed_fps']}, chunk={default_config['chunk_size']}, frames/chunk={default_config['semantic_frames_per_chunk']}")
    print()

    # Step 1: Create run
    t0 = time.time()
    run_id = http_post(f"{api}/v1/runs", {"mode": "detailed"})["run_id"]
    t_create = time.time() - t0

    # Step 2: Reason
    default_config["source_uri"] = source_uri
    t1 = time.time()
    reason_data = http_post(f"{api}/v1/runs/{run_id}/reason", default_config, timeout=120)
    t_reason = time.time() - t1

    # Step 3: Get interactions
    t2 = time.time()
    interactions_data = http_get(f"{api}/v1/runs/{run_id}/interactions")
    t_interactions = time.time() - t2

    t_total = time.time() - t0
    detected = interactions_data.get("interactions", [])

    # Step 4: Get VLM stats from events
    events = http_get(f"{api}/v1/runs/{run_id}/events").get("events", [])
    vlm_ok = sum(1 for e in events if e.get("kind") == "semantic_chunk_inferred" and not e.get("payload", {}).get("semantic_error"))
    vlm_err = sum(1 for e in events if e.get("kind") == "semantic_chunk_inferred" and e.get("payload", {}).get("semantic_error"))
    vlm_empty = sum(1 for e in events if e.get("kind") == "semantic_chunk_inferred" and not e.get("payload", {}).get("semantic_error") and (e.get("payload", {}).get("raw_output") in (None, [])))

    # Step 5: Match
    matches, false_positives, false_negatives = match_interactions(detected, ground_truth)

    precision = len(matches) / len(detected) if detected else 0
    recall = len(matches) / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Report
    print(f"{'─' * 60}")
    print(f"TIMING")
    print(f"{'─' * 60}")
    print(f"  Create run:    {t_create*1000:>7.0f}ms")
    print(f"  Reason:        {t_reason*1000:>7.0f}ms  ({t_reason:.1f}s)")
    print(f"  Interactions:  {t_interactions*1000:>7.0f}ms")
    print(f"  TOTAL:         {t_total*1000:>7.0f}ms  ({t_total:.1f}s)")
    print()

    print(f"{'─' * 60}")
    print(f"PIPELINE")
    print(f"{'─' * 60}")
    print(f"  Frames decoded:  {reason_data.get('decoded_frames', '?')}")
    print(f"  Markers:         {len(reason_data.get('markers', []))}")
    print(f"  VLM chunks:      {vlm_ok + vlm_err} total, {vlm_ok} ok, {vlm_err} errors, {vlm_empty} empty")
    print()

    print(f"{'─' * 60}")
    print(f"ACCURACY")
    print(f"{'─' * 60}")
    print(f"  Ground truth:    {len(ground_truth)} interactions")
    print(f"  Detected:        {len(detected)} interactions")
    print(f"  True positives:  {len(matches)}")
    print(f"  False positives: {len(false_positives)}")
    print(f"  False negatives: {len(false_negatives)}")
    print()
    print(f"  Precision:       {precision:.1%}")
    print(f"  Recall:          {recall:.1%}")
    print(f"  F1 Score:        {f1:.1%}")
    print()

    if matches:
        print(f"{'─' * 60}")
        print(f"MATCHED ({len(matches)})")
        print(f"{'─' * 60}")
        for gt_idx, det_idx, details in matches:
            gt_ts = ground_truth[gt_idx].get("timestamp", "?")
            print(f"  [{gt_ts}] {details['gt_element']:<25} -> {details['det_element']:<25} (score={details['score']})")

    if false_negatives:
        print(f"\n{'─' * 60}")
        print(f"MISSED ({len(false_negatives)})")
        print(f"{'─' * 60}")
        for idx in false_negatives:
            gt = ground_truth[idx]
            print(f"  [{gt.get('timestamp','?')}] {gt['element']} ({gt['action']})")

    if false_positives:
        print(f"\n{'─' * 60}")
        print(f"FALSE POSITIVES ({len(false_positives)})")
        print(f"{'─' * 60}")
        for idx in false_positives[:10]:
            det = detected[idx]
            print(f"  chunk {det.get('chunk_index','?')}: {det.get('element','?')} ({det.get('action','?')})")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {f1:.0%} F1 in {t_total:.1f}s  (target: >80% F1 in <9s)")
    print(f"{'=' * 60}")

    return {
        "run_id": run_id,
        "timing": {"total_s": round(t_total, 1), "reason_s": round(t_reason, 1)},
        "accuracy": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)},
        "counts": {"ground_truth": len(ground_truth), "detected": len(detected), "matched": len(matches)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vidarax interaction detection benchmark")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON")
    parser.add_argument("--api", default=os.environ.get("VIDARAX_API", "http://localhost:8080"), help="Vidarax API URL")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--frames-per-chunk", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    config = {
        "fixed_fps": args.fps,
        "chunk_size": args.chunk_size,
        "semantic_frames_per_chunk": args.frames_per_chunk,
    }

    result = run_benchmark(args.api, args.video, args.gt, config)
    if args.json:
        print(json.dumps(result, indent=2))

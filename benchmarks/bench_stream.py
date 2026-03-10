#!/usr/bin/env python3
"""Vidarax streaming benchmark.

Simulates real-time streaming by running /reason in background while
polling /events to show results appearing progressively. Measures
detection latency: how long after video-time does each interaction appear.

Usage:
    python benchmarks/bench_stream.py --video /tmp/vidarax-uploads/Wiki.mp4 --gt benchmarks/wiki_ground_truth.json
"""
import argparse
import json
import os
import sys
import time
import threading
import urllib.request
from difflib import SequenceMatcher


def http_post(url, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url, timeout=10):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fuzzy_match(a, b, threshold=0.6):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def run_reason(api, run_id, config, result_holder):
    """Background thread: fires /reason and stores result."""
    try:
        resp = http_post(f"{api}/v1/runs/{run_id}/reason", config, timeout=180)
        result_holder["response"] = resp
        result_holder["done"] = True
    except Exception as e:
        result_holder["error"] = str(e)
        result_holder["done"] = True


def run_streaming_bench(api, video_path, gt_path, config_overrides=None):
    with open(gt_path) as f:
        gt_data = json.load(f)
    ground_truth = gt_data["interactions"]
    gt_by_element = {g["element"].lower(): g for g in ground_truth}

    config = {
        "source_uri": f"file://{video_path}",
        "mode": "detailed",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
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
    if config_overrides:
        config.update(config_overrides)

    print(f"\033[1m{'=' * 70}\033[0m")
    print(f"\033[1mVIDARAX STREAMING BENCHMARK\033[0m")
    print(f"\033[1m{'=' * 70}\033[0m")
    print(f"Video:   {video_path}")
    print(f"GT:      {len(ground_truth)} interactions")
    print(f"API:     {api}")
    print()

    # Create run
    run_id = http_post(f"{api}/v1/runs", {"mode": "detailed"})["run_id"]
    print(f"Run:     {run_id}")
    print()

    # Start reason in background
    result_holder = {"done": False}
    t_start = time.time()

    thread = threading.Thread(target=run_reason, args=(api, run_id, config, result_holder))
    thread.start()

    # Poll events while processing
    seen_chunks = set()
    seen_interactions = []
    detection_times = {}  # element -> wall_time_since_start
    last_event_count = 0

    print(f"\033[1m{'TIME':>7}  {'CHUNK':>5}  {'ACTION':>10}  {'ELEMENT':<30}  {'GT MATCH'}\033[0m")
    print(f"{'─' * 70}")

    while not result_holder.get("done"):
        time.sleep(0.3)
        elapsed = time.time() - t_start

        try:
            events_data = http_get(f"{api}/v1/runs/{run_id}/events")
            events = events_data.get("events", [])
        except Exception:
            continue

        if len(events) == last_event_count:
            continue
        last_event_count = len(events)

        for e in events:
            if e.get("kind") != "semantic_chunk_inferred":
                continue
            p = e.get("payload", {})
            chunk_idx = p.get("chunk_index")
            if chunk_idx in seen_chunks:
                continue
            seen_chunks.add(chunk_idx)

            raw = p.get("raw_output")
            if not raw or not isinstance(raw, list) or len(raw) == 0:
                continue

            for item in raw:
                if not isinstance(item, dict):
                    continue
                element = item.get("element", "?")
                action = item.get("action", "?")
                ts = item.get("timestamp", "?")

                # Check GT match
                gt_match = ""
                for gt in ground_truth:
                    if fuzzy_match(element, gt["element"]):
                        gt_match = f"\033[32m= {gt['element']} [{gt['timestamp']}]\033[0m"
                        if element.lower() not in detection_times:
                            detection_times[element.lower()] = elapsed
                        break

                if not gt_match:
                    gt_match = "\033[33m(no match)\033[0m"

                wall_s = f"{elapsed:.1f}s"
                print(f"{wall_s:>7}  {chunk_idx:>5}  {action:>10}  {element:<30}  {gt_match}")
                seen_interactions.append(item)

    thread.join()
    t_total = time.time() - t_start

    # Final fetch of interactions
    interactions_data = http_get(f"{api}/v1/runs/{run_id}/interactions")
    final_interactions = interactions_data.get("interactions", [])

    # Match results
    matched = set()
    for gt in ground_truth:
        for det in final_interactions:
            if fuzzy_match(gt["element"], det.get("element", "")) and gt["element"].lower() not in matched:
                matched.add(gt["element"].lower())
                break

    precision = len(matched) / len(final_interactions) if final_interactions else 0
    recall = len(matched) / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'─' * 70}")
    print()

    # Reason response info
    reason = result_holder.get("response", {})
    if "error" in reason:
        print(f"\033[31mReason error: {json.dumps(reason['error'])[:200]}\033[0m")

    print(f"\033[1m{'─' * 70}\033[0m")
    print(f"\033[1mRESULTS\033[0m")
    print(f"\033[1m{'─' * 70}\033[0m")
    print(f"  Total time:      {t_total:.1f}s")
    print(f"  Frames decoded:  {reason.get('decoded_frames', '?')}")
    print(f"  VLM chunks:      {len(seen_chunks)} processed")
    print(f"  Detected:        {len(final_interactions)} interactions")
    print(f"  Matched GT:      {len(matched)}/{len(ground_truth)}")
    print(f"  Precision:       {precision:.0%}")
    print(f"  Recall:          {recall:.0%}")
    print(f"  F1:              {f1:.0%}")
    print()

    # Detection latency
    if detection_times:
        print(f"\033[1mDETECTION LATENCY\033[0m")
        print(f"{'─' * 70}")
        latencies = []
        for element, wall_time in sorted(detection_times.items(), key=lambda x: x[1]):
            # Find GT timestamp
            gt_ts_ms = None
            for gt in ground_truth:
                if fuzzy_match(element, gt["element"]):
                    gt_ts_ms = gt.get("timestamp_ms", 0)
                    break
            if gt_ts_ms is not None:
                # Detection latency = wall_time - (video_time / total_time * total_wall_time)
                # But since batch processing, latency is just wall_time
                print(f"  {element:<30}  detected at {wall_time:.1f}s  (video: {gt_ts_ms/1000:.0f}s)")
                latencies.append(wall_time)
        if latencies:
            print(f"\n  Avg detection time: {sum(latencies)/len(latencies):.1f}s")
            print(f"  First detection:    {min(latencies):.1f}s")
            print(f"  Last detection:     {max(latencies):.1f}s")

    print(f"\n\033[1m{'=' * 70}\033[0m")
    print(f"\033[1mSUMMARY: {f1:.0%} F1, {len(matched)}/{len(ground_truth)} matched, {t_total:.1f}s total\033[0m")
    print(f"\033[1m{'=' * 70}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--api", default=os.environ.get("VIDARAX_API", "http://localhost:8080"))
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--frames-per-chunk", type=int, default=1)
    args = parser.parse_args()

    run_streaming_bench(args.api, args.video, args.gt, {
        "fixed_fps": args.fps,
        "chunk_size": args.chunk_size,
        "semantic_frames_per_chunk": args.frames_per_chunk,
    })

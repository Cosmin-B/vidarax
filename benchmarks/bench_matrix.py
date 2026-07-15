#!/usr/bin/env python3
"""Vidarax benchmark matrix.

Runs multiple configurations against ground truth and produces a comparison table.

Usage:
    python benchmarks/bench_matrix.py
    python benchmarks/bench_matrix.py --api http://localhost:8080 --model Qwen/Qwen3.5-4B
"""
import argparse
import json
import os
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_API = os.environ.get("VIDARAX_API", "http://localhost:8080")
DEFAULT_API_KEY = os.environ.get("VIDARAX_API_KEY", "")
DEFAULT_GT = str(Path(__file__).parent / "wiki_ground_truth.json")
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

_UPLOAD_DIR = os.environ.get("VIDARAX_UPLOAD_DIR", "/tmp/vidarax-uploads")

# ─── Pricing ─────────────────────────────────────────────────────────────────
# USD per 1,000,000 tokens, (input, output). These are ILLUSTRATIVE defaults for
# ranking configs against each other — verify against the live rate card before
# quoting an absolute dollar figure. Matched by substring against the model id;
# anything unmatched (i.e. a locally-hosted model) is priced at $0 tokens, since
# its cost is GPU-time, not per-token billing. Override wholesale with
# VIDARAX_PRICE_INPUT / VIDARAX_PRICE_OUTPUT (USD per 1M) to price one run.
PRICES_PER_MTOK = {
    # Approximate (input, output) USD per 1M tokens for the recognised GA Gemini
    # models; output tokens include thinking tokens. These are illustrative list
    # prices, not a live quote: override them per run with VIDARAX_PRICE_INPUT /
    # VIDARAX_PRICE_OUTPUT for exact accounting. Longest matching key wins in
    # price_for_model(), so version-qualified ids resolve exactly.
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-flash-lite-latest": (0.25, 1.50),     # alias -> 3.1-flash-lite
    "gemini-flash-lite": (0.25, 1.50),
    "gemini-flash-latest": (0.25, 1.50),          # alias -> 3.1-flash-lite
    "gemini-flash": (0.25, 1.50),
    "gemini": (0.25, 1.50),  # generic fallback
}


def price_for_model(model):
    """(input, output) USD per 1M tokens for `model`. Env override wins; then
    the longest matching substring key; else $0 (local model, GPU-time cost)."""
    env_in = os.environ.get("VIDARAX_PRICE_INPUT")
    env_out = os.environ.get("VIDARAX_PRICE_OUTPUT")
    if env_in is not None and env_out is not None:
        return (float(env_in), float(env_out))
    m = model.lower()
    best = None
    for key, rate in PRICES_PER_MTOK.items():
        if key in m and (best is None or len(key) > len(best[0])):
            best = (key, rate)
    return best[1] if best else (0.0, 0.0)


def estimate_cost_usd(model, prompt_tokens, output_tokens):
    """Billable cost. `output_tokens` must be the FULL billed output — visible
    completion plus any hidden thinking tokens — since Google bills thinking at
    the output rate. See billed_output_tokens()."""
    in_rate, out_rate = price_for_model(model)
    return (prompt_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0


def billed_output_tokens(prompt_tokens, completion_tokens, total_tokens):
    """Output tokens Google actually bills = candidates + thoughts. Gemini's
    usageMetadata reports totalTokenCount = prompt + candidates + thoughts, but
    exposes only candidates as completion_tokens; thinking (thoughtsTokenCount)
    is billed at the output rate yet omitted from completion. So the true billed
    output is total - prompt, which collapses to completion for non-thinking
    models. max() guards the fallback where total == prompt + completion."""
    if total_tokens > prompt_tokens:
        return max(completion_tokens, total_tokens - prompt_tokens)
    return completion_tokens


def pareto_frontier(results):
    """Return the subset of runs not dominated on all three objectives at once:
    higher F1, lower est_cost_usd, lower time_s. A run is dominated when another
    is at least as good on every objective and strictly better on one. These are
    the configs worth choosing between — the rest are beaten outright."""
    ok = [r for r in results if "error" not in r]
    frontier = []
    for a in ok:
        dominated = False
        for b in ok:
            if b is a:
                continue
            no_worse = (
                b["f1"] >= a["f1"]
                and b["est_cost_usd"] <= a["est_cost_usd"]
                and b["time_s"] <= a["time_s"]
            )
            strictly_better = (
                b["f1"] > a["f1"]
                or b["est_cost_usd"] < a["est_cost_usd"]
                or b["time_s"] < a["time_s"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    return frontier

RESOLUTIONS = {
    "1080p": f"file://{_UPLOAD_DIR}/Wiki.mp4",
    "720p": f"file://{_UPLOAD_DIR}/Wiki_720p.mp4",
    "480p": f"file://{_UPLOAD_DIR}/Wiki_480p.mp4",
}

# ─── Preset configs ──────────────────────────────────────────────────────────

PRESETS = {
    # Clip mode presets
    "clip_snappy": {
        "label": "Clip snappy (0.3s, 5f, 15fps)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 15,
            "chunk_size": 5,
            "semantic_frames_per_chunk": 3,
            "semantic_timeout_ms": 30000,
        },
    },
    "clip_balanced": {
        "label": "Clip balanced (1s, 8f, 8fps)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 8,
            "chunk_size": 8,
            "semantic_frames_per_chunk": 4,
            "semantic_timeout_ms": 30000,
        },
    },
    "clip_detailed": {
        "label": "Clip detailed (2s, 20f, 10fps)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 10,
            "chunk_size": 20,
            "semantic_frames_per_chunk": 4,
            "semantic_timeout_ms": 30000,
        },
    },
    # Frame mode presets
    "frame_single": {
        "label": "Frame single (1f/chunk, 2fps)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
        },
    },
    "frame_temporal": {
        "label": "Frame temporal chain (1f, sequential)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "temporal_chain": True,
        },
    },
    "frame_visual_diff": {
        "label": "Visual diff (2 images, sequential)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "visual_diff": True,
        },
    },
    "clip_custom_fast": {
        "label": "Clip fast (0.5s, 2f, 4fps)",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 4,
            "chunk_size": 5,
            "semantic_frames_per_chunk": 2,
            "semantic_timeout_ms": 30000,
        },
    },
    # Video clip mode presets (sends MP4 segments instead of JPEG frames)
    "video_02s": {
        "label": "Video clip 0.2s",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "video_clip_mode": True,
            "video_clip_duration_s": 0.2,
        },
    },
    "video_05s": {
        "label": "Video clip 0.5s",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "video_clip_mode": True,
            "video_clip_duration_s": 0.5,
        },
    },
    "video_1s": {
        "label": "Video clip 1.0s",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "video_clip_mode": True,
            "video_clip_duration_s": 1.0,
        },
    },
    "video_2s": {
        "label": "Video clip 2.0s",
        "config": {
            "sampling_policy": "fixed",
            "fixed_fps": 2,
            "chunk_size": 10,
            "semantic_frames_per_chunk": 1,
            "semantic_timeout_ms": 30000,
            "video_clip_mode": True,
            "video_clip_duration_s": 2.0,
        },
    },
}


def api_headers(content_type=False):
    headers = {"Content-Type": "application/json"} if content_type else {}
    if DEFAULT_API_KEY:
        headers["x-api-key"] = DEFAULT_API_KEY
    return headers


def http_post(url, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=api_headers(content_type=True))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers=api_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fuzzy_match(a, b, threshold=0.6):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def run_single(api, model, source_uri, gt, preset_name, preset):
    config = {
        "source_uri": source_uri,
        "mode": "detailed",
        "model": model,
        "max_frames": 512,
        "semantic_inference": True,
        "semantic_prompt": (
            "Describe what is visible in this frame in one sentence. "
            "Focus on: page title, active menu items, visible UI elements, "
            "any person/object/scene."
        ),
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "element": {"type": "string"},
                    "action": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                },
                "required": ["timestamp", "element", "action", "duration_seconds"],
            },
        },
    }
    config.update(preset["config"])

    try:
        run_id = http_post(f"{api}/v1/runs", {"mode": "detailed"})["run_id"]
    except Exception as e:
        return {"preset": preset_name, "error": f"create_run: {e}"}

    t0 = time.time()
    try:
        reason = http_post(f"{api}/v1/runs/{run_id}/reason", config, timeout=300)
    except Exception as e:
        return {"preset": preset_name, "error": f"reason: {e}", "time_s": time.time() - t0}
    t_reason = time.time() - t0

    if "error" in reason:
        return {"preset": preset_name, "error": json.dumps(reason["error"])[:200], "time_s": t_reason}

    try:
        interactions = http_get(f"{api}/v1/runs/{run_id}/interactions")
    except Exception:
        interactions = {"interactions": []}

    try:
        events = http_get(f"{api}/v1/runs/{run_id}/events")
    except Exception:
        events = {"events": []}

    detected = interactions.get("interactions", [])
    evts = events.get("events", [])
    vlm_ok = sum(1 for e in evts if e.get("kind") == "semantic_chunk_inferred" and not e.get("payload", {}).get("semantic_error"))
    vlm_err = sum(1 for e in evts if e.get("kind") == "semantic_chunk_inferred" and e.get("payload", {}).get("semantic_error"))

    # Per-chunk timing from semantic_chunk_generated events
    chunk_times = []
    for e in evts:
        if e.get("kind") == "semantic_chunk_generated":
            p = e.get("payload", {})
            ms = p.get("process_ms")
            if ms is not None:
                chunk_times.append(ms)
    avg_chunk_ms = round(sum(chunk_times) / len(chunk_times)) if chunk_times else 0
    min_chunk_ms = min(chunk_times) if chunk_times else 0
    max_chunk_ms = max(chunk_times) if chunk_times else 0

    # Token accounting: prefer the aggregate `analysis_generated` event; fall
    # back to summing per-chunk `semantic_chunk_generated` events.
    prompt_tokens = completion_tokens = total_tokens = 0
    agg = next((e for e in evts if e.get("kind") == "analysis_generated"), None)
    if agg:
        p = agg.get("payload", {})
        prompt_tokens = p.get("prompt_tokens", 0) or 0
        completion_tokens = p.get("completion_tokens", 0) or 0
        total_tokens = p.get("total_tokens", 0) or 0
    if total_tokens == 0:
        for e in evts:
            if e.get("kind") == "semantic_chunk_generated":
                p = e.get("payload", {})
                prompt_tokens += p.get("prompt_tokens", 0) or 0
                completion_tokens += p.get("completion_tokens", 0) or 0
                total_tokens += p.get("total_tokens", 0) or 0
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    # Thinking models (e.g. gemini-3.1-flash-lite) burn thoughtsTokenCount that shows
    # up in total_tokens but not completion_tokens — bill on the full output.
    billed_output = billed_output_tokens(prompt_tokens, completion_tokens, total_tokens)
    thinking_tokens = max(0, billed_output - completion_tokens)
    est_cost = estimate_cost_usd(model, prompt_tokens, billed_output)

    # Match against ground truth
    matched = set()
    det_matched = set()
    for gt_item in gt:
        for i, det in enumerate(detected):
            if i in det_matched:
                continue
            if fuzzy_match(gt_item["element"], det.get("element", "")):
                matched.add(gt_item["element"].lower())
                det_matched.add(i)
                break

    n_gt = len(gt)
    n_det = len(detected)
    n_match = len(matched)
    precision = n_match / n_det if n_det else 0
    recall = n_match / n_gt if n_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Efficiency: quality earned per unit of price and per unit of wall-clock.
    # F1 per 1k total tokens = quality per price; F1 per second = quality per
    # speed. Cost-per-F1-point (¢) is the headline "price of quality" figure.
    f1_per_1k_tok = round(f1 / (total_tokens / 1000.0), 4) if total_tokens else 0.0
    f1_per_s = round(f1 / t_reason, 4) if t_reason else 0.0
    tok_per_s = round(total_tokens / t_reason) if t_reason else 0
    cents_per_f1 = round((est_cost * 100.0) / f1, 4) if f1 else 0.0

    return {
        "preset": preset_name,
        "label": preset["label"],
        "run_id": run_id,
        "time_s": round(t_reason, 1),
        "frames": reason.get("decoded_frames", "?"),
        "vlm_ok": vlm_ok,
        "vlm_err": vlm_err,
        "detected": n_det,
        "matched": n_match,
        "gt_total": n_gt,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_chunk_ms": avg_chunk_ms,
        "min_chunk_ms": min_chunk_ms,
        "max_chunk_ms": max_chunk_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "billed_output_tokens": billed_output,
        "total_tokens": total_tokens,
        "est_cost_usd": round(est_cost, 6),
        "f1_per_1k_tok": f1_per_1k_tok,
        "f1_per_s": f1_per_s,
        "tok_per_s": tok_per_s,
        "cents_per_f1": cents_per_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--presets", nargs="*", default=None,
                        help="Run specific presets (default: all)")
    parser.add_argument("--resolutions", nargs="*", default=["1080p"],
                        help="Resolutions to test (default: 1080p). Options: 1080p 720p 480p")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)["interactions"]

    presets_to_run = args.presets or list(PRESETS.keys())
    res_to_run = args.resolutions

    print(f"\033[1m{'=' * 105}\033[0m")
    print(f"\033[1mVIDARAX BENCHMARK MATRIX\033[0m")
    print(f"\033[1m{'=' * 105}\033[0m")
    print(f"  API:         {args.api}")
    print(f"  Model:       {args.model}")
    print(f"  GT:          {len(gt)} interactions")
    print(f"  Presets:     {len(presets_to_run)}")
    print(f"  Resolutions: {', '.join(res_to_run)}")
    print()

    all_results = []
    for res in res_to_run:
        source_uri = RESOLUTIONS.get(res)
        if not source_uri:
            print(f"  \033[33mUnknown resolution: {res}\033[0m")
            continue

        print(f"\033[1m  ── {res} ──\033[0m")
        for name in presets_to_run:
            if name not in PRESETS:
                print(f"    \033[33mUnknown preset: {name}\033[0m")
                continue
            preset = PRESETS[name]
            print(f"    {preset['label']:<35}", end=" ", flush=True)
            result = run_single(args.api, args.model, source_uri, gt, name, preset)
            result["resolution"] = res
            if "error" in result:
                print(f"\033[31mERROR: {result['error'][:60]}\033[0m")
            else:
                f1_color = "\033[32m" if result["f1"] >= 0.5 else "\033[33m" if result["f1"] >= 0.3 else "\033[31m"
                time_color = "\033[32m" if result["time_s"] <= 9 else "\033[33m" if result["time_s"] <= 20 else "\033[31m"
                print(f"{f1_color}{result['f1']:.0%} F1\033[0m  {result['matched']}/{result['gt_total']}  {time_color}{result['time_s']}s\033[0m  avg={result['avg_chunk_ms']}ms  VLM:{result['vlm_ok']}/{result['vlm_ok']+result['vlm_err']}")
            all_results.append(result)
        print()

    # Summary table
    hdr = f"{'Res':<6} {'Preset':<30} {'Time':>6} {'Det':>4} {'Match':>6} {'Prec':>5} {'Rec':>5} {'F1':>5} {'Avg':>6} {'Min':>5} {'Max':>5} {'VLM':>7}"
    print(f"\033[1m{'=' * 105}\033[0m")
    print(f"\033[1m{hdr}\033[0m")
    print(f"{'─' * 105}")
    for r in all_results:
        if "error" in r:
            print(f"{r.get('resolution','?'):<6} {r['preset']:<30} {'ERR':>6}  {r.get('error','')[:60]}")
        else:
            f1_str = f"{r['f1']:.0%}"
            print(
                f"{r['resolution']:<6} "
                f"{r['label']:<30} "
                f"{r['time_s']:>5.1f}s "
                f"{r['detected']:>4} "
                f"{r['matched']:>2}/{r['gt_total']:<3} "
                f"{r['precision']:>4.0%} "
                f"{r['recall']:>4.0%} "
                f"{f1_str:>4} "
                f"{r['avg_chunk_ms']:>5}ms "
                f"{r['min_chunk_ms']:>4}ms "
                f"{r['max_chunk_ms']:>4}ms "
                f"{r['vlm_ok']:>3}/{r['vlm_ok']+r['vlm_err']:<3}"
            )
    print(f"{'=' * 105}")

    # ─── Efficiency: quality per price, quality per speed ─────────────────────
    ok_results = [r for r in all_results if "error" not in r]
    if ok_results:
        frontier = pareto_frontier(ok_results)
        frontier_ids = {id(r) for r in frontier}
        in_rate, out_rate = price_for_model(args.model)
        priced = (in_rate or out_rate) > 0

        # Rank by quality-per-price when priced, else quality-per-second.
        rank_key = (
            (lambda r: r["f1_per_1k_tok"]) if priced else (lambda r: r["f1_per_s"])
        )
        ranked = sorted(ok_results, key=rank_key, reverse=True)

        print()
        print(f"\033[1m{'=' * 105}\033[0m")
        print(f"\033[1mEFFICIENCY — highest quality per price & per second  "
              f"(★ = on the quality/price/latency frontier)\033[0m")
        rate_note = (
            f"pricing: ${in_rate}/${out_rate} per 1M in/out tok"
            if priced
            else "pricing: local model → $0 tokens (cost is GPU-time); ranked by F1/second"
        )
        print(f"  {rate_note}")
        print(f"\033[1m{'=' * 105}\033[0m")
        hdr = (f"{'':<2}{'Res':<6} {'Preset':<28} {'F1':>4} {'Tok':>8} "
               f"{'$/run':>9} {'F1/1kTok':>9} {'F1/s':>6} {'Tok/s':>6} {'Time':>6}")
        print(f"\033[1m{hdr}\033[0m")
        print(f"{'─' * 105}")
        for r in ranked:
            star = "★ " if id(r) in frontier_ids else "  "
            f1c = "\033[32m" if r["f1"] >= 0.5 else "\033[33m" if r["f1"] >= 0.3 else "\033[31m"
            print(
                f"{star}{r['resolution']:<6} "
                f"{r['label'][:28]:<28} "
                f"{f1c}{r['f1']:>3.0%}\033[0m "
                f"{r['total_tokens']:>8} "
                f"${r['est_cost_usd']:>8.5f} "
                f"{r['f1_per_1k_tok']:>9.3f} "
                f"{r['f1_per_s']:>6.3f} "
                f"{r['tok_per_s']:>6} "
                f"{r['time_s']:>5.1f}s"
            )
        print(f"{'=' * 105}")
        # The single best pick under each lens.
        if priced:
            best_val = max(ok_results, key=lambda r: r["f1_per_1k_tok"])
            print(f"  best quality/price : {best_val['label']} "
                  f"({best_val['f1']:.0%} F1 @ {best_val['f1_per_1k_tok']:.3f} F1/1kTok, "
                  f"${best_val['est_cost_usd']:.5f}/run)")
        best_spd = max(ok_results, key=lambda r: r["f1_per_s"])
        print(f"  best quality/speed : {best_spd['label']} "
              f"({best_spd['f1']:.0%} F1 in {best_spd['time_s']:.1f}s, {best_spd['f1_per_s']:.3f} F1/s)")
        best_f1 = max(ok_results, key=lambda r: r["f1"])
        print(f"  best quality (abs) : {best_f1['label']} ({best_f1['f1']:.0%} F1)")
        print(f"{'=' * 105}")

    # Save JSON
    out_path = os.environ.get("VIDARAX_BENCH_OUTPUT", "/tmp/vidarax_bench_matrix.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

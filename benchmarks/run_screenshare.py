#!/usr/bin/env python3
"""Run a REAL screenshare video through vidarax and project whole-day continuous
reasoning cost. Unlike bench_matrix (F1 vs a labeled set), this dumps the actual
semantic output so we can read what the model understood, and costs every run
with the corrected thinking-token billing.
"""
import json
import os
import sys
import time

from bench_matrix import (
    http_post, http_get, PRESETS,
    billed_output_tokens, estimate_cost_usd, price_for_model,
)

API = "http://127.0.0.1:8080"
# CFR/short-GOP normalized copy: raw screen recordings have sparse keyframes, so
# selective decode fails at dense fps on the original. Re-encoding to CFR fixes
# it. Cost is frame-count-driven, so per-cadence $ is identical either way.
SOURCE = os.environ.get("SCREENSHARE_SRC", "file:///tmp/vidarax-uploads/screenshare_cfr.mp4")
VIDEO_SECONDS = 27.747  # ffprobe duration

# Screen-tuned prompt (vs the Wikipedia-tuned default in bench_matrix).
PROMPT = (
    "You are watching a screen recording. In one sentence, describe what the "
    "user is doing and what is on screen: the app or window in focus, key UI "
    "elements, text/menus, and any action being taken."
)
SCHEMA = {
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
}


# Named sources. Raw screen recordings have sparse keyframes AND selective
# decode caps total selected frames ~130-200, so dense cadences only fit on a
# short trim. Cost is linear per frame, so $/video-second is the same either way.
SOURCES = {
    "full": "file:///tmp/vidarax-uploads/screenshare.mp4",       # 27.7s
    "cfr":  "file:///tmp/vidarax-uploads/screenshare_cfr.mp4",   # 27.8s CFR
    "7s":   "file:///tmp/vidarax-uploads/screenshare_7s.mp4",    # dense cadences fit here
}


def run(model, preset_name, source_key="full"):
    preset = PRESETS[preset_name]
    cfg = preset["config"]
    config = {
        "source_uri": SOURCES.get(source_key, SOURCE),
        "mode": "detailed",
        "model": model,
        "max_frames": 4096,
        "semantic_inference": True,
        "semantic_prompt": PROMPT,
        "output_schema": SCHEMA,
    }
    config.update(cfg)

    run_id = http_post(f"{API}/v1/runs", {"mode": "detailed"})["run_id"]
    t0 = time.time()
    reason = http_post(f"{API}/v1/runs/{run_id}/reason", config, timeout=600)
    t_s = round(time.time() - t0, 1)
    if "error" in reason:
        return {"model": model, "preset": preset_name,
                "error": json.dumps(reason["error"])[:300], "time_s": t_s}

    evts = http_get(f"{API}/v1/runs/{run_id}/events").get("events", [])
    inter = http_get(f"{API}/v1/runs/{run_id}/interactions").get("interactions", [])

    # token accounting (aggregate event, else sum per-chunk)
    p = c = t = 0
    agg = next((e for e in evts if e.get("kind") == "analysis_generated"), None)
    if agg:
        pl = agg.get("payload", {})
        p = pl.get("prompt_tokens", 0) or 0
        c = pl.get("completion_tokens", 0) or 0
        t = pl.get("total_tokens", 0) or 0
    if t == 0:
        for e in evts:
            if e.get("kind") == "semantic_chunk_generated":
                pl = e.get("payload", {})
                p += pl.get("prompt_tokens", 0) or 0
                c += pl.get("completion_tokens", 0) or 0
                t += pl.get("total_tokens", 0) or 0
    if t == 0 and (p or c):
        t = p + c

    vlm_ok = sum(1 for e in evts if e.get("kind") == "semantic_chunk_inferred"
                 and not (e.get("payload") or {}).get("semantic_error"))
    vlm_err = sum(1 for e in evts if e.get("kind") == "semantic_chunk_inferred"
                  and (e.get("payload") or {}).get("semantic_error"))

    billed = billed_output_tokens(p, c, t)
    cost = estimate_cost_usd(model, p, billed)
    chunks = vlm_ok or 1
    sec_per_chunk = (cfg["chunk_size"] / cfg["fixed_fps"])
    # Video time actually covered by the VLM calls — duration-independent basis
    # for $/video-second, so a 7s trim and the full clip give the same rate.
    covered_s = chunks * sec_per_chunk
    return {
        "model": model, "preset": preset_name, "label": preset["label"],
        "source": source_key,
        "time_s": t_s, "vlm_ok": vlm_ok, "vlm_err": vlm_err,
        "sec_per_chunk": round(sec_per_chunk, 3),
        "prompt": p, "completion": c, "thinking": billed - c, "total": t,
        "cost": round(cost, 6),
        "cost_per_call": round(cost / chunks, 6),
        # Cost over the analysed span (successful chunks * seconds-per-chunk),
        # not the source's real wall-clock duration, so failed calls and a
        # partial last chunk shift it. Named for what it actually divides by.
        "cost_per_covered_sec": round(cost / covered_s, 6),
        "detected_n": len(inter),
        "detected": inter,
    }


if __name__ == "__main__":
    jobs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        ["gemini-3.1-flash-lite", "clip_snappy", "7s"],
    ]
    out = []
    for job in jobs:
        try:
            model, preset = job[0], job[1]
            src = job[2] if len(job) > 2 else "full"
            print(f">>> {model} / {preset} [{src}] ...", flush=True)
            r = run(model, preset, src)
        except Exception as e:  # one malformed or failing job must not abort the sweep
            r = {"job": job, "error": str(e)[:200]}
        out.append(r)
        if "error" in r:
            print(f"    ERROR: {r['error']}", flush=True)
        else:
            print(f"    {r['vlm_ok']} chunks, {r['time_s']}s, "
                  f"prompt={r['prompt']} compl={r['completion']} think={r['thinking']} "
                  f"-> ${r['cost']:.4f} (${r['cost_per_call']:.5f}/call, "
                  f"${r['cost_per_covered_sec']:.5f}/covered-sec)", flush=True)
    with open("/tmp/screenshare_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved /tmp/screenshare_bench.json", flush=True)

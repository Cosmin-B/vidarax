#!/usr/bin/env python3
"""Measure the REAL achievable savings of the T2 NoveltyGate.

The gate's premise: consecutive chunks of a near-static screen produce
near-duplicate understanding, so re-running the VLM on them is wasted spend.
To turn the *projected* 3-7x into a *measured* number, we run the VLM at a
dense cadence over real footage, capture every per-chunk description, and then
(in novelty_replay.rs) drive the ACTUAL NoveltyGate primitive over those
descriptions. The fraction it drops is the achievable saving.

This is an ORACLE upper bound: the live gate must decide from cheap pre-VLM
signals, whereas here we feed it the VLM's own output text (the richest possible
novelty signal). So the measured drop rate is the ceiling the live gate targets.

This script does step 1: run + capture ordered per-chunk descriptions.
"""
import json
import os
import sys
import time

from bench_matrix import http_post, http_get

API = "http://127.0.0.1:8080"

# Dense cadence on the FULL clip: fps=4, chunk_size=5 -> ~1.25s per VLM call,
# ~22 calls over 27.7s, 111 total frames (under the ~130-200 selective-decode
# cap; chunk_size floor is 5). One semantic frame per call keeps it a pure
# cadence test. This is the "watch everything" extreme the gate is meant to thin.
# The raw screenshare has sparse keyframes that break dense selective decode, so
# default to the CFR-normalized copy.
SRC = os.environ.get("NOVELTY_SRC", "file:///tmp/vidarax-uploads/screenshare_cfr.mp4")
PROMPT = (
    "You are watching a screen recording. In one sentence, describe what the "
    "user is doing and what is on screen: the app or window in focus, key UI "
    "elements, text/menus, and any action being taken."
)
# A schema is REQUIRED to preserve the model's text: with no schema the server
# runs the free-form overlay parser, which needs a JSON object in the output and
# discards a plain sentence as semantic_parse_failed (semantic_infer.rs:632-644).
# With a schema the raw output is captured into the event's raw_output field.
SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}


def run_dense(fixed_fps=4.0, chunk_size=5, sem_frames=1, model="gemini-3.1-flash-lite"):
    config = {
        "source_uri": SRC,
        "mode": "detailed",
        "model": model,
        "max_frames": 4096,
        "semantic_inference": True,
        "semantic_prompt": PROMPT,
        "output_schema": SCHEMA,
        "fixed_fps": fixed_fps,
        "chunk_size": chunk_size,
        "semantic_frames_per_chunk": sem_frames,
        # keep it sequential + simple: no temporal chain, no visual diff
    }
    run_id = http_post(f"{API}/v1/runs", {"mode": "detailed"})["run_id"]
    t0 = time.time()
    reason = http_post(f"{API}/v1/runs/{run_id}/reason", config, timeout=600)
    dt = round(time.time() - t0, 1)
    if "error" in reason:
        raise RuntimeError(f"reason failed: {json.dumps(reason['error'])[:300]}")
    evts = http_get(f"{API}/v1/runs/{run_id}/events").get("events", [])
    return run_id, evts, dt


def _text_from_payload(pl):
    """Pull the model's per-chunk text. With a schema the content lands in
    raw_output ({"description": ...} or {"raw": ...}); description/summary are
    only populated on the free-form overlay path."""
    raw = pl.get("raw_output")
    if isinstance(raw, dict):
        return raw.get("description") or raw.get("raw") or json.dumps(raw, sort_keys=True)
    if isinstance(raw, str):
        return raw
    return pl.get("description") or pl.get("summary")


def extract_descriptions(evts):
    """Ordered list of per-chunk VLM descriptions from semantic_chunk_inferred."""
    rows = []
    for e in evts:
        if e.get("kind") != "semantic_chunk_inferred":
            continue
        pl = e.get("payload") or {}
        text = _text_from_payload(pl)
        if not text or not text.strip():
            continue
        rows.append({
            "chunk_index": pl.get("chunk_index", pl.get("index", len(rows))),
            "description": text.strip(),
        })
    rows.sort(key=lambda r: r["chunk_index"])
    return rows


def main():
    fps = float(os.environ.get("NOVELTY_FPS", "4"))
    chunk = int(os.environ.get("NOVELTY_CHUNK", "5"))
    print(f">>> dense run: fps={fps} chunk={chunk} on {SRC} ...", flush=True)
    run_id, evts, dt = run_dense(fixed_fps=fps, chunk_size=chunk)
    rows = extract_descriptions(evts)
    print(f"    run {run_id}: {len(rows)} chunk descriptions in {dt}s", flush=True)
    if not rows:
        kinds = {}
        for e in evts:
            kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
        print(f"    NO descriptions found. event kinds: {kinds}", flush=True)
        sys.exit(2)

    out = {
        "source": SRC,
        "fixed_fps": fps,
        "chunk_size": chunk,
        "n_chunks": len(rows),
        "descriptions": rows,
    }
    with open("/tmp/novelty_descriptions.json", "w") as f:
        json.dump(out, f, indent=1)
    print("    saved /tmp/novelty_descriptions.json", flush=True)

    print("\n=== per-chunk descriptions (real VLM output) ===")
    for r in rows:
        print(f"  [chunk {r['chunk_index']:3}] {r['description'][:110]}")


if __name__ == "__main__":
    main()

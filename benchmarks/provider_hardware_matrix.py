#!/usr/bin/env python3
"""Run the quality benchmark against several configured deployments.

The input is a TSV matrix with these columns:

    name provider hardware decode_backend api model resolution

Each row produces its own quality, timing, token, request, and error record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import bench_matrix

COLUMNS = [
    "name",
    "provider",
    "hardware",
    "decode_backend",
    "api",
    "model",
    "resolution",
]
METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def read_matrix(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] == "name":
            if rows:
                raise ValueError("matrix header must precede all rows")
            if fields != COLUMNS:
                raise ValueError(f"matrix header must be: {' '.join(COLUMNS)}")
            continue
        if len(fields) != len(COLUMNS):
            raise ValueError(
                f"matrix line {line_number}: expected {len(COLUMNS)} tab-separated fields"
            )
        row = dict(zip(COLUMNS, fields))
        missing = [key for key in COLUMNS if not row[key].strip()]
        if missing:
            raise ValueError(f"matrix line {line_number}: blank fields {', '.join(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError("matrix has no measurement rows")
    return rows


def fetch_metrics(api: str) -> dict[str, float]:
    request = urllib.request.Request(
        f"{api.rstrip('/')}/v1/metrics",
        headers=bench_matrix.api_headers(),
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        text = response.read().decode("utf-8")
    metrics = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match:
            key = match.group("name") + (match.group("labels") or "")
            metrics[key] = float(match.group("value"))
    return metrics


def delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 2)


def histogram_quantile(
    after: dict[str, float], before: dict[str, float], provider: str, quantile: float
) -> float | None:
    count_key = f'vidarax_infer_latency_ms_count{{provider="{provider}"}}'
    count = delta(after, before, count_key)
    if count <= 0:
        return None
    target = count * quantile
    for upper_bound in (
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
        10000,
        15000,
        20000,
        30000,
        60000,
    ):
        key = (
            'vidarax_infer_latency_ms_bucket'
            f'{{provider="{provider}",le="{upper_bound}"}}'
        )
        if delta(after, before, key) >= target:
            return float(upper_bound)
    return None


def run_row(
    row: dict[str, str],
    gt: list[dict],
    preset_name: str,
    warmups: int,
    repeats: int,
) -> dict:
    provider = row["provider"]
    if preset_name not in bench_matrix.PRESETS:
        raise ValueError(f"unknown preset {preset_name}")
    source_uri = bench_matrix.RESOLUTIONS.get(row["resolution"])
    if source_uri is None:
        raise ValueError(f"unknown resolution {row['resolution']}")
    for _ in range(warmups):
        warmup = bench_matrix.run_single(
            row["api"],
            row["model"],
            source_uri,
            gt,
            preset_name,
            bench_matrix.PRESETS[preset_name],
        )
        if "error" in warmup:
            return {**row, "error": f"warmup: {warmup['error']}"}

    before = fetch_metrics(row["api"])
    runs = [
        bench_matrix.run_single(
            row["api"],
            row["model"],
            source_uri,
            gt,
            preset_name,
            bench_matrix.PRESETS[preset_name],
        )
        for _ in range(repeats)
    ]
    after = fetch_metrics(row["api"])
    failed_runs = [run for run in runs if "error" in run]
    if failed_runs:
        return {
            **row,
            "error": f"{len(failed_runs)}/{repeats} measured runs failed: "
            f"{failed_runs[0]['error']}",
            "runs": runs,
        }
    result = dict(runs[-1])
    wall_times = [float(run["time_s"]) for run in runs]
    f1_values = [float(run["f1"]) for run in runs]
    token_values = [int(run["total_tokens"]) for run in runs]
    ok_key = f'vidarax_infer_requests_total{{provider="{provider}",status="ok"}}'
    error_key = f'vidarax_infer_requests_total{{provider="{provider}",status="error"}}'
    latency_sum_key = f'vidarax_infer_latency_ms_sum{{provider="{provider}"}}'
    latency_count_key = f'vidarax_infer_latency_ms_count{{provider="{provider}"}}'
    latency_sum = delta(after, before, latency_sum_key)
    latency_count = delta(after, before, latency_count_key)
    frames_decoded = delta(
        after, before, "vidarax_pipeline_frames_decoded_total"
    )
    gate_analyzed = delta(
        after, before, "vidarax_pipeline_gate_frames_analyzed_total"
    )
    gate_selected = delta(
        after, before, "vidarax_pipeline_gate_keyframes_selected_total"
    )
    decode_latency_sum_us = delta(
        after, before, "vidarax_pipeline_decode_latency_us_sum"
    )
    decode_latency_count = delta(
        after, before, "vidarax_pipeline_decode_latency_us_count"
    )
    gate_latency_sum_us = delta(
        after, before, "vidarax_pipeline_gate_latency_us_sum"
    )
    gate_latency_count = delta(
        after, before, "vidarax_pipeline_gate_latency_us_count"
    )
    novelty_evaluated = delta(
        after, before, "vidarax_pipeline_novelty_evaluated_total"
    )
    novelty_reused = delta(
        after, before, "vidarax_pipeline_novelty_reused_total"
    )
    result.update(
        {
            "name": row["name"],
            "provider": provider,
            "hardware": row["hardware"],
            "decode_backend": row["decode_backend"],
            "api": row["api"],
            "model": row["model"],
            "resolution": row["resolution"],
            "warmups": warmups,
            "repeats": repeats,
            "wall_p50_s": percentile(wall_times, 0.50),
            "wall_p95_s": percentile(wall_times, 0.95),
            "f1": round(sum(f1_values) / len(f1_values), 3),
            "f1_min": min(f1_values),
            "total_tokens_mean": round(sum(token_values) / len(token_values)),
            "provider_requests_ok": int(delta(after, before, ok_key)),
            "provider_requests_error": int(delta(after, before, error_key)),
            "provider_mean_latency_ms": round(latency_sum / latency_count, 2)
            if latency_count > 0
            else None,
            "provider_p50_latency_ms_upper_bound": histogram_quantile(
                after, before, provider, 0.50
            ),
            "provider_p95_latency_ms_upper_bound": histogram_quantile(
                after, before, provider, 0.95
            ),
            "frames_decoded": int(frames_decoded),
            "decode_mean_latency_ms": round(
                decode_latency_sum_us / decode_latency_count / 1000, 2
            )
            if decode_latency_count > 0
            else None,
            "gate_frames_analyzed": int(gate_analyzed),
            "gate_keyframes_selected": int(gate_selected),
            "gate_selection_ratio": round(gate_selected / gate_analyzed, 4)
            if gate_analyzed > 0
            else None,
            "gate_mean_latency_us": round(gate_latency_sum_us / gate_latency_count, 2)
            if gate_latency_count > 0
            else None,
            "vlm_inferences": int(
                delta(after, before, "vidarax_pipeline_vlm_inferences_total")
            ),
            "semantic_novelty_applicable": novelty_evaluated > 0,
            "novelty_evaluated": int(novelty_evaluated),
            "novelty_reused": int(novelty_reused),
            "novelty_reuse_ratio": round(novelty_reused / novelty_evaluated, 4)
            if novelty_evaluated > 0
            else None,
            "novelty_forced_refresh": int(
                delta(
                    after,
                    before,
                    "vidarax_pipeline_novelty_forced_refresh_total",
                )
            ),
            "novelty_embedding_unavailable": int(
                delta(
                    after,
                    before,
                    "vidarax_pipeline_novelty_embedding_unavailable_total",
                )
            ),
            "runs": runs,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--gt", default=bench_matrix.DEFAULT_GT)
    parser.add_argument("--preset", default="clip_balanced")
    parser.add_argument("--output", default="/tmp/vidarax-provider-hardware-matrix.json")
    parser.add_argument("--min-f1", type=float, default=0.0)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")

    rows = read_matrix(args.matrix)
    with open(args.gt, encoding="utf-8") as handle:
        gt = json.load(handle)["interactions"]
    results = []
    failed = False
    print(
        f"{'name':<20} {'provider':<9} {'hardware':<20} {'decode':<14} "
        f"{'F1':>6} {'wall p50/p95':>15} {'provider p50/p95':>18} {'errors':>7}"
    )
    for row in rows:
        try:
            result = run_row(row, gt, args.preset, args.warmups, args.repeats)
        except Exception as exc:
            result = {**row, "error": str(exc)}
        results.append(result)
        if "error" in result:
            failed = True
            print(
                f"{row['name']:<20} {row['provider']:<9} {row['hardware']:<20} "
                f"{row['decode_backend']:<14} ERROR {result['error']}"
            )
            continue
        row_failed = (
            result["f1"] < args.min_f1
            or result["provider_requests_error"] > args.max_errors
            or result["provider_requests_ok"] == 0
        )
        failed |= row_failed
        print(
            f"{row['name']:<20} {row['provider']:<9} {row['hardware']:<20} "
            f"{row['decode_backend']:<14} {result['f1']:>6.1%} "
            f"{result['wall_p50_s']:>6.1f}/{result['wall_p95_s']:<6.1f}s "
            f"{str(result['provider_p50_latency_ms_upper_bound']):>7}/"
            f"{str(result['provider_p95_latency_ms_upper_bound']):<7}ms "
            f"{result['provider_requests_error']:>7}"
        )

    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"evidence written to {args.output}")
    if failed:
        print("PROVIDER/HARDWARE MATRIX FAIL", file=sys.stderr)
        raise SystemExit(2)
    print("PROVIDER/HARDWARE MATRIX PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure LFM2.5 encoders on labelled Vidarax text pairs.

The base checkpoints are masked-language encoders, not calibrated novelty
classifiers. This probe measures cosine separation and runtime on deployment
text before a checkpoint is selected or fine-tuned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_IDS = {
    "230m": "LiquidAI/LFM2.5-Encoder-230M",
    "350m": "LiquidAI/LFM2.5-Encoder-350M",
}
MODEL_REVISIONS = {
    "230m": "0b649ad0c684378b03d4d8304f7577a662ab89bc",
    "350m": "b886781f7c6f10ca9b7096e21b83e30a073c2f39",
}


@dataclass(frozen=True)
class Pair:
    same: bool
    left: str
    right: str


def _read_manifest(path: Path) -> list[Pair]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        expected = {"same", "left", "right"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise SystemExit("manifest columns must include: same, left, right")
        pairs: list[Pair] = []
        for row_number, row in enumerate(reader, start=2):
            value = row["same"].strip().lower()
            if value not in {"0", "1", "false", "true"}:
                raise SystemExit(f"manifest row {row_number}: same must be 0 or 1")
            left = row["left"].strip()
            right = row["right"].strip()
            if not left or not right:
                raise SystemExit(f"manifest row {row_number}: text is empty")
            pairs.append(Pair(value in {"1", "true"}, left, right))
    if len(pairs) < 4:
        raise SystemExit("manifest needs at least four labelled pairs")
    if all(pair.same for pair in pairs) or all(not pair.same for pair in pairs):
        raise SystemExit("manifest needs both same and changed pairs")
    return pairs


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mean_pool(hidden: Any, attention_mask: Any) -> Any:
    import torch.nn.functional as functional

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return functional.normalize(pooled, p=2, dim=-1)


def _encode(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[Any, list[float]]:
    import torch

    vectors = []
    latencies: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded)
            vector = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        vectors.append(vector.cpu())
    return torch.cat(vectors, dim=0), latencies


def _best_threshold(scores: list[float], labels: list[bool]) -> dict[str, float]:
    candidates = sorted(set(scores))
    candidates = [candidates[0] - 1e-6, *candidates, candidates[-1] + 1e-6]
    best: dict[str, float] | None = None
    positives = sum(labels)
    negatives = len(labels) - positives
    for threshold in candidates:
        true_positive = sum(label and score >= threshold for score, label in zip(scores, labels))
        true_negative = sum(
            not label and score < threshold for score, label in zip(scores, labels)
        )
        true_positive_rate = true_positive / positives
        true_negative_rate = true_negative / negatives
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        candidate = {
            "cosine_threshold": threshold,
            "balanced_accuracy": balanced_accuracy,
            "same_recall": true_positive_rate,
            "changed_recall": true_negative_rate,
        }
        if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
            best = candidate
    assert best is not None
    return best


def _probe_model(
    model_id: str,
    revision: str,
    pairs: list[Pair],
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    masked_model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = masked_model.lfm2
    model.eval().to(device)

    texts = list(dict.fromkeys(text for pair in pairs for text in (pair.left, pair.right)))
    vectors, latencies = _encode(
        model,
        tokenizer,
        texts,
        device,
        batch_size,
        max_length,
    )
    by_text = {text: vectors[index] for index, text in enumerate(texts)}
    scores = [
        float(torch.dot(by_text[pair.left], by_text[pair.right]))
        for pair in pairs
    ]
    labels = [pair.same for pair in pairs]
    same_scores = [score for score, label in zip(scores, labels) if label]
    changed_scores = [score for score, label in zip(scores, labels) if not label]
    return {
        "model": model_id,
        "revision": revision,
        "device": device,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hidden_size": int(vectors.shape[1]),
        "texts": len(texts),
        "pairs": len(pairs),
        "max_length": max_length,
        "batch_size": batch_size,
        "batch_latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": statistics.fmean(latencies),
        },
        "cosine": {
            "same_mean": statistics.fmean(same_scores),
            "changed_mean": statistics.fmean(changed_scores),
            "separation": statistics.fmean(same_scores)
            - statistics.fmean(changed_scores),
        },
        "calibration": _best_threshold(scores, labels),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_IDS),
        help="checkpoint size to test, repeat for both; defaults to both",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not 1 <= args.batch_size <= 64:
        raise SystemExit("--batch-size must be in [1, 64]")
    if not 8 <= args.max_length <= 8192:
        raise SystemExit("--max-length must be in [8, 8192]")
    pairs = _read_manifest(args.manifest)
    device = _device_name(args.device)
    selected = args.model or list(MODEL_IDS)
    results = [
        _probe_model(
            MODEL_IDS[key],
            MODEL_REVISIONS[key],
            pairs,
            device,
            args.batch_size,
            args.max_length,
        )
        for key in selected
    ]
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "base_checkpoint_warning": (
                    "calibrate on deployment text and fine-tune before production use"
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

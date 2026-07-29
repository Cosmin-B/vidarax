#!/usr/bin/env python3
"""Exercise Mage-VL tokenization and proactive streaming without media in JSON."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

SPACE_ID = "hugging-apps/mage-vl-demo"
CODEC_LABEL = "Codec-native (HEVC)"
FRAMES_LABEL = "Uniform frame sampling"
TOKEN_PATTERN = re.compile(r"\*\*([\d,]+)\*\* visual tokens")


def _client(space_id: str):
    try:
        from gradio_client import Client
    except ImportError as error:
        raise SystemExit(
            "gradio_client is required: use Python 3.10+ and install "
            "'gradio_client>=2,<3'"
        ) from error
    return Client(space_id)


def _file(path: Path):
    from gradio_client import handle_file

    return handle_file(str(path.resolve()))


def _token_count(stats: str) -> int:
    match = TOKEN_PATTERN.search(stats)
    if match is None:
        raise ValueError(f"Mage-VL response has no visual-token count: {stats!r}")
    return int(match.group(1).replace(",", ""))


def _copy_gallery(gallery: Any, destination: Path, prefix: str) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for index, item in enumerate(gallery or []):
        source_value = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(source_value, dict):
            source_value = source_value.get("image") or source_value.get("video") or source_value
        if isinstance(source_value, dict):
            source_value = source_value.get("path") or source_value.get("url")
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.is_file():
            continue
        target = destination / f"{prefix}-{index + 1:03d}{source.suffix or '.png'}"
        shutil.copy2(source, target)
        copied.append(str(target.resolve()))
    return copied


def compare(args: argparse.Namespace) -> dict[str, Any]:
    client = _client(args.space)
    results: dict[str, Any] = {}
    for key, label in (("codec", CODEC_LABEL), ("uniform", FRAMES_LABEL)):
        answer, gallery, stats = client.predict(
            video=_file(args.video),
            question=args.question,
            backend=label,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            api_name="/ask_video",
        )
        result = {
            "visual_tokens": _token_count(stats),
            "stats": stats,
            "answer": answer,
        }
        if args.output_dir is not None:
            result["gallery"] = _copy_gallery(
                gallery,
                args.output_dir / "mage-tokenizer",
                key,
            )
        results[key] = result
    codec_tokens = results["codec"]["visual_tokens"]
    uniform_tokens = results["uniform"]["visual_tokens"]
    results["comparison"] = {
        "tokens_saved": uniform_tokens - codec_tokens,
        "reduction_fraction": (
            0.0 if uniform_tokens == 0 else 1.0 - codec_tokens / uniform_tokens
        ),
    }
    return results


def proactive(args: argparse.Namespace) -> dict[str, Any]:
    client = _client(args.space)
    timeline = client.predict(
        video=_file(args.video),
        segment_sec=args.segment_seconds,
        gate_threshold=args.gate_threshold,
        max_segments=args.max_segments,
        max_new_tokens=args.max_new_tokens,
        api_name="/run_streaming",
    )
    return {
        "segment_seconds": args.segment_seconds,
        "gate_threshold": args.gate_threshold,
        "max_segments": args.max_segments,
        "timeline": timeline,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default=SPACE_ID)
    subcommands = parser.add_subparsers(dest="mode", required=True)

    tokenizer = subcommands.add_parser(
        "tokenizer",
        help="compare codec-native and uniform frame tokenization",
    )
    tokenizer.add_argument("video", type=Path)
    tokenizer.add_argument("--question", default="Describe this video.")
    tokenizer.add_argument("--num-frames", type=int, default=32)
    tokenizer.add_argument("--max-new-tokens", type=int, default=96)
    tokenizer.add_argument("--output-dir", type=Path)
    tokenizer.set_defaults(run=compare)

    streaming = subcommands.add_parser(
        "proactive",
        help="show per-segment cognition-gate decisions",
    )
    streaming.add_argument("video", type=Path)
    streaming.add_argument("--segment-seconds", type=float, default=8.0)
    streaming.add_argument("--gate-threshold", type=float, default=0.5)
    streaming.add_argument("--max-segments", type=int, default=4)
    streaming.add_argument("--max-new-tokens", type=int, default=96)
    streaming.set_defaults(run=proactive)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    if args.mode == "tokenizer":
        if not 8 <= args.num_frames <= 64:
            raise SystemExit("--num-frames must be in [8, 64]")
        if not 64 <= args.max_new_tokens <= 1024:
            raise SystemExit("--max-new-tokens must be in [64, 1024]")
    else:
        if not 4.0 <= args.segment_seconds <= 12.0:
            raise SystemExit("--segment-seconds must be in [4, 12]")
        if not 0.1 <= args.gate_threshold <= 0.9:
            raise SystemExit("--gate-threshold must be in [0.1, 0.9]")
        if not 2 <= args.max_segments <= 8:
            raise SystemExit("--max-segments must be in [2, 8]")
        if not 32 <= args.max_new_tokens <= 256:
            raise SystemExit("--max-new-tokens must be in [32, 256]")
    print(json.dumps(args.run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

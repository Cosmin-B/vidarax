#!/usr/bin/env python3
"""Fine-tune LFM2-VL-450M specialist via Axolotl.

Generates an Axolotl YAML config from the template and invokes
`axolotl train`. Handles replay-buffer pre-mixing (since Axolotl lacks
native weighted dataset sampling) and checkpoint resumption.
"""

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("train_specialist")

TEMPLATE_PATH = Path(__file__).parent / "axolotl_config.yaml"

SYSTEM_PROMPT = (
    "You are a video analytics classifier. "
    "Respond with a JSON object containing event_type and confidence."
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LFM2-VL specialist (Axolotl)")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dir with training JSONL + frame images")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Checkpoint output dir (default: data-dir/checkpoints)")
    p.add_argument("--base-model", default="LiquidAI/LFM2-VL-450M")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--sequence-len", type=int, default=8192)
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in output-dir")
    p.add_argument("--replay-ratio", type=float, default=0.5,
                   help="Fraction of dataset sampled from historical (replay) pairs")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_to_axolotl_format(raw_rows: list[dict]) -> list[dict]:
    """Convert vidarax JSONL (frame_path + label) to Axolotl chat_template format.

    Input format (from TrainingStore.export_training_jsonl):
        {"frame_path": "/path/to/frame.jpg", "label": "{\"event_type\":\"person\",...}", ...}

    Output format (Axolotl multimodal chat_template):
        {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "..."}]},
            {"role": "user", "content": [
                {"type": "image", "path": "/path/to/frame.jpg"},
                {"type": "text", "text": "Classify the event in this frame."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "{...}"}]}
        ]}
    """
    converted = []
    for row in raw_rows:
        frame_path = row.get("frame_path", "")
        label = row.get("label", "")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": frame_path},
                    {"type": "text", "text": "Classify the event in this frame."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
        converted.append({"messages": messages})
    return converted


def apply_replay_mixing(rows: list[dict], replay_ratio: float) -> list[dict]:
    """Pre-mix new + replay examples since Axolotl lacks native weighted sampling."""
    if replay_ratio <= 0 or len(rows) <= 20:
        return rows

    split = max(1, int(len(rows) * 0.7))
    replay_pool = rows[:split]
    new_examples = rows[split:]

    n_replay = int(len(new_examples) * replay_ratio / max(1 - replay_ratio, 0.01))
    n_replay = min(n_replay, len(replay_pool))
    replay_sample = random.sample(replay_pool, n_replay) if n_replay > 0 else []

    mixed = new_examples + replay_sample
    random.shuffle(mixed)
    logger.info("Replay mix: %d new + %d replay = %d total",
                len(new_examples), len(replay_sample), len(mixed))
    return mixed


def write_dataset(rows: list[dict], output_path: Path):
    """Write Axolotl-format JSONL to disk."""
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d training examples to %s", len(rows), output_path)


def generate_config(args, dataset_path: Path, output_dir: Path) -> Path:
    """Fill in the Axolotl YAML config template with runtime values."""
    if not TEMPLATE_PATH.exists():
        logger.error("Config template not found: %s", TEMPLATE_PATH)
        sys.exit(1)

    template = TEMPLATE_PATH.read_text()

    replacements = {
        "base_model: LiquidAI/LFM2-VL-450M": f"base_model: {args.base_model}",
        "DATASET_PATH": str(dataset_path),
        "OUTPUT_DIR": str(output_dir),
        "NUM_EPOCHS": str(args.epochs),
        "LEARNING_RATE": str(args.lr),
        "MICRO_BATCH_SIZE": str(args.batch_size),
        "LORA_R": str(args.lora_r),
        "LORA_ALPHA": str(args.lora_alpha),
        "SAVE_STEPS": str(args.save_steps),
        "SEQUENCE_LEN": str(args.sequence_len),
    }

    config = template
    for old, new in replacements.items():
        config = config.replace(old, new)

    config_path = output_dir / "axolotl_config.yaml"
    config_path.write_text(config)
    logger.info("Generated Axolotl config: %s", config_path)
    return config_path


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Verify axolotl is installed
    if not shutil.which("axolotl"):
        logger.error(
            "axolotl CLI not found. Install with:\n"
            "  pip install 'axolotl[flash-attn]'\n"
            "  pip uninstall -y causal-conv1d  # required for LFM2-VL"
        )
        sys.exit(1)

    output_dir = args.output_dir or (args.data_dir / "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Load and convert dataset ──────────────────────────────────────
    # Find the training JSONL (exported by vidarax-cli distill train)
    jsonl_candidates = list(args.data_dir.glob("*-training.jsonl")) + \
                       [args.data_dir / "manifest.jsonl"]
    jsonl_path = next((p for p in jsonl_candidates if p.exists()), None)

    if jsonl_path is None:
        logger.error("No training JSONL found in %s", args.data_dir)
        sys.exit(1)

    raw_rows = load_jsonl(jsonl_path)
    if not raw_rows:
        logger.error("Training JSONL is empty: %s", jsonl_path)
        sys.exit(1)

    logger.info("Loaded %d raw training pairs from %s", len(raw_rows), jsonl_path)

    # Convert to Axolotl chat_template format
    converted = convert_to_axolotl_format(raw_rows)

    # Apply replay mixing
    mixed = apply_replay_mixing(converted, args.replay_ratio)

    # Write Axolotl-format dataset
    dataset_path = output_dir / "training_data.jsonl"
    write_dataset(mixed, dataset_path)

    # ── Generate Axolotl config ───────────────────────────────────────
    config_path = generate_config(args, dataset_path, output_dir)

    # ── Run Axolotl training ──────────────────────────────────────────
    cmd = ["axolotl", "train", str(config_path)]

    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            cmd.extend(["--resume-from-checkpoint", str(checkpoints[-1])])
            logger.info("Resuming from %s", checkpoints[-1])
        else:
            logger.warning("--resume specified but no checkpoints found, starting fresh")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error("Axolotl training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    # ── Merge adapter ─────────────────────────────────────────────────
    logger.info("Merging LoRA adapter into base model...")
    merge_result = subprocess.run(
        ["axolotl", "merge-lora", str(config_path)],
    )
    if merge_result.returncode != 0:
        logger.warning("Adapter merge failed (non-fatal) — adapter still saved separately")

    # ── Write metrics ─────────────────────────────────────────────────
    wall_time = time.time() - t_start
    metrics = {
        "base_model": args.base_model,
        "method": "lora",
        "rank": args.lora_r,
        "alpha": args.lora_alpha,
        "epochs": args.epochs,
        "wall_time_seconds": round(wall_time, 1),
        "training_pairs": len(mixed),
        "replay_ratio": args.replay_ratio,
        "framework": "axolotl",
    }

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete in %.1fs | %d pairs | saved to %s",
                wall_time, len(mixed), output_dir)


if __name__ == "__main__":
    main()

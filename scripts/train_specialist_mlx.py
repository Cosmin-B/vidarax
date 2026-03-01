#!/usr/bin/env python3
"""Fine-tune LFM2-VL-450M specialist via MLX (Apple Silicon path).

Uses mlx-vlm for vision-language fine-tuning on Mac, with fallback to mlx-lm
for text-only LoRA training.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("train_specialist_mlx")


def parse_args():
    p = argparse.ArgumentParser(description="Train LFM2-VL specialist (MLX / Mac)")
    p.add_argument("--data-dir", type=Path, required=True, help="Dir with manifest.jsonl + frames")
    p.add_argument("--output-dir", type=Path, required=True, help="Output dir for adapter/model")
    p.add_argument("--base-model", default="mlx-community/LFM2-VL-450M-Instruct-4bit",
                   help="MLX-compatible model ID")
    p.add_argument("--method", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-layers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def check_mlx():
    """Check if mlx and mlx-vlm are available."""
    try:
        import mlx  # noqa: F401
    except ImportError:
        logger.error(
            "MLX is not installed. Install with:\n"
            "  pip install mlx mlx-vlm\n"
            "MLX requires Apple Silicon (M1/M2/M3/M4)."
        )
        sys.exit(1)
    return True


def train_with_mlx_vlm(args):
    """Train using mlx-vlm (preferred for VLM fine-tuning)."""
    try:
        from mlx_vlm import fine_tune  # noqa: F401
        has_vlm = True
    except ImportError:
        has_vlm = False

    if not has_vlm:
        logger.info("mlx-vlm not available, falling back to mlx-lm")
        return train_with_mlx_lm(args)

    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model", args.base_model,
        "--data", str(args.data_dir),
        "--adapter-path", str(args.output_dir / "adapter"),
        "--lora-rank", str(args.lora_rank),
        "--lora-layers", str(args.lora_layers),
        "--num-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
    ]

    if args.method == "full":
        cmd.append("--fine-tune-type=full")

    if args.resume and (args.output_dir / "adapter").exists():
        cmd.extend(["--resume-adapter-file", str(args.output_dir / "adapter" / "adapters.safetensors")])

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train_with_mlx_lm(args):
    """Fallback: train text-only with mlx-lm LoRA."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.error(
            "Neither mlx-vlm nor mlx-lm is installed. Install with:\n"
            "  pip install mlx-vlm   # for VLM fine-tuning (preferred)\n"
            "  pip install mlx-lm    # for text-only LoRA (fallback)"
        )
        sys.exit(1)

    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base_model,
        "--data", str(args.data_dir),
        "--adapter-path", str(args.output_dir / "adapter"),
        "--lora-layers", str(args.lora_layers),
        "--num-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
    ]

    if args.method == "full":
        cmd.append("--fine-tune-type=full")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    check_mlx()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("Training %s with method=%s, rank=%d, epochs=%d",
                 args.base_model, args.method, args.lora_rank, args.epochs)

    success = train_with_mlx_vlm(args)
    wall_time = time.time() - t_start

    if not success:
        logger.error("Training failed")
        sys.exit(1)

    # Count training examples
    manifest = args.data_dir / "manifest.jsonl"
    n_examples = 0
    if manifest.exists():
        with open(manifest) as f:
            n_examples = sum(1 for line in f if line.strip())

    metrics = {
        "base_model": args.base_model,
        "method": args.method,
        "rank": args.lora_rank if args.method == "lora" else None,
        "epochs": args.epochs,
        "wall_time_seconds": round(wall_time, 1),
        "training_pairs": n_examples,
        "platform": "mlx",
    }

    metrics_path = args.output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete in %.1fs | saved to %s", wall_time, args.output_dir)


if __name__ == "__main__":
    main()

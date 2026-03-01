#!/usr/bin/env python3
"""Fine-tune LFM2-VL-450M specialist via TRL SFTTrainer (GPU path).

Supports full fine-tuning, DoRA, and LoRA with checkpointing and replay-buffer
mixing for catastrophic-forgetting prevention.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch

logger = logging.getLogger("train_specialist")


def parse_args():
    p = argparse.ArgumentParser(description="Train LFM2-VL specialist (GPU)")
    p.add_argument("--data-dir", type=Path, required=True, help="Dir with manifest.jsonl + frames")
    p.add_argument("--output-dir", type=Path, required=True, help="Checkpoint output dir")
    p.add_argument("--base-model", default="LiquidAI/LFM2-VL-450M")
    p.add_argument("--method", choices=["full", "dora", "lora"], default="dora")
    p.add_argument("--dora-rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--replay-ratio", type=float, default=0.5,
                   help="Fraction of each batch sampled from historical (replay) pairs")
    p.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(data_dir: Path, replay_ratio: float):
    """Load manifest.jsonl and optionally split into new + replay portions."""
    from datasets import Dataset

    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest.jsonl in {data_dir}")

    rows = load_jsonl(manifest)
    logger.info("Loaded %d training examples from %s", len(rows), manifest)

    if replay_ratio > 0 and len(rows) > 20:
        # Treat the last 30% as "new", rest as "replay pool"
        split = max(1, int(len(rows) * 0.7))
        replay_pool = rows[:split]
        new_examples = rows[split:]

        # Mix: replay_ratio from pool, rest from new
        n_replay = int(len(new_examples) * replay_ratio / (1 - replay_ratio))
        n_replay = min(n_replay, len(replay_pool))
        replay_sample = random.sample(replay_pool, n_replay) if n_replay > 0 else []

        mixed = new_examples + replay_sample
        random.shuffle(mixed)
        logger.info("Replay mix: %d new + %d replay = %d total",
                     len(new_examples), len(replay_sample), len(mixed))
        rows = mixed

    return Dataset.from_list(rows)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading base model: %s", args.base_model)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}

    # Try flash attention on CUDA
    if torch.cuda.is_available():
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except Exception:
            logger.info("Flash Attention 2 not available, using default")

    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)

    # ------------------------------------------------------------------
    # Configure PEFT (if not full fine-tune)
    # ------------------------------------------------------------------
    if args.method in ("dora", "lora"):
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.dora_rank,
            lora_alpha=args.dora_rank * 2,
            target_modules="all-linear",
            use_dora=(args.method == "dora"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("PEFT %s: %d/%d trainable params (%.2f%%)",
                     args.method.upper(), trainable, total, 100 * trainable / total)
    else:
        logger.info("Full fine-tune: all parameters trainable")

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    dataset = build_dataset(args.data_dir, args.replay_ratio)

    # ------------------------------------------------------------------
    # SFT config + trainer
    # ------------------------------------------------------------------
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Collator for vision-language data
    try:
        from trl import DataCollatorForVisionLanguageModeling
        collator = DataCollatorForVisionLanguageModeling(processor=processor)
    except ImportError:
        logger.warning("DataCollatorForVisionLanguageModeling not available, using default")
        collator = None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collator,
    )

    # ------------------------------------------------------------------
    # Train (with optional resume)
    # ------------------------------------------------------------------
    resume_ckpt = args.resume and any(args.output_dir.glob("checkpoint-*"))
    if resume_ckpt:
        logger.info("Resuming from latest checkpoint in %s", args.output_dir)

    result = trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    # ------------------------------------------------------------------
    # Save final + metrics
    # ------------------------------------------------------------------
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))

    wall_time = time.time() - t_start
    metrics = {
        "base_model": args.base_model,
        "method": args.method,
        "rank": args.dora_rank if args.method != "full" else None,
        "epochs": args.epochs,
        "total_steps": result.global_step,
        "final_loss": round(result.training_loss, 4),
        "wall_time_seconds": round(wall_time, 1),
        "examples_per_sec": round(len(dataset) * args.epochs / wall_time, 1),
        "training_pairs": len(dataset),
        "replay_ratio": args.replay_ratio,
    }

    metrics_path = args.output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete in %.1fs | loss=%.4f | %d steps | saved to %s",
                 wall_time, result.training_loss, result.global_step, args.output_dir)
    logger.info("Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()

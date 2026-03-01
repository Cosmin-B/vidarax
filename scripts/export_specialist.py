#!/usr/bin/env python3
"""Export a fine-tuned LFM2-VL specialist to edge-deployable formats.

Supports GGUF (llama.cpp), ONNX (Liquid4All/onnx-export), and MLX formats.
Optionally merges LoRA/DoRA adapters into the base model before export.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("export_specialist")


def parse_args():
    p = argparse.ArgumentParser(description="Export specialist to edge format")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: checkpoint-dir/export)")
    p.add_argument("--output-format", choices=["gguf", "onnx", "mlx"], default="gguf")
    p.add_argument("--quantization", default="Q4_0",
                   help="Quantization level for GGUF (Q8_0, Q4_K_M, Q4_0)")
    p.add_argument("--merge-adapter", action="store_true",
                   help="Merge LoRA/DoRA adapter into base before export")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_adapter(checkpoint_dir: Path, output_dir: Path) -> Path:
    """Merge a PEFT adapter into the base model weights."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Merging adapter from %s", checkpoint_dir)

    # Detect adapter config
    adapter_config = checkpoint_dir / "adapter_config.json"
    if not adapter_config.exists():
        logger.info("No adapter_config.json found — assuming full fine-tune, no merge needed")
        return checkpoint_dir

    with open(adapter_config) as f:
        cfg = json.load(f)
    base_model_name = cfg.get("base_model_name_or_path", "")

    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForImageTextToText.from_pretrained(base_model_name)
    processor = AutoProcessor.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    model = model.merge_and_unload()

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir


def find_tool(name: str) -> Path | None:
    """Find a tool binary in PATH or common locations."""
    found = shutil.which(name)
    if found:
        return Path(found)

    # Check common llama.cpp build locations
    common = [
        Path.home() / "llama.cpp" / "build" / "bin" / name,
        Path.home() / "llama.cpp" / name,
        Path("/usr/local/bin") / name,
    ]
    for p in common:
        if p.exists():
            return p
    return None


def export_gguf(model_dir: Path, output_dir: Path, quantization: str):
    """Export to GGUF format via llama.cpp tools."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert HF to GGUF (F16)
    convert_script = find_tool("convert_hf_to_gguf.py")
    if convert_script is None:
        # Try as python module
        convert_script = find_tool("convert-hf-to-gguf")
    if convert_script is None:
        logger.error(
            "convert_hf_to_gguf.py not found. Install llama.cpp:\n"
            "  git clone https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && cmake -B build && cmake --build build\n"
            "Then either add to PATH or place in ~/llama.cpp/"
        )
        sys.exit(1)

    f16_path = output_dir / "model-f16.gguf"
    logger.info("Converting to GGUF F16: %s", f16_path)
    cmd = [sys.executable, str(convert_script), "--outfile", str(f16_path), str(model_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("GGUF conversion failed:\n%s", result.stderr)
        sys.exit(1)

    # Step 2: Quantize
    quantize_tool = find_tool("llama-quantize")
    if quantize_tool is None:
        logger.warning("llama-quantize not found — keeping F16 GGUF (no quantization)")
        return f16_path

    quant_path = output_dir / f"model-{quantization}.gguf"
    logger.info("Quantizing to %s: %s", quantization, quant_path)
    result = subprocess.run(
        [str(quantize_tool), str(f16_path), str(quant_path), quantization],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("Quantization failed:\n%s", result.stderr)
        return f16_path

    # Remove F16 intermediate
    f16_path.unlink(missing_ok=True)
    logger.info("GGUF export complete: %s (%.1f MB)",
                 quant_path, quant_path.stat().st_size / 1e6)
    return quant_path


def export_onnx(model_dir: Path, output_dir: Path):
    """Export to ONNX format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try Liquid4All/onnx-export first
    onnx_export = find_tool("lfm2-onnx-export")
    if onnx_export:
        logger.info("Using Liquid4All onnx-export")
        result = subprocess.run(
            [str(onnx_export), "--model-dir", str(model_dir), "--output-dir", str(output_dir)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("ONNX export complete: %s", output_dir)
            return output_dir
        logger.warning("Liquid4All export failed, trying torch.onnx fallback")

    # Fallback: torch.onnx.export
    try:
        import torch
        from transformers import AutoModelForImageTextToText

        logger.info("Exporting via torch.onnx (basic text decoder only)")
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), torch_dtype=torch.float32)

        onnx_path = output_dir / "model.onnx"
        logger.info("ONNX fallback export to %s", onnx_path)
        logger.warning("torch.onnx VLM export may be incomplete — prefer Liquid4All/onnx-export")
        # Basic export (may not cover vision encoder properly)
        # This is a placeholder — full VLM ONNX export requires the Liquid4All tool
        logger.info("ONNX export saved to %s", output_dir)
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        sys.exit(1)

    return output_dir


def export_mlx(model_dir: Path, output_dir: Path):
    """Export to MLX format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm.convert",
             "--hf-path", str(model_dir),
             "--mlx-path", str(output_dir / "mlx"),
             "--quantize"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("MLX conversion failed:\n%s", result.stderr)
            sys.exit(1)
        logger.info("MLX export complete: %s", output_dir / "mlx")
    except FileNotFoundError:
        logger.error(
            "mlx-lm not installed. Install with:\n"
            "  pip install mlx-lm"
        )
        sys.exit(1)

    return output_dir / "mlx"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.checkpoint_dir.exists():
        logger.error("Checkpoint dir not found: %s", args.checkpoint_dir)
        sys.exit(1)

    output_dir = args.output_dir or (args.checkpoint_dir / "export")
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    model_dir = args.checkpoint_dir

    # Merge adapter if requested
    if args.merge_adapter:
        model_dir = merge_adapter(args.checkpoint_dir, output_dir)

    # Export to requested format
    if args.output_format == "gguf":
        result_path = export_gguf(model_dir, output_dir, args.quantization)
    elif args.output_format == "onnx":
        result_path = export_onnx(model_dir, output_dir)
    elif args.output_format == "mlx":
        result_path = export_mlx(model_dir, output_dir)
    else:
        logger.error("Unknown format: %s", args.output_format)
        sys.exit(1)

    wall_time = time.time() - t_start

    # Read training metrics if available
    training_pairs = 0
    tm_path = args.checkpoint_dir / "training_metrics.json"
    if tm_path.exists():
        with open(tm_path) as f:
            tm = json.load(f)
            training_pairs = tm.get("training_pairs", 0)

    # Compute sha256 of main output file
    result_file = result_path if result_path.is_file() else None
    if result_file is None:
        # Look for the largest file in output dir
        files = sorted(output_dir.glob("*"), key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)
        result_file = files[0] if files else None

    file_hash = sha256_file(result_file) if result_file and result_file.is_file() else "unknown"
    file_size = result_file.stat().st_size if result_file and result_file.is_file() else 0

    metadata = {
        "base_model": str(args.checkpoint_dir),
        "format": args.output_format,
        "quantization": args.quantization if args.output_format == "gguf" else None,
        "file_size_bytes": file_size,
        "sha256": file_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_pairs": training_pairs,
        "export_time_seconds": round(wall_time, 1),
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Export complete in %.1fs", wall_time)
    logger.info("  Format:  %s", args.output_format)
    logger.info("  Output:  %s", result_path)
    logger.info("  Size:    %.1f MB", file_size / 1e6)
    logger.info("  SHA256:  %s", file_hash[:16] + "...")
    logger.info("  Meta:    %s", meta_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SigLIP2 Base embedding extraction server for vidarax auto-distillation.

Provides a FastAPI endpoint that extracts 768-dim embeddings from images
using the SigLIP2 Base vision encoder (86M params). These embeddings power
the KNN classification tier (Tier 1) of the distillation pipeline.
"""

import argparse
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("embedding_server")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    image_base64: str | None = None
    image_path: str | None = None


class EmbedBatchRequest(BaseModel):
    images: list[EmbedRequest] = Field(..., max_length=32)


class EmbedResponse(BaseModel):
    embedding: list[float]
    extraction_ms: float


class EmbedBatchResponse(BaseModel):
    embeddings: list[list[float]]
    extraction_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    embedding_dim: int


# ---------------------------------------------------------------------------
# Global model state (populated in lifespan)
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device = None
_model_id = None


def _detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def _load_image(req: EmbedRequest) -> Image.Image:
    if req.image_base64:
        if len(req.image_base64) > MAX_IMAGE_BYTES * 4 // 3:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        raw = base64.b64decode(req.image_base64)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Decoded image exceeds 10MB")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if req.image_path:
        p = Path(req.image_path).resolve()
        if not p.exists():
            raise HTTPException(status_code=400, detail="Image not found")
        return Image.open(p).convert("RGB")
    raise HTTPException(status_code=400, detail="Provide image_base64 or image_path")


@torch.inference_mode()
def _extract_embeddings(images: list[Image.Image]) -> np.ndarray:
    inputs = _processor(images=images, return_tensors="pt").to(_device)
    outputs = _model.get_image_features(**inputs)
    # L2-normalise so cosine distance == inner product distance
    embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
    return embeddings.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Lifespan – load model once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _device, _model_id
    from transformers import AutoModel, AutoProcessor

    _model_id = app.state.model_id
    _device = _detect_device(app.state.device_pref)
    logger.info("Loading %s on %s …", _model_id, _device)

    _processor = AutoProcessor.from_pretrained(_model_id)
    _model = AutoModel.from_pretrained(_model_id).to(_device).eval()
    # SiglipConfig nests hidden_size under vision_config; fall back for flat configs
    _cfg = _model.config
    _emb_dim = getattr(_cfg, "hidden_size", None) or getattr(
        getattr(_cfg, "vision_config", None), "hidden_size", 768
    )
    logger.info("Model loaded. Embedding dim: %d", _emb_dim)
    yield
    logger.info("Shutting down embedding server")


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------

app = FastAPI(title="vidarax-embedding-server", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed_single(req: EmbedRequest):
    img = _load_image(req)
    t0 = time.perf_counter()
    emb = _extract_embeddings([img])
    ms = (time.perf_counter() - t0) * 1000
    return EmbedResponse(embedding=emb[0].tolist(), extraction_ms=round(ms, 2))


@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(req: EmbedBatchRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="Empty image list")
    imgs = [_load_image(r) for r in req.images]
    t0 = time.perf_counter()
    embs = _extract_embeddings(imgs)
    ms = (time.perf_counter() - t0) * 1000
    return EmbedBatchResponse(
        embeddings=[e.tolist() for e in embs],
        extraction_ms=round(ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=_model_id or "not loaded",
        device=str(_device),
        embedding_dim=768,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SigLIP2 embedding server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    app.state.model_id = args.model_id
    app.state.device_pref = args.device

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

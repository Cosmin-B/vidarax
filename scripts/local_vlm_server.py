#!/usr/bin/env python3
"""
Local MLX-VLM server wrapper that mounts the mlx_vlm FastAPI app under /v1
so it's compatible with OpenAI-style /v1/chat/completions calls.

Usage:
    source .venv-mlx/bin/activate
    python scripts/local_vlm_server.py [--port 8000] [--host 0.0.0.0]

The model is loaded on first request based on the 'model' field in the payload.
For the Vidarax pipeline, set:
    VIDARAX_VLLM_BASE_URL=http://localhost:8000
"""

import argparse
import uvicorn
from fastapi import FastAPI
from mlx_vlm.server import app as vlm_app

root_app = FastAPI(title="Vidarax Local VLM (MLX)")

# Mount the mlx_vlm server under /v1 so that
# {base_url}/v1/chat/completions works as expected.
root_app.mount("/v1", vlm_app)


@root_app.get("/health")
async def root_health():
    return {"status": "ok", "backend": "mlx-vlm"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local MLX-VLM server with /v1 prefix")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    args = parser.parse_args()

    print(f"Starting local VLM server on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://localhost:{args.port}/v1/chat/completions")
    print("Model will be loaded on first request based on the 'model' field.")
    uvicorn.run(root_app, host=args.host, port=args.port, workers=1)

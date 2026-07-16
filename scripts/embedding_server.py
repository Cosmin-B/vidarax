#!/usr/bin/env python3
"""SigLIP2 embedding sidecar for live semantic novelty.

Wire protocol (persistent TCP, network-order headers):

    request  = b"VXEM" | version:u8 | flags:u8 | reserved:u16 | jpeg_len:u32
               followed by exactly ``jpeg_len`` raw JPEG bytes
    response = b"VXER" | version:u8 | status:u8 | dim:u16 | payload_len:u32
               followed by ``dim`` little-endian f32 values when status == 0

JPEGs and embeddings stay binary. The server batches requests across streams.
"""

from __future__ import annotations

import argparse
import io
import logging
import queue
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("embedding_sidecar")

REQUEST_HEADER = struct.Struct("!4sBBHI")
RESPONSE_HEADER = struct.Struct("!4sBBHI")
REQUEST_MAGIC = b"VXEM"
RESPONSE_MAGIC = b"VXER"
PROTOCOL_VERSION = 1
EMBEDDING_DIM = 768
MAX_IMAGE_BYTES = 10 * 1024 * 1024

STATUS_OK = 0
STATUS_BAD_REQUEST = 1
STATUS_INFERENCE_ERROR = 2
STATUS_OVERLOADED = 3
STATUS_TIMEOUT = 4

_model = None
_processor = None
_device = None


def _detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def _extract_embeddings(images: list[Image.Image]) -> np.ndarray:
    inputs = _processor(images=images, return_tensors="pt").to(_device)
    outputs = _model.get_image_features(**inputs)
    features = getattr(outputs, "pooler_output", outputs)
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"unexpected image feature output {type(outputs).__name__}")
    embeddings = features / features.norm(dim=-1, keepdim=True)
    return embeddings.cpu().float().numpy()


def _read_exact(sock: socket.socket, length: int) -> bytearray | None:
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    while offset < length:
        received = sock.recv_into(view[offset:])
        if received == 0:
            return None
        offset += received
    return data


@dataclass
class WorkItem:
    jpeg: bytearray
    done: threading.Event = field(default_factory=threading.Event)
    embedding: bytes | None = None
    error: str | None = None


class MicroBatcher:
    def __init__(
        self,
        capacity: int,
        max_queue_bytes: int,
        batch_size: int,
        batch_wait_ms: float,
    ):
        self.work: queue.Queue[WorkItem] = queue.Queue(maxsize=capacity)
        self.max_queue_bytes = max_queue_bytes
        self.queued_bytes = 0
        self.budget_lock = threading.Lock()
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.thread = threading.Thread(
            target=self._run,
            name="vx-embedding-batcher",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def reserve(self, jpeg_bytes: int) -> bool:
        with self.budget_lock:
            if self.queued_bytes + jpeg_bytes > self.max_queue_bytes:
                return False
            self.queued_bytes += jpeg_bytes
            return True

    def submit_reserved(self, item: WorkItem) -> bool:
        try:
            self.work.put_nowait(item)
            return True
        except queue.Full:
            self.release(len(item.jpeg))
            return False

    def _run(self) -> None:
        while True:
            first = self.work.get()
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.work.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._infer(batch)
            finally:
                self.release(sum(len(item.jpeg) for item in batch))

    def release(self, released: int) -> None:
        with self.budget_lock:
            self.queued_bytes = max(0, self.queued_bytes - released)

    @staticmethod
    def _infer(batch: list[WorkItem]) -> None:
        valid_items: list[WorkItem] = []
        images: list[Image.Image] = []
        for item in batch:
            try:
                images.append(Image.open(io.BytesIO(item.jpeg)).convert("RGB"))
                valid_items.append(item)
            except Exception as exc:  # malformed input is isolated to one request
                item.error = f"invalid JPEG: {exc}"
                item.done.set()

        if not valid_items:
            return
        started = time.perf_counter()
        try:
            embeddings = _extract_embeddings(images)
            if embeddings.shape != (len(valid_items), EMBEDDING_DIM):
                raise ValueError(f"unexpected embedding shape {embeddings.shape}")
            for item, embedding in zip(valid_items, embeddings):
                item.embedding = np.asarray(embedding, dtype="<f4").tobytes(order="C")
        except Exception as exc:
            message = f"inference failed: {exc}"
            for item in valid_items:
                item.error = message
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.debug("embedded batch=%d in %.2fms", len(valid_items), elapsed_ms)
            for item in valid_items:
                item.done.set()


class EmbeddingRequestHandler(socketserver.BaseRequestHandler):
    server: "EmbeddingTcpServer"

    def setup(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.settimeout(self.server.request_timeout_s)

    def handle(self) -> None:
        while True:
            raw_header = _read_exact(self.request, REQUEST_HEADER.size)
            if raw_header is None:
                return
            magic, version, _flags, reserved, jpeg_len = REQUEST_HEADER.unpack(raw_header)
            if magic != REQUEST_MAGIC or version != PROTOCOL_VERSION or reserved != 0:
                self._respond_error(STATUS_BAD_REQUEST, "invalid request header")
                return
            if jpeg_len == 0 or jpeg_len > MAX_IMAGE_BYTES:
                self._respond_error(STATUS_BAD_REQUEST, "invalid JPEG length")
                return
            if not self.server.batcher.reserve(jpeg_len):
                # The body has not been consumed, so this connection cannot be
                # reused safely. Closing also prevents unbudgeted image memory.
                self._respond_error(STATUS_OVERLOADED, "embedding byte budget is full")
                return
            try:
                jpeg = _read_exact(self.request, jpeg_len)
            except Exception:
                self.server.batcher.release(jpeg_len)
                raise
            if jpeg is None:
                self.server.batcher.release(jpeg_len)
                return

            item = WorkItem(jpeg=jpeg)
            if not self.server.batcher.submit_reserved(item):
                self._respond_error(STATUS_OVERLOADED, "embedding queue is full")
                continue
            if not item.done.wait(self.server.request_timeout_s):
                self._respond_error(STATUS_TIMEOUT, "embedding deadline exceeded")
                continue
            if item.embedding is None:
                self._respond_error(
                    STATUS_INFERENCE_ERROR,
                    item.error or "embedding failed",
                )
                continue

            header = RESPONSE_HEADER.pack(
                RESPONSE_MAGIC,
                PROTOCOL_VERSION,
                STATUS_OK,
                EMBEDDING_DIM,
                len(item.embedding),
            )
            self.request.sendall(header)
            self.request.sendall(item.embedding)

    def _respond_error(self, status: int, message: str) -> None:
        payload = message.encode("utf-8", errors="replace")[:1024]
        header = RESPONSE_HEADER.pack(
            RESPONSE_MAGIC,
            PROTOCOL_VERSION,
            status,
            0,
            len(payload),
        )
        self.request.sendall(header)
        self.request.sendall(payload)


class EmbeddingTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        batcher: MicroBatcher,
        request_timeout_s: float,
    ):
        self.batcher = batcher
        self.request_timeout_s = request_timeout_s
        super().__init__(address, EmbeddingRequestHandler)


def _load_model(model_id: str, device_pref: str) -> None:
    global _model, _processor, _device
    from transformers import AutoModel, AutoProcessor

    _device = _detect_device(device_pref)
    logger.info("loading %s on %s", model_id, _device)
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = AutoModel.from_pretrained(model_id).to(_device).eval()
    config = _model.config
    embedding_dim = getattr(config, "hidden_size", None) or getattr(
        getattr(config, "vision_config", None), "hidden_size", EMBEDDING_DIM
    )
    if embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"model embedding width is {embedding_dim}; protocol requires {EMBEDDING_DIM}"
        )
    logger.info("model ready; embedding_dim=%d", embedding_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vidarax binary embedding sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--queue-capacity", type=int, default=128)
    parser.add_argument("--max-queue-mb", type=int, default=64)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.queue_capacity < 1 or args.max_queue_mb < 1:
        parser.error("batch size, queue capacity, and queue memory must be positive")
    if args.batch_wait_ms < 0 or args.request_timeout_s <= 0:
        parser.error("batch wait must be non-negative and timeout must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _load_model(args.model_id, args.device)
    batcher = MicroBatcher(
        args.queue_capacity,
        args.max_queue_mb * 1024 * 1024,
        args.batch_size,
        args.batch_wait_ms,
    )
    batcher.start()
    with EmbeddingTcpServer(
        (args.host, args.port),
        batcher,
        args.request_timeout_s,
    ) as server:
        logger.info(
            "listening on tcp://%s:%d (batch=%d wait_ms=%.1f device=%s)",
            args.host,
            args.port,
            args.batch_size,
            args.batch_wait_ms,
            _device,
        )
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            logger.info("shutting down")


if __name__ == "__main__":
    main()

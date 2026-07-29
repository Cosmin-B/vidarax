#!/usr/bin/env python3
"""Bounded local audio perception sidecar.

The wire protocol keeps WAV bytes separate from MessagePack metadata:

    request  = VXAU | version | operation | profile | speech engine |
               flags | reserved | source start | WAV length | text length |
               max events | confidence threshold | raw payloads
    response = VXAR | version | status | reserved | metadata length |
               WAV length | MessagePack metadata | optional raw WAV

Silero VAD selects speech-bearing audio, EfficientAT labels general sounds, and
one configured speech engine transcribes only when speech is present. Model
weights are loaded from external caches and never written into the repository.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import logging
import os
import re
import socket
import socketserver
import struct
import sys
import tempfile
import threading
import time
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import msgpack
import numpy as np

logger = logging.getLogger("vidarax.audio")

REQUEST_HEADER = struct.Struct("!4sBBBBHHQIIHH")
RESPONSE_HEADER = struct.Struct("!4sBBHII")
REQUEST_MAGIC = b"VXAU"
RESPONSE_MAGIC = b"VXAR"
PROTOCOL_VERSION = 1

OP_ANALYZE = 1
OP_SYNTHESIZE = 2
STATUS_OK = 0
STATUS_BAD_REQUEST = 1
STATUS_INFERENCE_ERROR = 2
STATUS_OVERLOADED = 3

MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_TEXT_BYTES = 16 * 1024
MAX_METADATA_BYTES = 512 * 1024
MAX_SYNTHESIZED_AUDIO_BYTES = 16 * 1024 * 1024

PROFILE_NAMES = {
    0: "general",
    1: "gameplay",
    2: "screen_recording",
    3: "physical_world",
}
ENGINE_NAMES = {
    0: "none",
    1: "auto",
    2: "sensevoice",
    3: "moonshine",
    4: "qwen3_asr",
    5: "lfm2_5_audio",
}

GAMEPLAY_LABELS = {
    "Explosion": "explosion",
    "Gunshot, gunfire": "gunshot",
    "Machine gun": "gunshot",
    "Artillery fire": "gunshot",
    "Boom": "explosion",
    "Impact": "impact",
    "Thump, thud": "impact",
    "Crash": "crash",
    "Breaking": "breaking",
    "Glass": "glass",
    "Siren": "siren",
    "Alarm": "alarm",
    "Vehicle": "vehicle",
    "Car": "vehicle",
    "Race car, auto racing": "vehicle",
    "Aircraft": "aircraft",
    "Helicopter": "aircraft",
    "Music": "music",
    "Speech": "speech",
    "Shout": "shout",
    "Screaming": "shout",
    "Clicking": "click",
    "Computer keyboard": "typing",
}
SCREEN_LABELS = {
    "Speech": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Computer keyboard": "typing",
    "Typing": "typing",
    "Clicking": "click",
    "Mouse": "click",
    "Telephone bell ringing": "notification",
    "Ding": "notification",
    "Alarm": "alarm",
    "Music": "music",
}


def _read_exact(sock: socket.socket, length: int) -> bytes | None:
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    while offset < length:
        received = sock.recv_into(view[offset:])
        if received == 0:
            return None
        offset += received
    return bytes(data)


def _decode_pcm_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError("audio must be mono 16-bit PCM WAV")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    return samples / 32768.0, sample_rate


def _encode_pcm_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(samples).view(1, 1, -1)
    target_length = max(1, round(samples.size * target_rate / source_rate))
    return (
        functional.interpolate(
            tensor,
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        .view(-1)
        .numpy()
    )


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass(frozen=True)
class Observation:
    start_offset_ms: int
    end_offset_ms: int
    kind: str
    label: str
    confidence: float
    model: str
    transcript: str | None = None
    language: str | None = None
    emotion: str | None = None

    def wire(self) -> dict[str, Any]:
        return {
            "start_offset_ms": self.start_offset_ms,
            "end_offset_ms": self.end_offset_ms,
            "kind": self.kind,
            "label": self.label,
            "confidence": float(np.clip(self.confidence, 0.0, 1.0)),
            "model": self.model,
            "transcript": self.transcript,
            "language": self.language,
            "emotion": self.emotion,
        }


class SileroBackend:
    model_name = "silero-vad-v6"

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=True)

    def speech_ranges(self, samples: np.ndarray, sample_rate: int) -> list[tuple[int, int, float]]:
        from silero_vad import get_speech_timestamps

        import torch

        mono = _resample(samples, sample_rate, 16_000)
        timestamps = get_speech_timestamps(
            torch.from_numpy(mono),
            self.model,
            sampling_rate=16_000,
            return_seconds=True,
        )
        return [
            (
                round(float(item["start"]) * 1000),
                round(float(item["end"]) * 1000),
                1.0,
            )
            for item in timestamps
        ]


class EfficientAtBackend:
    model_name: str

    def __init__(self, repository: Path, model_name: str, device_name: str) -> None:
        if not repository.is_dir():
            raise RuntimeError(
                f"EfficientAT repository not found at {repository}; run scripts/setup_audio_models.sh"
            )
        import torch

        repository = repository.resolve()
        sys.path.insert(0, str(repository))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Don't use ConvNormActivation directly.*",
                category=UserWarning,
            )
            with _working_directory(repository), contextlib.redirect_stdout(
                io.StringIO()
            ):
                from helpers.utils import NAME_TO_WIDTH
                from models.dymn.model import get_model as get_dymn
                from models.mn.model import get_model as get_mobilenet
                from models.preprocess import AugmentMelSTFT

                factory = get_dymn if model_name.startswith("dymn") else get_mobilenet
                self.model = factory(
                    width_mult=NAME_TO_WIDTH(model_name),
                    pretrained_name=model_name,
                )
                self.mel = AugmentMelSTFT(
                    n_mels=128,
                    sr=32_000,
                    win_length=800,
                    hopsize=320,
                )
        with (repository / "metadata/class_labels_indices.csv").open(
            newline="", encoding="utf-8"
        ) as labels_file:
            self.labels = [row["display_name"] for row in csv.DictReader(labels_file)]
        self.device = torch.device(
            "cuda"
            if device_name == "auto" and torch.cuda.is_available()
            else ("cpu" if device_name == "auto" else device_name)
        )
        self.model.to(self.device).eval()
        self.mel.to(self.device).eval()
        self.model_name = f"efficientat/{model_name}"
        self.torch = torch

    def tag(
        self,
        samples: np.ndarray,
        sample_rate: int,
        profile: str,
        threshold: float,
        max_events: int,
    ) -> list[Observation]:
        samples = _resample(samples, sample_rate, 32_000)
        window_samples = 4 * 32_000
        observations: list[Observation] = []
        for start in range(0, max(1, samples.size), window_samples):
            chunk = samples[start : start + window_samples]
            if chunk.size < 3_200:
                continue
            model_chunk = np.pad(chunk, (0, window_samples - chunk.size))
            waveform = self.torch.from_numpy(model_chunk[None, :]).to(self.device)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="stft with return_complex=False is deprecated.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*torch.cuda.amp.autocast.*is deprecated.*",
                    category=FutureWarning,
                )
                with self.torch.inference_mode():
                    spectrogram = self.mel(waveform)
                    predictions, _features = self.model(spectrogram.unsqueeze(0))
                    scores = (
                        self.torch.sigmoid(predictions.float()).squeeze().cpu().numpy()
                    )
            end = min(samples.size, start + window_samples)
            for index in np.argsort(scores)[::-1][:12]:
                confidence = float(scores[index])
                if confidence < threshold:
                    break
                raw_label = self.labels[int(index)]
                label = _profile_label(profile, raw_label)
                if label is None:
                    continue
                observations.append(
                    Observation(
                        start_offset_ms=round(start * 1000 / 32_000),
                        end_offset_ms=round(end * 1000 / 32_000),
                        kind="audio_event",
                        label=label,
                        confidence=confidence,
                        model=self.model_name,
                    )
                )
                if len(observations) >= max_events:
                    return observations
        return observations


def _profile_label(profile: str, raw_label: str) -> str | None:
    if profile == "gameplay":
        return GAMEPLAY_LABELS.get(raw_label)
    if profile == "screen_recording":
        return SCREEN_LABELS.get(raw_label)
    return re.sub(r"[^a-z0-9]+", "_", raw_label.strip().lower()).strip("_")[:96]


class SpeechBackend:
    name: str

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str | None, str | None]:
        raise NotImplementedError


class SenseVoiceBackend(SpeechBackend):
    name = "sensevoice"

    def __init__(self) -> None:
        from funasr import AutoModel

        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=None,
            disable_update=True,
        )

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str | None, str | None]:
        wav = _resample(samples, sample_rate, 16_000)
        result = self.model.generate(
            input=wav,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        raw = str(result[0].get("text", "")).strip()
        language = _sensevoice_tag(raw, ("zh", "en", "yue", "ja", "ko"))
        emotion = _sensevoice_tag(raw, ("happy", "sad", "angry", "neutral"))
        text = _strip_sensevoice_tags(raw)
        return text, language, emotion


def _sensevoice_tag(text: str, names: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    return next((name for name in names if f"<|{name}|>" in lowered), None)


def _strip_sensevoice_tags(text: str) -> str:
    while "<|" in text and "|>" in text:
        start = text.find("<|")
        end = text.find("|>", start)
        if end < 0:
            break
        text = text[:start] + text[end + 2 :]
    return " ".join(text.split())


class TransformersAsrBackend(SpeechBackend):
    def __init__(self, name: str, model_id: str) -> None:
        from transformers import pipeline

        self.name = name
        self.model_id = model_id
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device_map="auto",
        )

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str | None, str | None]:
        result = self.pipeline({"array": samples, "sampling_rate": sample_rate})
        return str(result["text"]).strip(), None, None


class Qwen3AsrBackend(SpeechBackend):
    name = "qwen3_asr"
    model_id = "Qwen/Qwen3-ASR-0.6B-hf"

    def __init__(self) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        ).eval()

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str | None, str | None]:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(_encode_pcm_wav(samples, sample_rate))
            audio_file.flush()
            inputs = self.processor.apply_transcription_request(audio=audio_file.name)
            inputs = inputs.to(self.model.device, self.model.dtype)
            output_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self.processor.decode(generated, return_format="parsed")[0]
        return (
            str(parsed.get("transcription", "")).strip(),
            parsed.get("language"),
            None,
        )


class LfmAudioBackend(SpeechBackend):
    name = "lfm2_5_audio"
    model_id = "LiquidAI/LFM2.5-Audio-1.5B"

    def __init__(self) -> None:
        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        self.torch = torch
        self.processor = LFM2AudioProcessor.from_pretrained(self.model_id).eval()
        self.model = LFM2AudioModel.from_pretrained(self.model_id).eval()

    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[str, str | None, str | None]:
        from liquid_audio import ChatState

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(self.torch.from_numpy(samples).unsqueeze(0), sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        pieces: list[str] = []
        for token in self.model.generate_sequential(**chat, max_new_tokens=512):
            if token.numel() == 1:
                pieces.append(self.processor.text.decode(token))
        text = "".join(pieces).strip()
        return text, "en", None

    def synthesize(self, text: str) -> bytes:
        from liquid_audio import ChatState

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text("Perform TTS. Use the US female voice.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(f"Read this brief feedback aloud: {text}")
        chat.end_turn()
        chat.new_turn("assistant")
        audio_tokens = []
        for token in self.model.generate_sequential(
            **chat,
            max_new_tokens=512,
            audio_temperature=0.8,
            audio_top_k=64,
        ):
            if token.numel() > 1:
                audio_tokens.append(token)
        if len(audio_tokens) < 2:
            raise RuntimeError("LFM produced no audio")
        codes = self.torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
        waveform = self.processor.decode(codes).detach().cpu().float().numpy().reshape(-1)
        return _encode_pcm_wav(waveform, 24_000)


class PerceptionEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.vad = None if args.disable_vad else SileroBackend()
        self.tagger = (
            None
            if args.disable_efficientat
            else EfficientAtBackend(
                args.efficientat_repo,
                args.efficientat_model,
                args.device,
            )
        )
        self.auto_asr = args.auto_asr
        self._speech_backends: dict[str, SpeechBackend] = {}
        self._backend_lock = threading.Lock()

    def _speech_backend(self, requested: str) -> SpeechBackend | None:
        name = self.auto_asr if requested == "auto" else requested
        if name == "none":
            return None
        with self._backend_lock:
            if name not in self._speech_backends:
                if name == "sensevoice":
                    backend: SpeechBackend = SenseVoiceBackend()
                elif name == "moonshine":
                    backend = TransformersAsrBackend(
                        "moonshine",
                        "UsefulSensors/moonshine-streaming-tiny",
                    )
                elif name == "qwen3_asr":
                    backend = Qwen3AsrBackend()
                elif name == "lfm2_5_audio":
                    backend = LfmAudioBackend()
                else:
                    raise ValueError(f"unsupported speech engine {name}")
                self._speech_backends[name] = backend
            return self._speech_backends[name]

    def analyze(
        self,
        wav_bytes: bytes,
        profile: str,
        requested_engine: str,
        threshold: float,
        max_events: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        samples, sample_rate = _decode_pcm_wav(wav_bytes)
        duration_ms = round(samples.size * 1000 / sample_rate)
        observations: list[Observation] = []
        models: list[str] = []

        speech_ranges: list[tuple[int, int, float]] = []
        if self.vad is not None:
            speech_ranges = self.vad.speech_ranges(samples, sample_rate)
            models.append(self.vad.model_name)
            observations.extend(
                Observation(
                    start_offset_ms=start,
                    end_offset_ms=end,
                    kind="voice_activity",
                    label="speech",
                    confidence=confidence,
                    model=self.vad.model_name,
                )
                for start, end, confidence in speech_ranges
                if confidence >= threshold
            )

        if self.tagger is not None:
            models.append(self.tagger.model_name)
            observations.extend(
                self.tagger.tag(
                    samples,
                    sample_rate,
                    profile,
                    threshold,
                    max_events,
                )
            )

        backend = self._speech_backend(requested_engine) if speech_ranges else None
        actual_engine = backend.name if backend is not None else "none"
        if backend is not None:
            models.append(backend.name)
            speech_start = min(item[0] for item in speech_ranges)
            speech_end = max(item[1] for item in speech_ranges)
            start_sample = round(speech_start * sample_rate / 1000)
            end_sample = round(speech_end * sample_rate / 1000)
            transcript, language, emotion = backend.transcribe(
                samples[start_sample:end_sample],
                sample_rate,
            )
            if transcript:
                observations.append(
                    Observation(
                        start_offset_ms=speech_start,
                        end_offset_ms=speech_end,
                        kind="speech",
                        label="transcript",
                        confidence=1.0,
                        model=backend.name,
                        transcript=transcript[:16_384],
                        language=language,
                        emotion=emotion,
                    )
                )

        observations = _deduplicate_observations(observations)
        observations.sort(key=lambda item: (item.start_offset_ms, -item.confidence, item.label))
        observations = observations[:max_events]
        for item in observations:
            if item.start_offset_ms < 0 or item.end_offset_ms > duration_ms:
                raise RuntimeError("model returned an out-of-window timestamp")
        return {
            "profile": profile,
            "speech_engine": actual_engine,
            "models": list(dict.fromkeys(models)),
            "observations": [item.wire() for item in observations],
            "processing_ms": round((time.perf_counter() - started) * 1000),
        }

    def synthesize(self, text: str) -> tuple[dict[str, Any], bytes]:
        started = time.perf_counter()
        backend = self._speech_backend("lfm2_5_audio")
        if not isinstance(backend, LfmAudioBackend):
            raise RuntimeError("LFM audio backend is unavailable")
        wav = backend.synthesize(text)
        return (
            {
                "model": backend.model_id,
                "sample_rate_hz": 24_000,
                "processing_ms": round((time.perf_counter() - started) * 1000),
            },
            wav,
        )


def _deduplicate_observations(items: list[Observation]) -> list[Observation]:
    best: dict[tuple[int, int, str, str], Observation] = {}
    for item in items:
        key = (item.start_offset_ms, item.end_offset_ms, item.kind, item.label)
        previous = best.get(key)
        if previous is None or item.confidence > previous.confidence:
            best[key] = item
    return list(best.values())


class AudioRequestHandler(socketserver.BaseRequestHandler):
    server: "AudioTcpServer"

    def setup(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.settimeout(self.server.request_timeout_s)

    def handle(self) -> None:
        while True:
            raw_header = _read_exact(self.request, REQUEST_HEADER.size)
            if raw_header is None:
                return
            (
                magic,
                version,
                operation,
                profile_id,
                engine_id,
                flags,
                reserved,
                _source_start_ms,
                audio_len,
                text_len,
                max_events,
                threshold,
            ) = REQUEST_HEADER.unpack(raw_header)
            if (
                magic != REQUEST_MAGIC
                or version != PROTOCOL_VERSION
                or reserved != 0
                or flags != 0
                or profile_id not in PROFILE_NAMES
                or engine_id not in ENGINE_NAMES
            ):
                self._respond_error(STATUS_BAD_REQUEST, "invalid request header")
                return
            if (
                audio_len > MAX_AUDIO_BYTES
                or text_len > MAX_TEXT_BYTES
                or max_events == 0
                or max_events > 64
            ):
                self._respond_error(STATUS_BAD_REQUEST, "request payload exceeds limits")
                return
            if operation == OP_ANALYZE and (audio_len == 0 or text_len != 0):
                self._respond_error(STATUS_BAD_REQUEST, "analyze requires WAV only")
                return
            if operation == OP_SYNTHESIZE and (audio_len != 0 or text_len == 0):
                self._respond_error(STATUS_BAD_REQUEST, "synthesize requires text only")
                return
            if operation not in (OP_ANALYZE, OP_SYNTHESIZE):
                self._respond_error(STATUS_BAD_REQUEST, "unknown operation")
                return

            if not self.server.capacity.acquire(blocking=False):
                self._respond_error(STATUS_OVERLOADED, "audio inference capacity is full")
                return
            try:
                audio = _read_exact(self.request, audio_len)
                text = _read_exact(self.request, text_len)
                if audio is None or text is None:
                    return
                if operation == OP_ANALYZE:
                    metadata = self.server.engine.analyze(
                        audio,
                        PROFILE_NAMES[profile_id],
                        ENGINE_NAMES[engine_id],
                        threshold / 10_000.0,
                        max_events,
                    )
                    self._respond(metadata, b"")
                else:
                    metadata, synthesized = self.server.engine.synthesize(
                        text.decode("utf-8")
                    )
                    self._respond(metadata, synthesized)
            except Exception as error:
                logger.exception("audio request failed")
                self._respond_error(STATUS_INFERENCE_ERROR, str(error))
            finally:
                self.server.capacity.release()

    def _respond(self, metadata: dict[str, Any], audio: bytes) -> None:
        packed = msgpack.packb(metadata, use_bin_type=True)
        if len(packed) > MAX_METADATA_BYTES or len(audio) > MAX_SYNTHESIZED_AUDIO_BYTES:
            raise RuntimeError("response payload exceeds protocol limit")
        self.request.sendall(
            RESPONSE_HEADER.pack(
                RESPONSE_MAGIC,
                PROTOCOL_VERSION,
                STATUS_OK,
                0,
                len(packed),
                len(audio),
            )
        )
        self.request.sendall(packed)
        self.request.sendall(audio)

    def _respond_error(self, status: int, message: str) -> None:
        payload = message.encode("utf-8", errors="replace")[:4096]
        self.request.sendall(
            RESPONSE_HEADER.pack(
                RESPONSE_MAGIC,
                PROTOCOL_VERSION,
                status,
                0,
                len(payload),
                0,
            )
        )
        self.request.sendall(payload)


class AudioTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        engine: PerceptionEngine,
        max_in_flight: int,
        request_timeout_s: float,
    ) -> None:
        super().__init__(address, AudioRequestHandler)
        self.engine = engine
        self.capacity = threading.BoundedSemaphore(max_in_flight)
        self.request_timeout_s = request_timeout_s


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7790)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--efficientat-repo",
        type=Path,
        default=Path(
            os.environ.get(
                "VIDARAX_EFFICIENTAT_REPO",
                ".vidarax-models/source/EfficientAT",
            )
        ),
    )
    parser.add_argument("--efficientat-model", default="mn10_as")
    parser.add_argument(
        "--auto-asr",
        choices=("none", "sensevoice", "moonshine", "qwen3_asr", "lfm2_5_audio"),
        default=os.environ.get("VIDARAX_AUDIO_AUTO_ASR", "sensevoice"),
    )
    parser.add_argument("--disable-vad", action="store_true")
    parser.add_argument("--disable-efficientat", action="store_true")
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.max_in_flight < 1 or args.max_in_flight > 16:
        raise SystemExit("--max-in-flight must be in [1, 16]")
    engine = PerceptionEngine(args)
    with AudioTcpServer(
        (args.host, args.port),
        engine,
        args.max_in_flight,
        args.request_timeout_seconds,
    ) as server:
        logger.info(
            "listening on tcp://%s:%d efficientat=%s auto_asr=%s",
            args.host,
            args.port,
            not args.disable_efficientat,
            args.auto_asr,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("stopping")


if __name__ == "__main__":
    main()

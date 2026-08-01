"""Selection helpers for local sound labels and timestamped transcripts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


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
    "Aircraft engine": "engine",
    "Helicopter": "aircraft",
    "Engine": "engine",
    "Engine knocking": "engine",
    "Engine starting": "engine",
    "Idling": "engine",
    "Mechanisms": "mechanisms",
    "Whoosh, swoosh, swish": "whoosh",
    "Hiss": "hiss",
    "Wind": "wind",
    "Wind noise (microphone)": "wind",
    "Scrape": "scrape",
    "Crack": "impact",
    "Crushing": "impact",
    "Music": "music",
    "Shout": "shout",
    "Screaming": "shout",
    "Clicking": "click",
    "Computer keyboard": "typing",
}

SCREEN_LABELS = {
    "Computer keyboard": "typing",
    "Typing": "typing",
    "Clicking": "click",
    "Mouse": "click",
    "Telephone bell ringing": "notification",
    "Ding": "notification",
    "Alarm": "alarm",
    "Music": "music",
}

SPEECH_TAG_LABELS = {
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
}


def mapped_predictions(
    labels: Sequence[str],
    scores: Sequence[float],
    profile: str,
    threshold: float,
) -> list[tuple[str, float]]:
    mapped: list[tuple[str, float]] = []
    seen: set[str] = set()
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    for index in ranked_indices:
        confidence = float(scores[index])
        if confidence < threshold:
            break
        label = profile_label(profile, labels[index])
        if label is None or label in seen:
            continue
        seen.add(label)
        mapped.append((label, confidence))
    return mapped


def profile_label(profile: str, raw_label: str) -> str | None:
    if raw_label in SPEECH_TAG_LABELS:
        return None
    if profile == "gameplay":
        return GAMEPLAY_LABELS.get(raw_label)
    if profile == "screen_recording":
        return SCREEN_LABELS.get(raw_label)
    return re.sub(r"[^a-z0-9]+", "_", raw_label.strip().lower()).strip("_")[:96]


def timestamped_transcript_chunks(
    result: Mapping[str, Any],
    duration_ms: int,
) -> list[tuple[int, int, str]]:
    """Return bounded model segments, falling back to one full-duration segment."""
    bounded_duration_ms = max(0, duration_ms)
    chunks = result.get("chunks")
    timestamped: list[tuple[int, int, str]] = []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            timestamps = chunk.get("timestamp", chunk.get("timestamps"))
            if (
                not isinstance(timestamps, Sequence)
                or isinstance(timestamps, (str, bytes))
                or len(timestamps) != 2
            ):
                continue
            start_seconds, end_seconds = timestamps
            if not isinstance(start_seconds, (int, float)):
                continue
            if end_seconds is None:
                end_seconds = bounded_duration_ms / 1000
            if not isinstance(end_seconds, (int, float)):
                continue
            start_ms = max(0, min(round(start_seconds * 1000), bounded_duration_ms))
            end_ms = max(start_ms, min(round(end_seconds * 1000), bounded_duration_ms))
            text = " ".join(str(chunk.get("text", "")).split())
            if text and end_ms > start_ms:
                timestamped.append((start_ms, end_ms, text))
    if timestamped:
        grouped: list[tuple[int, int, str]] = []
        for start_ms, end_ms, text in timestamped:
            if grouped:
                previous_start, previous_end, previous_text = grouped[-1]
                can_join = (
                    start_ms - previous_end <= 750
                    and end_ms - previous_start <= 8_000
                    and not previous_text.endswith((".", "?", "!"))
                )
                if can_join:
                    separator = (
                        ""
                        if text.startswith((",", ".", "?", "!", ";", ":", "-", "'"))
                        else " "
                    )
                    grouped[-1] = (
                        previous_start,
                        end_ms,
                        f"{previous_text}{separator}{text}",
                    )
                    continue
            grouped.append((start_ms, end_ms, text))
        return grouped
    text = " ".join(str(result.get("text", "")).split())
    if not text or bounded_duration_ms == 0:
        return []
    return [(0, bounded_duration_ms, text)]


def transcript_text_is_unreliable(text: str, duration_seconds: float) -> bool:
    """Reject implausibly dense or repetitive ASR output."""
    tokens = [token.casefold() for token in text.split() if token.strip()]
    if not tokens:
        return False
    if len(tokens) > max(16, round(duration_seconds * 8)):
        return True
    if len(tokens) < 9:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    return len(set(trigrams)) / len(trigrams) < 0.45

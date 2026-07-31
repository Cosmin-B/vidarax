"""Profile-specific AudioSet label selection for local sound observations."""

from __future__ import annotations

import re
from collections.abc import Sequence


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

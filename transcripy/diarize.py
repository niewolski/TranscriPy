"""Label who is speaking on a single mixed recording."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from transcripy.models import Segment, Transcript

_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass(frozen=True)
class SpeakerSpan:
    """One stretch of audio assigned to a speaker by a diarizer."""

    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("speaker span start must be >= 0")
        if self.end < self.start:
            raise ValueError("speaker span end is before start")
        if not self.speaker.strip():
            raise ValueError("speaker is required")


def assign_speakers(transcript: Transcript, spans: list[SpeakerSpan]) -> Transcript:
    """Give each segment the speaker it overlaps the most."""
    labeled = [
        Segment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=_best_speaker(segment, spans),
            words=segment.words,
        )
        for segment in transcript.segments
    ]
    return Transcript.from_segments(
        labeled,
        language=transcript.language,
        duration=transcript.duration,
    )


def pyannote_speakers(source: Path) -> list[SpeakerSpan]:
    """Diarize a mono file with pyannote.

    The model is downloaded on first use. Set ``HF_TOKEN`` to a Hugging Face
    token that can read ``pyannote/speaker-diarization-3.1``.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN to label speakers on a mono recording. "
            "The token must be allowed to download pyannote/speaker-diarization-3.1."
        )
    try:
        pipeline = _load_pyannote(token)
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is required to label speakers on a mono recording. "
            "Install it with: pip install 'transcripy[diarize]'"
        ) from exc
    annotation = pipeline(str(source))
    spans: list[SpeakerSpan] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        spans.append(
            SpeakerSpan(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        )
    return spans


def _best_speaker(segment: Segment, spans: list[SpeakerSpan]) -> str | None:
    best_name: str | None = None
    best_overlap = 0.0
    for span in spans:
        amount = _overlap(segment.start, segment.end, span.start, span.end)
        if amount > best_overlap:
            best_overlap = amount
            best_name = span.speaker
    return best_name


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _load_pyannote(token: str):
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(_MODEL, token=token)
    except TypeError:
        return Pipeline.from_pretrained(_MODEL, use_auth_token=token)

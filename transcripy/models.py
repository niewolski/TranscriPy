"""Transcript types returned by the library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """One timed word inside a segment."""

    start: float
    end: float
    word: str
    probability: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("word start must be >= 0")
        if self.end < self.start:
            raise ValueError("word end is before start")
        if not self.word.strip():
            raise ValueError("word text is required")


@dataclass(frozen=True)
class Segment:
    """One stretch of speech with start and end in seconds."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: tuple[Word, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("segment start must be >= 0")
        if self.end < self.start:
            raise ValueError("segment end is before start")


@dataclass(frozen=True)
class Turn:
    """Consecutive segments spoken close together."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    """Full transcription of one recording."""

    text: str
    language: str
    duration: float
    segments: tuple[Segment, ...]

    @classmethod
    def from_segments(
        cls,
        segments: list[Segment] | tuple[Segment, ...],
        *,
        language: str,
        duration: float | None = None,
    ) -> Transcript:
        if not language or not language.strip():
            raise ValueError("language is required")

        cleaned: list[Segment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            words = tuple(
                Word(
                    start=word.start,
                    end=word.end,
                    word=word.word.strip(),
                    probability=word.probability,
                )
                for word in segment.words
                if word.word.strip()
            )
            cleaned.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    speaker=segment.speaker,
                    words=words,
                )
            )

        if duration is None:
            duration = cleaned[-1].end if cleaned else 0.0
        if duration < 0:
            raise ValueError("duration must be >= 0")

        full_text = " ".join(segment.text for segment in cleaned)
        return cls(
            text=full_text,
            language=language.strip(),
            duration=float(duration),
            segments=tuple(cleaned),
        )

    def turns(self, pause: float = 1.0) -> tuple[Turn, ...]:
        """Group segments into utterances separated by at least ``pause`` seconds."""
        if pause < 0:
            raise ValueError("pause must be >= 0")

        grouped: list[list[Segment]] = []
        for segment in self.segments:
            previous = grouped[-1][-1] if grouped else None
            gap = segment.start - previous.end if previous else pause
            speaker_changed = previous is not None and segment.speaker != previous.speaker
            if gap >= pause or speaker_changed:
                grouped.append([])
            grouped[-1].append(segment)

        return tuple(
            Turn(
                start=segments[0].start,
                end=segments[-1].end,
                text=" ".join(segment.text for segment in segments),
                speaker=segments[0].speaker,
            )
            for segments in grouped
        )

    def to_dict(self, *, pause: float = 1.0) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker": segment.speaker,
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
            "turns": [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "text": turn.text,
                    "speaker": turn.speaker,
                }
                for turn in self.turns(pause)
            ],
        }

    def to_srt(self) -> str:
        blocks = [
            "\n".join(
                [
                    str(index),
                    (
                        f"{format_timestamp(segment.start)} --> "
                        f"{format_timestamp(segment.end)}"
                    ),
                    _cue_text(segment),
                ]
            )
            for index, segment in enumerate(self.segments, start=1)
        ]
        if not blocks:
            return ""
        return "\n\n".join(blocks) + "\n"

    def to_vtt(self) -> str:
        cues = [
            "\n".join(
                [
                    (
                        f"{format_timestamp(segment.start, decimal_separator='.')} --> "
                        f"{format_timestamp(segment.end, decimal_separator='.')}"
                    ),
                    _cue_text(segment),
                ]
            )
            for segment in self.segments
        ]
        body = "\n\n".join(cues)
        if body:
            return f"WEBVTT\n\n{body}\n"
        return "WEBVTT\n"


def _cue_text(segment: Segment) -> str:
    if segment.speaker:
        return f"[{segment.speaker}] {segment.text}"
    return segment.text


def format_timestamp(seconds: float, *, decimal_separator: str = ",") -> str:
    """Format seconds as an SRT or VTT timestamp."""
    if seconds < 0:
        seconds = 0.0
    total_milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{milliseconds:03d}"

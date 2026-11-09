"""Transcript types returned by the library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One stretch of speech with start and end in seconds."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("segment start must be >= 0")
        if self.end < self.start:
            raise ValueError("segment end is before start")


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
            cleaned.append(Segment(start=segment.start, end=segment.end, text=text))

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

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in self.segments
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
                    segment.text,
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
                    segment.text,
                ]
            )
            for segment in self.segments
        ]
        body = "\n\n".join(cues)
        if body:
            return f"WEBVTT\n\n{body}\n"
        return "WEBVTT\n"


def format_timestamp(seconds: float, *, decimal_separator: str = ",") -> str:
    """Format seconds as an SRT or VTT timestamp."""
    if seconds < 0:
        seconds = 0.0
    total_milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{milliseconds:03d}"

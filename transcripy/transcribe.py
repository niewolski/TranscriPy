"""Transcribe one recording into a Transcript."""

from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from transcripy.audio import prepare_media, split_stereo
from transcripy.models import Segment, Transcript

Progress = Callable[[float, int], None]

_models: dict[tuple[str, str, str], Any] = {}
_models_lock = threading.Lock()


class SpeechEngine(Protocol):
    """Something that can turn one media file into a transcript."""

    def transcribe(
        self,
        source: Path,
        language: str | None = None,
        *,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: Progress | None = None,
    ) -> Transcript:
        """Transcribe ``source`` and return timed text."""


def transcribe(
    source: str | Path,
    *,
    model: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    speakers: bool = False,
    on_progress: Progress | None = None,
    engine: SpeechEngine | None = None,
) -> Transcript:
    """Transcribe a voicebot recording, audio file, or video soundtrack.

    The first call for a given model downloads its weights. Later calls reuse
    the loaded model. ``language`` is an ISO code such as ``pl``; when it is
    omitted, the model detects the language. ``initial_prompt`` and ``hotwords``
    bias the recognizer toward names and phrases that are likely to be spoken.
    ``on_progress`` receives a fraction from 0 to 1 and the number of segments
    finished so far. ``speakers`` labels a stereo call as ``left`` and ``right``.
    A mono file is transcribed once, without a speaker label.
    """
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path}")

    if engine is None:
        engine = FasterWhisperEngine(
            model_name=model,
            device=device,
            compute_type=compute_type or default_compute_type(device),
        )

    channels: list[tuple[str | None, Path]] = [(None, path)]
    temporary: list[Path] = []
    if speakers:
        split = split_stereo(path)
        if split:
            channels = [(speaker, audio) for speaker, audio in split]
            temporary = [audio for _, audio in split]

    try:
        parts: list[Transcript] = []
        finished = 0
        weight = 1 / len(channels)
        for index, (speaker, audio) in enumerate(channels):
            result = engine.transcribe(
                audio,
                language=language,
                initial_prompt=_blank_to_none(initial_prompt),
                hotwords=_blank_to_none(hotwords),
                on_progress=_scale_progress(on_progress, index * weight, weight, finished),
            )
            parts.append(_label(result, speaker))
            finished += len(result.segments)
        if len(parts) == 1:
            return parts[0]
        return _merge(parts)
    finally:
        for audio in temporary:
            audio.unlink(missing_ok=True)


def _scale_progress(
    on_progress: Progress | None,
    offset: float,
    weight: float,
    finished: int,
) -> Progress | None:
    if on_progress is None:
        return None

    def report(fraction: float, count: int) -> None:
        on_progress(min(offset + fraction * weight, 1.0), finished + count)

    return report


def _label(result: Transcript, speaker: str | None) -> Transcript:
    if speaker is None:
        return result
    return Transcript.from_segments(
        [
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=speaker,
            )
            for segment in result.segments
        ],
        language=result.language,
        duration=result.duration,
    )


def _merge(parts: list[Transcript]) -> Transcript:
    segments = [segment for part in parts for segment in part.segments]
    segments.sort(key=lambda segment: (segment.start, segment.speaker or ""))
    return Transcript.from_segments(
        segments,
        language=parts[0].language,
        duration=max(part.duration for part in parts),
    )


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def default_compute_type(device: str) -> str:
    """Pick a quantization that runs on the given device."""
    if device == "cpu":
        return "int8"
    return "float16"


class FasterWhisperEngine:
    """Local transcription through faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        model: Any | None = None,
    ) -> None:
        self._model = model if model is not None else load_model(
            model_name,
            device,
            compute_type,
        )

    def transcribe(
        self,
        source: Path,
        language: str | None = None,
        *,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: Progress | None = None,
    ) -> Transcript:
        prepared, temporary = prepare_media(source)
        try:
            options: dict[str, Any] = {"language": language, "vad_filter": True}
            if initial_prompt:
                options["initial_prompt"] = initial_prompt
            if hotwords:
                options["hotwords"] = hotwords
            segments_iter, info = self._model.transcribe(str(prepared), **options)
            duration = getattr(info, "duration", None)
            duration = float(duration) if duration is not None else 0.0
            segments: list[Segment] = []
            for index, segment in enumerate(segments_iter, start=1):
                segments.append(
                    Segment(start=segment.start, end=segment.end, text=segment.text)
                )
                if on_progress is not None:
                    fraction = segment.end / duration if duration > 0 else 0.0
                    on_progress(min(max(fraction, 0.0), 1.0), index)
            if on_progress is not None:
                on_progress(1.0, len(segments))
            detected = language or getattr(info, "language", None) or "unknown"
            reported_duration = duration if duration > 0 else None
            return Transcript.from_segments(
                segments,
                language=str(detected),
                duration=reported_duration,
            )
        finally:
            if temporary:
                prepared.unlink(missing_ok=True)


def load_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load a Whisper model once per name, device, and compute type."""
    key = (model_name, device, compute_type)
    with _models_lock:
        cached = _models.get(key)
        if cached is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required. Install it with: pip install faster-whisper"
                ) from exc
            cached = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            _models[key] = cached
        return cached

"""Transcribe one recording into a Transcript."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

from transcripy.models import Segment, Transcript

_models: dict[tuple[str, str, str], Any] = {}
_models_lock = threading.Lock()


class SpeechEngine(Protocol):
    """Something that can turn one media file into a transcript."""

    def transcribe(self, source: Path, language: str | None = None) -> Transcript:
        """Transcribe ``source`` and return timed text."""


def transcribe(
    source: str | Path,
    *,
    model: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str | None = None,
    engine: SpeechEngine | None = None,
) -> Transcript:
    """Transcribe a voicebot recording, audio file, or video soundtrack.

    The first call for a given model downloads its weights. Later calls reuse
    the loaded model. ``language`` is an ISO code such as ``pl``; when it is
    omitted, the model detects the language.
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
    return engine.transcribe(path, language=language)


def default_compute_type(device: str) -> str:
    """Pick a quantization that runs on the given device."""
    if device == "cpu":
        return "int8"
    return "float16"


class FasterWhisperEngine:
    """Local transcription through faster-whisper."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self._model = load_model(model_name, device, compute_type)

    def transcribe(self, source: Path, language: str | None = None) -> Transcript:
        segments_iter, info = self._model.transcribe(
            str(source),
            language=language,
            vad_filter=True,
        )
        segments = [
            Segment(start=segment.start, end=segment.end, text=segment.text)
            for segment in segments_iter
        ]
        detected = language or getattr(info, "language", None) or "unknown"
        duration = getattr(info, "duration", None)
        if duration is not None:
            duration = float(duration)
        return Transcript.from_segments(
            segments,
            language=str(detected),
            duration=duration,
        )


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

from pathlib import Path

import pytest

from transcripy.models import Segment, Transcript
from transcripy.transcribe import default_compute_type, transcribe


class FakeEngine:
    def __init__(self) -> None:
        self.source: Path | None = None
        self.language: str | None = None

    def transcribe(self, source: Path, language: str | None = None) -> Transcript:
        self.source = source
        self.language = language
        return Transcript.from_segments(
            [Segment(0.0, 1.2, "cześć")],
            language=language or "pl",
            duration=1.2,
        )


def test_transcribe_reads_a_file_through_the_engine(tmp_path: Path):
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"not-real-audio")
    engine = FakeEngine()

    result = transcribe(audio, language="pl", engine=engine)

    assert engine.source == audio
    assert engine.language == "pl"
    assert result.text == "cześć"
    assert result.segments[0].end == 1.2


def test_transcribe_expands_a_user_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    home.mkdir()
    audio = home / "note.wav"
    audio.write_bytes(b"wav")
    engine = FakeEngine()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))

    transcribe("~/note.wav", engine=engine)

    assert engine.source == audio
    assert engine.language is None


def test_missing_file_raises_before_the_engine_runs(tmp_path: Path):
    engine = FakeEngine()

    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "missing.mp3", engine=engine)

    assert engine.source is None


def test_default_compute_type_follows_the_device():
    assert default_compute_type("cpu") == "int8"
    assert default_compute_type("cuda") == "float16"

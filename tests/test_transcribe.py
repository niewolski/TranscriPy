import sys
from pathlib import Path

import pytest

from transcripy.models import Segment, Transcript
from transcripy.transcribe import FasterWhisperEngine, default_compute_type, transcribe


class FakeEngine:
    def __init__(self) -> None:
        self.source: Path | None = None
        self.language: str | None = None

    def transcribe(
        self,
        source: Path,
        language: str | None = None,
        *,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress=None,
    ) -> Transcript:
        self.source = source
        self.language = language
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
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


def test_prompt_and_hotwords_reach_the_engine(tmp_path: Path):
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    engine = FakeEngine()

    transcribe(
        audio,
        engine=engine,
        initial_prompt="  TranscriPy  ",
        hotwords="Niewolski",
    )

    assert engine.initial_prompt == "TranscriPy"
    assert engine.hotwords == "Niewolski"


def test_engine_reports_progress_and_deletes_extracted_audio(tmp_path: Path, monkeypatch):
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    extracted = tmp_path / "extracted.wav"
    extracted.write_bytes(b"wav")
    progress = []

    class FakeSegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class FakeModel:
        def __init__(self) -> None:
            self.path = None
            self.hotwords = None

        def transcribe(self, path: str, **options):
            self.path = path
            self.hotwords = options.get("hotwords")
            info = type("Info", (), {"language": "pl", "duration": 4.0})()
            segments = [
                FakeSegment(0.0, 1.0, "hej"),
                FakeSegment(2.5, 4.0, "bot"),
            ]
            return iter(segments), info

    def fake_prepare(path: Path):
        assert path == source
        return extracted, True

    monkeypatch.setattr(sys.modules["transcripy.transcribe"], "prepare_media", fake_prepare)
    model = FakeModel()
    engine = FasterWhisperEngine(model=model)

    result = engine.transcribe(
        source,
        language="pl",
        hotwords="TranscriPy",
        on_progress=lambda fraction, count: progress.append((fraction, count)),
    )

    assert result.text == "hej bot"
    assert model.path == str(extracted)
    assert model.hotwords == "TranscriPy"
    assert progress == [(0.25, 1), (1.0, 2), (1.0, 2)]
    assert not extracted.exists()


def test_default_compute_type_follows_the_device():
    assert default_compute_type("cpu") == "int8"
    assert default_compute_type("cuda") == "float16"

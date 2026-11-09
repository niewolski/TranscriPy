import sys
from pathlib import Path

import pytest

from transcripy.diarize import SpeakerSpan
from transcripy.models import Segment, Transcript, Word
from transcripy.transcribe import (
    DEFAULT_MODEL,
    FasterWhisperEngine,
    default_compute_type,
    transcribe,
)


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
        word_timestamps: bool = True,
        on_progress=None,
    ) -> Transcript:
        self.source = source
        self.language = language
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.word_timestamps = word_timestamps
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


def test_stereo_recording_labels_each_channel(tmp_path: Path, monkeypatch):
    source = tmp_path / "call.wav"
    source.write_bytes(b"stereo")
    left = tmp_path / "left.wav"
    right = tmp_path / "right.wav"
    left.write_bytes(b"L")
    right.write_bytes(b"R")
    progress = []

    class ChannelEngine:
        def __init__(self) -> None:
            self.sources: list[Path] = []

        def transcribe(self, source: Path, language: str | None = None, **kwargs):
            self.sources.append(source)
            if kwargs.get("on_progress") is not None:
                kwargs["on_progress"](1.0, 1)
            if source == left:
                segment = Segment(0.5, 1.2, "lewy")
            else:
                segment = Segment(0.0, 0.4, "prawy")
            return Transcript.from_segments(
                [segment],
                language=language or "pl",
                duration=segment.end,
            )

    def fake_split(path: Path, labels=("left", "right")):
        assert path == source
        assert labels == ("agent", "customer")
        return [("agent", left), ("customer", right)]

    monkeypatch.setattr(sys.modules["transcripy.transcribe"], "split_stereo", fake_split)
    engine = ChannelEngine()

    result = transcribe(
        source,
        language="pl",
        speakers=True,
        channel_labels=("agent", "customer"),
        engine=engine,
        on_progress=lambda fraction, count: progress.append((fraction, count)),
    )

    assert [(segment.speaker, segment.text) for segment in result.segments] == [
        ("customer", "prawy"),
        ("agent", "lewy"),
    ]
    assert result.turns(1.0)[0].speaker == "customer"
    assert engine.sources == [left, right]
    assert progress == [(0.5, 1), (1.0, 2)]
    assert not left.exists()
    assert not right.exists()


def test_invalid_channel_labels_raise(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"wav")

    with pytest.raises(ValueError, match="channel_labels"):
        transcribe(audio, speakers=True, channel_labels=("only",), engine=FakeEngine())


def test_mono_recording_uses_the_diarizer(tmp_path: Path, monkeypatch):
    source = tmp_path / "call.wav"
    source.write_bytes(b"mono")
    engine = FakeEngine()
    seen: list[Path] = []
    monkeypatch.setattr(
        sys.modules["transcripy.transcribe"],
        "split_stereo",
        lambda path, labels=("left", "right"): None,
    )

    def diarizer(path: Path):
        seen.append(path)
        return [SpeakerSpan(0.0, 2.0, "SPEAKER_00")]

    result = transcribe(source, speakers=True, engine=engine, diarizer=diarizer)

    assert seen == [source]
    assert result.segments[0].speaker == "SPEAKER_00"


def test_prompt_and_hotwords_reach_the_engine(tmp_path: Path):
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    engine = FakeEngine()

    transcribe(
        audio,
        engine=engine,
        initial_prompt="  TranscriPy  ",
        hotwords="Niewolski",
        word_timestamps=False,
    )

    assert engine.initial_prompt == "TranscriPy"
    assert engine.hotwords == "Niewolski"
    assert engine.word_timestamps is False


def test_engine_reports_progress_and_keeps_word_timings(tmp_path: Path, monkeypatch):
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    extracted = tmp_path / "extracted.wav"
    extracted.write_bytes(b"wav")
    progress = []

    class FakeWord:
        def __init__(self, start: float, end: float, word: str, probability: float) -> None:
            self.start = start
            self.end = end
            self.word = word
            self.probability = probability

    class FakeSegment:
        def __init__(self, start: float, end: float, text: str, words) -> None:
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    class FakeModel:
        def __init__(self) -> None:
            self.path = None
            self.options = None

        def transcribe(self, path: str, **options):
            self.path = path
            self.options = options
            info = type("Info", (), {"language": "pl", "duration": 4.0})()
            segments = [
                FakeSegment(
                    0.0,
                    1.0,
                    "hej",
                    [FakeWord(0.0, 1.0, "hej", 0.95)],
                ),
                FakeSegment(2.5, 4.0, "bot", [FakeWord(2.5, 4.0, "bot", 0.8)]),
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
    assert model.options["hotwords"] == "TranscriPy"
    assert model.options["word_timestamps"] is True
    assert result.segments[0].words == (Word(0.0, 1.0, "hej", 0.95),)
    assert progress == [(0.25, 1), (1.0, 2), (1.0, 2)]
    assert not extracted.exists()


def test_default_compute_type_follows_the_device():
    assert default_compute_type("cpu") == "int8"
    assert default_compute_type("cuda") == "float16"
    assert DEFAULT_MODEL == "large-v3"

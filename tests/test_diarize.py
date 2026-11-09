from pathlib import Path

import pytest

from transcripy.diarize import SpeakerSpan, assign_speakers, pyannote_speakers
from transcripy.models import Segment, Transcript


def _transcript() -> Transcript:
    return Transcript.from_segments(
        [
            Segment(0.0, 1.0, "cześć"),
            Segment(1.2, 3.0, "mówi bot"),
            Segment(5.0, 5.4, "halo"),
        ],
        language="pl",
        duration=5.4,
    )


def test_assign_speakers_uses_the_largest_overlap():
    spans = [
        SpeakerSpan(0.0, 1.5, "SPEAKER_00"),
        SpeakerSpan(1.4, 3.2, "SPEAKER_01"),
    ]

    labeled = assign_speakers(_transcript(), spans)

    assert [segment.speaker for segment in labeled.segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        None,
    ]
    assert labeled.turns(0.5)[0].speaker == "SPEAKER_00"
    assert labeled.turns(0.5)[1].speaker == "SPEAKER_01"


def test_pyannote_requires_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"mono")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        pyannote_speakers(audio)


def test_pyannote_requires_the_optional_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"mono")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def missing(token: str):
        raise ImportError(token)

    monkeypatch.setattr("transcripy.diarize._load_pyannote", missing)

    with pytest.raises(RuntimeError, match="pyannote.audio"):
        pyannote_speakers(audio)


def test_pyannote_reads_annotation_tracks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"mono")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class Annotation:
        def itertracks(self, yield_label: bool = False):
            yield Turn(0.2, 1.0), "track", "SPEAKER_00"

    def loader(token: str):
        assert token == "test-token"
        return lambda path: Annotation()

    monkeypatch.setattr("transcripy.diarize._load_pyannote", loader)

    assert pyannote_speakers(audio) == [SpeakerSpan(0.2, 1.0, "SPEAKER_00")]

from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from transcripy.models import Segment, Transcript


class RecordingTranscriber:
    def __init__(self) -> None:
        self.source: Path | None = None
        self.model: str | None = None
        self.language: str | None = None
        self.device: str | None = None

    def __call__(
        self,
        source: str | Path,
        *,
        model: str = "small",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str | None = None,
        engine: object | None = None,
    ) -> Transcript:
        self.source = Path(source)
        assert self.source.is_file()
        self.model = model
        self.language = language
        self.device = device
        return Transcript.from_segments(
            [Segment(0.0, 1.5, "cześć")],
            language=language or "pl",
            duration=1.5,
        )


def test_transcribe_upload_returns_json(monkeypatch):
    recorder = RecordingTranscriber()
    monkeypatch.setattr("api.routes.transcribe", recorder)
    client = TestClient(app)

    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("call.mp3", b"not-real-audio", "audio/mpeg")},
        data={"language": "pl", "model": "tiny", "device": "cpu"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "cześć",
        "language": "pl",
        "duration": 1.5,
        "segments": [{"start": 0.0, "end": 1.5, "text": "cześć"}],
    }
    assert recorder.model == "tiny"
    assert recorder.language == "pl"
    assert recorder.device == "cpu"
    assert recorder.source is not None
    assert not recorder.source.exists()


def test_transcribe_upload_can_return_subtitles(monkeypatch):
    monkeypatch.setattr("api.routes.transcribe", RecordingTranscriber())
    client = TestClient(app)
    upload = {"file": ("film.mp4", b"not-real-video", "video/mp4")}

    srt = client.post("/api/v1/transcribe?format=srt", files=upload)
    vtt = client.post("/api/v1/transcribe?format=vtt", files=upload)
    text = client.post("/api/v1/transcribe?format=txt", files=upload)

    assert srt.status_code == 200
    assert srt.text == "1\n00:00:00,000 --> 00:00:01,500\ncześć\n"
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert vtt.text.startswith("WEBVTT\n")
    assert text.text == "cześć"


def test_empty_upload_is_rejected(monkeypatch):
    monkeypatch.setattr("api.routes.transcribe", RecordingTranscriber())
    client = TestClient(app)

    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("call.mp3", b"", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "file is empty"


def test_unknown_format_is_rejected():
    client = TestClient(app)

    response = client.post(
        "/api/v1/transcribe?format=pdf",
        files={"file": ("call.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "format must be json, txt, srt, or vtt"

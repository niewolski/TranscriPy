import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.jobs import JobStore
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
        model: str = "large-v3",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str | None = None,
        engine: object | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        speakers: bool = False,
        channel_labels=("left", "right"),
        word_timestamps: bool = True,
        on_progress=None,
    ) -> Transcript:
        self.source = Path(source)
        assert self.source.is_file()
        self.model = model
        self.language = language
        self.device = device
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.speakers = speakers
        self.channel_labels = channel_labels
        self.word_timestamps = word_timestamps
        return Transcript.from_segments(
            [Segment(0.0, 1.5, "hello")],
            language=language or "en",
            duration=1.5,
        )


def test_transcribe_upload_returns_json(monkeypatch):
    recorder = RecordingTranscriber()
    monkeypatch.setattr("api.routes.transcribe", recorder)
    client = TestClient(app)

    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("call.mp3", b"not-real-audio", "audio/mpeg")},
        data={"language": "en", "model": "tiny", "device": "cpu"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "hello",
        "language": "en",
        "duration": 1.5,
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": None}],
        "turns": [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": None}],
    }
    assert recorder.model == "tiny"
    assert recorder.language == "en"
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
    assert srt.text == "1\n00:00:00,000 --> 00:00:01,500\nhello\n"
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert vtt.text.startswith("WEBVTT\n")
    assert text.text == "hello"


def test_speakers_flag_is_forwarded(monkeypatch):
    recorder = RecordingTranscriber()
    monkeypatch.setattr("api.routes.transcribe", recorder)
    client = TestClient(app)

    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("call.wav", b"stereo", "audio/wav")},
        data={"speakers": "true"},
    )

    assert response.status_code == 200
    assert recorder.speakers is True


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


def test_job_returns_before_transcription_finishes(monkeypatch):
    release = threading.Event()
    reported = threading.Event()
    recorder = RecordingTranscriber()

    def slow(*args, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress(0.4, 2)
        reported.set()
        release.wait(2)
        return recorder(*args, on_progress=on_progress, **kwargs)

    monkeypatch.setattr("api.routes.jobs", JobStore(slow))
    client = TestClient(app)

    response = client.post(
        "/api/v1/jobs?format=srt",
        files={"file": ("film.mp4", b"video", "video/mp4")},
        data={"language": "en", "model": "tiny", "device": "cpu"},
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["status"] == "queued"
    assert reported.wait(2)
    running = client.get(f"/api/v1/jobs/{job_id}").json()
    assert running["status"] == "running"
    assert running["progress"] == 0.4
    assert running["segments_done"] == 2
    assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 409

    release.set()
    finished = _wait_for_job(client, job_id)

    assert finished["status"] == "done"
    assert finished["error"] is None
    assert finished["progress"] == 1.0
    assert finished["segments_done"] == 1
    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.text == "1\n00:00:00,000 --> 00:00:01,500\nhello\n"
    assert recorder.model == "tiny"
    assert recorder.language == "en"
    assert recorder.source is not None
    assert not recorder.source.exists()


def test_failed_job_reports_the_error_and_deletes_the_file(monkeypatch):
    seen: dict[str, Path] = {}

    def fail(source, **kwargs):
        seen["path"] = Path(source)
        raise RuntimeError("model missing")

    monkeypatch.setattr("api.routes.jobs", JobStore(fail))
    client = TestClient(app)
    created = client.post(
        "/api/v1/jobs",
        files={"file": ("call.mp3", b"audio", "audio/mpeg")},
    )
    job_id = created.json()["id"]
    finished = _wait_for_job(client, job_id)

    assert finished["status"] == "failed"
    assert finished["error"] == "model missing"
    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 500
    assert result.json()["detail"] == "model missing"
    assert not seen["path"].exists()


def test_unknown_job_is_not_found():
    client = TestClient(app)

    assert client.get("/api/v1/jobs/missing").status_code == 404
    assert client.get("/api/v1/jobs/missing/result").status_code == 404


def test_job_rejects_an_empty_file(monkeypatch):
    monkeypatch.setattr("api.routes.jobs", JobStore(RecordingTranscriber()))
    client = TestClient(app)

    response = client.post(
        "/api/v1/jobs",
        files={"file": ("call.mp3", b"", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "file is empty"


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    last = {}
    for _ in range(100):
        last = client.get(f"/api/v1/jobs/{job_id}").json()
        if last["status"] in {"done", "failed"}:
            return last
        time.sleep(0.01)
    raise AssertionError(last)

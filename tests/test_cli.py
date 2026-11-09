import json
import subprocess
import sys
from pathlib import Path

from transcripy.cli import main
from transcripy.models import Segment, Transcript


class RecordingTranscriber:
    def __init__(self) -> None:
        self.source: str | Path | None = None
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
        self.source = source
        self.model = model
        self.language = language
        self.device = device
        return Transcript.from_segments(
            [Segment(0.0, 1.5, "cześć")],
            language=language or "pl",
            duration=1.5,
        )


def test_output_extension_selects_srt(tmp_path: Path):
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    destination = tmp_path / "napisy.srt"
    recorder = RecordingTranscriber()

    code = main(
        [str(source), "-o", str(destination), "--language", "pl", "--model", "tiny"],
        transcriber=recorder,
    )

    assert code == 0
    assert destination.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\ncześć\n"
    )
    assert recorder.language == "pl"
    assert recorder.model == "tiny"
    assert recorder.device == "cpu"


def test_writes_vtt_txt_and_json(tmp_path: Path):
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")
    recorder = RecordingTranscriber()

    assert main([str(source), "-o", str(tmp_path / "a.vtt")], transcriber=recorder) == 0
    assert main([str(source), "-o", str(tmp_path / "a.txt")], transcriber=recorder) == 0
    assert main([str(source), "-o", str(tmp_path / "a.json")], transcriber=recorder) == 0

    assert (tmp_path / "a.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "cześć\n"
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["text"] == "cześć"


def test_prints_text_when_there_is_no_output_file(tmp_path: Path, capsys):
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")

    code = main([str(source), "--device", "cuda"], transcriber=RecordingTranscriber())

    assert code == 0
    assert capsys.readouterr().out == "cześć\n"


def test_format_flag_overrides_the_extension(tmp_path: Path):
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "napisy.txt"

    code = main(
        [str(source), "-o", str(destination), "--format", "srt"],
        transcriber=RecordingTranscriber(),
    )

    assert code == 0
    assert destination.read_text(encoding="utf-8").startswith("1\n")


def test_unknown_extension_fails_before_transcription(tmp_path: Path, capsys):
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")
    recorder = RecordingTranscriber()

    code = main([str(source), "-o", str(tmp_path / "napisy.doc")], transcriber=recorder)

    assert code == 1
    assert recorder.source is None
    assert "cannot tell the format" in capsys.readouterr().err


def test_missing_file_returns_an_error(tmp_path: Path, capsys):
    missing = tmp_path / "brak.mp3"
    destination = tmp_path / "napisy.srt"

    code = main([str(missing), "-o", str(destination)])

    assert code == 1
    assert not destination.exists()
    assert "not found" in capsys.readouterr().err


def test_module_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "transcripy", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "video soundtrack" in result.stdout

from pathlib import Path
from types import SimpleNamespace

import pytest

from transcripy.audio import prepare_media


def test_plain_audio_skips_ffmpeg(tmp_path: Path):
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")

    def unused(name: str) -> str:
        raise AssertionError(name)

    prepared, temporary = prepare_media(audio, which=unused)

    assert prepared == audio
    assert temporary is False


def test_video_is_extracted_to_wav(tmp_path: Path):
    video = tmp_path / "film.mp4"
    video.write_bytes(b"video")

    def runner(command, capture_output, check):
        assert command[0] == "ffmpeg"
        assert command[command.index("-i") + 1] == str(video)
        assert "-vn" in command
        Path(command[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stderr=b"")

    prepared, temporary = prepare_media(video, which=lambda name: "ffmpeg", runner=runner)

    assert temporary is True
    assert prepared.suffix == ".wav"
    assert prepared.read_bytes() == b"wav"
    prepared.unlink()


def test_missing_ffmpeg_explains_what_to_install(tmp_path: Path):
    video = tmp_path / "film.mp4"
    video.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="Install ffmpeg"):
        prepare_media(video, which=lambda name: None)


def test_failed_extraction_deletes_the_temporary_wav(tmp_path: Path):
    video = tmp_path / "film.mkv"
    video.write_bytes(b"video")
    created: list[Path] = []

    def runner(command, capture_output, check):
        path = Path(command[-1])
        created.append(path)
        path.write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr=b"Invalid data found\n")

    with pytest.raises(RuntimeError, match="Invalid data found"):
        prepare_media(video, which=lambda name: "ffmpeg", runner=runner)

    assert created
    assert not created[0].exists()

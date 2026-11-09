from pathlib import Path
from types import SimpleNamespace

import pytest

from transcripy.audio import prepare_media, split_stereo


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


def test_mono_recording_is_not_split(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"mono")
    commands: list[str] = []

    def runner(command, capture_output, check):
        commands.append(command[0])
        return SimpleNamespace(returncode=0, stdout=b"1\n", stderr=b"")

    assert split_stereo(audio, which=lambda name: name, runner=runner) is None
    assert commands == ["ffprobe"]


def test_stereo_recording_extracts_both_channels(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"stereo")

    def runner(command, capture_output, check):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=b"2\n", stderr=b"")
        assert "pan=mono|c0=c0" in command or "pan=mono|c0=c1" in command
        Path(command[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    split = split_stereo(audio, which=lambda name: name, runner=runner)

    assert split is not None
    assert [speaker for speaker, _ in split] == ["left", "right"]
    for _, path in split:
        assert path.read_bytes() == b"wav"
        path.unlink()


def test_stereo_recording_uses_custom_labels(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"stereo")

    def runner(command, capture_output, check):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=b"2\n", stderr=b"")
        Path(command[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    split = split_stereo(
        audio,
        labels=("agent", "customer"),
        which=lambda name: name,
        runner=runner,
    )

    assert split is not None
    assert [speaker for speaker, _ in split] == ["agent", "customer"]
    for _, path in split:
        path.unlink()


def test_separating_speakers_requires_ffmpeg(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"stereo")

    with pytest.raises(RuntimeError, match="separate speakers"):
        split_stereo(audio, which=lambda name: None)


def test_failed_channel_extract_deletes_temporary_files(tmp_path: Path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"stereo")
    created: list[Path] = []

    def runner(command, capture_output, check):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=b"2\n", stderr=b"")
        path = Path(command[-1])
        created.append(path)
        path.write_bytes(b"partial")
        returncode = 0 if len(created) == 1 else 1
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=b"boom\n")

    with pytest.raises(RuntimeError, match="right channel"):
        split_stereo(audio, which=lambda name: name, runner=runner)

    assert created
    assert all(not path.exists() for path in created)

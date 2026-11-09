"""Read a soundtrack from a video or an unusual recording."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Formats faster-whisper can open directly through PyAV.
PASSTHROUGH_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def prepare_media(
    source: Path,
    *,
    which=shutil.which,
    runner=subprocess.run,
) -> tuple[Path, bool]:
    """Return audio for the recognizer and whether that file is temporary.

    Plain audio is returned unchanged. Video and other containers are converted
    with ffmpeg to 16 kHz mono WAV. The second value is true when the caller
    must delete the returned file.
    """
    if source.suffix.lower() in PASSTHROUGH_SUFFIXES:
        return source, False

    executable = which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "ffmpeg is required to read this file. Install ffmpeg and add it to PATH."
        )

    descriptor, name = tempfile.mkstemp(suffix=".wav")
    os.close(descriptor)
    destination = Path(name)
    try:
        completed = runner(
            [
                executable,
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(destination),
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = _last_line(getattr(completed, "stderr", b""))
            message = "ffmpeg could not read the soundtrack"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination, True


def split_stereo(
    source: Path,
    *,
    which=shutil.which,
    runner=subprocess.run,
) -> list[tuple[str, Path]] | None:
    """Split a stereo recording into left and right 16 kHz mono files.

    Returns nothing when the soundtrack has one channel. Both returned files
    are temporary and the caller deletes them.
    """
    ffmpeg = which("ffmpeg")
    ffprobe = which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "ffmpeg is required to separate speakers. Install ffmpeg and add it to PATH."
        )

    probe = runner(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = _last_line(getattr(probe, "stderr", b""))
        message = "ffprobe could not read the audio channels"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    channels = _channel_count(getattr(probe, "stdout", b""))
    if channels < 2:
        return None

    created: list[Path] = []
    try:
        for index, speaker in enumerate(("left", "right")):
            descriptor, name = tempfile.mkstemp(suffix=".wav")
            os.close(descriptor)
            destination = Path(name)
            created.append(destination)
            completed = runner(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-af",
                    f"pan=mono|c0=c{index}",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(destination),
                ],
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = _last_line(getattr(completed, "stderr", b""))
                message = f"ffmpeg could not extract the {speaker} channel"
                if detail:
                    message = f"{message}: {detail}"
                raise RuntimeError(message)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return [("left", created[0]), ("right", created[1])]


def _channel_count(stdout: bytes | str) -> int:
    if isinstance(stdout, bytes):
        text = stdout.decode("utf-8", errors="replace")
    else:
        text = stdout
    token = text.strip().splitlines()
    if not token:
        return 1
    try:
        return int(token[0].strip())
    except ValueError:
        return 1


def _last_line(stderr: bytes | str) -> str:
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = stderr
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""

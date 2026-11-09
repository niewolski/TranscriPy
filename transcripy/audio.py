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


def _last_line(stderr: bytes | str) -> str:
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = stderr
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""

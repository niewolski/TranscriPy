"""Integration checks for the stereo call path with real ffmpeg when present."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from transcripy.audio import split_stereo
from transcripy.models import Segment, Transcript
from transcripy.transcribe import transcribe

pytestmark = pytest.mark.integration


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return True


def _write_stereo_wav(path: Path, seconds: float = 0.25, rate: int = 16_000) -> None:
    """Write a short stereo wav with different tones on each channel."""
    frames = int(seconds * rate)
    samples = bytearray()
    for index in range(frames):
        left = int(12000 * math.sin(2 * math.pi * 440 * index / rate))
        right = int(12000 * math.sin(2 * math.pi * 660 * index / rate))
        samples.extend(struct.pack("<hh", left, right))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(samples))


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe required")
def test_real_stereo_wav_splits_into_labeled_channels(tmp_path: Path):
    source = tmp_path / "call.wav"
    _write_stereo_wav(source)

    split = split_stereo(source, labels=("agent", "customer"))

    assert split is not None
    assert [name for name, _ in split] == ["agent", "customer"]
    for name, path in split:
        assert path.is_file()
        assert path.stat().st_size > 0
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16_000
        path.unlink()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe required")
def test_transcribe_labels_stereo_channels_with_hotword_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "call.wav"
    _write_stereo_wav(source)
    seen: list[tuple[str | None, str]] = []

    class ChannelEngine:
        def transcribe(self, audio: Path, language: str | None = None, **kwargs):
            assert language == "en"
            assert kwargs["hotwords"] == "TranscriPy, Acme"
            assert kwargs["initial_prompt"] == "A call about the TranscriPy product."
            assert kwargs["word_timestamps"] is True
            # Identify channel by reading which temporary file we got after split.
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnchannels() == 1
            # Order follows channel_labels: agent then customer.
            label = "agent" if not seen else "customer"
            if label == "agent":
                segment = Segment(0.05, 0.2, "hello from TranscriPy")
            else:
                segment = Segment(0.0, 0.15, "listening Acme")
            seen.append((label, segment.text))
            return Transcript.from_segments(
                [segment],
                language="en",
                duration=0.25,
            )

    monkeypatch.setattr(
        sys.modules["transcripy.transcribe"],
        "FasterWhisperEngine",
        lambda **kwargs: ChannelEngine(),
    )

    result = transcribe(
        source,
        language="en",
        speakers=True,
        channel_labels=("agent", "customer"),
        hotwords="TranscriPy, Acme",
        initial_prompt="A call about the TranscriPy product.",
        engine=ChannelEngine(),
    )

    assert [segment.speaker for segment in result.segments] == ["customer", "agent"]
    assert "TranscriPy" in result.text
    assert "Acme" in result.text
    turns = result.turns(pause=1.0)
    assert {turn.speaker for turn in turns} == {"agent", "customer"}

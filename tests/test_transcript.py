import pytest

from transcripy.models import Segment, Transcript, format_timestamp


def _sample() -> Transcript:
    return Transcript.from_segments(
        [
            Segment(0.0, 1.5, "  dzień dobry  "),
            Segment(1.5, 3.0, "   "),
            Segment(3.0, 4.25, "tu voicebot"),
        ],
        language="pl",
        duration=4.25,
    )


def test_from_segments_cleans_text_and_drops_blanks():
    transcript = _sample()

    assert transcript.text == "dzień dobry tu voicebot"
    assert transcript.language == "pl"
    assert transcript.duration == 4.25
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "dzień dobry"


def test_duration_defaults_to_the_last_segment():
    transcript = Transcript.from_segments(
        [Segment(0.0, 2.0, "halo")],
        language="pl",
    )

    assert transcript.duration == 2.0


def test_empty_transcript():
    transcript = Transcript.from_segments([], language="pl")

    assert transcript.text == ""
    assert transcript.segments == ()
    assert transcript.duration == 0.0
    assert transcript.to_srt() == ""
    assert transcript.to_vtt() == "WEBVTT\n"


def test_srt_and_vtt_timestamps():
    transcript = _sample()

    assert transcript.to_srt() == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "dzień dobry\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,250\n"
        "tu voicebot\n"
    )
    assert transcript.to_vtt() == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.500\n"
        "dzień dobry\n"
        "\n"
        "00:00:03.000 --> 00:00:04.250\n"
        "tu voicebot\n"
    )


def test_to_dict():
    transcript = _sample()

    assert transcript.to_dict() == {
        "text": "dzień dobry tu voicebot",
        "language": "pl",
        "duration": 4.25,
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "dzień dobry"},
            {"start": 3.0, "end": 4.25, "text": "tu voicebot"},
        ],
        "turns": [
            {"start": 0.0, "end": 1.5, "text": "dzień dobry"},
            {"start": 3.0, "end": 4.25, "text": "tu voicebot"},
        ],
    }


def test_turns_follow_the_pause():
    transcript = Transcript.from_segments(
        [
            Segment(0.0, 1.0, "dzień dobry"),
            Segment(1.4, 2.0, "tu voicebot"),
            Segment(4.0, 5.0, "do widzenia"),
        ],
        language="pl",
    )

    assert transcript.turns(1.0)[0].text == "dzień dobry tu voicebot"
    assert transcript.turns(1.0)[1].text == "do widzenia"
    assert len(transcript.turns(3.0)) == 1
    with pytest.raises(ValueError):
        transcript.turns(-0.1)


def test_format_timestamp_over_an_hour():
    assert format_timestamp(3661.25) == "01:01:01,250"
    assert format_timestamp(-1, decimal_separator=".") == "00:00:00.000"


def test_invalid_segment_and_duration():
    with pytest.raises(ValueError):
        Segment(-0.1, 1.0, "halo")
    with pytest.raises(ValueError):
        Segment(2.0, 1.0, "halo")
    with pytest.raises(ValueError):
        Transcript.from_segments([Segment(0.0, 1.0, "halo")], language="  ")
    with pytest.raises(ValueError):
        Transcript.from_segments(
            [Segment(0.0, 1.0, "halo")],
            language="pl",
            duration=-1,
        )

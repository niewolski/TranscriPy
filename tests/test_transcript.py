import pytest

from transcripy.models import Segment, Transcript, Word, format_timestamp


def _sample() -> Transcript:
    return Transcript.from_segments(
        [
            Segment(0.0, 1.5, "  good morning  "),
            Segment(1.5, 3.0, "   "),
            Segment(3.0, 4.25, "this is the voicebot"),
        ],
        language="en",
        duration=4.25,
    )


def test_from_segments_cleans_text_and_drops_blanks():
    transcript = _sample()

    assert transcript.text == "good morning this is the voicebot"
    assert transcript.language == "en"
    assert transcript.duration == 4.25
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "good morning"


def test_duration_defaults_to_the_last_segment():
    transcript = Transcript.from_segments(
        [Segment(0.0, 2.0, "hello")],
        language="en",
    )

    assert transcript.duration == 2.0


def test_empty_transcript():
    transcript = Transcript.from_segments([], language="en")

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
        "good morning\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,250\n"
        "this is the voicebot\n"
    )
    assert transcript.to_vtt() == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.500\n"
        "good morning\n"
        "\n"
        "00:00:03.000 --> 00:00:04.250\n"
        "this is the voicebot\n"
    )


def test_to_dict():
    transcript = _sample()

    assert transcript.to_dict() == {
        "text": "good morning this is the voicebot",
        "language": "en",
        "duration": 4.25,
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "good morning",
                "speaker": None,
                "words": [],
            },
            {
                "start": 3.0,
                "end": 4.25,
                "text": "this is the voicebot",
                "speaker": None,
                "words": [],
            },
        ],
        "turns": [
            {"start": 0.0, "end": 1.5, "text": "good morning", "speaker": None},
            {"start": 3.0, "end": 4.25, "text": "this is the voicebot", "speaker": None},
        ],
    }


def test_words_are_kept_on_segments():
    transcript = Transcript.from_segments(
        [
            Segment(
                0.0,
                1.0,
                "good morning",
                words=(
                    Word(0.0, 0.4, "good", 0.9),
                    Word(0.4, 1.0, "morning", 0.8),
                ),
            )
        ],
        language="en",
    )

    assert transcript.segments[0].words[0].word == "good"
    assert transcript.to_dict()["segments"][0]["words"] == [
        {"start": 0.0, "end": 0.4, "word": "good", "probability": 0.9},
        {"start": 0.4, "end": 1.0, "word": "morning", "probability": 0.8},
    ]


def test_turns_follow_the_pause():
    transcript = Transcript.from_segments(
        [
            Segment(0.0, 1.0, "good morning"),
            Segment(1.4, 2.0, "this is the voicebot"),
            Segment(4.0, 5.0, "goodbye"),
        ],
        language="en",
    )

    assert transcript.turns(1.0)[0].text == "good morning this is the voicebot"
    assert transcript.turns(1.0)[1].text == "goodbye"
    assert len(transcript.turns(3.0)) == 1


def test_turns_do_not_cross_speakers():
    transcript = Transcript.from_segments(
        [
            Segment(0.0, 1.0, "hello", speaker="left"),
            Segment(1.1, 2.0, "listening", speaker="right"),
        ],
        language="en",
    )

    turns = transcript.turns(2.0)

    assert [turn.speaker for turn in turns] == ["left", "right"]
    assert transcript.to_srt().splitlines()[2] == "[left] hello"
    with pytest.raises(ValueError):
        transcript.turns(-0.1)


def test_format_timestamp_over_an_hour():
    assert format_timestamp(3661.25) == "01:01:01,250"
    assert format_timestamp(-1, decimal_separator=".") == "00:00:00.000"


def test_invalid_segment_and_duration():
    with pytest.raises(ValueError):
        Segment(-0.1, 1.0, "hello")
    with pytest.raises(ValueError):
        Segment(2.0, 1.0, "hello")
    with pytest.raises(ValueError):
        Transcript.from_segments([Segment(0.0, 1.0, "hello")], language="  ")
    with pytest.raises(ValueError):
        Transcript.from_segments(
            [Segment(0.0, 1.0, "hello")],
            language="en",
            duration=-1,
        )

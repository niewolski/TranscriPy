"""Command line entry point for one-file transcription."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transcripy.models import Transcript
from transcripy.transcribe import transcribe

_FORMATS = ("txt", "srt", "vtt", "json")


def main(argv: list[str] | None = None, *, transcriber=transcribe) -> int:
    """Transcribe one file and print it or write it to ``--output``."""
    parser = argparse.ArgumentParser(
        prog="transcripy",
        description="Transcribe a voicebot recording, audio file, or video soundtrack.",
    )
    parser.add_argument("source", help="audio or video file")
    parser.add_argument(
        "-o",
        "--output",
        help="file to write; the extension picks txt, srt, vtt, or json",
    )
    parser.add_argument("--language", help="ISO language code, for example pl")
    parser.add_argument("--model", default="small", help="Whisper model name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--prompt", help="words the recording is likely to contain")
    parser.add_argument("--hotwords", help="names or phrases to bias recognition toward")
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="label stereo channels, or separate voices on a mono recording",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="seconds of silence that start a new utterance",
    )
    parser.add_argument(
        "--format",
        choices=_FORMATS,
        help="output format; overrides the extension of --output",
    )
    args = parser.parse_args(argv)

    if args.pause < 0:
        print("pause must be >= 0", file=sys.stderr)
        return 1

    try:
        output_format = choose_format(args.output, args.format)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    reported = False

    def on_progress(fraction: float, segments_done: int) -> None:
        nonlocal reported
        reported = True
        percent = int(round(min(max(fraction, 0.0), 1.0) * 100))
        print(
            f"\r{percent:3d}%  {segments_done} segments",
            file=sys.stderr,
            end="",
            flush=True,
        )

    try:
        result = transcriber(
            args.source,
            model=args.model,
            language=args.language,
            device=args.device,
            initial_prompt=args.prompt,
            hotwords=args.hotwords,
            speakers=args.speakers,
            on_progress=on_progress,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        if reported:
            print(file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    if reported:
        print(file=sys.stderr)

    rendered = render(result, output_format, args.pause)
    if args.output is None:
        sys.stdout.write(rendered)
        return 0

    destination = Path(args.output).expanduser()
    try:
        destination.write_text(rendered, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def choose_format(output: str | None, explicit: str | None) -> str:
    """Pick txt, srt, vtt, or json from a flag or a file name."""
    if explicit:
        return explicit
    if not output:
        return "txt"
    suffix = Path(output).suffix.lower().lstrip(".")
    if suffix in _FORMATS:
        return suffix
    if suffix == "":
        return "txt"
    raise ValueError(
        f"cannot tell the format from .{suffix}; pass --format txt, srt, vtt, or json"
    )


def render(result: Transcript, output_format: str, pause: float = 1.0) -> str:
    """Turn a transcript into the text that should be stored."""
    if output_format == "txt":
        if not result.text or result.text.endswith("\n"):
            return result.text
        return result.text + "\n"
    if output_format == "srt":
        return result.to_srt()
    if output_format == "vtt":
        return result.to_vtt()
    if output_format == "json":
        return json.dumps(result.to_dict(pause=pause), ensure_ascii=False, indent=2) + "\n"
    raise ValueError(f"unsupported format: {output_format}")


if __name__ == "__main__":
    raise SystemExit(main())

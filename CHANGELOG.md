# Changelog

## 1.0.0

### Breaking

- Default Whisper model is now `large-v3` (was `small`). Pass `model="small"` or `--model small` to keep the old default.
- Stereo speaker labels stay `left` / `right` by default, but can be overridden with `channel_labels` / `--channel-labels` (for example `agent,customer`).
- Package distribution is library + CLI only. FastAPI code under `api/` / `main.py` is experimental and not installed by `pip install transcripy`.

### Added

- Per-word timestamps on segments (`Segment.words` / JSON `words`), enabled by default.
- Configurable stereo channel labels for voicebot and call-center recordings.
- PyPI metadata, NOTICE for third-party models, CI workflows, and focused call-recording docs.

### Changed

- Public positioning: local transcription of call recordings into timed speaker turns, not a general “transcribe any file” wrapper.

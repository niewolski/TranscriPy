# TranscriPy

Python library that transcribes voicebot recordings, audio files, and video soundtracks.

Pass a path to an mp3, wav, m4a, or a video file. The result contains the full text, timed segments, and subtitle exports.

## Install

```bash
pip install -e ".[dev]"
```

The first transcription downloads a Whisper model, so that call needs a network connection.

## Use

```python
from transcripy import transcribe

result = transcribe("call.mp3", language="pl")
print(result.text)
print(result.to_srt())
```

`language` is an ISO code such as `pl`. Leave it out and the model detects the language. The default model is `small` on CPU. `model="tiny"` is faster to try, and `device="cuda"` uses a GPU when one is available.

`result.to_vtt()` writes WebVTT subtitles. `result.segments` holds each phrase with `start` and `end` in seconds.

The FastAPI app in this repository still only exposes health checks. Transcription goes through the `transcripy` package.

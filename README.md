# TranscriPy

Local transcription of **voicebot calls and phone recordings** into timed turns with speaker labels. Audio stays on your machine.

```bash
pip install transcripy
```

```python
from transcripy import transcribe

result = transcribe(
    "call.wav",
    language="pl",
    speakers=True,
    channel_labels=("agent", "customer"),
    hotwords="TranscriPy, Niewolski",
    initial_prompt="Rozmowa z voicebotem o produkcie TranscriPy.",
)

for turn in result.turns():
    print(f"[{turn.speaker}] {turn.text}")
```

```bash
transcripy call.wav \
  --language pl \
  --speakers \
  --channel-labels agent,customer \
  --hotwords "TranscriPy, Niewolski" \
  --prompt "Rozmowa z voicebotem o produkcie TranscriPy." \
  -o call.json
```

## What it is for

- Stereo call recordings where each party has a channel (agent / customer)
- Mono mixed calls when you install the optional diarization extra
- Offline / self-hosted pipelines that must not upload audio to a cloud ASR API

The public contract is simple: one media file in, a `Transcript` with `segments`, `turns`, and optional per-word timings out. Export as text, SRT, VTT, or JSON.

## Requirements

- Python 3.10+
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (installed with the package)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` for video containers and stereo channel split
- Optional NVIDIA CUDA for `--device cuda`
- Optional Hugging Face token for mono speaker diarization (see below)

## Speakers

**Stereo (recommended for voicebots).** `--speakers` (or `speakers=True`) splits the two channels and labels them with `--channel-labels` / `channel_labels`. Default labels are `left` and `right`.

**Mono.** The same flag runs [pyannote](https://github.com/pyannote/pyannote-audio) and assigns each segment the speaker with the largest time overlap. Overlapping speech and similar voices reduce quality; this is not WhisperX-level alignment.

```bash
pip install "transcripy[diarize]"
```

1. Accept the user agreements for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).
2. Create a Hugging Face access token.
3. Export it as `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`).

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| `model` | `large-v3` | Pass a smaller Whisper name (`medium`, `small`, `tiny`) to trade quality for speed |
| `language` | auto-detect | Prefer an ISO code such as `pl` for call center audio |
| `word_timestamps` | on | Disable with `--no-word-timestamps` / `word_timestamps=False` |
| `device` | `cpu` | Use `cuda` when a GPU is available |

Compared with [WhisperX](https://github.com/m-bain/whisperX): TranscriPy focuses on a small call-recording API and reliable **stereo channel labels**. WhisperX is stronger when you need batched GPU throughput and forced word alignment on mixed mono audio.

## Non-goals (1.0)

- Live / streaming microphone transcription
- Hosted cloud ASR
- Production HTTP API (the `api/` folder in this repo is experimental; not shipped in the wheel)
- High-throughput batched GPU pipelines
- Subtitle tooling for films as the primary use case

## Install options

```bash
pip install transcripy              # library + CLI
pip install "transcripy[diarize]"   # + pyannote for mono speakers
pip install "transcripy[dev]"       # tests and the experimental server deps
```

## Development

```bash
git clone https://github.com/niewolski/TranscriPy.git
cd TranscriPy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Model weight notices are in [NOTICE](NOTICE).

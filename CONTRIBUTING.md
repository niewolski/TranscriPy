# Contributing to TranscriPy

Thanks for your interest in the project. Bug reports, ideas, documentation fixes, and code changes are all welcome.

TranscriPy is a Python library and CLI for local transcription of voicebot calls and phone recordings into timed speaker turns.

## Reporting bugs

1. Check **Issues** to see if the problem was already reported.
2. If not, open a new issue using the **Bug report** template.
3. Include your Python version, operating system, steps to reproduce, and the error message.

## Feature requests

Have an idea for a new feature? Open an issue using the **Feature request** template and describe the problem it should solve. Please check the README non-goals for 1.0 first.

## Pull requests

1. Fork the repository.
2. Create a branch from `main`, for example: `git checkout -b feat/my-change`
3. Keep the change focused on one thing.
4. Run the tests: `pytest`
5. Commit with a clear message, for example: `git commit -m "Add support for X"`
6. Push the branch and open a pull request into `main`.

## Local setup

```bash
git clone https://github.com/niewolski/TranscriPy.git
cd TranscriPy
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
pytest
```

For speaker diarization support, also install the optional extra:

```bash
pip install -e ".[diarize]"
```

The FastAPI app under `api/` / `main.py` is experimental and not part of the published 1.0 wheel. Install `.[dev]` (or `.[server]`) if you need to run those tests.

## Code of Conduct

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

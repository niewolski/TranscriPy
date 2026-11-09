"""Local transcription of voicebot calls and phone recordings."""

from transcripy.models import Segment, Transcript, Turn, Word
from transcripy.transcribe import DEFAULT_MODEL, transcribe

__all__ = ["DEFAULT_MODEL", "Segment", "Transcript", "Turn", "Word", "transcribe"]
__version__ = "1.0.0"

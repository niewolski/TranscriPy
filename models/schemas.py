from pydantic import BaseModel

# base response model
class MessageResponse(BaseModel):
    """simple message response model"""
    message: str


# health check response model
class HealthResponse(BaseModel):
    """health check response model"""
    status: str


class SegmentResponse(BaseModel):
    """one timed phrase from a transcription"""
    start: float
    end: float
    text: str


class TranscriptResponse(BaseModel):
    """transcription returned by the api"""
    text: str
    language: str
    duration: float
    segments: list[SegmentResponse]


class JobCreatedResponse(BaseModel):
    """accepted transcription job"""
    id: str
    status: str


class JobStatusResponse(BaseModel):
    """current state of a transcription job"""
    id: str
    status: str
    error: str | None = None


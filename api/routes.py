import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from models.schemas import MessageResponse, TranscriptResponse
from transcripy import transcribe
from transcripy.models import Transcript

# create router for api endpoints
router = APIRouter(prefix="/api/v1", tags=["api"])

_OUTPUT_FORMATS = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}


@router.get("/", response_model=MessageResponse)
async def api_root():
    """api root endpoint"""
    return {"message": "transcripy api v1"}


@router.post("/transcribe", response_model=None)
def transcribe_file(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    model: str = Form(default="small"),
    device: str = Form(default="cpu"),
    output_format: str = Query(default="json", alias="format"),
) -> TranscriptResponse | PlainTextResponse:
    """transcribe an uploaded audio file or video soundtrack"""
    if output_format not in _OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="format must be json, txt, srt, or vtt",
        )

    saved = _save_upload(file)
    try:
        if saved.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="file is empty")
        try:
            result = transcribe(
                saved,
                model=model.strip() or "small",
                language=_blank_to_none(language),
                device=device.strip() or "cpu",
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        saved.unlink(missing_ok=True)

    return _render(result, output_format)


def _save_upload(file: UploadFile) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix(file.filename)) as tmp:
        while chunk := file.file.read(1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name)


def _suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".bin"


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _render(result: Transcript, output_format: str) -> TranscriptResponse | PlainTextResponse:
    if output_format == "json":
        return TranscriptResponse(**result.to_dict())
    if output_format == "txt":
        body = result.text
    elif output_format == "srt":
        body = result.to_srt()
    else:
        body = result.to_vtt()
    return PlainTextResponse(body, media_type=_OUTPUT_FORMATS[output_format])


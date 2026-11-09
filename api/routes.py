import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from api.jobs import JobStore
from models.schemas import (
    JobCreatedResponse,
    JobStatusResponse,
    MessageResponse,
    TranscriptResponse,
)
from transcripy import transcribe
from transcripy.models import Transcript

# create router for api endpoints
router = APIRouter(prefix="/api/v1", tags=["api"])
jobs = JobStore()

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
    _require_format(output_format)

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


@router.post("/jobs", status_code=202, response_model=JobCreatedResponse)
def create_job(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    model: str = Form(default="small"),
    device: str = Form(default="cpu"),
    output_format: str = Query(default="json", alias="format"),
) -> JobCreatedResponse:
    """queue a transcription and return while it is still running"""
    _require_format(output_format)
    saved = _save_upload(file)
    if saved.stat().st_size == 0:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="file is empty")
    job = jobs.create(
        saved,
        model=model.strip() or "small",
        language=_blank_to_none(language),
        device=device.strip() or "cpu",
        output_format=output_format,
    )
    return JobCreatedResponse(id=job.id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """return whether a transcription is queued, running, done, or failed"""
    job = _get_job(job_id)
    return JobStatusResponse(id=job.id, status=job.status, error=job.error)


@router.get("/jobs/{job_id}/result", response_model=None)
def job_result(job_id: str) -> TranscriptResponse | PlainTextResponse:
    """return the transcript once the job has finished"""
    job = _get_job(job_id)
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="job is still running")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=500, detail=job.error or "transcription failed")
    return _render(job.result, job.output_format)


def _get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _require_format(output_format: str) -> None:
    if output_format not in _OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="format must be json, txt, srt, or vtt",
        )


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


"""In-memory transcription jobs for the API process."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from transcripy.models import Transcript
from transcripy.transcribe import transcribe


@dataclass
class Job:
    """One transcription request and its current state."""

    id: str
    status: str
    output_format: str
    error: str | None = None
    result: Transcript | None = None


class JobStore:
    """Run transcriptions on background threads and remember their state.

    Jobs live in this process only. Restarting the API clears them.
    """

    def __init__(self, transcriber=transcribe) -> None:
        self._transcriber = transcriber
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        source: Path,
        *,
        model: str,
        language: str | None,
        device: str,
        output_format: str,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex, status="queued", output_format=output_format)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job.id, source, model, language, device),
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return Job(
                id=job.id,
                status=job.status,
                output_format=job.output_format,
                error=job.error,
                result=job.result,
            )

    def _run(
        self,
        job_id: str,
        source: Path,
        model: str,
        language: str | None,
        device: str,
    ) -> None:
        self._mark(job_id, status="running")
        try:
            result = self._transcriber(
                source,
                model=model,
                language=language,
                device=device,
            )
        except Exception as exc:
            self._mark(job_id, status="failed", error=str(exc))
        else:
            self._mark(job_id, status="done", result=result)
        finally:
            source.unlink(missing_ok=True)

    def _mark(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        result: Transcript | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.error = error
            job.result = result

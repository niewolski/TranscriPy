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
    model: str
    language: str | None
    device: str
    initial_prompt: str | None = None
    hotwords: str | None = None
    speakers: bool = False
    error: str | None = None
    result: Transcript | None = None
    progress: float = 0.0
    segments_done: int = 0


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
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        speakers: bool = False,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            status="queued",
            output_format=output_format,
            model=model,
            language=language,
            device=device,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            speakers=speakers,
        )
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job.id, source),
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
                model=job.model,
                language=job.language,
                device=job.device,
                initial_prompt=job.initial_prompt,
                hotwords=job.hotwords,
                speakers=job.speakers,
                error=job.error,
                result=job.result,
                progress=job.progress,
                segments_done=job.segments_done,
            )

    def _run(self, job_id: str, source: Path) -> None:
        job = self.get(job_id)
        if job is None:
            source.unlink(missing_ok=True)
            return
        self._mark(job_id, status="running")
        try:
            result = self._transcriber(
                source,
                model=job.model,
                language=job.language,
                device=job.device,
                initial_prompt=job.initial_prompt,
                hotwords=job.hotwords,
                speakers=job.speakers,
                on_progress=lambda fraction, count: self._set_progress(
                    job_id,
                    fraction,
                    count,
                ),
            )
        except Exception as exc:
            self._mark(job_id, status="failed", error=str(exc))
        else:
            self._mark(job_id, status="done", result=result)
        finally:
            source.unlink(missing_ok=True)

    def _set_progress(self, job_id: str, fraction: float, segments_done: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.progress = min(max(float(fraction), 0.0), 1.0)
            job.segments_done = segments_done

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
            if status == "done" and result is not None:
                job.progress = 1.0
                job.segments_done = len(result.segments)

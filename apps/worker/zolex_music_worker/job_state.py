from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .errors import WorkerError
from .utils import atomic_write_json


class CancelledError(WorkerError):
    pass


class JobJournal:
    def __init__(self, job_dir: Path, job_id: str) -> None:
        self.job_dir = job_dir
        self.job_id = job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.job_dir / "status.json"
        self.events_path = self.job_dir / "events.jsonl"
        self.cancel_path = self.job_dir / "CANCEL"
        self._write_lock = Lock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def update(self, state: str, **details: Any) -> None:
        payload = {"job_id": self.job_id, "state": state, "updated_at": self._now(), **details}
        with self._write_lock:
            atomic_write_json(self.status_path, payload)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def check_cancelled(self) -> None:
        if self.cancel_path.exists():
            self.update("cancelled")
            raise CancelledError(f"Job {self.job_id} was cancelled")

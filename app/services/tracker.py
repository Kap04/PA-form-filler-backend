from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


class TrackerStore:
    def __init__(self, tracker_path: str | Path) -> None:
        self.tracker_path = Path(tracker_path)
        self.lock = Lock()
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.tracker_path.exists():
            self.tracker_path.write_text("[]", encoding="utf-8")

    def add_entry(self, **payload: Any) -> dict[str, Any]:
        with self.lock:
            entries = self._read_all()
            entry = {
                "job_id": payload.get("job_id") or str(uuid4()),
                "status": payload.get("status", "completed"),
                "template_name": payload.get("template_name", ""),
                "created_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            entries.append(entry)
            self.tracker_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            return entry

    def list_entries(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._read_all()

    def _read_all(self) -> list[dict[str, Any]]:
        content = self.tracker_path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)

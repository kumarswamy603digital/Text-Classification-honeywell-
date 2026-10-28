"""Optional prediction history stored in SQLite (stdlib sqlite3).

Privacy policy: raw input text is NOT stored unless HISTORY_STORE_TEXT=true
is explicitly configured. Records hold only id, timestamp, model, label and
confidence. The core application works fully with history disabled.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_id TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL,
    input_text TEXT
);
"""


class HistoryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._db_path: Path = self.settings.history_db
        if self.enabled:
            self._init_db()

    @property
    def enabled(self) -> bool:
        return self.settings.enable_history

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(self, prediction: dict) -> None:
        if not self.enabled:
            return
        row = (
            prediction["prediction_id"],
            prediction["timestamp"],
            prediction["model_id"],
            prediction["predicted_label"],
            prediction.get("confidence"),
            prediction.get("input") if self.settings.history_store_text else None,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions "
                "(prediction_id, created_at, model_id, predicted_label, confidence, input_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )

    def list_recent(self, limit: int = 50) -> list[dict]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT prediction_id, created_at, model_id, predicted_label, confidence "
                "FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def total(self) -> int:
        if not self.enabled:
            return 0
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def get(self, prediction_id: str) -> dict | None:
        if not self.enabled:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prediction_id, created_at, model_id, predicted_label, confidence "
                "FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete(self, prediction_id: str) -> bool:
        if not self.enabled:
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM predictions WHERE prediction_id = ?", (prediction_id,)
            )
            return cur.rowcount > 0

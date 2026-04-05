"""
src/database.py
----------------
SQLite database layer for SmartRoad report persistence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "smartroad_reports.db"


@dataclass
class ReportRow:
    id: int
    report_id: str
    location_description: str
    latitude: float | None
    longitude: float | None
    issue_type: str
    severity_score: int
    severity_level: str
    status: str
    created_at: str
    updated_at: str
    report_text: str


class DatabaseManager:
    """Simple SQLite manager for SmartRoad incident reports."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    location_description TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    issue_type TEXT NOT NULL,
                    severity_score INTEGER NOT NULL,
                    severity_level TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    report_text TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def insert_report(
        self,
        report_id: str,
        location_description: str,
        latitude: float | None,
        longitude: float | None,
        issue_type: str,
        severity_score: int,
        severity_level: str,
        report_text: str,
        status: str = "open",
    ) -> int:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reports (
                    report_id, location_description, latitude, longitude,
                    issue_type, severity_score, severity_level, status,
                    created_at, updated_at, report_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    location_description,
                    latitude,
                    longitude,
                    issue_type,
                    severity_score,
                    severity_level,
                    status,
                    now,
                    now,
                    report_text,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_reports(self, status: str | None = None) -> list[ReportRow]:
        query = "SELECT * FROM reports"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [ReportRow(**dict(row)) for row in rows]

    def get_open_reports(self) -> list[ReportRow]:
        return self.get_reports(status="open")

    def resolve_report(self, report_id: str) -> None:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET status = 'resolved', updated_at = ? WHERE report_id = ?",
                (now, report_id),
            )
            conn.commit()

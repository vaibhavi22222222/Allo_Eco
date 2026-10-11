"""Additive optimizer result storage; does not alter input/version schemas."""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from .repository import init_db, get_dataset_version


def save_optimization_run(
    dataset_version_id, request, response, db_path=None, run_id=None
):
    path = init_db(db_path)
    version = get_dataset_version(dataset_version_id, path)
    if not version or version["status"] != "accepted":
        raise ValueError(
            "Optimization runs require an accepted, complete input version"
        )
    identifier = run_id or "al-" + uuid4().hex[:12]
    # JSON serialization also rejects nonfinite values.
    payload = json.loads(json.dumps(response, allow_nan=False))
    payload["run_id"] = identifier
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""CREATE TABLE IF NOT EXISTS optimization_runs (
            run_id TEXT PRIMARY KEY,
            dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id),
            created_at TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL)""")
        conn.execute(
            "INSERT INTO optimization_runs VALUES (?,?,?,?,?)",
            (
                identifier,
                dataset_version_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(request, allow_nan=False),
                json.dumps(payload, allow_nan=False),
            ),
        )
    return payload


def load_optimization_run(run_id, db_path=None):
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='optimization_runs'"
        ).fetchone()
        row = (
            conn.execute(
                "SELECT response_json FROM optimization_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if exists
            else None
        )
    if not row:
        raise KeyError(f"Unknown optimization run: {run_id}")
    return json.loads(row[0])

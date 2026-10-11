"""Dataset access for the application, built ONLY on the existing data module.

Three jobs:

1. ``active_dataset`` - find the newest *accepted* dataset version in SQLite
   (``storage.repository.list_dataset_versions``) and load its four tables
   (``storage.repository.load_dataset_from_db``).
2. ``ensure_seeded`` - if the database is empty, ingest the team's synthetic
   pack with the existing ``services.ingestion.ingest_all``.
3. ``ingest_table`` - support POST /ingest. The guide uploads ONE table at a
   time, but the data module's single-table ingestion (``ingest_dataset``)
   creates a *partial* version that the optimizer cannot use (it requires an
   ``accepted`` four-table version). So we take the active version's other
   three tables, swap in the uploaded table, and run the unchanged
   ``ingest_all`` on that folder. All validation rules stay the data module's.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.bootstrap import ENGINE_DIR  # noqa: F401  (adds carbonwise-data to sys.path)
from services.ingestion import DataLoadError, DatasetIngestionError, ingest_all
from services.validation import FILE_NAMES, SCHEMAS
from storage.repository import list_dataset_versions, load_dataset_from_db


class NoDatasetError(RuntimeError):
    """No accepted dataset exists and none could be seeded."""


@dataclass
class ActiveDataset:
    """The tables the optimizer will use, plus their version id."""

    dataset_version_id: str
    tables: dict[str, pd.DataFrame]


def latest_accepted_version(db_path: Optional[Path]) -> Optional[str]:
    """Newest accepted dataset version id, or None."""
    for row in list_dataset_versions(db_path):  # already newest first
        if row.get("status") == "accepted":
            return row["dataset_version_id"]
    return None


def ensure_seeded(db_path: Optional[Path], seed_dir: Optional[Path]) -> Optional[str]:
    """Ingest ``seed_dir`` with the existing loader if no accepted version exists."""
    version = latest_accepted_version(db_path)
    if version or seed_dir is None:
        return version
    result = ingest_all(seed_dir, db_path)  # raises DatasetIngestionError if invalid
    return result["dataset_version_id"]


def active_dataset(db_path: Optional[Path], seed_dir: Optional[Path]) -> ActiveDataset:
    """Load the newest accepted dataset (seeding first if the DB is empty)."""
    version = ensure_seeded(db_path, seed_dir)
    if version is None:
        raise NoDatasetError("No accepted dataset is loaded. Use POST /ingest first.")
    return ActiveDataset(version, load_dataset_from_db(version, db_path))


def _rejected_rows(report: dict[str, Any], table: str) -> int:
    """Count distinct rows of ``table`` named in error issues (0 if row-less errors)."""
    rows = {e.get("row_id") for e in report.get("errors", [])
            if e.get("table") == table and e.get("row_id") is not None}
    return len(rows)


def ingest_table(
    table: str,
    csv_bytes: bytes,
    db_path: Optional[Path],
    seed_dir: Optional[Path],
) -> dict[str, Any]:
    """Replace one table and re-ingest the full dataset with ``ingest_all``.

    Returns a dict matching ``app.contracts.IngestResponse``. Validation is
    all-or-nothing in the data module, so a rejected upload loads 0 rows.

    Raises:
        ValueError: unknown table name or unreadable CSV (API answers 422).
    """
    if table not in SCHEMAS:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(SCHEMAS)}")
    if not csv_bytes.strip():
        raise ValueError("Uploaded file is empty")

    # The other three tables come from the current accepted version, if any.
    current: dict[str, pd.DataFrame] = {}
    version = ensure_seeded(db_path, seed_dir)
    if version:
        current = load_dataset_from_db(version, db_path)

    with tempfile.TemporaryDirectory(prefix="carbonwise-ingest-") as tmp:
        folder = Path(tmp)
        (folder / FILE_NAMES[table]).write_bytes(csv_bytes)
        for name, frame in current.items():
            if name != table:
                # Dates were loaded as datetimes; write them back as YYYY-MM-DD.
                frame.to_csv(folder / FILE_NAMES[name], index=False, date_format="%Y-%m-%d")
        try:
            result = ingest_all(folder, db_path)
        except DatasetIngestionError as exc:
            uploaded = _count_rows(folder / FILE_NAMES[table])
            return {
                "rows_loaded": 0,
                "rows_rejected": _rejected_rows(exc.report, table) or uploaded,
                "status": "rejected", "table": table,
                "dataset_version_id": None,
                "errors": exc.report.get("errors", []),
                "warnings": exc.report.get("warnings", []),
            }
        except DataLoadError as exc:
            raise ValueError(str(exc)) from exc

    return {
        "rows_loaded": int(result["row_counts"][table]),
        "rows_rejected": 0, "status": "accepted", "table": table,
        "dataset_version_id": result["dataset_version_id"],
        "errors": [], "warnings": result["validation_report"].get("warnings", []),
    }


def _count_rows(path: Path) -> int:
    """Data rows in a CSV, used only to report how many rows were rejected."""
    try:
        return len(pd.read_csv(path))
    except (OSError, ValueError, pd.errors.ParserError):
        return 0

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from services.validation import (
    FILE_NAMES,
    SCHEMAS,
    canonicalize_table,
    normalize_columns,
    validate_data_quality,
    validate_schema,
    validate_table_data_quality,
)
from storage import repository


class DataLoadError(RuntimeError):
    """Raised when an input CSV cannot be read safely."""


class DatasetIngestionError(RuntimeError):
    """Raised when validation rejects input; includes its persisted diagnostics."""

    def __init__(self, message: str, report: dict[str, Any], dataset_version_id: str):
        self.report = report
        self.dataset_version_id = dataset_version_id
        super().__init__(message)


def load_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_file():
        raise DataLoadError(f"CSV file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DataLoadError(f"Could not read CSV header from {path}: {exc}") from exc

    if not header:
        raise DataLoadError(f"CSV file has no header row: {path}")
    if len(set(header)) != len(header):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        raise DataLoadError(
            f"CSV file contains duplicate column headers in {path}: "
            + ", ".join(repr(name) for name in duplicates)
        )

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataLoadError(f"Could not read CSV file {path}: {exc}") from exc


def load_dataset(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    if not root.is_dir():
        raise DataLoadError(f"Dataset directory does not exist: {root}")

    loaded: dict[str, pd.DataFrame] = {}
    for table_name, filename in FILE_NAMES.items():
        file_path = root / filename
        try:
            loaded[table_name] = normalize_columns(load_csv(file_path))
        except ValueError as exc:
            raise DataLoadError(f"Invalid column headers in {file_path}: {exc}") from exc
    return loaded


def _merge_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    errors = [issue for report in reports for issue in report.get("errors", [])]
    warnings = [issue for report in reports for issue in report.get("warnings", [])]
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": "Dataset passed validation."
        if not errors
        else f"Dataset rejected with {len(errors)} validation error(s).",
    }


def _metadata(
    tables: dict[str, pd.DataFrame],
    source_id: str | None,
    summary: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "summary": summary,
    }


def _persist_validation_failure(
    metadata: dict[str, Any],
    report: dict[str, Any],
    db_path: str | Path | None,
) -> str:
    version_id = repository.record_failed_dataset_version(metadata, report, db_path)
    return version_id


def ingest_dataset(
    table_name: str,
    df: pd.DataFrame,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one table as a partial, non-solver-ready version."""
    if table_name not in SCHEMAS:
        raise ValueError(f"Unknown table name: {table_name!r}.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    schema_report = validate_schema(df, table_name)
    if schema_report["valid"]:
        canonical = canonicalize_table(df, table_name)
        quality_report = validate_table_data_quality(table_name, canonical)
    else:
        canonical = None
        quality_report = {"valid": True, "errors": [], "warnings": []}
    report = _merge_reports(schema_report, quality_report)
    tables = {table_name: canonical if canonical is not None else df}
    metadata = _metadata(tables, source_id=f"single-table:{table_name}")

    if not report["valid"]:
        version_id = _persist_validation_failure(metadata, report, db_path)
        raise DatasetIngestionError(
            f"Validation rejected table {table_name!r}.", report, version_id
        )

    version_id = repository.create_dataset_version(metadata, db_path)
    try:
        repository.save_dataset(
            {table_name: canonical},
            version_id,
            db_path,
            additional_logs=report["warnings"],
        )
    except (sqlite3.Error, ValueError, TypeError) as exc:
        failure = _merge_reports(
            report,
            {
                "errors": [
                    {
                        "severity": "error",
                        "table": table_name,
                        "row_id": None,
                        "rule": "persistence",
                        "message": f"Could not persist input table: {exc}",
                    }
                ],
                "warnings": [],
            },
        )
        repository.mark_dataset_failed(version_id, failure, db_path)
        raise DatasetIngestionError(
            f"Could not persist table {table_name!r}.", failure, version_id
        ) from exc

    return {
        "table_name": table_name,
        "row_count": len(canonical),
        "dataset_version_id": version_id,
        "validation_result": report,
    }


def ingest_all(
    data_dir: str | Path,
    db_path: str | Path | None = None,
    planning_start: str | None = None,
) -> dict[str, Any]:
    """Load, validate and persist a complete four-table version atomically.

    Returns canonical pandas DataFrames, a real persisted version identifier,
    row counts and the validation report. Rejected input raises
    DatasetIngestionError and leaves a rejected version plus its diagnostics.
    """
    root = Path(data_dir)
    try:
        raw_tables = load_dataset(root)
    except DataLoadError as exc:
        report = {
            "valid": False,
            "errors": [
                {
                    "severity": "error",
                    "table": None,
                    "row_id": None,
                    "rule": "csv_load",
                    "message": str(exc),
                }
            ],
            "warnings": [],
            "summary": "Dataset rejected because an input file could not be loaded.",
        }
        version_id = _persist_validation_failure(
            _metadata({}, source_id=str(root), summary=report["summary"]),
            report,
            db_path,
        )
        raise DatasetIngestionError(
            "Could not load the complete dataset.", report, version_id
        ) from exc

    canonical_tables: dict[str, pd.DataFrame] = {}
    schema_reports: list[dict[str, Any]] = []
    for table_name in SCHEMAS:
        report = validate_schema(raw_tables[table_name], table_name)
        schema_reports.append(report)
        if report["valid"]:
            canonical_tables[table_name] = canonicalize_table(
                raw_tables[table_name], table_name
            )

    schema_report = _merge_reports(*schema_reports)
    if schema_report["valid"]:
        quality_report = validate_data_quality(canonical_tables, planning_start)
    else:
        quality_report = {"valid": True, "errors": [], "warnings": []}
    report = _merge_reports(schema_report, quality_report)
    report["summary"] = (
        "Dataset passed schema and cross-table validation."
        if report["valid"]
        else f"Dataset rejected with {len(report['errors'])} validation error(s)."
    )
    metadata = _metadata(
        canonical_tables if len(canonical_tables) == len(SCHEMAS) else raw_tables,
        source_id=str(root.resolve()),
        summary=report["summary"],
    )
    metadata["planning_start"] = planning_start

    if not report["valid"]:
        version_id = _persist_validation_failure(metadata, report, db_path)
        raise DatasetIngestionError(
            "Dataset validation failed.", report, version_id
        )

    version_id = repository.create_dataset_version(metadata, db_path)
    try:
        repository.save_dataset(
            canonical_tables,
            version_id,
            db_path,
            additional_logs=report["warnings"],
        )
    except (sqlite3.Error, ValueError, TypeError) as exc:
        failure = _merge_reports(
            report,
            {
                "errors": [
                    {
                        "severity": "error",
                        "table": None,
                        "row_id": None,
                        "rule": "persistence",
                        "message": f"Database transaction failed: {exc}",
                    }
                ],
                "warnings": [],
            },
        )
        failure["summary"] = "Dataset rejected because database persistence failed."
        repository.mark_dataset_failed(version_id, failure, db_path)
        raise DatasetIngestionError(
            "Dataset persistence failed; no input rows were committed.",
            failure,
            version_id,
        ) from exc

    return {
        "tables": canonical_tables,
        "dataset_version_id": version_id,
        "row_counts": {name: len(frame) for name, frame in canonical_tables.items()},
        "validation_report": report,
    }

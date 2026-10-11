from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from services.validation import SCHEMAS, canonicalize_table

BASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
TABLE_ORDER = ("orders", "plants", "transport_duties", "capacity_events")


def default_db_path() -> Path:
    """Return the configurable persistent database path."""
    configured = os.environ.get("CARBONWISE_DB_PATH")
    return Path(configured).expanduser() if configured else BASE_DIR / "data" / "carbonwise.sqlite3"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _managed_connection(db_path: str | Path):
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else default_db_path()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with _managed_connection(path) as connection:
        connection.executescript(schema_sql)
    return path


def _new_version_id() -> str:
    return f"cw-{uuid4()}"


def _metadata_value(metadata: dict[str, Any], name: str, default: int = 0) -> int:
    value = metadata.get(name, default)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _insert_version(
    connection: sqlite3.Connection,
    metadata: dict[str, Any],
    dataset_version_id: str | None = None,
    status: str = "processing",
    validation_status: str = "pending",
) -> str:
    version_id = dataset_version_id or _new_version_id()
    counts = metadata.get("row_counts", {})
    connection.execute(
        """
        INSERT INTO dataset_versions (
            dataset_version_id, created_at, status, source_id,
            orders_row_count, plants_row_count, transport_duties_row_count,
            capacity_events_row_count, validation_status, summary, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            _timestamp(),
            status,
            metadata.get("source_id"),
            _metadata_value(counts, "orders"),
            _metadata_value(counts, "plants"),
            _metadata_value(counts, "transport_duties"),
            _metadata_value(counts, "capacity_events"),
            validation_status,
            metadata.get("summary"),
            json.dumps(metadata, default=str, sort_keys=True),
        ),
    )
    return version_id


def create_dataset_version(metadata: dict[str, Any], db_path: str | Path | None = None) -> str:
    """Create a pending version. Use save_dataset or mark_dataset_failed next."""
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        return _insert_version(connection, metadata)


def _sql_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_dataset(
    tables: dict[str, pd.DataFrame],
    dataset_version_id: str,
    db_path: str | Path | None = None,
    additional_logs: list[dict[str, Any]] | None = None,
) -> None:
    """Atomically store one or more canonical input tables for an existing version."""
    unknown = set(tables) - set(TABLE_ORDER)
    if unknown:
        raise ValueError(f"Unknown table names: {', '.join(sorted(unknown))}.")
    if not tables:
        raise ValueError("At least one table is required.")

    canonical = {
        table_name: canonicalize_table(frame, table_name)
        for table_name, frame in tables.items()
    }
    path = init_db(db_path)
    columns_by_table = {
        table_name: tuple(SCHEMAS[table_name])
        for table_name in TABLE_ORDER
    }

    with _managed_connection(path) as connection:
        version = connection.execute(
            "SELECT status FROM dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
        if version is None:
            raise ValueError(f"Dataset version {dataset_version_id!r} does not exist.")
        if version["status"] != "processing":
            raise ValueError(
                f"Dataset version {dataset_version_id!r} is not in processing state."
            )

        row_counts: dict[str, int] = {name: 0 for name in TABLE_ORDER}
        for table_name in TABLE_ORDER:
            frame = canonical.get(table_name)
            if frame is None:
                continue
            columns = columns_by_table[table_name]
            placeholders = ", ".join("?" for _ in range(len(columns) + 2))
            sql_columns = ", ".join(("dataset_version_id", "row_order", *columns))
            values = [
                (dataset_version_id, row_order, *(_sql_value(value) for value in row))
                for row_order, row in enumerate(
                    frame.loc[:, list(columns)].itertuples(index=False, name=None)
                )
            ]
            connection.executemany(
                f"INSERT INTO {table_name} ({sql_columns}) VALUES ({placeholders})",
                values,
            )
            row_counts[table_name] = len(frame)

        final_status = "accepted" if set(canonical) == set(TABLE_ORDER) else "partial"
        validation_status = "valid"
        connection.execute(
            """
            UPDATE dataset_versions
            SET status = ?, validation_status = ?,
                orders_row_count = ?, plants_row_count = ?,
                transport_duties_row_count = ?, capacity_events_row_count = ?
            WHERE dataset_version_id = ?
            """,
            (
                final_status,
                validation_status,
                row_counts["orders"],
                row_counts["plants"],
                row_counts["transport_duties"],
                row_counts["capacity_events"],
                dataset_version_id,
            ),
        )
        _insert_logs(
            connection,
            dataset_version_id,
            [
                {
                    "severity": "info",
                    "table": name,
                    "rule": "table_persisted",
                    "message": f"Persisted {count} rows.",
                }
                for name, count in row_counts.items()
                if name in canonical
            ]
            + (additional_logs or []),
        )


def _insert_logs(
    connection: sqlite3.Connection,
    dataset_version_id: str | None,
    issues: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO ingestion_logs (
            dataset_version_id, created_at, severity, table_name, row_id,
            rule_name, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dataset_version_id,
                _timestamp(),
                issue["severity"],
                issue.get("table"),
                issue.get("row_id"),
                issue["rule"],
                issue["message"],
            )
            for issue in issues
        ],
    )


def save_ingestion_log(
    dataset_version_id: str | None,
    severity: str,
    table_name: str | None,
    row_id: str | None,
    rule_name: str,
    message: str,
    db_path: str | Path | None = None,
) -> None:
    if severity not in {"info", "warning", "error"}:
        raise ValueError("severity must be 'info', 'warning', or 'error'.")
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        _insert_logs(
            connection,
            dataset_version_id,
            [
                {
                    "severity": severity,
                    "table": table_name,
                    "row_id": row_id,
                    "rule": rule_name,
                    "message": message,
                }
            ],
        )


def mark_dataset_failed(
    dataset_version_id: str,
    report: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    """Mark a pending version rejected and persist its diagnostics."""
    path = init_db(db_path)
    issues = [*report.get("errors", []), *report.get("warnings", [])]
    with _managed_connection(path) as connection:
        current = connection.execute(
            "SELECT status FROM dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
        if current is None:
            raise ValueError(f"Dataset version {dataset_version_id!r} does not exist.")
        connection.execute(
            """
            UPDATE dataset_versions
            SET status = 'rejected', validation_status = 'invalid', summary = ?
            WHERE dataset_version_id = ?
            """,
            (report.get("summary", "Dataset ingestion failed."), dataset_version_id),
        )
        _insert_logs(connection, dataset_version_id, issues)


def record_failed_dataset_version(
    metadata: dict[str, Any],
    report: dict[str, Any],
    db_path: str | Path | None = None,
) -> str:
    """Persist a rejected version and its diagnostics without any input rows."""
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        version_id = _insert_version(
            connection,
            metadata,
            status="rejected",
            validation_status="invalid",
        )
        issues = [*report.get("errors", []), *report.get("warnings", [])]
        _insert_logs(connection, version_id, issues)
        return version_id


def get_dataset_version(
    dataset_version_id: str, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        row = connection.execute(
            "SELECT * FROM dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def list_dataset_versions(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM dataset_versions ORDER BY created_at DESC, dataset_version_id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def load_dataset_from_db(
    dataset_version_id: str, db_path: str | Path | None = None
) -> dict[str, pd.DataFrame]:
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        version = connection.execute(
            "SELECT status FROM dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
        if version is None:
            raise ValueError(f"Dataset version {dataset_version_id!r} does not exist.")
        if version["status"] != "accepted":
            raise ValueError(
                f"Dataset version {dataset_version_id!r} is not a complete accepted dataset."
            )

        result: dict[str, pd.DataFrame] = {}
        for table_name in TABLE_ORDER:
            columns = list(SCHEMAS[table_name])
            selected = ", ".join(columns)
            frame = pd.read_sql_query(
                f"""
                SELECT {selected} FROM {table_name}
                WHERE dataset_version_id = ?
                ORDER BY row_order
                """,
                connection,
                params=(dataset_version_id,),
            )
            for column, kind in SCHEMAS[table_name].items():
                if kind == "string":
                    frame[column] = frame[column].astype("string")
                elif kind == "int":
                    frame[column] = frame[column].astype("int64")
                elif kind == "float":
                    frame[column] = frame[column].astype("float64")
                else:
                    frame[column] = pd.to_datetime(frame[column], errors="raise")
            result[table_name] = frame
    return result


def list_ingestion_logs(
    dataset_version_id: str | None = None, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    path = init_db(db_path)
    with _managed_connection(path) as connection:
        if dataset_version_id is None:
            rows = connection.execute(
                "SELECT * FROM ingestion_logs ORDER BY ingestion_log_id"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM ingestion_logs
                WHERE dataset_version_id = ?
                ORDER BY ingestion_log_id
                """,
                (dataset_version_id,),
            ).fetchall()
    return [dict(row) for row in rows]

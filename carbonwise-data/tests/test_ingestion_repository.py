from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.ingestion import DatasetIngestionError, ingest_all, ingest_dataset
from storage import repository

MAIN = ROOT / "synthetic_data" / "main"


def test_successful_ingestion_persists_and_reloads_dataframes(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "carbonwise.sqlite3"
    result = ingest_all(MAIN, db_path)

    version = repository.get_dataset_version(result["dataset_version_id"], db_path)
    assert version is not None
    assert version["status"] == "accepted"
    assert version["validation_status"] == "valid"
    assert result["row_counts"] == {
        "orders": 125,
        "plants": 13,
        "transport_duties": 30,
        "capacity_events": 3,
    }

    reloaded = repository.load_dataset_from_db(result["dataset_version_id"], db_path)
    for table_name, frame in result["tables"].items():
        pd.testing.assert_frame_equal(frame, reloaded[table_name])


def test_reingestion_creates_isolated_immutable_versions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(MAIN, data_dir)
    db_path = tmp_path / "versions.sqlite3"

    first = ingest_all(data_dir, db_path)
    original_qty = first["tables"]["orders"].loc[0, "qty"]
    changed = pd.read_csv(data_dir / "orders.csv")
    changed.loc[0, "qty"] = int(original_qty) + 1
    changed.to_csv(data_dir / "orders.csv", index=False)
    second = ingest_all(data_dir, db_path)

    assert first["dataset_version_id"] != second["dataset_version_id"]
    earlier = repository.load_dataset_from_db(first["dataset_version_id"], db_path)
    later = repository.load_dataset_from_db(second["dataset_version_id"], db_path)
    assert earlier["orders"].loc[0, "qty"] == original_qty
    assert later["orders"].loc[0, "qty"] == original_qty + 1
    assert len(repository.list_dataset_versions(db_path)) == 2


def test_database_write_failure_rolls_back_all_input_rows_and_logs_failure(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rollback.sqlite3"
    repository.init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER injected_plant_failure
            BEFORE INSERT ON plants
            BEGIN
                SELECT RAISE(ABORT, 'injected plant failure');
            END;
            """
        )

    with pytest.raises(DatasetIngestionError, match="no input rows were committed") as raised:
        ingest_all(MAIN, db_path)

    version_id = raised.value.dataset_version_id
    version = repository.get_dataset_version(version_id, db_path)
    assert version is not None
    assert version["status"] == "rejected"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM orders WHERE dataset_version_id = ?", (version_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM plants WHERE dataset_version_id = ?", (version_id,)
        ).fetchone()[0] == 0
    logs = repository.list_ingestion_logs(version_id, db_path)
    assert any(log["rule_name"] == "persistence" for log in logs)


def test_validation_failure_is_persisted_as_rejected_with_errors(tmp_path: Path) -> None:
    data_dir = tmp_path / "bad-data"
    shutil.copytree(MAIN, data_dir)
    transport = pd.read_csv(data_dir / "transport_duties.csv")
    transport.loc[0, "duty_pct"] = 1.5
    transport.to_csv(data_dir / "transport_duties.csv", index=False)
    db_path = tmp_path / "rejected.sqlite3"

    with pytest.raises(DatasetIngestionError) as raised:
        ingest_all(data_dir, db_path)

    version = repository.get_dataset_version(raised.value.dataset_version_id, db_path)
    assert version is not None
    assert version["status"] == "rejected"
    assert version["validation_status"] == "invalid"
    assert any(
        issue["rule"] == "percentage_range"
        for issue in raised.value.report["errors"]
    )
    assert any(
        log["severity"] == "error"
        for log in repository.list_ingestion_logs(raised.value.dataset_version_id, db_path)
    )


def test_single_table_ingestion_is_partial_and_not_solver_ready(tmp_path: Path) -> None:
    orders = pd.read_csv(MAIN / "orders.csv")
    result = ingest_dataset("orders", orders, tmp_path / "partial.sqlite3")

    version = repository.get_dataset_version(result["dataset_version_id"], tmp_path / "partial.sqlite3")
    assert version is not None
    assert version["status"] == "partial"
    with pytest.raises(ValueError, match="not a complete accepted dataset"):
        repository.load_dataset_from_db(result["dataset_version_id"], tmp_path / "partial.sqlite3")


def test_unknown_table_and_invalid_input_type_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown table"):
        ingest_dataset("unexpected", pd.DataFrame(), tmp_path / "unknown.sqlite3")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ingest_dataset("orders", object(), tmp_path / "type.sqlite3")


def test_warning_logs_are_persisted_with_accepted_version(tmp_path: Path) -> None:
    orders = pd.read_csv(MAIN / "orders.csv")
    orders["optional_upload_note"] = "extra"
    result = ingest_dataset("orders", orders, tmp_path / "warnings.sqlite3")

    logs = repository.list_ingestion_logs(
        result["dataset_version_id"], tmp_path / "warnings.sqlite3"
    )
    assert any(log["rule_name"] == "extra_columns" for log in logs)

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.ingestion import DataLoadError, load_csv, load_dataset
from services.validation import (
    ValidationError,
    canonicalize_table,
    normalize_columns,
    validate_data_quality,
    validate_schema,
    validate_table_data_quality,
)

MAIN = ROOT / "synthetic_data" / "main"


def _tables() -> dict[str, pd.DataFrame]:
    return load_dataset(MAIN)


def test_csv_loading_and_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "small.csv"
    source.write_text("order_id,qty\nO-1,3\n", encoding="utf-8")
    assert load_csv(source).to_dict(orient="records") == [{"order_id": "O-1", "qty": 3}]
    with pytest.raises(DataLoadError, match="does not exist"):
        load_csv(tmp_path / "missing.csv")


def test_csv_duplicate_headers_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    source.write_text("order_id,order_id\nO-1,O-2\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="duplicate column headers"):
        load_csv(source)


def test_column_whitespace_is_trimmed() -> None:
    actual = normalize_columns(pd.DataFrame({" order_id ": ["O-1"], " qty ": [3]}))
    assert list(actual.columns) == ["order_id", "qty"]


def test_normalization_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="creates duplicate names"):
        normalize_columns(pd.DataFrame([["O-1", "O-2"]], columns=["order_id", " order_id "]))


def test_load_dataset_requires_all_four_csvs(tmp_path: Path) -> None:
    shutil.copy(MAIN / "orders.csv", tmp_path / "orders.csv")
    with pytest.raises(DataLoadError, match="does not exist"):
        load_dataset(tmp_path)


def test_missing_required_column_is_reported() -> None:
    result = validate_schema(pd.DataFrame({"order_id": ["O-1"]}), "orders")
    assert not result["valid"]
    assert {issue["rule"] for issue in result["errors"]} == {"required_column"}


@pytest.mark.parametrize(
    ("column", "value"),
    [("qty", "not-a-number"), ("qty", 2.5)],
)
def test_invalid_order_numeric_values_are_rejected(column: str, value: object) -> None:
    row = {"order_id": "O-1", "product": "P1", "qty": 2, "destination": "D1", "due": "2026-02-01"}
    row[column] = value
    result = validate_schema(pd.DataFrame([row]), "orders")
    assert not result["valid"]
    assert any(issue["rule"] == "value_type" for issue in result["errors"])


def test_invalid_date_is_reported() -> None:
    orders = _tables()["orders"]
    orders.loc[0, "due"] = "not-a-date"
    result = validate_schema(orders, "orders")
    assert not result["valid"]
    assert any(issue["rule"] == "value_type" for issue in result["errors"])


def test_non_iso_date_is_rejected() -> None:
    orders = _tables()["orders"]
    orders.loc[0, "due"] = "01/25/2026"
    result = validate_schema(orders, "orders")
    assert not result["valid"]


def test_duplicate_identifiers_are_rejected() -> None:
    orders = canonicalize_table(_tables()["orders"].iloc[:2], "orders")
    orders.loc[1, "order_id"] = orders.loc[0, "order_id"]
    report = validate_table_data_quality("orders", orders)
    assert not report["valid"]
    assert any(issue["rule"] == "unique_key" for issue in report["errors"])


@pytest.mark.parametrize(
    ("table_name", "column", "value"),
    [
        ("orders", "qty", 0),
        ("plants", "weekly_capacity", 0),
        ("plants", "unit_cost", -1.0),
    ],
)
def test_non_positive_quantity_or_invalid_capacity_cost_rejected(
    table_name: str, column: str, value: object
) -> None:
    frame = _tables()[table_name].iloc[:1].copy()
    frame.loc[frame.index[0], column] = value
    report = validate_table_data_quality(table_name, frame)
    assert not report["valid"]
    assert any(issue["rule"] == "numeric_range" for issue in report["errors"])


def test_invalid_duty_percentage_is_rejected() -> None:
    transport = _tables()["transport_duties"].iloc[:1].copy()
    transport.loc[transport.index[0], "duty_pct"] = 1.2
    report = validate_table_data_quality("transport_duties", transport)
    assert not report["valid"]


def test_unknown_plant_reference_is_rejected() -> None:
    tables = _tables()
    tables["capacity_events"].loc[0, "plant"] = "UNKNOWN"
    report = validate_data_quality(tables)
    assert not report["valid"]
    assert any(issue["rule"] == "plant_reference" for issue in report["errors"])


def test_missing_plant_destination_route_is_rejected() -> None:
    tables = _tables()
    tables["transport_duties"] = tables["transport_duties"].iloc[1:].copy()
    report = validate_data_quality(tables)
    assert not report["valid"]
    assert any(issue["rule"] == "complete_route_coverage" for issue in report["errors"])


def test_product_without_eligible_plant_is_a_reported_warning() -> None:
    tables = _tables()
    tables["orders"].loc[0, "product"] = "P-NOT-MADE"
    report = validate_data_quality(tables)
    assert report["valid"]
    assert any(
        issue["rule"] == "product_without_eligible_plant"
        and issue["row_id"] == tables["orders"].loc[0, "order_id"]
        for issue in report["warnings"]
    )


def test_invalid_capacity_event_date_range_is_rejected() -> None:
    events = _tables()["capacity_events"].iloc[:1].copy()
    events.loc[events.index[0], "from_date"] = "2026-03-15"
    events.loc[events.index[0], "to_date"] = "2026-03-01"
    report = validate_table_data_quality("capacity_events", events)
    assert not report["valid"]
    assert any(issue["rule"] == "event_date_range" for issue in report["errors"])


def test_extra_columns_are_accepted_with_warning_and_omitted() -> None:
    orders = _tables()["orders"].assign(source_note="extension")
    report = validate_schema(orders, "orders")
    assert report["valid"]
    assert any(issue["rule"] == "extra_columns" for issue in report["warnings"])
    assert "source_note" not in canonicalize_table(orders, "orders").columns


def test_planning_start_is_optional_but_enforced_when_given() -> None:
    tables = _tables()
    tables["orders"].loc[0, "due"] = "2025-12-31"
    assert validate_data_quality(tables)["valid"]
    report = validate_data_quality(tables, planning_start="2026-01-01")
    assert not report["valid"]
    assert any(issue["rule"] == "due_after_planning_start" for issue in report["errors"])

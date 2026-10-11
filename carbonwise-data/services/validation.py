from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd


SCHEMAS: dict[str, dict[str, str]] = {
    "orders": {
        "order_id": "string",
        "product": "string",
        "qty": "int",
        "destination": "string",
        "due": "date",
    },
    "plants": {
        "plant": "string",
        "product": "string",
        "weekly_capacity": "int",
        "unit_cost": "float",
        "energy_kwh_per_unit": "float",
        "carbon_kg_per_kwh": "float",
    },
    "transport_duties": {
        "plant": "string",
        "destination": "string",
        "transport_cost_per_unit": "float",
        "transit_days": "int",
        "duty_pct": "float",
        "carbon_kg_per_unit": "float",
    },
    "capacity_events": {
        "event_id": "string",
        "plant": "string",
        "from_date": "date",
        "to_date": "date",
        "capacity_loss_pct": "float",
    },
}

FILE_NAMES = {
    "orders": "orders.csv",
    "plants": "plants.csv",
    "transport_duties": "transport_duties.csv",
    "capacity_events": "capacity_events.csv",
}


class ValidationError(ValueError):
    """Raised when supplied data fails its documented validation rules."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        messages = "; ".join(issue["message"] for issue in report["errors"])
        super().__init__(messages or "Dataset validation failed.")


def _issue(
    severity: str,
    table: str | None,
    rule: str,
    message: str,
    row_id: Any = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "table": table,
        "row_id": None if pd.isna(row_id) else str(row_id),
        "rule": rule,
        "message": message,
    }


def _result(errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim header whitespace and reject duplicate or colliding labels."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    labels = list(df.columns)
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("All column names must be strings.")
    normalized = [label.strip() for label in labels]
    if any(not label for label in normalized):
        raise ValueError("Column names cannot be empty after trimming whitespace.")
    if len(set(normalized)) != len(normalized):
        duplicates = sorted({label for label in normalized if normalized.count(label) > 1})
        raise ValueError(
            "Column normalization creates duplicate names: " + ", ".join(duplicates)
        )

    result = df.copy()
    result.columns = normalized
    return result


def validate_schema(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    """Check required fields and whether values can be safely converted."""
    if table_name not in SCHEMAS:
        raise ValueError(f"Unknown table name: {table_name!r}.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        normalized = normalize_columns(df)
    except (TypeError, ValueError) as exc:
        return _result(
            [_issue("error", table_name, "column_normalization", str(exc))], warnings
        )

    required = SCHEMAS[table_name]
    missing = [column for column in required if column not in normalized.columns]
    for column in missing:
        errors.append(
            _issue(
                "error",
                table_name,
                "required_column",
                f"Required column {column!r} is missing.",
            )
        )
    extras = [column for column in normalized.columns if column not in required]
    if extras:
        warnings.append(
            _issue(
                "warning",
                table_name,
                "extra_columns",
                "Unsupported extra columns are accepted but omitted from the canonical "
                f"stored dataset: {', '.join(extras)}.",
            )
        )

    for column, kind in required.items():
        if column not in normalized.columns:
            continue
        series = normalized[column]
        if kind == "string":
            invalid = series.isna() | series.astype("string").str.strip().eq("")
        elif kind in {"int", "float"}:
            converted = pd.to_numeric(series, errors="coerce")
            invalid = series.isna() | converted.isna() | ~np.isfinite(converted)
            if kind == "int":
                invalid |= converted.notna() & (converted % 1 != 0)
        else:
            converted = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
            invalid = series.isna() | converted.isna()

        for index in normalized.index[invalid.fillna(True)]:
            row_id = normalized.at[index, _id_column(table_name)] if _id_column(table_name) in normalized else index
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "value_type",
                    f"Column {column!r} contains a null or invalid {kind} value.",
                    row_id,
                )
            )

    return _result(errors, warnings)


def _id_column(table_name: str) -> str:
    return {
        "orders": "order_id",
        "plants": "plant",
        "transport_duties": "plant",
        "capacity_events": "event_id",
    }[table_name]


def canonicalize_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Return only official fields, trimmed and converted to stable pandas types."""
    report = validate_schema(df, table_name)
    if not report["valid"]:
        raise ValidationError(report)

    normalized = normalize_columns(df)
    columns = list(SCHEMAS[table_name])
    result = normalized.loc[:, columns].copy()
    for column, kind in SCHEMAS[table_name].items():
        if kind == "string":
            result[column] = result[column].astype("string").str.strip()
        elif kind == "int":
            result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
        elif kind == "float":
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
        else:
            result[column] = pd.to_datetime(
                result[column], errors="raise", format="%Y-%m-%d"
            ).dt.normalize()
    return result.reset_index(drop=True)


def validate_table_data_quality(table_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Validate rules that can be checked without the other three input tables."""
    canonical = canonicalize_table(frame, table_name)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    unique_keys = {
        "orders": ("order_id",),
        "plants": ("plant", "product"),
        "transport_duties": ("plant", "destination"),
        "capacity_events": ("event_id",),
    }[table_name]
    duplicate_rows = canonical.duplicated(list(unique_keys), keep=False)
    for row in canonical.loc[duplicate_rows].itertuples(index=False):
        key = ", ".join(str(getattr(row, key)) for key in unique_keys)
        errors.append(
            _issue(
                "error",
                table_name,
                "unique_key",
                f"Duplicate key ({key}).",
                getattr(row, _id_column(table_name)),
            )
        )

    checks_by_table: dict[str, dict[str, tuple[float, float | None]]] = {
        "orders": {"qty": (1, None)},
        "plants": {
            "weekly_capacity": (1, None),
            "unit_cost": (0, None),
            "energy_kwh_per_unit": (0, None),
            "carbon_kg_per_kwh": (0, None),
        },
        "transport_duties": {
            "transport_cost_per_unit": (0, None),
            "transit_days": (0, None),
            "duty_pct": (0, 1),
            "carbon_kg_per_unit": (0, None),
        },
        "capacity_events": {"capacity_loss_pct": (0, 1)},
    }
    checks = checks_by_table[table_name]
    for column, (minimum, maximum) in checks.items():
        invalid = canonical[column] < minimum
        if maximum is not None:
            invalid |= canonical[column] > maximum
        for row in canonical.loc[invalid].itertuples(index=False):
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "numeric_range",
                    f"{column} must be in the range [{minimum}, "
                    f"{maximum if maximum is not None else 'infinity'}].",
                    getattr(row, _id_column(table_name)),
                )
            )

    if table_name == "capacity_events":
        invalid_ranges = canonical["from_date"] > canonical["to_date"]
        for event_id in canonical.loc[invalid_ranges, "event_id"]:
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "event_date_range",
                    "Capacity event from_date must be on or before to_date.",
                    event_id,
                )
            )
    return _result(errors, warnings)


def validate_data_quality(
    tables: dict[str, pd.DataFrame], planning_start: str | None = None
) -> dict[str, Any]:
    """Validate row-level rules and relationships across all four inputs."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_tables = [name for name in SCHEMAS if name not in tables]
    for table_name in missing_tables:
        errors.append(
            _issue("error", table_name, "required_table", f"Table {table_name!r} is missing.")
        )
    if missing_tables:
        return _result(errors, warnings)

    for table_name, frame in tables.items():
        if table_name not in SCHEMAS:
            errors.append(
                _issue("error", table_name, "unknown_table", f"Unknown table {table_name!r}.")
            )
            continue
        report = validate_schema(frame, table_name)
        errors.extend(report["errors"])
        warnings.extend(report["warnings"])
    if errors:
        return _result(errors, warnings)

    orders = {name: canonicalize_table(tables[name], name) for name in SCHEMAS}
    order_df = orders["orders"]
    plant_df = orders["plants"]
    transport_df = orders["transport_duties"]
    events_df = orders["capacity_events"]

    for table_name, frame, key in (
        ("orders", order_df, "order_id"),
        ("capacity_events", events_df, "event_id"),
    ):
        duplicates = frame.loc[frame[key].duplicated(keep=False), key].drop_duplicates()
        for value in duplicates:
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "unique_identifier",
                    f"Identifier {value!r} is duplicated.",
                    value,
                )
            )

    if order_df.empty:
        errors.append(_issue("error", "orders", "non_empty", "Orders cannot be empty."))
    if plant_df.empty:
        errors.append(_issue("error", "plants", "non_empty", "Plants cannot be empty."))
    if transport_df.empty:
        errors.append(
            _issue(
                "error",
                "transport_duties",
                "non_empty",
                "Transport and duties cannot be empty.",
            )
        )

    duplicate_plant_products = plant_df.duplicated(["plant", "product"], keep=False)
    for row in plant_df.loc[duplicate_plant_products].itertuples(index=False):
        errors.append(
            _issue(
                "error",
                "plants",
                "unique_plant_product",
                f"Plant/product pair ({row.plant!r}, {row.product!r}) is duplicated.",
                row.plant,
            )
        )
    duplicate_routes = transport_df.duplicated(["plant", "destination"], keep=False)
    for row in transport_df.loc[duplicate_routes].itertuples(index=False):
        errors.append(
            _issue(
                "error",
                "transport_duties",
                "unique_plant_destination",
                f"Plant/destination pair ({row.plant!r}, {row.destination!r}) is duplicated.",
                row.plant,
            )
        )

    for column, minimum, label in (
        ("qty", 1, "Order quantity"),
    ):
        invalid = order_df[column] < minimum
        for value in order_df.loc[invalid, "order_id"]:
            errors.append(
                _issue(
                    "error",
                    "orders",
                    "positive_quantity",
                    f"{label} must be positive.",
                    value,
                )
            )
    for column, minimum, table_name, label in (
        ("weekly_capacity", 1, "plants", "Weekly capacity"),
        ("unit_cost", 0, "plants", "Unit cost"),
        ("energy_kwh_per_unit", 0, "plants", "Energy per unit"),
        ("carbon_kg_per_kwh", 0, "plants", "Plant carbon intensity"),
        ("transport_cost_per_unit", 0, "transport_duties", "Transport cost"),
        ("transit_days", 0, "transport_duties", "Transit days"),
        ("carbon_kg_per_unit", 0, "transport_duties", "Transport carbon"),
    ):
        frame = orders[table_name]
        invalid = frame[column] < minimum
        id_column = "plant" if table_name == "plants" else "plant"
        for row_id in frame.loc[invalid, id_column]:
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "valid_non_negative_value",
                    f"{label} must be {'positive' if minimum else 'non-negative'}.",
                    row_id,
                )
            )

    for table_name, frame, column in (
        ("transport_duties", transport_df, "duty_pct"),
        ("capacity_events", events_df, "capacity_loss_pct"),
    ):
        invalid = ~frame[column].between(0, 1, inclusive="both")
        for row_id in frame.loc[invalid, _id_column(table_name)]:
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "percentage_range",
                    f"{column} must be expressed as a fraction between 0 and 1.",
                    row_id,
                )
            )

    plant_names = set(plant_df["plant"])
    destinations = set(order_df["destination"])
    for table_name, frame in (
        ("transport_duties", transport_df),
        ("capacity_events", events_df),
    ):
        for row in frame.loc[~frame["plant"].isin(plant_names)].itertuples(index=False):
            errors.append(
                _issue(
                    "error",
                    table_name,
                    "plant_reference",
                    f"Unknown plant reference {row.plant!r}.",
                    _id_for_row(row, table_name),
                )
            )

    product_to_plants = plant_df.groupby("product")["plant"].apply(set).to_dict()
    for row in order_df.itertuples(index=False):
        if row.product not in product_to_plants:
            warnings.append(
                _issue(
                    "warning",
                    "orders",
                    "product_without_eligible_plant",
                    f"Order product {row.product!r} has no eligible plant; this order is "
                    "business-infeasible and is reported, not treated as malformed input.",
                    row.order_id,
                )
            )

    routes = set(zip(transport_df["plant"], transport_df["destination"]))
    for plant, destination in product(sorted(plant_names), sorted(destinations)):
        if (plant, destination) not in routes:
            errors.append(
                _issue(
                    "error",
                    "transport_duties",
                    "complete_route_coverage",
                    f"Missing transport route for plant {plant!r} and destination "
                    f"{destination!r}.",
                )
            )

    invalid_ranges = events_df["from_date"] > events_df["to_date"]
    for event_id in events_df.loc[invalid_ranges, "event_id"]:
        errors.append(
            _issue(
                "error",
                "capacity_events",
                "event_date_range",
                "Capacity event from_date must be on or before to_date.",
                event_id,
            )
        )

    if planning_start is not None:
        start = pd.to_datetime(planning_start, errors="coerce", format="%Y-%m-%d")
        if pd.isna(start):
            errors.append(
                _issue(
                    "error",
                    "orders",
                    "planning_start",
                    f"Invalid planning_start date {planning_start!r}.",
                )
            )
        else:
            before_start = order_df["due"] < start.normalize()
            for order_id in order_df.loc[before_start, "order_id"]:
                errors.append(
                    _issue(
                        "error",
                        "orders",
                        "due_after_planning_start",
                        "Order due date is earlier than the configured planning start.",
                        order_id,
                    )
                )

    for plant_name, frame in events_df.groupby("plant"):
        ordered = frame.sort_values("from_date")
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and row.from_date <= previous_end:
                warnings.append(
                    _issue(
                        "warning",
                        "capacity_events",
                        "overlapping_capacity_events",
                        f"Capacity events overlap for plant {plant_name!r}; event "
                        "combination policy is not applied by ingestion.",
                        row.event_id,
                    )
                )
            previous_end = max(previous_end, row.to_date) if previous_end is not None else row.to_date

    return _result(errors, warnings)


def _id_for_row(row: Any, table_name: str) -> Any:
    return getattr(row, _id_column(table_name), None)

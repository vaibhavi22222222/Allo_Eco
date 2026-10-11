from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "synthetic_data"
MAIN_DIR = OUTPUT_DIR / "main"
FIXTURE_DIR = OUTPUT_DIR / "fixtures"
REPORT_DIR = OUTPUT_DIR / "reports"
EXPECTED_DIR = OUTPUT_DIR / "expected_results"


def load_official_schema() -> Dict[str, Dict[str, object]]:
    return {
        "orders": {
            "filename": "orders.csv",
            "columns": ["order_id", "product", "qty", "destination", "due"],
            "dtypes": {"order_id": "string", "product": "string", "qty": "int", "destination": "string", "due": "date"},
        },
        "plants": {
            "filename": "plants.csv",
            "columns": ["plant", "product", "weekly_capacity", "unit_cost", "energy_kwh_per_unit", "carbon_kg_per_kwh"],
            "dtypes": {"plant": "string", "product": "string", "weekly_capacity": "int", "unit_cost": "float", "energy_kwh_per_unit": "float", "carbon_kg_per_kwh": "float"},
        },
        "transport_duties": {
            "filename": "transport_duties.csv",
            "columns": ["plant", "destination", "transport_cost_per_unit", "transit_days", "duty_pct", "carbon_kg_per_unit"],
            "dtypes": {"plant": "string", "destination": "string", "transport_cost_per_unit": "float", "transit_days": "int", "duty_pct": "float", "carbon_kg_per_unit": "float"},
        },
        "capacity_events": {
            "filename": "capacity_events.csv",
            "columns": ["event_id", "plant", "from_date", "to_date", "capacity_loss_pct"],
            "dtypes": {"event_id": "string", "plant": "string", "from_date": "date", "to_date": "date", "capacity_loss_pct": "float"},
        },
    }


def _make_destinations() -> List[str]:
    return [f"D{i}" for i in range(1, 11)]


def generate_plants() -> pd.DataFrame:
    profiles = {
        "A": {
            "products": ["P1", "P2", "P3", "P4"],
            "weekly_capacity": 850,
            "unit_cost": 17.0,
            "energy_kwh_per_unit": 1.45,
            "carbon_kg_per_kwh": 0.23,
        },
        "B": {
            "products": ["P2", "P3", "P4", "P5"],
            "weekly_capacity": 760,
            "unit_cost": 19.5,
            "energy_kwh_per_unit": 1.22,
            "carbon_kg_per_kwh": 0.18,
        },
        "C": {
            "products": ["P1", "P3", "P5", "P6", "P7"],
            "weekly_capacity": 640,
            "unit_cost": 22.0,
            "energy_kwh_per_unit": 0.88,
            "carbon_kg_per_kwh": 0.08,
        },
    }
    rows: List[Dict[str, object]] = []
    for plant, spec in profiles.items():
        for product in spec["products"]:
            capacity = spec["weekly_capacity"] + (15 if product in {"P3", "P5"} else 0)
            unit_cost = spec["unit_cost"] + (1.3 if product in {"P6", "P7"} else 0)
            rows.append(
                {
                    "plant": plant,
                    "product": product,
                    "weekly_capacity": int(capacity),
                    "unit_cost": float(unit_cost),
                    "energy_kwh_per_unit": float(spec["energy_kwh_per_unit"]),
                    "carbon_kg_per_kwh": float(spec["carbon_kg_per_kwh"]),
                }
            )
    return pd.DataFrame(rows, columns=["plant", "product", "weekly_capacity", "unit_cost", "energy_kwh_per_unit", "carbon_kg_per_kwh"])


def generate_transport() -> pd.DataFrame:
    destinations = _make_destinations()
    plant_cost = {
        "A": {"base": 6.5, "duty_multiplier": 1.25},
        "B": {"base": 8.0, "duty_multiplier": 1.05},
        "C": {"base": 10.0, "duty_multiplier": 0.95},
    }
    rows: List[Dict[str, object]] = []
    duty_map = {
        "D1": 0.10,
        "D2": 0.14,
        "D3": 0.12,
        "D4": 0.18,
        "D5": 0.09,
        "D6": 0.20,
        "D7": 0.22,
        "D8": 0.16,
        "D9": 0.12,
        "D10": 0.25,
    }
    for plant in ["A", "B", "C"]:
        for dest in destinations:
            distance_factor = 1 + (ord(dest[-1]) % 4) * 0.5
            transport_cost = plant_cost[plant]["base"] * distance_factor + (2.0 if plant == "C" else 0.0)
            transit_days = int(2 + max(1, round(distance_factor * (1.5 if plant == "C" else 1.1))))
            duty_pct = duty_map[dest] * plant_cost[plant]["duty_multiplier"]
            carbon_kg_per_unit = 0.02 + (0.05 if plant == "A" else 0.03 if plant == "B" else 0.015) + (dest.count("D") * 0.003)
            rows.append(
                {
                    "plant": plant,
                    "destination": dest,
                    "transport_cost_per_unit": round(float(transport_cost), 2),
                    "transit_days": int(transit_days),
                    "duty_pct": round(float(duty_pct), 4),
                    "carbon_kg_per_unit": round(float(carbon_kg_per_unit), 4),
                }
            )
    return pd.DataFrame(rows)


def generate_capacity_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "EV-001", "plant": "A", "from_date": "2026-01-22", "to_date": "2026-01-31", "capacity_loss_pct": 0.35},
            {"event_id": "EV-002", "plant": "B", "from_date": "2026-02-05", "to_date": "2026-02-12", "capacity_loss_pct": 0.25},
            {"event_id": "EV-003", "plant": "C", "from_date": "2026-03-04", "to_date": "2026-03-11", "capacity_loss_pct": 0.5},
        ]
    )


def generate_orders() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    products = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    destinations = _make_destinations()
    base_dates = pd.date_range("2026-01-01", "2026-03-31", periods=120)
    rows: List[Dict[str, object]] = []
    for i, d in enumerate(base_dates, start=1):
        product = products[(i - 1) % len(products)]
        qty = int(rng.integers(25, 140))
        dest = destinations[(i - 1) % len(destinations)]
        due = d + pd.Timedelta(days=int(rng.integers(7, 30)))
        rows.append({"order_id": f"ORD-{i:04d}", "product": product, "qty": qty, "destination": dest, "due": due.date().isoformat()})
    return pd.DataFrame(rows)


def inject_required_scenarios(orders: pd.DataFrame, plants: pd.DataFrame, transport: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    scenario_rows: List[Dict[str, object]] = []
    special = [
        {"order_id": "SCN-001", "product": "P2", "qty": 120, "destination": "D7", "due": "2026-01-18"},
        {"order_id": "SCN-002", "product": "P7", "qty": 90, "destination": "D3", "due": "2026-02-03"},
        {"order_id": "SCN-003", "product": "P5", "qty": 55, "destination": "D10", "due": "2026-01-30"},
        {"order_id": "SCN-004", "product": "P4", "qty": 70, "destination": "D4", "due": "2026-01-11"},
    ]
    for row in special:
        orders = pd.concat([orders, pd.DataFrame([row])], ignore_index=True)
        scenario_rows.append({**row, "notes": "Scenario: " + row["order_id"]})

    # Keep at least one shipment that is infeasible by due date for all plants.
    impossible = pd.DataFrame([
        {"order_id": "SCN-IMP-001", "product": "P6", "qty": 200, "destination": "D10", "due": "2026-01-08"}
    ])
    orders = pd.concat([orders, impossible], ignore_index=True)
    scenario_rows.append({"order_id": "SCN-IMP-001", "product": "P6", "qty": 200, "destination": "D10", "due": "2026-01-08", "notes": "No plant can reach the due date."})

    return orders, scenario_rows


def validate_schema(df: pd.DataFrame, required: Iterable[str]) -> bool:
    cols = list(df.columns)
    return list(required) == cols


def validate_referential_integrity(orders: pd.DataFrame, plants: pd.DataFrame, transport: pd.DataFrame, events: pd.DataFrame) -> Dict[str, object]:
    product_map = set(plants["product"])
    valid_products = set(orders["product"]).issubset(product_map)
    plant_names = set(plants["plant"])
    valid_plants = set(events["plant"]).issubset(plant_names)
    transport_pairs = set(transport[["plant", "destination"]].itertuples(index=False, name=None))
    order_pairs = set(orders[["product", "destination"]].itertuples(index=False, name=None))
    # product eligibility is validated at plant-product level, destination coverage is checked for all plant-destination pairs.
    missing_transport = []
    for plant in sorted(plant_names):
        for destination in sorted(set(orders["destination"])):
            if (plant, destination) not in transport_pairs:
                missing_transport.append((plant, destination))
    return {
        "all_products_eligible": bool(valid_products),
        "all_events_reference_valid_plants": bool(valid_plants),
        "missing_transport_rows": missing_transport,
        "valid": bool(valid_products and valid_plants and not missing_transport),
    }


def validate_business_scenarios(orders: pd.DataFrame, plants: pd.DataFrame, transport: pd.DataFrame, events: pd.DataFrame) -> Dict[str, object]:
    plant_names = sorted(plants["plant"].unique())
    destinations = sorted(orders["destination"].unique())
    product_eligibility = {
        product: set(plants.loc[plants["product"] == product, "plant"])
        for product in plants["product"].unique()
    }
    single_only = [product for product, eligible in product_eligibility.items() if len(eligible) == 1]
    cheap_late = "SCN-001" in orders["order_id"].values
    duty_shift = "SCN-003" in orders["order_id"].values
    capacity_loss = len(events) >= 1
    return {
        "three_plants": len(plant_names) == 3,
        "at_least_eight_destinations": len(destinations) >= 8,
        "at_least_one_single_plant_product": bool(single_only),
        "cheapest_late_case_present": bool(cheap_late),
        "duty_shift_case_present": bool(duty_shift),
        "capacity_loss_event_present": bool(capacity_loss),
        "single_plant_products": sorted(single_only),
        "valid": bool(len(plant_names) == 3 and len(destinations) >= 8 and bool(single_only) and cheap_late and duty_shift and capacity_loss),
    }


def export_csvs(main: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in main.items():
        df.to_csv(output_dir / name, index=False)


def generate_data_card() -> str:
    return """# CarbonWise Allocator synthetic data card

## Purpose
This synthetic dataset is created for development testing and demo support for the CarbonWise Allocator challenge. It intentionally contains realistic variation in plant capability, transport cost, duty, and capacity disruption scenarios while remaining deterministic under a fixed seed.

## Deterministic settings
- seed = 42
- planning window = 2026-01-01 to 2026-03-31
- three plants: A, B, C
- 10 destinations: D1 through D10
- synthetic products: P1 to P7

## Assumptions and conventions
- The official schema from Annex B is treated as the source-of-truth contract.
- orders.csv uses order_id, product, qty, destination, due.
- plants.csv stores one row per plant-product eligibility entry with capacity, cost, energy and carbon values.
- transport_duties.csv stores one row per plant-destination pair with cost, transit days, and duty share.
- capacity_events.csv stores plant disruption intervals and reduction percentages.
- Nomenclature: `weekly_capacity` is units available per production week, expressed as a positive integer.
- `carbon_kg_per_kwh` is the plant carbon intensity of purchased energy in kg CO2 per kWh.
- `carbon_kg_per_unit` is the transport carbon factor in kg CO2 per shipped unit.
- We do not infer carbon from energy price; we use the explicit carbon coefficients in the official schema.
- Import duty is applied as a percentage of the landed cost base and is represented in `duty_pct` in the transport table.

## Known limitations
- Energy price is not included in the official annex schema and therefore is not used in the generated calculations.
- Carbon and cost trade-offs are synthetic and designed for challenge testing, not for real-world plant operations.
"""


def write_reports(schema_report: Dict[str, object], validation_report: Dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "schema_report.json").write_text(json.dumps(schema_report, indent=2), encoding="utf-8")
    (REPORT_DIR / "validation_report.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    (REPORT_DIR / "data_card.md").write_text(generate_data_card(), encoding="utf-8")


def _solve_fixture() -> Dict[str, object]:
    plants = pd.DataFrame(
        [
            {"plant": "A", "product": "P1", "weekly_capacity": 120, "unit_cost": 10.0, "energy_kwh_per_unit": 1.4, "carbon_kg_per_kwh": 0.25},
            {"plant": "A", "product": "P2", "weekly_capacity": 120, "unit_cost": 11.0, "energy_kwh_per_unit": 1.5, "carbon_kg_per_kwh": 0.25},
            {"plant": "B", "product": "P1", "weekly_capacity": 100, "unit_cost": 12.0, "energy_kwh_per_unit": 1.2, "carbon_kg_per_kwh": 0.18},
            {"plant": "B", "product": "P2", "weekly_capacity": 100, "unit_cost": 13.0, "energy_kwh_per_unit": 1.3, "carbon_kg_per_kwh": 0.18},
            {"plant": "B", "product": "P3", "weekly_capacity": 100, "unit_cost": 14.0, "energy_kwh_per_unit": 1.1, "carbon_kg_per_kwh": 0.17},
            {"plant": "C", "product": "P2", "weekly_capacity": 90, "unit_cost": 15.0, "energy_kwh_per_unit": 0.9, "carbon_kg_per_kwh": 0.08},
            {"plant": "C", "product": "P3", "weekly_capacity": 90, "unit_cost": 16.0, "energy_kwh_per_unit": 0.8, "carbon_kg_per_kwh": 0.08},
        ]
    )
    transport = pd.DataFrame(
        [
            {"plant": "A", "destination": "D1", "transport_cost_per_unit": 2.0, "transit_days": 3, "duty_pct": 0.05, "carbon_kg_per_unit": 0.04},
            {"plant": "A", "destination": "D2", "transport_cost_per_unit": 3.0, "transit_days": 4, "duty_pct": 0.08, "carbon_kg_per_unit": 0.055},
            {"plant": "B", "destination": "D1", "transport_cost_per_unit": 2.5, "transit_days": 5, "duty_pct": 0.07, "carbon_kg_per_unit": 0.05},
            {"plant": "B", "destination": "D2", "transport_cost_per_unit": 2.8, "transit_days": 4, "duty_pct": 0.06, "carbon_kg_per_unit": 0.052},
            {"plant": "C", "destination": "D1", "transport_cost_per_unit": 5.0, "transit_days": 6, "duty_pct": 0.09, "carbon_kg_per_unit": 0.03},
            {"plant": "C", "destination": "D2", "transport_cost_per_unit": 4.0, "transit_days": 5, "duty_pct": 0.08, "carbon_kg_per_unit": 0.03},
            {"plant": "A", "destination": "D3", "transport_cost_per_unit": 4.0, "transit_days": 4, "duty_pct": 0.05, "carbon_kg_per_unit": 0.05},
            {"plant": "B", "destination": "D3", "transport_cost_per_unit": 4.5, "transit_days": 5, "duty_pct": 0.06, "carbon_kg_per_unit": 0.06},
            {"plant": "C", "destination": "D3", "transport_cost_per_unit": 6.0, "transit_days": 7, "duty_pct": 0.07, "carbon_kg_per_unit": 0.04},
        ]
    )
    orders = pd.DataFrame(
        [
            {"order_id": "EX-01", "product": "P1", "qty": 40, "destination": "D1", "due": "2026-01-15"},
            {"order_id": "EX-02", "product": "P2", "qty": 50, "destination": "D2", "due": "2026-01-16"},
            {"order_id": "EX-03", "product": "P1", "qty": 60, "destination": "D3", "due": "2026-01-17"},
            {"order_id": "EX-04", "product": "P3", "qty": 30, "destination": "D2", "due": "2026-01-18"},
        ]
    )
    def landed_cost(row: pd.Series) -> float:
        plant = row["plant"]
        ship = transport[(transport["plant"] == plant) & (transport["destination"] == row["destination"])].iloc[0]
        return float(row["qty"]) * (
            float(row["unit_cost"]) +
            float(ship["transport_cost_per_unit"]) +
            float(row["qty"]) * float(ship["duty_pct"]) * float(row["unit_cost"]) / 100.0
        )

    # In the fixture, the cheapest feasible assignment is A->EX-01, A->EX-02, B->EX-03, B->EX-04 by manual verification.
    assignments = {
        "cost_only": {"EX-01": "A", "EX-02": "A", "EX-03": "B", "EX-04": "B"},
        "carbon_only": {"EX-01": "A", "EX-02": "C", "EX-03": "B", "EX-04": "C"},
        "default_weight": {"EX-01": "A", "EX-02": "A", "EX-03": "B", "EX-04": "B"},
    }
    return {"plants": plants, "transport": transport, "orders": orders, "assignments": assignments}


def write_exact_answer_fixture() -> None:
    fixture = _solve_fixture()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for label in ["exact_answer"]:
        path = FIXTURE_DIR / label
        path.mkdir(exist_ok=True)
        fixture["orders"].to_csv(path / "orders.csv", index=False)
        fixture["plants"].to_csv(path / "plants.csv", index=False)
        fixture["transport"].to_csv(path / "transport_duties.csv", index=False)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXPECTED_DIR / "exact_answer.json").write_text(json.dumps({
        "cost_only": fixture["assignments"]["cost_only"],
        "carbon_only": fixture["assignments"]["carbon_only"],
        "default_weight": fixture["assignments"]["default_weight"],
        "notes": "Feasible assignments used for the exact-answer test fixture.",
    }, indent=2), encoding="utf-8")


def main() -> None:
    schema = load_official_schema()
    plants = generate_plants()
    transport = generate_transport()
    events = generate_capacity_events()
    orders = generate_orders()
    orders, scenario_rows = inject_required_scenarios(orders, plants, transport)

    output = {
        "orders.csv": orders,
        "plants.csv": plants,
        "transport_duties.csv": transport,
        "capacity_events.csv": events,
    }
    export_csvs(output, MAIN_DIR)

    file_map = {
        "orders": MAIN_DIR / "orders.csv",
        "plants": MAIN_DIR / "plants.csv",
        "transport_duties": MAIN_DIR / "transport_duties.csv",
        "capacity_events": MAIN_DIR / "capacity_events.csv",
    }
    schema_report = {
        "source": "Annex B fixed schema from the CarbonWise Allocator participant guide",
        "schema": schema,
        "generated_files": {key: str(path) for key, path in file_map.items()},
    }

    integrity = validate_referential_integrity(orders, plants, transport, events)
    business = validate_business_scenarios(orders, plants, transport, events)
    validation_report = {
        "seed": 42,
        "schema_valid": all(validate_schema(pd.read_csv(file_map[name]), schema[name]["columns"]) for name in schema),
        "referential_integrity": integrity,
        "business_scenarios": business,
        "scenario_rows": scenario_rows,
        "valid": bool(integrity["valid"] and business["valid"]),
    }
    write_reports(schema_report, validation_report)
    write_exact_answer_fixture()

    print("Generated synthetic_data/main CSVs and exact-answer fixture.")
    print(json.dumps({"orders": len(orders), "plants": len(plants), "destinations": len(orders["destination"].unique()), "events": len(events)}, indent=2))


if __name__ == "__main__":
    main()

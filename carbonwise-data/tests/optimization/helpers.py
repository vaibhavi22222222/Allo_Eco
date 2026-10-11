from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from services.validation import SCHEMAS
from optimization import prepare_problem, SolvePolicy


def tables(orders=None, plants=None, routes=None, events=None):
    rows = {
        "orders": orders
        if orders is not None
        else [
            {
                "order_id": "O1",
                "product": "P1",
                "qty": 1,
                "destination": "D",
                "due": "2026-10-13",
            }
        ],
        "plants": plants
        if plants is not None
        else [
            {
                "plant": "A",
                "product": "P1",
                "weekly_capacity": 100,
                "unit_cost": 100,
                "energy_kwh_per_unit": 1,
                "carbon_kg_per_kwh": 100,
            }
        ],
        "transport_duties": routes
        if routes is not None
        else [
            {
                "plant": "A",
                "destination": "D",
                "transport_cost_per_unit": 0,
                "transit_days": 1,
                "duty_pct": 0,
                "carbon_kg_per_unit": 0,
            }
        ],
        "capacity_events": events if events is not None else [],
    }
    result = {n: pd.DataFrame(r, columns=SCHEMAS[n]) for n, r in rows.items()}
    for n, frame in result.items():
        for col, kind in SCHEMAS[n].items():
            if kind == "float":
                frame[col] = frame[col].astype(float)
    return result


def config(t, **updates):
    result = {
        "planning_start": "2026-10-12",
        "horizon_days": 7,
        "energy_price_per_kwh": {p: 0 for p in t["plants"]["plant"].unique()},
    }
    result.update(updates)
    return result


def problem(t=None, **updates):
    t = tables() if t is None else t
    return prepare_problem(t, config(t, **updates), dataset_version_id="fixture-v1")


def policy(**kw):
    return SolvePolicy(time_limit_seconds=3, total_time_limit_seconds=12, **kw)


def tradeoff():
    plants = []
    routes = []
    for p, c, e in [("A", 100, 100), ("B", 110, 50), ("C", 150, 10)]:
        plants.append(
            {
                "plant": p,
                "product": "P1",
                "weekly_capacity": 100,
                "unit_cost": c,
                "energy_kwh_per_unit": 1,
                "carbon_kg_per_kwh": e,
            }
        )
        routes.append(
            {
                "plant": p,
                "destination": "D",
                "transport_cost_per_unit": 0,
                "transit_days": 1,
                "duty_pct": 0,
                "carbon_kg_per_unit": 0,
            }
        )
    t = tables(plants=plants, routes=routes)
    return problem(t, cost_scale=100, carbon_scale=100)


def joint():
    plants = []
    routes = []
    for p, costs in [("A", [100, 110]), ("B", [120, 200])]:
        for product, cost in zip(["P1", "P2"], costs):
            plants.append(
                {
                    "plant": p,
                    "product": product,
                    "weekly_capacity": 5,
                    "unit_cost": cost,
                    "energy_kwh_per_unit": 0,
                    "carbon_kg_per_kwh": 0,
                }
            )
        routes.append(
            {
                "plant": p,
                "destination": "D",
                "transport_cost_per_unit": 0,
                "transit_days": 1,
                "duty_pct": 0,
                "carbon_kg_per_unit": 0,
            }
        )
    orders = [
        {
            "order_id": "O1",
            "product": "P1",
            "qty": 1,
            "destination": "D",
            "due": "2026-10-13",
        },
        {
            "order_id": "O2",
            "product": "P2",
            "qty": 1,
            "destination": "D",
            "due": "2026-10-13",
        },
    ]
    return problem(tables(orders, plants, routes))

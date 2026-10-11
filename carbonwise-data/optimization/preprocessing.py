"""Adapt the teammate's canonical DataFrames without modifying input/storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Any

import pandas as pd

from services.validation import SCHEMAS, canonicalize_table, validate_data_quality
from .contracts import CapacityEvent, PlanningConfig


class OptimizationInputError(ValueError):
    def __init__(self, message: str, details=None):
        self.details = details or []
        super().__init__(message)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()[:20]


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True)
class PreparedProblem:
    dataset_version_id: str
    data_hash: str
    config_version: str
    scenario_hash: str
    config: PlanningConfig
    tables: dict[str, pd.DataFrame]
    orders: dict[str, dict]
    capabilities: dict[tuple[str, str], dict]
    routes: dict[tuple[str, str], dict]
    events: tuple[CapacityEvent, ...]
    dates: tuple[date, ...]
    production_dates: tuple[date, ...]
    plants: tuple[str, ...]
    weekly_capacity: dict[str, int]
    daily_capacity: dict[tuple[str, date], int]
    raw_daily_capacity: dict[tuple[str, date], float]
    prices: dict[tuple[str, date], float]
    carbon_factors: dict[tuple[str, str, date], float]
    scales: dict[str, float]
    warnings: tuple[str, ...]

    @property
    def end(self) -> date:
        return self.dates[-1]

    @property
    def allocation_input_hash(self) -> str:
        return digest(
            {
                name: sorted(
                    _records(self.tables[name]),
                    key=lambda r: json.dumps(r, sort_keys=True, default=str),
                )
                for name in ("orders", "plants", "transport_duties")
            }
        )


def _records(frame: pd.DataFrame):
    result = []
    for row in frame.to_dict("records"):
        result.append(
            {k: v.date() if isinstance(v, pd.Timestamp) else v for k, v in row.items()}
        )
    return result


def prepare_problem(
    tables: dict[str, pd.DataFrame],
    config: PlanningConfig | dict,
    scenario: dict | None = None,
    dataset_version_id: str = "in-memory",
) -> PreparedProblem:
    config = PlanningConfig.model_validate(config).model_copy(deep=True)
    report = validate_data_quality(tables, config.planning_start.isoformat())
    if not report["valid"]:
        raise OptimizationInputError(
            "Input tables failed existing data validation", report["errors"]
        )
    canonical = {name: canonicalize_table(tables[name], name) for name in SCHEMAS}
    records = {name: _records(frame) for name, frame in canonical.items()}
    order_rows = sorted(records["orders"], key=lambda row: row["order_id"])
    orders = {row["order_id"]: row for row in order_rows}
    caps = {(r["plant"], r["product"]): r for r in records["plants"]}
    routes = {(r["plant"], r["destination"]): r for r in records["transport_duties"]}
    plants = tuple(sorted({r["plant"] for r in records["plants"]}))
    for plant in plants:
        if plant not in config.energy_price_per_kwh:
            raise OptimizationInputError(
                f"Missing explicit energy price for plant {plant}"
            )
    unknown_config_plants = (
        set(config.energy_price_per_kwh)
        | set(config.plant_weekly_capacity)
        | {x.plant for x in config.weekly_carbon_overrides}
        | {x.plant for x in config.weekly_energy_price_overrides}
    ) - set(plants)
    if unknown_config_plants:
        raise OptimizationInputError(
            f"Configuration references unknown plants: {sorted(unknown_config_plants)}"
        )
    warnings = [i["message"] for i in report["warnings"]]
    weekly = {}
    for plant in plants:
        values = {int(r["weekly_capacity"]) for (p, _), r in caps.items() if p == plant}
        if plant in config.plant_weekly_capacity:
            weekly[plant] = config.plant_weekly_capacity[plant]
            warnings.append(
                f"Plant {plant} shared capacity explicitly configured as {weekly[plant]} units/week; CSV row values={sorted(values)}."
            )
        elif len(values) == 1:
            weekly[plant] = values.pop()
        else:
            raise OptimizationInputError(
                f"Plant {plant} has inconsistent product-row capacities {sorted(values)}; "
                "set plant_weekly_capacity explicitly. Capacities will not be summed."
            )
    events = {
        r["event_id"]: CapacityEvent.model_validate(r)
        for r in records["capacity_events"]
    }
    if scenario:
        unknown = set(scenario) - {"capacity_events"}
        if unknown:
            raise OptimizationInputError(
                f"Unknown preprocessing scenario keys: {sorted(unknown)}"
            )
        for raw in scenario.get("capacity_events", []):
            ev = CapacityEvent.model_validate(raw)
            if ev.plant not in plants:
                raise OptimizationInputError(f"Unknown event plant {ev.plant}")
            if ev.event_id in events and events[ev.event_id] != ev:
                raise OptimizationInputError(
                    f"Conflicting event_id {ev.event_id}; use a new id for a different event"
                )
            events[ev.event_id] = ev
    event_list = tuple(events[k] for k in sorted(events))
    dates = tuple(
        config.planning_start + timedelta(days=n) for n in range(config.horizon_days)
    )
    production_dates = tuple(
        d for d in dates if d.weekday() in config.production_weekdays
    )
    daily, raw_daily = {}, {}
    price_overrides = {
        (v.plant, v.week_start): v.value for v in config.weekly_energy_price_overrides
    }
    carbon_overrides = {
        (v.plant, v.week_start): v.value for v in config.weekly_carbon_overrides
    }
    prices, carbon = {}, {}
    for plant in plants:
        for d in dates:
            loss = max(
                (
                    v.capacity_loss_pct
                    for v in event_list
                    if v.plant == plant and v.from_date <= d <= v.to_date
                ),
                default=0.0,
            )
            quota = (
                weekly[plant] / len(config.production_weekdays) * (1 - loss)
                if d in production_dates
                else 0.0
            )
            raw_daily[plant, d] = quota
            daily[plant, d] = int(math.floor(quota + 1e-9))
            prices[plant, d] = price_overrides.get(
                (plant, week_start(d)), config.energy_price_per_kwh[plant]
            )
    for (plant, product), cap in caps.items():
        for d in dates:
            carbon[plant, product, d] = carbon_overrides.get(
                (plant, week_start(d)), cap["carbon_kg_per_kwh"]
            )
    # Fixed engineering scales use per-order minimum full-quantity candidate estimates.
    # They do NOT assert a jointly feasible reference plan and are held across comparisons.
    cscale = escale = 0.0
    for order in orders.values():
        cands, emissions = [], []
        for plant in plants:
            cap = caps.get((plant, order["product"]))
            if cap is None:
                continue
            route = routes[plant, order["destination"]]
            price = min(prices[plant, d] for d in dates)
            factor = min(carbon[plant, order["product"], d] for d in dates)
            cands.append(
                order["qty"]
                * (
                    (cap["unit_cost"] + cap["energy_kwh_per_unit"] * price)
                    * (1 + route["duty_pct"])
                    + route["transport_cost_per_unit"]
                )
            )
            emissions.append(
                order["qty"]
                * (cap["energy_kwh_per_unit"] * factor + route["carbon_kg_per_unit"])
            )
        cscale += min(cands, default=0)
        escale += min(emissions, default=0)
    scales = {
        "cost": config.cost_scale or max(cscale, 1.0),
        "carbon": config.carbon_scale or max(escale, 1.0),
        "lateness": float(max(1, len(orders) * config.horizon_days)),
    }
    normalized_records = {
        k: sorted(v, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        for k, v in records.items()
    }
    if max(r["due"] for r in orders.values()) > dates[-1]:
        warnings.append(
            "Some due dates extend beyond the production horizon; infeasibility is scoped to that horizon."
        )
    return PreparedProblem(
        dataset_version_id,
        digest(normalized_records),
        digest(config.model_dump(mode="json")),
        digest([v.model_dump(mode="json") for v in event_list]),
        config,
        canonical,
        orders,
        caps,
        routes,
        event_list,
        dates,
        production_dates,
        plants,
        weekly,
        daily,
        raw_daily,
        prices,
        carbon,
        scales,
        tuple(warnings),
    )


def eligible_plants(problem: PreparedProblem, order_id: str) -> list[str]:
    product = problem.orders[order_id]["product"]
    return [p for p in problem.plants if (p, product) in problem.capabilities]


def arrival_date(
    problem: PreparedProblem, order_id: str, plant: str, dispatch: date
) -> date:
    route = problem.routes[plant, problem.orders[order_id]["destination"]]
    return dispatch + timedelta(days=int(route["transit_days"]))

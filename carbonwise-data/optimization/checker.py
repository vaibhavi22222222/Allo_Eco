"""Independent result verification using canonical source rows (never PuLP vars)."""

from __future__ import annotations
from collections import defaultdict
from datetime import date, timedelta
import math

from .contracts import SolveResult
from .preprocessing import PreparedProblem, week_start


def validate_plan(problem: PreparedProblem, result: SolveResult | dict) -> dict:
    data = result.model_dump(mode="json") if isinstance(result, SolveResult) else result
    if data.get("kpis") is None:
        return {"applicable": False, "valid": False, "violations": ["No computed plan"]}
    errors = []
    orders = {r["order_id"]: r for r in problem.tables["orders"].to_dict("records")}
    caps = {
        (r["plant"], r["product"]): r
        for r in problem.tables["plants"].to_dict("records")
    }
    routes = {
        (r["plant"], r["destination"]): r
        for r in problem.tables["transport_duties"].to_dict("records")
    }
    allocations = {}
    for row in data["allocation"]:
        oid = row["order_id"]
        if oid in allocations:
            errors.append(f"Duplicate allocation: {oid}")
        allocations[oid] = row
    unassigned = [r["order_id"] for r in data.get("unassigned_orders", [])]
    if len(unassigned) != len(set(unassigned)):
        errors.append("Duplicate unassigned order")
    if set(allocations) & set(unassigned):
        errors.append("Order both assigned and unassigned")
    if set(allocations) | set(unassigned) != set(orders):
        errors.append("Order coverage mismatch")
    if (
        data.get("data_hash") != problem.data_hash
        or data.get("scenario_hash") != problem.scenario_hash
    ):
        errors.append("Dataset/scenario identity mismatch")
    if data.get("config_version") != problem.config_version:
        errors.append("Configuration identity mismatch")
    produced = defaultdict(int)
    usage = defaultdict(int)
    prod_by_order = defaultdict(list)
    prod_keys = set()
    for row in data["production_schedule"]:
        oid, plant = row["order_id"], row["plant"]
        try:
            d = date.fromisoformat(str(row["date"]))
            qty = row["qty"]
            if (
                isinstance(qty, bool)
                or not isinstance(qty, (int, float))
                or not math.isfinite(qty)
                or qty <= 0
                or int(qty) != qty
            ):
                raise ValueError("Production quantity must be a positive integer")
        except (ValueError, TypeError) as exc:
            errors.append(f"Bad production row: {exc}")
            continue
        if (oid, plant, d) in prod_keys:
            errors.append(f"Duplicate production row: {oid}/{plant}/{d}")
        prod_keys.add((oid, plant, d))
        if oid not in allocations or allocations[oid]["plant"] != plant:
            errors.append(f"Production without matching assignment: {oid}/{plant}")
        if (
            d < problem.config.planning_start
            or d > problem.end
            or d.weekday() not in problem.config.production_weekdays
        ):
            errors.append(f"Production outside calendar: {oid}/{d}")
        produced[oid, plant] += int(qty)
        usage[plant, d] += int(qty)
        prod_by_order[oid].append((plant, d, int(qty)))

    # Re-derive capacities separately from the preparation/model functions.
    usable = {}
    for plant in problem.plants:
        values = {int(r["weekly_capacity"]) for (p, _), r in caps.items() if p == plant}
        weekly = problem.config.plant_weekly_capacity.get(plant)
        if weekly is None:
            if len(values) != 1:
                errors.append(f"Ambiguous shared capacity: {plant}")
                continue
            weekly = next(iter(values))
        for d in problem.dates:
            active_loss = max(
                (
                    e.capacity_loss_pct
                    for e in problem.events
                    if e.plant == plant and e.from_date <= d <= e.to_date
                ),
                default=0.0,
            )
            available = (
                weekly / len(problem.config.production_weekdays) * (1 - active_loss)
                if d.weekday() in problem.config.production_weekdays
                else 0
            )
            usable[plant, d] = math.floor(available + 1e-9)
            if usage[plant, d] > usable[plant, d]:
                errors.append(f"Capacity exceeded: {plant}/{d}")
    for (plant, d), qty in usage.items():
        if qty and (plant, d) not in usable:
            errors.append(f"Unknown capacity bucket: {plant}/{d}")

    total_cost = total_carbon = 0.0
    late_count = late_days_total = fulfilled = 0
    carbon_overrides = {
        (r.plant, r.week_start): r.value for r in problem.config.weekly_carbon_overrides
    }
    price_overrides = {
        (r.plant, r.week_start): r.value
        for r in problem.config.weekly_energy_price_overrides
    }

    def equal(a, b, label):
        if (
            not isinstance(a, (int, float))
            or not math.isfinite(a)
            or not math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-8)
        ):
            errors.append(f"Incorrect {label}: reported={a}, computed={b}")

    for oid, row in allocations.items():
        if oid not in orders:
            errors.append(f"Unknown order: {oid}")
            continue
        order = orders[oid]
        plant = row["plant"]
        if (plant, order["product"]) not in caps:
            errors.append(f"Ineligible plant: {oid}/{plant}")
            continue
        cap = caps[plant, order["product"]]
        route = routes.get((plant, order["destination"]))
        if route is None:
            errors.append(f"Missing route: {oid}/{plant}")
            continue
        if row["qty"] != order["qty"] or produced[oid, plant] != order["qty"]:
            errors.append(f"Quantity balance mismatch: {oid}")
        try:
            dispatch = date.fromisoformat(row["dispatch"])
            due = order["due"].date()
            arrival = dispatch + timedelta(days=int(route["transit_days"]))
            late = max(0, (arrival - due).days)
            if not problem.config.planning_start <= dispatch <= problem.end:
                errors.append(f"Dispatch outside horizon: {oid}")
            if (
                row["arrival"] != arrival.isoformat()
                or row["late_days"] != late
                or row["on_time"] != (late == 0)
            ):
                errors.append(f"Incorrect delivery fields: {oid}")
        except (ValueError, TypeError, KeyError):
            errors.append(f"Invalid delivery dates: {oid}")
            continue
        p_cost = float(order["qty"] * cap["unit_cost"])
        e_cost = e_carbon = 0.0
        for production_plant, d, qty in prod_by_order[oid]:
            if d > dispatch:
                errors.append(f"Production after dispatch: {oid}")
            if production_plant != plant:
                errors.append(f"Order split between plants: {oid}")
            price = price_overrides.get(
                (plant, week_start(d)), problem.config.energy_price_per_kwh[plant]
            )
            factor = carbon_overrides.get(
                (plant, week_start(d)), cap["carbon_kg_per_kwh"]
            )
            e_cost += qty * cap["energy_kwh_per_unit"] * price
            e_carbon += qty * cap["energy_kwh_per_unit"] * factor
        transport = order["qty"] * route["transport_cost_per_unit"]
        duty = (p_cost + e_cost) * route["duty_pct"]
        cost = p_cost + e_cost + transport + duty
        emissions = e_carbon + order["qty"] * route["carbon_kg_per_unit"]
        equal(row["cost"], cost, f"cost {oid}")
        equal(row["carbon_kg"], emissions, f"carbon {oid}")
        for name, val in {
            "production": p_cost,
            "energy": e_cost,
            "transport": transport,
            "duty": duty,
        }.items():
            equal(row["cost_breakdown"].get(name), val, f"{name} {oid}")
        total_cost += cost
        total_carbon += emissions
        late_count += int(late > 0)
        late_days_total += late
        fulfilled += int(order["qty"])
    expected = {
        "total_cost": total_cost,
        "total_carbon_t": total_carbon / 1000,
        "late_orders": late_count,
        "total_lateness_days": late_days_total,
        "fulfilled_units": fulfilled,
        "unassigned_orders": len(unassigned),
        "unassigned_units": sum(
            int(orders[o]["qty"]) for o in unassigned if o in orders
        ),
    }
    for name, value in expected.items():
        equal(data["kpis"].get(name), value, name)
    if data["kpis"].get("currency") != problem.config.currency:
        errors.append("Currency mismatch")
    weekly_detail = []
    utilization = {}
    for p in problem.plants:
        capacity = sum(usable.get((p, d), 0) for d in problem.dates)
        amount = sum(usage[p, d] for d in problem.dates)
        utilization[p] = amount / capacity if capacity else None
        for w in sorted({week_start(d) for d in problem.dates}):
            ds = [d for d in problem.dates if week_start(d) == w]
            c = sum(usable.get((p, d), 0) for d in ds)
            n = sum(usage[p, d] for d in ds)
            weekly_detail.append(
                {
                    "plant": p,
                    "week_start": w.isoformat(),
                    "produced_units": n,
                    "effective_capacity": c,
                    "utilisation": n / c if c else None,
                    "zero_capacity": c == 0,
                }
            )
    # Check both total and per-week utilization; horizon averages must not hide overload.
    for p, val in utilization.items():
        reported = data["kpis"].get("plant_utilisation", {}).get(p)
        if val is None:
            if reported is not None:
                errors.append(f"Zero-capacity utilisation must be null: {p}")
        else:
            equal(reported, val, f"utilisation {p}")
    supplied_weekly = data["kpis"].get("plant_utilisation_by_week", [])
    if supplied_weekly != weekly_detail:
        errors.append("Weekly utilization mismatch")
    ctx = data.get("problem_context", {})
    for oid, plant in ctx.get("forced_assignments", {}).items():
        if oid not in allocations or allocations[oid]["plant"] != plant:
            errors.append(f"Forced assignment not satisfied: {oid}")
    budget = ctx.get("carbon_budget_kg")
    if budget is not None and total_carbon > budget + 1e-5:
        errors.append("Carbon budget exceeded")
    delivery_ok = late_count == 0 and not unassigned
    if not data.get("alternative") and not delivery_ok:
        errors.append("Strict plan has late or unassigned orders")
    if data.get("original_request_met") and not delivery_ok:
        errors.append("False claim that original delivery request was met")
    return {
        "applicable": True,
        "valid": not errors,
        "capacity_ok": not any(
            "Capacity" in e or "capacity bucket" in e for e in errors
        ),
        "eligibility_ok": not any("Ineligible" in e for e in errors),
        "production_balance_ok": not any(
            "Quantity" in e or "Production" in e or "split" in e for e in errors
        ),
        "delivery_ok": delivery_ok,
        "violations": errors,
        "recomputed_kpis": expected,
    }

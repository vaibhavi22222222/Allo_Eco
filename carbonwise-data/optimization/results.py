from __future__ import annotations
from collections import defaultdict
from .contracts import ObjectiveWeights, SolvePolicy, SolveResult
from .preprocessing import PreparedProblem, arrival_date, eligible_plants, week_start
from .costs import cost_carbon_for_order
from .checker import validate_plan


def empty_result(
    p: PreparedProblem,
    weights: ObjectiveWeights,
    status: str,
    reason: str,
    policy: SolvePolicy,
    trace=None,
) -> SolveResult:
    return SolveResult(
        dataset_version_id=p.dataset_version_id,
        data_hash=p.data_hash,
        config_version=p.config_version,
        scenario_hash=p.scenario_hash,
        planning_window={
            "start": p.config.planning_start.isoformat(),
            "end": p.end.isoformat(),
        },
        solver={
            "status": status,
            "termination_reason": reason,
            "has_feasible_solution": False,
            "optimality_proven": False,
            "solve_seconds": 0.0,
            "best_bound": None,
            "relative_gap": None,
        },
        answer_type="infeasible" if status == "infeasible" else "allocation",
        weights_used=weights.normalized(),
        normalization_scales=p.scales,
        assumptions=[
            "Duty and capacity_loss_pct are fractions: 0.30 means 30%, matching the existing loader.",
            "One plant per order; all orders available at planning start; production may span days.",
            "Shared plant capacity distributed equally across working days and rounded down.",
            "Duty applies to production plus energy; transport uses calendar days.",
            "Prices and weekly overrides are explicit configuration, not inferred from CSV.",
            "Overlapping events use the largest active loss; early inventory has no holding cost.",
            "Equal order priority; no execution/WIP state is supplied.",
        ],
        limitations=[
            "Coarse daily production model; no setups, machine routing or inventory cost.",
            "Feasibility is scoped to the configured finite horizon.",
            "Normalization uses fixed candidate estimates, not a proven feasible reference optimum.",
        ],
        warnings=list(p.warnings),
        trace=trace or [],
        problem_context={
            "resolved_config": p.config.model_dump(mode="json"),
            "allocation_input_hash": p.allocation_input_hash,
            "capacity_events": [e.model_dump(mode="json") for e in p.events],
            "carbon_budget_kg": policy.carbon_budget_kg,
            "forced_assignments": policy.forced_assignments,
        },
    )


def build_result(
    p, weights, policy, assignments, production, solver, alternative=False, trace=None
):
    result = empty_result(
        p,
        weights,
        solver["status"],
        solver.get("termination_reason", ""),
        policy,
        trace,
    )
    result.solver.update(solver)
    result.alternative = alternative
    result.answer_type = "infeasible" if alternative else "allocation"
    result.production_schedule = sorted(
        production, key=lambda r: (r["order_id"], r["plant"], r["date"])
    )
    grouped = defaultdict(list)
    usage = defaultdict(int)
    for row in result.production_schedule:
        grouped[row["order_id"]].append(row)
        usage[row["plant"], row["date"]] += row["qty"]
    for oid, (plant, dispatch) in sorted(assignments.items()):
        order = p.orders[oid]
        arrival = arrival_date(p, oid, plant, dispatch)
        late = max(0, (arrival - order["due"]).days)
        result.allocation.append(
            {
                "order_id": oid,
                "plant": plant,
                "qty": order["qty"],
                "dispatch": dispatch.isoformat(),
                "arrival": arrival.isoformat(),
                "on_time": late == 0,
                "late_days": late,
                **cost_carbon_for_order(p, oid, plant, grouped[oid]),
                "evidence_refs": [
                    f"{p.dataset_version_id}:orders:{oid}",
                    f"{p.dataset_version_id}:plants:{plant}:{order['product']}",
                    f"config:{p.config_version}",
                ],
            }
        )
    for oid, order in p.orders.items():
        if oid not in assignments:
            reason = (
                "NO_ELIGIBLE_PLANT"
                if not eligible_plants(p, oid)
                else "UNASSIGNED_IN_BEST_FOUND_ALTERNATIVE"
            )
            result.unassigned_orders.append(
                {
                    "order_id": oid,
                    "qty": order["qty"],
                    "reason_code": reason,
                    "scope": "configured production horizon",
                }
            )
    total_cost = sum(r["cost"] for r in result.allocation)
    total_carbon = sum(r["carbon_kg"] for r in result.allocation)
    lateness = sum(r["late_days"] for r in result.allocation)
    util, by_week = {}, []
    for plant in p.plants:
        cap = sum(p.daily_capacity[plant, d] for d in p.dates)
        made = sum(usage[plant, d.isoformat()] for d in p.dates)
        util[plant] = made / cap if cap else None
        for w in sorted({week_start(d) for d in p.dates}):
            ds = [d for d in p.dates if week_start(d) == w]
            c = sum(p.daily_capacity[plant, d] for d in ds)
            n = sum(usage[plant, d.isoformat()] for d in ds)
            by_week.append(
                {
                    "plant": plant,
                    "week_start": w.isoformat(),
                    "produced_units": n,
                    "effective_capacity": c,
                    "utilisation": n / c if c else None,
                    "zero_capacity": c == 0,
                }
            )
    result.kpis = {
        "total_cost": total_cost,
        "currency": p.config.currency,
        "total_carbon_t": total_carbon / 1000,
        "late_orders": sum(not r["on_time"] for r in result.allocation),
        "total_lateness_days": lateness,
        "unassigned_orders": len(result.unassigned_orders),
        "unassigned_units": sum(r["qty"] for r in result.unassigned_orders),
        "fulfilled_units": sum(r["qty"] for r in result.allocation),
        "plant_utilisation": util,
        "plant_utilisation_by_week": by_week,
    }
    w = result.weights_used
    result.solver["objective_value"] = (
        w["cost"] * total_cost / p.scales["cost"]
        + w["carbon"] * total_carbon / p.scales["carbon"]
        + w["lateness"] * lateness / p.scales["lateness"]
    )
    result.original_request_met = (
        not alternative
        and not result.unassigned_orders
        and not result.kpis["late_orders"]
    )
    result.solver["has_feasible_solution"] = True
    result.validator = validate_plan(p, result)
    if not result.validator["valid"]:
        invalid = empty_result(
            p, weights, "error", "independent_checker_rejected", policy, trace
        )
        invalid.solver["validation_violations"] = result.validator["violations"]
        invalid.validator = result.validator
        return invalid
    result.recommendations = [
        {
            "action": "Review the proposed allocation."
            if not alternative
            else "Review unmet commitments and the closest computed alternative.",
            "rationale": "The independent checker verified capacity, eligibility, quantities and reported arithmetic.",
            "evidence_refs": [
                f"{p.dataset_version_id}:optimizer:validator",
                f"config:{p.config_version}",
            ],
            "confidence": "medium",
            "estimated_impact": {
                "kind": "plan_total",
                "cost": total_cost,
                "currency": p.config.currency,
                "carbon_t": total_carbon / 1000,
            },
            "requires_approval": True,
        }
    ]
    return result

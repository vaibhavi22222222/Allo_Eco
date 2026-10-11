from __future__ import annotations
from dataclasses import replace
import time
from .contracts import SolvePolicy, SolveResult
from .preprocessing import OptimizationInputError, prepare_problem
from .solve import solve_allocation


def _ready(result):
    return result.kpis is not None and result.validator.get("valid", False)


def compare_plans(base: SolveResult, revised: SolveResult) -> dict:
    if not _ready(base) or not _ready(revised):
        return {
            "available": False,
            "reason": "Both plans must contain independently verified results",
        }
    if base.problem_context.get("allocation_input_hash") != revised.problem_context.get(
        "allocation_input_hash"
    ):
        raise OptimizationInputError(
            "Cannot compare plans with different orders, capabilities or transport data"
        )
    if (
        base.planning_window != revised.planning_window
        or base.kpis["currency"] != revised.kpis["currency"]
    ):
        raise OptimizationInputError(
            "Cannot compare different planning windows or currencies"
        )
    b = {r["order_id"]: r for r in base.allocation}
    r = {row["order_id"]: row for row in revised.allocation}
    changes = []
    plant_changes = dispatch_changes = 0
    for oid in sorted(set(b) | set(r)):
        old, new = b.get(oid), r.get(oid)
        if (
            old is None
            or new is None
            or old["plant"] != new["plant"]
            or old["dispatch"] != new["dispatch"]
        ):
            plant_changes += int(
                old is None or new is None or old["plant"] != new["plant"]
            )
            dispatch_changes += int(
                old is None or new is None or old["dispatch"] != new["dispatch"]
            )
            changes.append(
                {
                    "order_id": oid,
                    "before": {
                        k: old[k]
                        for k in (
                            "plant",
                            "dispatch",
                            "arrival",
                            "cost",
                            "carbon_kg",
                            "on_time",
                        )
                    }
                    if old
                    else None,
                    "after": {
                        k: new[k]
                        for k in (
                            "plant",
                            "dispatch",
                            "arrival",
                            "cost",
                            "carbon_kg",
                            "on_time",
                        )
                    }
                    if new
                    else None,
                    "reason": "Solver reallocated under the changed inputs; inspect the run evidence for constraints.",
                }
            )
    fields = (
        "total_cost",
        "total_carbon_t",
        "late_orders",
        "total_lateness_days",
        "unassigned_orders",
        "unassigned_units",
        "fulfilled_units",
    )
    delta = {key: revised.kpis[key] - base.kpis[key] for key in fields}
    percentages = {
        key: (100 * delta[key] / base.kpis[key] if base.kpis[key] else None)
        for key in ("total_cost", "total_carbon_t")
    }
    common = set(b) & set(r)
    comparable = set(b) == set(r)

    def production_map(plan):
        return {
            (row["order_id"], row["plant"], row["date"]): row["qty"]
            for row in plan.production_schedule
        }

    old_production, new_production = production_map(base), production_map(revised)
    production_changes = [
        {
            "order_id": key[0],
            "plant": key[1],
            "date": key[2],
            "before_qty": old_production.get(key, 0),
            "after_qty": new_production.get(key, 0),
        }
        for key in sorted(set(old_production) | set(new_production))
        if old_production.get(key, 0) != new_production.get(key, 0)
    ]
    return {
        "available": True,
        "sign_convention": "revised minus baseline",
        "currency": base.kpis["currency"],
        "delta": delta,
        "percentage_delta": percentages,
        "comparable_fulfillment": comparable,
        "interpretation": "Same assigned workload."
        if comparable
        else "Fulfillment changed; lower totals must not be labelled savings.",
        "matched_order_count": len(common),
        "matched_order_delta": {
            "cost": sum(r[i]["cost"] - b[i]["cost"] for i in common),
            "carbon_kg": sum(r[i]["carbon_kg"] - b[i]["carbon_kg"] for i in common),
        },
        "plant_assignment_changes": plant_changes,
        "dispatch_changes": dispatch_changes,
        "changes": changes,
        "production_bucket_changes": len(production_changes),
        "production_changes": production_changes,
        "plant_utilisation_by_week": {
            "before": base.kpis["plant_utilisation_by_week"],
            "after": revised.kpis["plant_utilisation_by_week"],
        },
    }


def replan_capacity(problem, base_plan, events, weights=None, policy=None):
    if (
        base_plan.data_hash != problem.data_hash
        or base_plan.scenario_hash != problem.scenario_hash
        or base_plan.config_version != problem.config_version
    ):
        raise OptimizationInputError(
            "Baseline must match the provided base problem snapshot/config/scenario"
        )
    scenario_events = [e.model_dump(mode="json") for e in problem.events] + list(events)
    # Equal IDs are deduplicated by prepare_problem; conflicting IDs are rejected.
    revised_problem = prepare_problem(
        problem.tables,
        problem.config,
        {"capacity_events": scenario_events},
        problem.dataset_version_id,
    )
    revised_problem = replace(revised_problem, scales=dict(problem.scales))
    result = solve_allocation(revised_problem, weights, policy, base_plan)
    if result.answer_type == "allocation":
        result.answer_type = "scenario"
    return {
        "base": base_plan,
        "revised": result,
        "comparison": compare_plans(base_plan, result),
    }


def compare_weight_settings(problem, weight_sets=None, policy=None):
    policy = SolvePolicy.model_validate(policy or {})
    weights = weight_sets or [
        {"cost": 1, "carbon": 0},
        {"cost": 0.7, "carbon": 0.3},
        {"cost": 0, "carbon": 1},
    ]
    if not 1 <= len(weights) <= 10:
        raise OptimizationInputError("Compare between 1 and 10 weight settings")
    deadline = time.monotonic() + policy.total_time_limit_seconds
    results = []
    points = []
    signatures = {}
    for raw in weights:
        result = solve_allocation(problem, raw, policy, deadline=deadline)
        index = len(results)
        results.append(result)
        if _ready(result):
            signature = tuple(
                (r["order_id"], r["plant"], r["dispatch"]) for r in result.allocation
            )
            duplicate = signatures.get(signature)
            signatures.setdefault(signature, index)
            points.append(
                {
                    "index": index,
                    "weights": result.weights_used,
                    "cost": result.kpis["total_cost"],
                    "carbon_t": result.kpis["total_carbon_t"],
                    "late_orders": result.kpis["late_orders"],
                    "unassigned_orders": result.kpis["unassigned_orders"],
                    "fulfilled_units": result.kpis["fulfilled_units"],
                    "duplicate_of": duplicate,
                    "dominated": False,
                    "solver_status": result.solver["status"],
                }
            )
    for point in points:
        a = results[point["index"]]
        a_ids = {r["order_id"] for r in a.allocation}
        for other in points:
            b = results[other["index"]]
            if a_ids != {r["order_id"] for r in b.allocation}:
                continue
            dims = ("cost", "carbon_t", "late_orders")
            if all(other[k] <= point[k] + 1e-8 for k in dims) and any(
                other[k] < point[k] - 1e-8 for k in dims
            ):
                point["dominated"] = True
    return {
        "results": results,
        "points": points,
        "normalization_scales": problem.scales,
        "limitations": [
            "Sampled weighted trade-offs; not a complete Pareto frontier.",
            "Different weights may produce the same plan.",
        ],
    }


def solve_carbon_target(problem, baseline, carbon_cut_pct, policy=None):
    if (
        not _ready(baseline)
        or not baseline.original_request_met
        or baseline.alternative
    ):
        raise OptimizationInputError(
            "Carbon target requires a complete on-time baseline with verified KPIs"
        )
    if (
        baseline.data_hash != problem.data_hash
        or baseline.config_version != problem.config_version
        or baseline.scenario_hash != problem.scenario_hash
    ):
        raise OptimizationInputError(
            "Carbon target baseline must match the same snapshot/config/scenario"
        )
    import math

    if (
        not isinstance(carbon_cut_pct, (int, float))
        or not math.isfinite(carbon_cut_pct)
        or not 0 <= carbon_cut_pct <= 100
    ):
        raise OptimizationInputError(
            "carbon_cut_pct must be finite and between 0 and 100"
        )
    initial = baseline.kpis["total_carbon_t"] * 1000
    target = initial * (1 - carbon_cut_pct / 100)
    policy = SolvePolicy.model_validate(policy or {}).model_copy(
        update={"carbon_budget_kg": target, "allow_diagnostic_alternative": False}
    )
    result = solve_allocation(problem, {"cost": 1, "carbon": 0}, policy)
    result.problem_context["carbon_target"] = {
        "baseline_carbon_kg": initial,
        "requested_cut_pct": carbon_cut_pct,
        "carbon_cap_kg": target,
        "actual_cut_pct": 100
        * (initial - result.kpis["total_carbon_t"] * 1000)
        / initial
        if result.kpis and initial
        else None,
    }
    if initial == 0:
        result.warnings.append(
            "Baseline carbon is zero; percentage reduction is undefined, and the enforced cap is zero."
        )
    return result

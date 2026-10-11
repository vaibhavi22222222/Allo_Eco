"""Deterministic evidence for the LLM; no unverified causal assertions."""

from __future__ import annotations
import time
from .contracts import SolvePolicy
from .preprocessing import OptimizationInputError, eligible_plants, arrival_date
from .solve import solve_allocation
from .costs import cost_carbon_for_order


def explain_order(problem, plan, order_id, policy=None, counterfactual=False):
    if order_id not in problem.orders:
        raise OptimizationInputError(f"Unknown order {order_id}")
    if (
        plan.data_hash != problem.data_hash
        or plan.config_version != problem.config_version
        or plan.scenario_hash != problem.scenario_hash
    ):
        raise OptimizationInputError(
            "Explanation requires the problem used to produce this plan"
        )
    if plan.kpis is None or not plan.validator.get("valid"):
        raise OptimizationInputError(
            "Explanation requires an independently verified plan"
        )
    policy = SolvePolicy.model_validate(policy or {})
    deadline = time.monotonic() + policy.total_time_limit_seconds
    chosen = next((row for row in plan.allocation if row["order_id"] == order_id), None)
    order = problem.orders[order_id]
    alternatives = []
    for plant in problem.plants:
        record = {"plant": plant}
        if plant not in eligible_plants(problem, order_id):
            record.update({"status": "ineligible", "reason": "Product not supported"})
        else:
            rows = []
            remaining = order["qty"]
            for d in problem.production_dates:
                if arrival_date(problem, order_id, plant, d) > order["due"]:
                    break
                n = min(remaining, problem.daily_capacity[plant, d])
                if n:
                    rows.append(
                        {
                            "order_id": order_id,
                            "plant": plant,
                            "date": d.isoformat(),
                            "qty": n,
                        }
                    )
                    remaining -= n
                if not remaining:
                    break
            if remaining:
                record.update(
                    {
                        "status": "no_on_time_capacity_even_in_isolation",
                        "shortfall_units_in_isolation": remaining,
                    }
                )
            else:
                record.update(
                    {
                        "status": "candidate_in_isolation",
                        "standalone_estimate": cost_carbon_for_order(
                            problem, order_id, plant, rows
                        ),
                        "caveat": "This estimate ignores competition with other orders; it is not a global feasibility claim.",
                    }
                )
                if counterfactual and (chosen is None or plant != chosen["plant"]):
                    forced = policy.model_copy(
                        update={
                            "forced_assignments": {
                                **policy.forced_assignments,
                                order_id: plant,
                            },
                            "allow_diagnostic_alternative": False,
                        }
                    )
                    result = solve_allocation(
                        problem, plan.weights_used, forced, deadline=deadline
                    )
                    record["counterfactual"] = {
                        "status": result.solver["status"],
                        "optimality_proven": result.solver["optimality_proven"],
                        "cost_delta": result.kpis["total_cost"]
                        - plan.kpis["total_cost"]
                        if result.kpis
                        else None,
                        "carbon_delta_kg": (
                            result.kpis["total_carbon_t"] - plan.kpis["total_carbon_t"]
                        )
                        * 1000
                        if result.kpis
                        else None,
                        "comparable_delivery_policy": not plan.alternative,
                        "termination_reason": result.solver["termination_reason"],
                    }
        alternatives.append(record)
    return {
        "order_id": order_id,
        "chosen": chosen,
        "candidates": alternatives,
        "evidence_refs": [
            f"{problem.dataset_version_id}:orders:{order_id}",
            f"config:{problem.config_version}",
        ],
        "limitations": [
            "A plant being full in one plan alone does not prove reallocation impossible.",
            "Counterfactual solves enforce all orders on time; a diagnostic baseline is not directly comparable.",
        ],
    }

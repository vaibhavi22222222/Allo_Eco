"""Due-date-ordered cheapest-feasible heuristic, explicitly NOT an optimizer."""

from __future__ import annotations
from .contracts import ObjectiveWeights, SolvePolicy
from .preprocessing import eligible_plants, arrival_date
from .results import build_result
from .costs import cost_carbon_for_order


def solve_greedy_baseline(problem, policy=None, weights=None):
    policy = SolvePolicy.model_validate(policy or {})
    weights = ObjectiveWeights.model_validate(weights or {})
    remaining = dict(problem.daily_capacity)
    assignments, production = {}, []
    carbon_used = 0.0
    for oid, order in sorted(problem.orders.items(), key=lambda v: (v[1]["due"], v[0])):
        candidates = []
        for plant in eligible_plants(problem, oid):
            if (
                oid in policy.forced_assignments
                and plant != policy.forced_assignments[oid]
            ):
                continue
            qty = order["qty"]
            rows = []
            for d in problem.production_dates:
                if arrival_date(problem, oid, plant, d) > order["due"]:
                    break
                amount = min(qty, remaining[plant, d])
                if amount:
                    rows.append(
                        {
                            "order_id": oid,
                            "plant": plant,
                            "date": d.isoformat(),
                            "qty": amount,
                        }
                    )
                    qty -= amount
                if qty == 0:
                    from datetime import date

                    dispatch = date.fromisoformat(rows[-1]["date"])
                    values = cost_carbon_for_order(problem, oid, plant, rows)
                    if (
                        policy.carbon_budget_kg is None
                        or carbon_used + values["carbon_kg"]
                        <= policy.carbon_budget_kg + 1e-8
                    ):
                        candidates.append(
                            (values["cost"], plant, dispatch, rows, values["carbon_kg"])
                        )
                    break
        if candidates:
            _, plant, dispatch, rows, emissions = min(
                candidates, key=lambda x: (x[0], x[1])
            )
            assignments[oid] = plant, dispatch
            production.extend(rows)
            carbon_used += emissions
            from datetime import date

            for row in rows:
                remaining[plant, date.fromisoformat(row["date"])] -= row["qty"]
    alternative = len(assignments) != len(problem.orders)
    result = build_result(
        problem,
        weights,
        policy,
        assignments,
        production,
        {
            "status": "feasible",
            "termination_reason": "greedy_heuristic",
            "optimality_proven": False,
            "solve_seconds": 0.0,
            "method": "due_date_ordered_cheapest_feasible",
        },
        alternative,
    )
    if alternative:
        # Failure of a heuristic is not a mathematical infeasibility proof.
        result.answer_type = "allocation"
        result.solver["strict_solver_status"] = "unknown"
        result.limitations.append(
            "Greedy left orders unassigned; this does not prove the full problem infeasible."
        )
    return result

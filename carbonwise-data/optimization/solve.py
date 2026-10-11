from __future__ import annotations
import time
import pulp
from .contracts import ObjectiveWeights, SolvePolicy, SolveResult
from .preprocessing import (
    PreparedProblem,
    OptimizationInputError,
    eligible_plants,
    arrival_date,
)
from .model import build_model, weighted_objective
from .results import empty_result, build_result
from .solver_adapter import run_stage, solver_available


def _direct_impossibilities(p, policy):
    issues = []
    for oid, order in p.orders.items():
        options = eligible_plants(p, oid)
        if oid in policy.forced_assignments:
            options = [q for q in options if q == policy.forced_assignments[oid]]
        can_fit = False
        for plant in options:
            timely_days = [
                d
                for d in p.production_dates
                if arrival_date(p, oid, plant, d) <= order["due"]
            ]
            if sum(p.daily_capacity[plant, d] for d in timely_days) >= order["qty"]:
                can_fit = True
                break
        if not can_fit:
            issues.append(
                {
                    "order_id": oid,
                    "reason": "NO_ELIGIBLE_ON_TIME_CAPACITY",
                    "scope": "each eligible plant checked in isolation within planning horizon",
                }
            )
    return issues


def _extract(bundle):
    assignments = {
        (i): (p, t)
        for (i, p, t), v in bundle.x.items()
        if v.varValue is not None and v.varValue > 0.5
    }
    production = [
        {
            "order_id": i,
            "plant": p,
            "date": d.isoformat(),
            "qty": int(round(v.varValue)),
        }
        for (i, p, d), v in bundle.z.items()
        if v.varValue is not None and v.varValue > 0.5
    ]
    return assignments, production


def _snapshot(bundle):
    return {v.name: v.varValue for v in bundle.model.variables()}


def _restore(bundle, values):
    for v in bundle.model.variables():
        if v.name in values:
            v.varValue = values[v.name]


def solve_allocation(
    problem: PreparedProblem,
    weights: ObjectiveWeights | dict | None = None,
    policy: SolvePolicy | dict | None = None,
    base_plan: SolveResult | None = None,
    *,
    deadline: float | None = None,
) -> SolveResult:
    p = problem
    weights = ObjectiveWeights.model_validate(weights or {})
    policy = SolvePolicy.model_validate(policy or {})
    if weights.cost + weights.carbon <= 0:
        raise OptimizationInputError(
            "Strict planning requires a positive cost or carbon weight; lateness-only is degenerate"
        )
    if set(policy.forced_assignments) - set(p.orders):
        raise OptimizationInputError("Forced assignment refers to an unknown order")
    if base_plan and (
        base_plan.problem_context.get("allocation_input_hash")
        != p.allocation_input_hash
        or base_plan.config_version != p.config_version
    ):
        raise OptimizationInputError(
            "Base plan must use the same data/configuration; scenario events may differ"
        )
    started = time.monotonic()
    deadline = min(deadline or float("inf"), started + policy.total_time_limit_seconds)
    trace = []
    if not solver_available():
        return empty_result(
            p,
            weights,
            "error",
            "CBC unavailable; install requirements-optimizer.txt",
            policy,
        )

    def run(bundle, obj, name, warm=False):
        seconds = min(policy.time_limit_seconds, deadline - time.monotonic())
        item = run_stage(bundle, obj, seconds, name, policy.integrality_tolerance, warm)
        trace.append({"tool": "cbc_solve", "output": item})
        return item

    direct = _direct_impossibilities(p, policy)
    strict_bundle = None
    if direct:
        strict = {
            "stage": "strict",
            "status": "infeasible",
            "optimality_proven": False,
            "termination_reason": "necessary_condition_failure",
            "solve_seconds": 0.0,
        }
        trace.append(
            {
                "tool": "strict_presolve",
                "output": {"status": "infeasible", "findings": direct},
            }
        )
    else:
        strict_bundle = build_model(p, policy, False, base_plan)
        strict = run(
            strict_bundle,
            weighted_objective(strict_bundle, weights, p.scales),
            "strict",
        )
    if strict["status"] in ("optimal", "feasible"):
        bundle = strict_bundle
        solver = dict(strict)
        primary_proven = strict["optimality_proven"]
        if (
            base_plan
            and policy.stability_tiebreak
            and primary_proven
            and time.monotonic() < deadline
        ):
            primary = weighted_objective(bundle, weights, p.scales)
            bundle.model += (
                primary <= float(pulp.value(primary)) + policy.objective_tolerance
            )
            saved = _snapshot(bundle)
            tie = run(bundle, bundle.stability, "stability", True)
            if tie["status"] not in ("optimal", "feasible"):
                _restore(bundle, saved)
            solver["stability_status"] = tie["status"]
        assignments, production = _extract(bundle)
        solver["optimality_proven"] = primary_proven
        solver["strict_solver_status"] = strict["status"]
        solver["solve_seconds"] = time.monotonic() - started
        return build_result(
            p, weights, policy, assignments, production, solver, False, trace
        )
    if strict["status"] != "infeasible" or not policy.allow_diagnostic_alternative:
        result = empty_result(
            p, weights, strict["status"], strict["termination_reason"], policy, trace
        )
        result.solver.update(strict)
        result.solver["solve_seconds"] = time.monotonic() - started
        return result

    bundle = build_model(p, policy, True, base_plan)
    solver = None
    saved = None
    stages = [
        ("minimize_missed", bundle.missed),
        ("minimize_unassigned", bundle.unassigned),
        ("weighted_alternative", weighted_objective(bundle, weights, p.scales)),
    ]
    all_proven = True
    for name, obj in stages:
        stage = run(bundle, obj, name, saved is not None)
        if stage["status"] not in ("optimal", "feasible"):
            all_proven = False
            if saved is not None:
                _restore(bundle, saved)
                solver["termination_reason"] = (
                    "later_stage_" + stage["termination_reason"]
                )
            break
        solver = dict(stage)
        saved = _snapshot(bundle)
        if not stage["optimality_proven"]:
            all_proven = False
            break
        if name in ("minimize_missed", "minimize_unassigned"):
            bundle.model += obj == round(pulp.value(obj) or 0)
    if solver is None:
        result = empty_result(
            p,
            weights,
            "infeasible",
            "strict_infeasible_alternative_unavailable",
            policy,
            trace,
        )
        result.solver["strict_solver_status"] = "infeasible"
        result.solver["alternative_solver_status"] = stage["status"]
        result.solver["solve_seconds"] = time.monotonic() - started
        return result
    solver["status"] = "optimal" if all_proven else "feasible"
    solver["optimality_proven"] = all_proven
    solver["strict_solver_status"] = "infeasible"
    solver["alternative_solver_status"] = solver["status"]
    solver["optimization_scope"] = "diagnostic_alternative"
    solver["solve_seconds"] = time.monotonic() - started
    if (
        base_plan
        and policy.stability_tiebreak
        and all_proven
        and time.monotonic() < deadline
    ):
        objective = weighted_objective(bundle, weights, p.scales)
        bundle.model += (
            objective <= float(pulp.value(objective)) + policy.objective_tolerance
        )
        saved = _snapshot(bundle)
        tie = run(bundle, bundle.stability, "alternative_stability", True)
        if tie["status"] not in ("optimal", "feasible"):
            _restore(bundle, saved)
        solver["stability_status"] = tie["status"]
    assignments, production = _extract(bundle)
    result = build_result(
        p, weights, policy, assignments, production, solver, True, trace
    )
    result.solver["solve_seconds"] = time.monotonic() - started
    if not all_proven:
        result.limitations.append(
            "Diagnostic lexicographic optimum not proved; this is the best verified incumbent."
        )
    return result

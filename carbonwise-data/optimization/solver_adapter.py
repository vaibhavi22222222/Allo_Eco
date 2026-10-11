"""CBC status handling: PuLP 'Optimal' alone can hide a timed-out incumbent."""

from __future__ import annotations
import os
from pathlib import Path
import shutil
import tempfile
import time
import pulp


def cbc_solver(seconds: float, log_path: str | None = None, warm_start=False):
    kwargs = {
        "msg": False,
        "timeLimit": max(0.01, seconds),
        "threads": 1,
        "gapRel": 0.0,
        "gapAbs": 0.0,
        "warmStart": warm_start,
        "options": ["randomSeed 42"],
    }
    if log_path:
        kwargs["logPath"] = log_path
    override = os.environ.get("CBC_PATH")
    if override:
        solver = pulp.COIN_CMD(path=override, **kwargs)
    else:
        solver = pulp.PULP_CBC_CMD(**kwargs)
        if not solver.available() and shutil.which("cbc"):
            solver = pulp.COIN_CMD(path=shutil.which("cbc"), **kwargs)
    if not solver.available():
        raise RuntimeError(
            "CBC unavailable. Install requirements-optimizer.txt or set CBC_PATH."
        )
    return solver


def solver_available() -> bool:
    try:
        return bool(cbc_solver(1).available())
    except (RuntimeError, pulp.PulpSolverError):
        return False


def verified_incumbent(model, tolerance=1e-5) -> bool:
    for variable in model.variables():
        if variable.name == "__dummy":
            continue
        value = variable.varValue
        if value is None:
            return False
        import math

        if not math.isfinite(value):
            return False
        if variable.lowBound is not None and value < variable.lowBound - tolerance:
            return False
        if variable.upBound is not None and value > variable.upBound + tolerance:
            return False
        if variable.cat == pulp.LpInteger and abs(value - round(value)) > tolerance:
            return False
    for constraint in model.constraints.values():
        residual = constraint.value()
        if residual is None:
            return False
        if constraint.sense == pulp.LpConstraintEQ and abs(residual) > tolerance:
            return False
        if constraint.sense == pulp.LpConstraintLE and residual > tolerance:
            return False
        if constraint.sense == pulp.LpConstraintGE and residual < -tolerance:
            return False
    return True


def run_stage(bundle, objective, seconds, name, tolerance=1e-5, warm_start=False):
    started = time.monotonic()
    if seconds <= 0:
        return {
            "stage": name,
            "status": "no_solution",
            "optimality_proven": False,
            "termination_reason": "time_budget_exhausted",
            "solve_seconds": 0.0,
        }
    bundle.model.setObjective(objective)
    with tempfile.TemporaryDirectory(prefix="carbonwise-cbc-") as folder:
        path = str(Path(folder) / "cbc.log")
        try:
            solver = cbc_solver(seconds, path, warm_start)
            bundle.model.solve(solver)
            log = Path(path).read_text(errors="replace") if Path(path).exists() else ""
        except (pulp.PulpSolverError, RuntimeError, OSError) as exc:
            return {
                "stage": name,
                "status": "error",
                "optimality_proven": False,
                "termination_reason": str(exc),
                "solve_seconds": time.monotonic() - started,
            }
    solution_status = bundle.model.sol_status
    has_incumbent = solution_status in (
        pulp.LpSolutionOptimal,
        pulp.LpSolutionIntegerFeasible,
    )
    valid = has_incumbent and verified_incumbent(bundle.model, tolerance)
    # Require both statuses and no explicit termination at time/gap/node limits.
    limited = any(
        s in log.lower()
        for s in (
            "stopped on time",
            "exiting on maximum",
            "within gap tolerance",
            "stopped on iterations",
        )
    )
    optimal = valid and solution_status == pulp.LpSolutionOptimal and not limited
    if valid:
        status = "optimal" if optimal else "feasible"
        reason = "optimality_proven" if optimal else "limit_with_verified_incumbent"
    elif bundle.model.status == pulp.LpStatusInfeasible:
        status, reason = "infeasible", "infeasibility_proven"
    elif has_incumbent:
        status, reason = "error", "solver_incumbent_failed_constraint_check"
    else:
        status, reason = "no_solution", "no_verified_incumbent_before_limit"
    interesting = [
        line.strip()
        for line in log.splitlines()
        if any(
            s in line
            for s in (
                "Result -",
                "Objective value:",
                "Lower bound:",
                "Gap:",
                "Enumerated nodes:",
                "Time (Wallclock seconds):",
            )
        )
    ]
    return {
        "stage": name,
        "status": status,
        "optimality_proven": optimal,
        "termination_reason": reason,
        "solve_seconds": time.monotonic() - started,
        "stage_objective": float(pulp.value(objective) or 0) if valid else None,
        "pulp_status": pulp.LpStatus.get(bundle.model.status),
        "pulp_solution_status": pulp.LpSolution.get(solution_status),
        "log_summary": interesting,
        "best_bound": None,
        "relative_gap": None,
    }

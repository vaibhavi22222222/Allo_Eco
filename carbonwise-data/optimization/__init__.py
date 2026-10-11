"""Vibhu's optimization public API."""

from .contracts import (
    PlanningConfig,
    ObjectiveWeights,
    SolvePolicy,
    SolveResult,
    CapacityEvent,
)
from .preprocessing import prepare_problem, PreparedProblem, OptimizationInputError
from .solve import solve_allocation
from .scenarios import (
    compare_plans,
    replan_capacity,
    compare_weight_settings,
    solve_carbon_target,
)
from .explanations import explain_order
from .checker import validate_plan
from .baseline import solve_greedy_baseline
from .solver_adapter import solver_available

__all__ = [
    "PlanningConfig",
    "ObjectiveWeights",
    "SolvePolicy",
    "SolveResult",
    "CapacityEvent",
    "PreparedProblem",
    "OptimizationInputError",
    "prepare_problem",
    "solve_allocation",
    "compare_plans",
    "replan_capacity",
    "compare_weight_settings",
    "solve_carbon_target",
    "explain_order",
    "validate_plan",
    "solve_greedy_baseline",
    "solver_available",
]

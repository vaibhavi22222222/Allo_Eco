"""Backend/LangGraph integration facade over the existing ingestion interface."""

from __future__ import annotations
from pathlib import Path
import yaml

from optimization import PlanningConfig, SolveResult, prepare_problem, solve_allocation
from storage.repository import load_dataset_from_db
from storage.optimizer_repository import save_optimization_run


def load_planning_config(path: str | Path) -> PlanningConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return PlanningConfig.model_validate(yaml.safe_load(stream))


def optimize_tables(
    tables,
    config,
    dataset_version_id="in-memory",
    weights=None,
    policy=None,
    scenario=None,
    base_plan=None,
) -> SolveResult:
    problem = prepare_problem(tables, config, scenario, dataset_version_id)
    return solve_allocation(problem, weights, policy, base_plan)


def optimize_dataset_version(
    dataset_version_id,
    config,
    db_path=None,
    weights=None,
    policy=None,
    scenario=None,
    base_plan=None,
    persist=True,
):
    tables = load_dataset_from_db(dataset_version_id, db_path)
    result = optimize_tables(
        tables, config, dataset_version_id, weights, policy, scenario, base_plan
    )
    payload = result.model_dump(mode="json")
    if persist:
        payload = save_optimization_run(
            dataset_version_id,
            {
                "operation": "allocate",
                "weights": result.weights_used,
                "scenario": scenario,
            },
            payload,
            db_path,
        )
    return payload

"""Executable optimizer demo/CLI. Run from carbonwise-data, or any directory."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ingestion import ingest_all
from services.optimization import load_planning_config
from storage.repository import load_dataset_from_db, default_db_path
from storage.optimizer_repository import save_optimization_run
from optimization import (
    prepare_problem,
    solve_allocation,
    compare_weight_settings,
    replan_capacity,
    solve_carbon_target,
    solve_greedy_baseline,
    explain_order,
    SolvePolicy,
    SolveResult,
)

ROOT = Path(__file__).resolve().parents[1]


def serialize(value):
    if isinstance(value, SolveResult):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Vibhu's deterministic allocation optimizer"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "synthetic_data/main")
    parser.add_argument("--db-path", type=Path, default=default_db_path())
    parser.add_argument(
        "--dataset-version",
        help="Use an existing accepted SQLite version instead of ingesting",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/optimizer.yaml")
    parser.add_argument(
        "--mode",
        choices=[
            "solve",
            "tradeoffs",
            "replan",
            "carbon-target",
            "baseline",
            "explain",
        ],
        default="solve",
    )
    parser.add_argument("--weights", default='{"cost":0.7,"carbon":0.3,"lateness":0}')
    parser.add_argument(
        "--events-json",
        type=Path,
        help="JSON list of fraction-based dated event records",
    )
    parser.add_argument("--carbon-cut", type=float, default=10)
    parser.add_argument("--order-id")
    parser.add_argument("--time-limit", type=float, default=15)
    parser.add_argument("--total-time-limit", type=float, default=60)
    parser.add_argument("--no-alternative", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/optimization_result.json"
    )
    args = parser.parse_args()
    try:
        config = load_planning_config(args.config)
        if args.dataset_version:
            version = args.dataset_version
            tables = load_dataset_from_db(version, args.db_path)
        else:
            ingested = ingest_all(
                args.data_dir, args.db_path, config.planning_start.isoformat()
            )
            tables, version = ingested["tables"], ingested["dataset_version_id"]
        weights = json.loads(args.weights)
        policy = SolvePolicy(
            time_limit_seconds=args.time_limit,
            total_time_limit_seconds=args.total_time_limit,
            allow_diagnostic_alternative=not args.no_alternative,
        )
        problem = prepare_problem(tables, config, dataset_version_id=version)
        import time

        deadline = time.monotonic() + policy.total_time_limit_seconds
        if args.mode == "tradeoffs":
            output = compare_weight_settings(problem, policy=policy)
        elif args.mode == "baseline":
            output = solve_greedy_baseline(problem, policy, weights)
        else:
            base = solve_allocation(problem, weights, policy, deadline=deadline)
            remaining = max(0.01, deadline - time.monotonic())
            rest = policy.model_copy(update={"total_time_limit_seconds": remaining})
            if args.mode == "replan":
                if not args.events_json:
                    raise ValueError("--events-json required for replan")
                events = json.loads(args.events_json.read_text())
                output = replan_capacity(problem, base, events, weights, rest)
            elif args.mode == "carbon-target":
                output = solve_carbon_target(problem, base, args.carbon_cut, rest)
            elif args.mode == "explain":
                if not args.order_id:
                    raise ValueError("--order-id required for explain")
                output = explain_order(
                    problem, base, args.order_id, rest, counterfactual=True
                )
            else:
                output = base
        payload = save_optimization_run(
            version,
            {"mode": args.mode, "weights": weights},
            serialize(output),
            args.db_path,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        top = payload.get("revised", payload)
        print(
            json.dumps(
                {
                    "run_id": payload["run_id"],
                    "dataset_version_id": version,
                    "output": str(args.output),
                    "solver": top.get("solver"),
                    "kpis": {
                        k: v
                        for k, v in (top.get("kpis") or {}).items()
                        if not k.startswith("plant_")
                    },
                },
                indent=2,
            )
        )
        status = top.get("solver", {}).get("status")
        return 2 if status in ("error", "no_solution") else 0
    except Exception as exc:
        # Preserve meaningful CLI diagnostics; backend integration should catch typed errors.
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

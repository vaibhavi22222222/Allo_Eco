"""Reproducible optimizer-only report on the supplied snapshot, with actual timings."""

from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pulp
from optimization import (
    prepare_problem,
    solve_allocation,
    solve_greedy_baseline,
    compare_plans,
    compare_weight_settings,
    replan_capacity,
    SolvePolicy,
)
from services.ingestion import ingest_all
from services.optimization import load_planning_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 20:
        parser.error("--repeat must be 1..20")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_planning_config(ROOT / "config/optimizer.yaml")
    ingested = ingest_all(
        ROOT / "synthetic_data/main",
        ROOT / "data/evaluation.sqlite3",
        cfg.planning_start.isoformat(),
    )
    p = prepare_problem(
        ingested["tables"], cfg, dataset_version_id=ingested["dataset_version_id"]
    )
    policy = SolvePolicy(time_limit_seconds=10, total_time_limit_seconds=45)
    latency = []
    runs = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        runs.append(solve_allocation(p, policy=policy))
        latency.append(time.perf_counter() - started)
    base = runs[0]
    greedy = solve_greedy_baseline(p, policy)
    tradeoffs = compare_weight_settings(p, policy=policy)
    events = json.loads((ROOT / "config/capacity_scenario.example.json").read_text())
    scenario = replan_capacity(p, base, events, policy=policy)
    shutdown = [
        {
            "event_id": "EVAL-SHUTDOWN-" + plant,
            "plant": plant,
            "from_date": cfg.planning_start.isoformat(),
            "to_date": p.end.isoformat(),
            "capacity_loss_pct": 1.0,
        }
        for plant in p.plants
    ]
    impossible_p = prepare_problem(
        p.tables, cfg, {"capacity_events": shutdown}, p.dataset_version_id
    )
    impossible = solve_allocation(impossible_p, policy=policy)
    for name, result in [
        ("allocation", base),
        ("scenario", scenario["revised"]),
        ("infeasible", impossible),
    ]:
        (out / f"sample_{name}.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False)
        )
    tests = []
    xml = ROOT / "data/optimizer_tests.xml"
    if xml.exists():
        for case in ET.parse(xml).findall(".//testcase"):
            tests.append(
                {
                    "name": case.attrib.get("name"),
                    "class": case.attrib.get("classname"),
                    "seconds": float(case.attrib.get("time", "0")),
                    "passed": case.find("failure") is None
                    and case.find("error") is None
                    and case.find("skipped") is None,
                }
            )
    (out / "test_cases.json").write_text(json.dumps(tests, indent=2))
    evaluated = (
        runs + [greedy] + tradeoffs["results"] + [scenario["revised"], impossible]
    )
    computed = [r for r in evaluated if r.kpis is not None]
    report = {
        "scope": "Vibhu optimizer only; no LLM/API/UI evaluation is claimed.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.machine(),
            "pulp": pulp.__version__,
            "cbc": "bundled with PuLP 3.3.0",
        },
        "dataset_version_id": p.dataset_version_id,
        "data_hash": p.data_hash,
        "config_version": p.config_version,
        "row_counts": ingested["row_counts"],
        "assumptions": base.assumptions,
        "warnings": base.warnings,
        "latency": {
            "sample_count": len(latency),
            "seconds": latency,
            "p50_seconds": float(np.percentile(latency, 50)),
            "p95_seconds": float(np.percentile(latency, 95)),
            "method": "Repeated in-process full solve+result-check calls; ingestion excluded.",
            "limitation": "Small local sample; not a production latency guarantee.",
        },
        "computed_plan_count": len(computed),
        "plan_validity_rate": sum(r.validator.get("valid", False) for r in computed)
        / len(computed)
        if computed
        else None,
        "base": {"solver": base.solver, "kpis": base.kpis},
        "greedy": {"solver": greedy.solver, "kpis": greedy.kpis},
        "greedy_to_optimizer": compare_plans(greedy, base),
        "tradeoffs": tradeoffs["points"],
        "capacity_scenario": scenario["comparison"],
        "infeasible_case": {
            "description": "All plants unavailable for entire configured production horizon.",
            "answer_type": impossible.answer_type,
            "solver": impossible.solver,
            "kpis": impossible.kpis,
        },
        "test_summary": {
            "count": len(tests),
            "passed": sum(t["passed"] for t in tests),
            "optimizer_count": sum(
                "tests.optimization" in (t["class"] or "") for t in tests
            ),
        },
        "llm_calls": 0,
        "llm_tokens": 0,
        "llm_cost": 0,
        "limitations": [
            "Optimizer outputs are checked; end-to-end application/LLM evaluation belongs to teammates.",
            "Duty uses input fractions; electricity prices and shared capacities are disclosed synthetic assumptions.",
            "The original exact_answer fixture is not used as optimizer ground truth.",
            "Docker runtime was not available in the implementation environment.",
        ],
    }
    (out / "optimizer_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    # Compact human-readable companion; full per-case evidence remains in JSON.
    lines = [
        "# Optimizer evaluation — measured results",
        "",
        "Scope: Vibhu's deterministic optimization module, not the full application.",
        "",
        f"- Tests recorded: {len(tests)}; passed: {sum(t['passed'] for t in tests)}.",
        f"- Computed plans checked: {len(computed)}; validity rate: {report['plan_validity_rate']:.0%}.",
        f"- Timing sample: {len(latency)} runs; p50 {report['latency']['p50_seconds']:.3f}s; p95 {report['latency']['p95_seconds']:.3f}s.",
        "- Timing includes model preparation inside solve, solving, extraction and independent checking; excludes CSV ingestion.",
        "- Small local sample; not a runtime guarantee.",
        "",
        "| Plan | Cost | Carbon t | Late | Unassigned | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, r in [
        ("Balanced optimizer", base),
        ("Greedy feasible heuristic", greedy),
    ]:
        k = r.kpis
        if k:
            lines.append(
                f"| {name} | {k['total_cost']:.2f} | {k['total_carbon_t']:.6f} | {k['late_orders']} | {k['unassigned_orders']} | {r.solver['status']} |"
            )
    lines += [
        "",
        "Different fulfilled workloads are not like-for-like savings.",
        "See optimizer_report.json for deltas, trade-offs, actual configuration identifiers and all qualifications.",
        "sample_allocation.json, sample_scenario.json and sample_infeasible.json contain solver traces and verification.",
        "test_cases.json lists individual regression/evaluation checks.",
        "No LLM faithfulness or full API/UI results are claimed. Docker was not run in this environment.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "report": str(out / "optimizer_report.json"),
                "validity": report["plan_validity_rate"],
                "tests": report["test_summary"],
                "latency": report["latency"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from .helpers import tables, config, problem, tradeoff, joint, policy
from optimization import (
    prepare_problem,
    solve_allocation,
    validate_plan,
    replan_capacity,
    compare_weight_settings,
    solve_carbon_target,
    solve_greedy_baseline,
    explain_order,
    ObjectiveWeights,
    OptimizationInputError,
    CapacityEvent,
)


def test_hand_calculated_cost_and_carbon():
    t = tables()
    t["orders"].loc[0, ["qty", "due"]] = [100, "2026-10-19"]
    t["plants"].loc[
        0, ["weekly_capacity", "unit_cost", "energy_kwh_per_unit", "carbon_kg_per_kwh"]
    ] = [1500, 40, 2, 0.8]
    t["transport_duties"].loc[
        0, ["transport_cost_per_unit", "transit_days", "duty_pct", "carbon_kg_per_unit"]
    ] = [3, 4, 0.05, 0.4]
    p = problem(t, energy_price_per_kwh={"A": 8})
    r = solve_allocation(p, policy=policy())
    assert r.kpis["total_cost"] == pytest.approx(6180)
    assert r.kpis["total_carbon_t"] == pytest.approx(0.2)
    assert r.validator["valid"]
    assert sum(r.allocation[0]["cost_breakdown"].values()) == pytest.approx(6180)


def test_joint_allocation_optimum_beats_greedy():
    p = joint()
    r = solve_allocation(p, {"cost": 1, "carbon": 0}, policy())
    g = solve_greedy_baseline(p)
    assert {x["order_id"]: x["plant"] for x in r.allocation} == {"O1": "B", "O2": "A"}
    assert r.kpis["total_cost"] == 230
    assert g.kpis["total_cost"] == 300
    assert r.solver["optimality_proven"]


@pytest.mark.parametrize(
    "weight,expected",
    [
        ({"cost": 1, "carbon": 0}, "A"),
        ({"cost": 0.7, "carbon": 0.3}, "B"),
        ({"cost": 0, "carbon": 1}, "C"),
    ],
)
def test_weight_settings(weight, expected):
    r = solve_allocation(tradeoff(), weight, policy())
    assert r.allocation[0]["plant"] == expected
    assert r.validator["valid"]


@pytest.mark.parametrize("transit", [2, 4, 20])
def test_cheapest_route_late(transit):
    p = tradeoff()
    t = {k: v.copy() for k, v in p.tables.items()}
    t["transport_duties"].loc[t["transport_duties"].plant == "A", "transit_days"] = (
        transit
    )
    r = solve_allocation(problem(t), {"cost": 1, "carbon": 0}, policy())
    assert r.allocation[0]["plant"] == "B"
    assert r.allocation[0]["on_time"]


def test_ineligible_plant_not_selected():
    p = tradeoff()
    t = {k: v.copy() for k, v in p.tables.items()}
    t["plants"] = t["plants"].loc[t["plants"].plant == "B"]
    t["transport_duties"] = t["transport_duties"].loc[
        t["transport_duties"].plant == "B"
    ]
    r = solve_allocation(problem(t), policy=policy())
    assert r.allocation[0]["plant"] == "B"


def test_unsupported_product_becomes_infeasible_not_fake_assignment():
    t = tables()
    t["orders"].loc[0, "product"] = "UNKNOWN"
    r = solve_allocation(problem(t), policy=policy())
    assert r.answer_type == "infeasible" and not r.original_request_met
    assert r.kpis["unassigned_orders"] == 1 and not r.allocation
    assert r.validator["valid"]


def test_due_boundary_and_production_before_dispatch():
    p = joint()
    r = solve_allocation(p, policy=policy())
    assert all(x["arrival"] == "2026-10-13" and x["on_time"] for x in r.allocation)
    assert all(x["date"] == "2026-10-12" for x in r.production_schedule)


def test_multi_day_order_is_not_forced_into_one_day():
    t = tables()
    t["orders"].loc[0, ["qty", "due"]] = [25, "2026-10-14"]
    r = solve_allocation(problem(t), policy=policy())
    assert r.original_request_met
    assert len(r.production_schedule) >= 2
    assert sum(x["qty"] for x in r.production_schedule) == 25


def test_shared_capacity_not_multiplied_by_products():
    p = joint()
    t = {k: v.copy() for k, v in p.tables.items()}
    t["plants"] = t["plants"].loc[t["plants"].plant == "A"]
    t["transport_duties"] = t["transport_duties"].loc[
        t["transport_duties"].plant == "A"
    ]
    p = problem(t, horizon_days=1)
    r = solve_allocation(p, policy=policy())
    assert r.answer_type == "infeasible"
    assert r.kpis["fulfilled_units"] == 1 and r.kpis["unassigned_orders"] == 1


def test_inconsistent_capacities_require_explicit_config():
    p = joint()
    t = {k: v.copy() for k, v in p.tables.items()}
    t["plants"].loc[1, "weekly_capacity"] = 10
    with pytest.raises(OptimizationInputError, match="inconsistent"):
        problem(t)
    p = problem(t, plant_weekly_capacity={"A": 5})
    assert p.weekly_capacity["A"] == 5


@pytest.mark.parametrize("loss,expected", [(0.3, 700), (0.5, 500), (1.0, 0)])
def test_two_week_capacity_loss(loss, expected):
    t = tables()
    t["plants"].loc[0, "weekly_capacity"] = 1000
    t["capacity_events"] = pd.DataFrame(
        [
            {
                "event_id": "E",
                "plant": "A",
                "from_date": "2026-10-19",
                "to_date": "2026-11-01",
                "capacity_loss_pct": loss,
            }
        ]
    )
    p = problem(t, horizon_days=28)
    for start, value in [
        (date(2026, 10, 12), 1000),
        (date(2026, 10, 19), expected),
        (date(2026, 10, 26), expected),
        (date(2026, 11, 2), 1000),
    ]:
        from datetime import timedelta

        assert (
            sum(p.daily_capacity["A", start + timedelta(days=i)] for i in range(7))
            == value
        )


def test_partial_week_and_overlap():
    t = tables()
    t["plants"].loc[0, "weekly_capacity"] = 1000
    events = [
        {
            "event_id": "E1",
            "plant": "A",
            "from_date": "2026-10-14",
            "to_date": "2026-10-16",
            "capacity_loss_pct": 0.5,
        },
        {
            "event_id": "E2",
            "plant": "A",
            "from_date": "2026-10-14",
            "to_date": "2026-10-16",
            "capacity_loss_pct": 0.3,
        },
    ]
    p = prepare_problem(t, config(t), {"capacity_events": events})
    assert sum(p.daily_capacity.values()) == 700


def test_shutdown_and_zero_utilization_denominator():
    t = tables(
        events=[
            {
                "event_id": "E",
                "plant": "A",
                "from_date": "2026-10-12",
                "to_date": "2026-10-18",
                "capacity_loss_pct": 1,
            }
        ]
    )
    r = solve_allocation(problem(t), policy=policy())
    assert r.answer_type == "infeasible" and r.kpis["unassigned_orders"] == 1
    assert r.kpis["plant_utilisation"]["A"] is None


def test_carbon_target_enforced():
    p = tradeoff()
    base = solve_allocation(p, {"cost": 1, "carbon": 0}, policy())
    r = solve_carbon_target(p, base, 10, policy())
    assert r.allocation[0]["plant"] == "B"
    assert r.kpis["total_carbon_t"] * 1000 <= 90 + 1e-6
    assert r.kpis["total_cost"] == 110


def test_impossible_carbon_target_does_not_drop_orders():
    p = tradeoff()
    base = solve_allocation(p, {"cost": 1, "carbon": 0}, policy())
    r = solve_carbon_target(p, base, 100, policy())
    assert r.answer_type == "infeasible" and r.kpis is None


def test_weekly_carbon_uses_actual_production_date():
    t = tables()
    t["orders"].loc[0, ["qty", "due"]] = [25, "2026-10-20"]
    p = problem(
        t,
        horizon_days=14,
        weekly_carbon_overrides=[
            {"plant": "A", "week_start": "2026-10-19", "value": 1}
        ],
    )
    r = solve_allocation(p, {"cost": 0, "carbon": 1}, policy())
    # 20 units on low-carbon Monday, remaining 5 in prior week at factor 100.
    assert r.kpis["total_carbon_t"] * 1000 == pytest.approx(520)
    assert r.validator["valid"]


def test_immutable_baseline_replan_and_event_deduplication():
    p = tradeoff()
    base = solve_allocation(p, {"cost": 1, "carbon": 0}, policy())
    before = base.model_dump(mode="json")
    ev = {
        "event_id": "E",
        "plant": "A",
        "from_date": "2026-10-12",
        "to_date": "2026-10-18",
        "capacity_loss_pct": 1,
    }
    output = replan_capacity(p, base, [ev, ev], {"cost": 1, "carbon": 0}, policy())
    assert base.model_dump(mode="json") == before
    assert output["revised"].allocation[0]["plant"] == "B"
    assert output["comparison"]["delta"]["total_cost"] == 10
    assert output["comparison"]["comparable_fulfillment"]


def test_conflicting_duplicate_event_rejected():
    t = tables(
        events=[
            {
                "event_id": "E",
                "plant": "A",
                "from_date": "2026-10-12",
                "to_date": "2026-10-18",
                "capacity_loss_pct": 0.3,
            }
        ]
    )
    with pytest.raises(OptimizationInputError, match="Conflicting"):
        prepare_problem(
            t,
            config(t),
            {
                "capacity_events": [
                    {
                        "event_id": "E",
                        "plant": "A",
                        "from_date": "2026-10-12",
                        "to_date": "2026-10-18",
                        "capacity_loss_pct": 0.5,
                    }
                ]
            },
        )


def test_comparison_marks_reduced_fulfillment():
    p = problem()
    base = solve_allocation(p, policy=policy())
    out = replan_capacity(
        p,
        base,
        [
            {
                "event_id": "E",
                "plant": "A",
                "from_date": "2026-10-12",
                "to_date": "2026-10-18",
                "capacity_loss_pct": 1,
            }
        ],
        policy=policy(),
    )
    assert not out["comparison"]["comparable_fulfillment"]
    assert "not be labelled savings" in out["comparison"]["interpretation"]


@pytest.mark.parametrize(
    "weights",
    [{"cost": -1}, {"cost": float("nan")}, {"cost": 0, "carbon": 0, "lateness": 0}],
)
def test_invalid_weights(weights):
    with pytest.raises(ValidationError):
        ObjectiveWeights.model_validate(weights)


def test_lateness_only_rejected():
    with pytest.raises(OptimizationInputError, match="lateness-only"):
        solve_allocation(problem(), {"cost": 0, "carbon": 0, "lateness": 1})


def test_missing_energy_price_and_fraction_scale():
    t = tables()
    with pytest.raises(OptimizationInputError, match="energy price"):
        problem(t, energy_price_per_kwh={})
    with pytest.raises(ValidationError):
        CapacityEvent(
            event_id="E",
            plant="A",
            from_date="2026-10-12",
            to_date="2026-10-18",
            capacity_loss_pct=30,
        )


def test_checker_catches_tampering():
    p = problem()
    r = solve_allocation(p, policy=policy())
    bad = r.model_dump(mode="json")
    bad["allocation"][0]["cost"] += 50
    assert not validate_plan(p, bad)["valid"]
    bad = r.model_dump(mode="json")
    bad["production_schedule"][0]["qty"] += 100
    assert not validate_plan(p, bad)["valid"]
    bad = r.model_dump(mode="json")
    bad["production_schedule"][0]["date"] = "2026-10-18"
    assert not validate_plan(p, bad)["valid"]


def test_permutation_invariance():
    p = joint()
    r = solve_allocation(p, {"cost": 1, "carbon": 0}, policy())
    shuffled = {
        n: f.sample(frac=1, random_state=22).reset_index(drop=True)
        for n, f in p.tables.items()
    }
    p2 = problem(shuffled)
    r2 = solve_allocation(p2, {"cost": 1, "carbon": 0}, policy())
    assert p.data_hash == p2.data_hash
    assert r.kpis["total_cost"] == r2.kpis["total_cost"]


def test_json_contract_serializable():
    r = solve_allocation(tradeoff(), policy=policy())
    payload = json.loads(json.dumps(r.model_dump(mode="json"), allow_nan=False))
    assert payload["schema_version"] == "vibhu-optimizer-1"
    assert payload["recommendations"][0]["requires_approval"] is True


def test_tradeoff_points_and_explanations():
    p = tradeoff()
    results = compare_weight_settings(p, policy=policy())
    assert [r.allocation[0]["plant"] for r in results["results"]] == ["A", "B", "C"]
    assert all(not x["dominated"] for x in results["points"])
    exp = explain_order(p, results["results"][1], "O1", policy(), counterfactual=True)
    assert exp["chosen"]["plant"] == "B"
    assert any("counterfactual" in c for c in exp["candidates"])


def test_unknown_explanation_and_stale_baseline():
    p = tradeoff()
    r = solve_allocation(p, policy=policy())
    with pytest.raises(OptimizationInputError):
        explain_order(p, r, "UNKNOWN")
    altered = r.model_copy(update={"config_version": "old"})
    with pytest.raises(OptimizationInputError):
        solve_carbon_target(p, altered, 10)


def test_timeout_status_is_not_infeasibility(monkeypatch):
    import optimization.solve as module

    def no_solution(*args, **kwargs):
        return {
            "stage": "strict",
            "status": "no_solution",
            "optimality_proven": False,
            "termination_reason": "no_verified_incumbent_before_limit",
            "solve_seconds": 0,
        }

    monkeypatch.setattr(module, "run_stage", no_solution)
    r = solve_allocation(problem(), policy=policy())
    assert (
        r.solver["status"] == "no_solution"
        and r.answer_type != "infeasible"
        and r.kpis is None
    )


def test_feasible_incumbent_is_not_optimal(monkeypatch):
    import optimization.solve as module

    original = module.run_stage

    def limited(*args, **kwargs):
        out = original(*args, **kwargs)
        out.update(
            status="feasible",
            optimality_proven=False,
            termination_reason="limit_with_verified_incumbent",
        )
        return out

    monkeypatch.setattr(module, "run_stage", limited)
    r = solve_allocation(problem(), policy=policy())
    assert (
        r.solver["status"] == "feasible"
        and not r.solver["optimality_proven"]
        and r.validator["valid"]
    )


def test_solver_unavailable(monkeypatch):
    import optimization.solve as module

    monkeypatch.setattr(module, "solver_available", lambda: False)
    r = solve_allocation(problem(), policy=policy())
    assert r.solver["status"] == "error" and not r.allocation


def test_existing_sqlite_version_integration(tmp_path):
    from services.ingestion import ingest_all
    from services.optimization import optimize_dataset_version, load_planning_config
    from storage.optimizer_repository import load_optimization_run

    root = Path(__file__).resolve().parents[2]
    db = tmp_path / "db.sqlite3"
    data = ingest_all(root / "synthetic_data/main", db)
    cfg = load_planning_config(root / "config/optimizer.yaml")
    out = optimize_dataset_version(
        data["dataset_version_id"],
        cfg,
        db,
        policy={"time_limit_seconds": 6, "total_time_limit_seconds": 15},
    )
    assert out["validator"]["valid"]
    assert out["kpis"]["fulfilled_units"] == 10677
    assert (
        load_optimization_run(out["run_id"], db)["dataset_version_id"]
        == data["dataset_version_id"]
    )


def test_duty_can_reverse_cheapest_choice():
    p = tradeoff()
    t = {k: v.copy() for k, v in p.tables.items()}
    t["transport_duties"].loc[t["transport_duties"].plant == "A", "duty_pct"] = 0.5
    r = solve_allocation(problem(t), {"cost": 1, "carbon": 0}, policy())
    assert r.allocation[0]["plant"] == "B"


def test_diagnostic_preserves_capacity_and_flags_late():
    t = tables()
    t["orders"].loc[0, "qty"] = 21
    r = solve_allocation(problem(t), policy=policy())
    assert r.answer_type == "infeasible" and r.alternative
    assert r.kpis["late_orders"] == 1
    assert not r.validator["delivery_ok"] and r.validator["capacity_ok"]
    assert r.solver["strict_solver_status"] == "infeasible"

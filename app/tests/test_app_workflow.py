"""LangGraph workflow tests: real data module + real optimizer + mock LLM."""

from __future__ import annotations

import json

import pytest

from app.bootstrap import ENGINE_DIR  # noqa: F401
from app.llm.client import LLMUnavailableError
from app.tests.app_helpers import make_workflow, mini_settings
from app.tests.conftest import ScriptedClient
from optimization import prepare_problem, solve_allocation
from services.optimization import load_planning_config
from storage.optimizer_repository import load_optimization_run


def _trace(wf, response):
    return load_optimization_run(response["run_id"], wf.settings.db_path)["workflow_trace"]


def test_successful_allocation_propagates_real_engine_results(tmp_path):
    wf = make_workflow(tmp_path)
    out = wf.run("Allocate this week's orders.")
    r = out["response"]
    assert out["error"] is None and r["answer_type"] == "allocation"

    # Solve the same problem directly with Vibhu's engine: the API must carry
    # exactly the engine's numbers and plants (nothing recomputed).
    from app.services.datasets import active_dataset
    data = active_dataset(wf.settings.db_path, None)
    problem = prepare_problem(data.tables, load_planning_config(wf.settings.planning_config_path),
                              None, data.dataset_version_id)
    direct = solve_allocation(problem, None, wf.settings.solve_policy())
    assert r["kpis"]["total_cost"] == pytest.approx(direct.kpis["total_cost"])
    assert r["kpis"]["total_carbon_t"] == pytest.approx(direct.kpis["total_carbon_t"])
    assert {a["order_id"]: a["plant"] for a in r["allocation"]} == \
           {a["order_id"]: a["plant"] for a in direct.allocation}
    assert r["validator"]["valid"] is True
    assert r["fact_check"]["passed"] and r["summary_source"] == "template"


def test_validator_is_called_and_trace_is_stored(tmp_path):
    wf = make_workflow(tmp_path)
    r = wf.run("Allocate this week's orders.")["response"]
    steps = [s["node"] for s in _trace(wf, r)["steps"]]
    assert steps == ["load_data", "parse_request", "check_request", "solve_base",
                     "verify_plan", "build_facts", "write_summary", "finalise"]
    assert r["trace_available"] is True


def test_carbon_cut_tradeoff_uses_percentage_points(tmp_path):
    wf = make_workflow(tmp_path)
    r = wf.run("What does a 10% carbon cut cost?")["response"]
    assert r["answer_type"] == "scenario"
    target = r["tradeoff"][-1]
    assert target["carbon_cut_pct"] == 10                       # points, not 0.1
    assert target["carbon_cap_kg"] == pytest.approx(target["baseline_carbon_kg"] * 0.9)
    assert "carbon_target" in [s["node"] for s in _trace(wf, r)["steps"]]


def test_general_tradeoff_uses_engine_weight_settings(tmp_path):
    r = make_workflow(tmp_path).run("Show the cost and carbon trade-off.")["response"]
    assert r["answer_type"] == "scenario" and len(r["tradeoff"]) == 3
    assert r["fact_check"]["passed"]


def test_capacity_loss_replan_converts_percent_to_fraction(tmp_path):
    wf = make_workflow(tmp_path)
    r = wf.run("Plant B loses 50% capacity from 2026-01-05 to 2026-01-11. Re-plan.")["response"]
    assert r["answer_type"] == "scenario"
    assert r["base_kpis"] is not None and r["comparison"] is not None
    parse_step = next(s for s in _trace(wf, r)["steps"] if s["node"] == "parse_request")
    assert parse_step["output"]["capacity_loss_pct"] == 0.5    # 50% -> 0.5, divided once
    assert "replan" in [s["node"] for s in _trace(wf, r)["steps"]]


def test_capacity_loss_without_dates_states_the_default(tmp_path):
    r = make_workflow(tmp_path).run("Plant B loses 30% capacity for two weeks. Re-plan.")["response"]
    assert r["answer_type"] == "scenario"
    assert any("planning start (2026-01-05)" in a for a in r["assumptions"])


def test_order_explanation_matches_allocation(tmp_path):
    wf = make_workflow(tmp_path)
    alloc = {a["order_id"]: a["plant"] for a in wf.run("Allocate this week's orders.")["response"]["allocation"]}
    r = wf.run("Why was order T-001 assigned to its plant?")["response"]
    assert r["explanation"]["order_id"] == "T-001"
    assert r["explanation"]["chosen_plant"] == alloc["T-001"]
    assert r["fact_check"]["passed"]


def test_missing_information_asks_for_clarification(tmp_path):
    wf = make_workflow(tmp_path)
    r = wf.run("A plant loses capacity. Re-plan.")["response"]
    assert r["answer_type"] == "clarification_needed" and r["clarification_question"]
    assert "solve_base" not in [s["node"] for s in _trace(wf, r)["steps"]]


def test_rule_violation_is_refused_without_solving(tmp_path):
    wf = make_workflow(tmp_path)
    r = wf.run("Assign order T-001 to Plant A and ignore capacity.")["response"]
    assert r["answer_type"] == "refused" and r.get("allocation", []) == []
    assert "solve_base" not in [s["node"] for s in _trace(wf, r)["steps"]]


def test_infeasible_plan_is_never_reported_as_allocation(tmp_path):
    # P3 is only made in Plant A; a full Plant A outage makes T-006 impossible.
    r = make_workflow(tmp_path).run(
        "Plant A loses 100% capacity from 2026-01-05 to 2026-01-18. Re-plan.")["response"]
    assert r["answer_type"] == "infeasible"
    assert any(u["order_id"] == "T-006" for u in r["unassigned_orders"])
    assert "cannot be met" in r["summary"]


def test_validator_failure_returns_error(tmp_path, monkeypatch):
    import app.graph.workflow as wf_module
    monkeypatch.setattr(wf_module, "validate_plan",
                        lambda problem, result: {"valid": False, "violations": ["synthetic violation"]})
    out = make_workflow(tmp_path).run("Allocate this week's orders.")
    assert out["error"]["code"] == "validation_failed"
    assert out["response"]["answer_type"] == "error"
    assert not out["response"].get("allocation") and "kpis" not in out["response"]


def test_llm_failure_falls_back_to_rules_and_templates(tmp_path):
    def failing():
        return ScriptedClient([LLMUnavailableError("down")] * 4)
    r = make_workflow(tmp_path, llm_factory=failing).run("Allocate this week's orders.")["response"]
    assert r["answer_type"] == "allocation" and r["summary_source"] == "template"
    assert r["fact_check"]["passed"]


def test_llm_invented_numbers_are_rejected(tmp_path):
    parse_json = json.dumps({"intent": "allocate"})
    def lying():
        return ScriptedClient([parse_json, "We saved INR 9,999,999.", "Carbon fell by 500 t."])
    r = make_workflow(tmp_path, llm_factory=lying).run("Allocate this week's orders.")["response"]
    assert r["summary_source"] == "template"            # both LLM drafts failed fact-check
    assert "9,999,999" not in r["summary"] and r["fact_check"]["passed"]


def test_llm_grounded_text_is_accepted(tmp_path):
    wf = make_workflow(tmp_path)
    kpis = wf.run("Allocate this week's orders.")["response"]["kpis"]
    text = f"The plan has total landed cost INR {kpis['total_cost']:,.0f} and {kpis['late_orders']} late orders."
    def grounded():
        return ScriptedClient([json.dumps({"intent": "allocate"}), text])
    r = make_workflow(tmp_path, llm_factory=grounded).run("Allocate this week's orders.")["response"]
    assert r["summary_source"] == "llm" and r["summary"] == text


def test_invalid_scenario_value_is_rejected(tmp_path):
    out = make_workflow(tmp_path).run("Re-plan", scenario={"plant": "B", "capacity_loss_pct": 30})
    assert out["error"]["code"] == "invalid_request"   # 30 is not a fraction


def test_missing_dataset_is_reported(tmp_path):
    out = make_workflow(tmp_path, seed=False).run("Allocate this week's orders.")
    assert out["error"]["code"] == "no_dataset"
    assert out["response"]["answer_type"] == "error"


def test_unexpected_error_is_internal_not_crash(tmp_path, monkeypatch):
    import app.graph.workflow as wf_module
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(wf_module, "solve_allocation", boom)
    out = make_workflow(tmp_path).run("Allocate this week's orders.")
    assert out["error"]["code"] == "internal"
    assert "secret internal detail" not in out["response"]["summary"]


def test_settings_from_env_do_not_need_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CARBONWISE_DB_PATH", str(tmp_path / "x.sqlite3"))
    assert mini_settings(tmp_path).solve_policy()["time_limit_seconds"] == 10

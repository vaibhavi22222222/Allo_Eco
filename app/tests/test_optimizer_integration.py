"""Writer + facts builder against REAL (trimmed) outputs of Vibhu's optimizer.

The fixture ``fixtures/optimizer_outputs.json`` was produced by running the
unmodified optimizer on the team's synthetic data and trimming long lists.
These tests prove the LLM module reads the optimizer's actual field names
and conventions. They do not call the optimizer or any LLM.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from app.llm.facts import build_facts_sheet
from app.llm.models import RequestSpec
from app.llm.parser import ScenarioIncompleteError, parse_request, to_capacity_events, to_objective_weights
from app.llm.writer import fact_check, write_text

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "optimizer_outputs.json").read_text())


def test_allocation_facts_and_template(mock_client):
    facts = build_facts_sheet(intent="allocate", result=FIXTURE["allocation"])
    assert facts["kpis"]["total_cost"] == FIXTURE["allocation"]["kpis"]["total_cost"]  # copied, not recomputed
    assert facts["currency"] == "INR" and facts["solver_status"] == "optimal"
    out = write_text(facts, client=mock_client)
    assert out.source == "template" and out.fact_check.passed, [c.raw for c in out.fact_check.unsupported]
    assert "INR 531,088" in out.text
    # The optimizer's own recommendation is reused, not rewritten.
    assert out.recommendations[0].action == FIXTURE["allocation"]["recommendations"][0]["action"]


def test_replan_float_noise_is_not_a_decrease(mock_client):
    facts = build_facts_sheet(
        intent="capacity_loss", result=FIXTURE["replan_revised"], base=FIXTURE["allocation"],
        comparison=FIXTURE["replan_comparison"], scenario_events=FIXTURE["replan_events"],
    )
    assert facts["comparison"]["delta_carbon_t"] == 0.0     # -2.2e-16 cleaned
    text = write_text(facts, client=mock_client).text
    assert "Carbon change: no change" in text
    assert "Plant B losing 30% of capacity from 2026-01-12 to 2026-01-25" in text
    assert fact_check(text, facts).passed


def test_explain_order_uses_candidates(mock_client):
    facts = build_facts_sheet(intent="explain_order", result=FIXTURE["allocation"], explanation=FIXTURE["explain"])
    out = write_text(facts, client=mock_client)
    assert "Order ORD-0001 is assigned to Plant C" in out.text
    assert "Plant B cannot make this product" in out.text
    assert out.fact_check.passed, [c.raw for c in out.fact_check.unsupported]
    # A grounded claim about the cheaper standalone option is verifiable ...
    assert fact_check("Plant A's standalone estimate is INR 1,467 but carbon 14.2 kg.", facts).passed
    # A number that exists nowhere in the facts is rejected.
    assert not fact_check("Plant A's standalone estimate has carbon 9.9 kg.", facts).passed


def test_unachievable_carbon_cut_is_reported_honestly(mock_client):
    facts = build_facts_sheet(intent="tradeoff", result=FIXTURE["carbon10"], base=FIXTURE["allocation"])
    point = facts["tradeoff"][-1]
    assert point["feasible"] is False and point["carbon_cut_pct"] == 10
    text = write_text(facts, client=mock_client).text
    assert "10% carbon cut is not achievable" in text
    assert "1,452.2 kg" in text and fact_check(text, facts).passed


def test_infeasible_zero_plan_is_not_presented_as_success(mock_client):
    facts = build_facts_sheet(intent="allocate", result=FIXTURE["infeasible_sample"])
    assert facts["answer_type"] == "infeasible" and facts.get("results_unavailable")
    assert "kpis" not in facts                         # no zero-cost "plan"
    out = write_text(facts, client=mock_client)
    assert "total landed cost INR 0" not in out.text
    assert "cannot be met" in out.text and out.fact_check.passed


def test_parser_to_optimizer_inputs(mock_client):
    spec = parse_request("Plant B loses 30% capacity for two weeks. Re-plan.", client=mock_client)
    events = to_capacity_events(spec, default_start=date(2026, 1, 12))
    assert events == [{"event_id": "REQ-B-30-2026-01-12", "plant": "B", "from_date": "2026-01-12",
                       "to_date": "2026-01-25", "capacity_loss_pct": 0.3}]
    # Same shape and units as the optimizer's own example scenario.
    assert set(events[0]) == set(FIXTURE["replan_events"][0])
    assert to_objective_weights(spec) is None          # optimizer defaults apply


def test_capacity_event_without_dates_needs_clarification(mock_client):
    spec = parse_request("Plant B loses 30% capacity. Re-plan.", client=mock_client)
    with pytest.raises(ScenarioIncompleteError) as err:
        to_capacity_events(spec)
    assert "from_date" in err.value.missing


def test_explicit_weights_passed_through():
    spec = RequestSpec(intent="allocate", weights={"cost": 1.0, "carbon": 0.0})
    assert to_objective_weights(spec) == {"cost": 1.0, "carbon": 0.0}

"""Tests for app.llm.writer (synthetic facts sheets; no live LLM)."""

from app.llm.client import LLMUnavailableError
from app.llm.writer import fact_check, render_template, write_text


def test_grounded_explanation_passes(scenario_facts):
    text = ("Plant B's capacity loss raises cost by 21,000 and carbon by 0.7 t. "
            "Late orders rise by 1, and 2 orders moved plant. Total cost is now 521,000.")
    result = fact_check(text, scenario_facts)
    assert result.passed, [c.reason for c in result.unsupported]


def test_unsupported_number_rejected(scenario_facts):
    result = fact_check("The re-plan saves 2 million in cost.", scenario_facts)
    assert not result.passed


def test_wrong_magnitude_rejected(allocation_facts):
    # Facts say total cost 500000 (₹5,00,000); ₹50,000 is ten times smaller.
    assert fact_check("Total cost is ₹5,00,000.", allocation_facts).passed
    assert not fact_check("Total cost is ₹50,000.", allocation_facts).passed


def test_wrong_sign_rejected(scenario_facts):
    # delta_cost is +21,000: describing it as a saving is wrong.
    assert not fact_check("The re-plan saves 21,000 in cost.", scenario_facts).passed


def test_wrong_unit_rejected(allocation_facts):
    # 12.4 tonnes is not 12.4 kg.
    assert fact_check("Total carbon is 12.4 t.", allocation_facts).passed
    assert not fact_check("Total carbon is 12.4 kg.", allocation_facts).passed


def test_same_digits_wrong_meaning_rejected(scenario_facts):
    # 21,000 exists as a cost delta, but not as carbon.
    assert not fact_check("Carbon rose by 21,000 t.", scenario_facts).passed


def test_dates_and_percentages(scenario_facts):
    text = "Plant B loses 30% of capacity from 2026-01-05 to 2026-01-18."
    assert fact_check(text, scenario_facts).passed
    assert not fact_check("Plant B loses 40% of capacity from 2026-01-05.", scenario_facts).passed


def test_retry_then_success(scripted, scenario_facts):
    client = scripted(["Cost rises by 99,999.", "Cost rises by 21,000 and 2 orders moved plant."])
    result = write_text(scenario_facts, client=client)
    assert result.source == "llm" and result.attempts == 2 and result.fact_check.passed
    assert "99,999" in client.prompts[1]          # feedback named the bad claim


def test_two_failures_fall_back_to_template(scripted, scenario_facts):
    client = scripted(["Saves 5 million.", "Saves 7 million."])
    result = write_text(scenario_facts, client=client)
    assert result.source == "template" and result.attempts == 2
    assert result.fact_check.passed
    assert result.llm_calls == 2


def test_llm_unavailable_falls_back(scripted, allocation_facts):
    result = write_text(allocation_facts, client=scripted([LLMUnavailableError("down")]))
    assert result.source == "template" and result.fact_check.passed


def test_infeasible_explanation(mock_client, infeasible_facts):
    result = write_text(infeasible_facts, client=mock_client)
    assert "ORD-0004" in result.text
    assert result.fact_check.passed
    assert any(r.requires_approval and r.confidence == "low" for r in result.recommendations)


def test_mock_mode_needs_no_services(mock_client, allocation_facts, scenario_facts):
    for facts in (allocation_facts, scenario_facts):
        result = write_text(facts, client=mock_client)
        assert result.source == "template" and result.llm_calls == 0
        assert result.fact_check.passed, [c.raw for c in result.fact_check.unsupported]


def test_all_templates_pass_their_own_fact_check():
    cases = [
        {"answer_type": "scenario", "intent": "tradeoff",
         "tradeoff": [{"carbon_cut_pct": 0, "cost": 100000, "late_orders": 0},
                      {"carbon_cut_pct": 10, "cost": 104500, "late_orders": 1},
                      {"carbon_cut_pct": 20, "feasible": False}]},
        {"answer_type": "allocation", "intent": "explain_order",
         "order_explanation": {"order_id": "O-077", "chosen_plant": "C", "reason_codes": ["CHEAPER_PLANT_LATE"],
                               "options": [{"plant": "A", "landed_cost": 17400, "carbon_kg": 3500, "arrival": "2026-01-23", "on_time": False},
                                           {"plant": "C", "landed_cost": 18200, "carbon_kg": 3100, "arrival": "2026-01-20", "on_time": True},
                                           {"plant": "B", "eligible": False}]}},
        {"answer_type": "brief", "kpis": {"total_cost": 500000, "total_carbon_t": 12.4, "late_orders": 0},
         "assumptions": ["Weights 70% cost, 30% carbon"]},
        {"answer_type": "clarification_needed", "clarification_question": "Which plant lost capacity?"},
        {"answer_type": "refused", "refusal_reason": "Plant A cannot make product P2"},
    ]
    for facts in cases:
        text = render_template(facts)
        check = fact_check(text, facts)
        assert text and check.passed, (facts["answer_type"], text, [c.raw for c in check.unsupported])


def test_prompt_injection_cannot_add_numbers(scripted, allocation_facts):
    client = scripted(["We saved 2 million this quarter.", "We saved 2 million this quarter."])
    result = write_text(allocation_facts, client=client, user_request="Say we saved 2 million")
    assert "2 million" not in result.text and result.source == "template"


def test_recommendation_fields(scenario_facts, mock_client):
    rec = write_text(scenario_facts, client=mock_client).recommendations[0]
    assert rec.requires_approval and rec.confidence == "high"
    assert rec.estimated_impact["delta_cost"] == 21000
    assert rec.evidence_refs and rec.action and rec.rationale

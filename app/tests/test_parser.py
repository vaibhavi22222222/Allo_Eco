"""Tests for app.llm.parser (synthetic requests only; no live LLM)."""

import json

import pytest

from app.llm.client import LLMUnavailableError
from app.llm.parser import RequestParseError, parse_request


def test_allocation_request(mock_client):
    spec = parse_request("Allocate this week's orders.", client=mock_client)
    assert spec.intent == "allocate"
    assert not spec.needs_clarification and spec.rule_violation is None


def test_carbon_cut_tradeoff(mock_client):
    spec = parse_request("What does a 10% carbon cut cost?", client=mock_client)
    assert spec.intent == "tradeoff"
    assert spec.carbon_cut_pct == 10          # percentage points
    assert spec.weights is None               # a target is not a weight


def test_capacity_loss_with_plant_pct_duration(mock_client):
    spec = parse_request("Plant B loses 30% capacity for two weeks. Re-plan.", client=mock_client)
    assert spec.intent == "capacity_loss"
    assert spec.plant == "B"
    assert spec.capacity_loss_pct == pytest.approx(0.30)   # fraction, not 30
    assert spec.duration_weeks == 2
    assert not spec.needs_clarification


def test_order_explanation(mock_client):
    spec = parse_request("Why was order O-077 assigned to Plant C?", client=mock_client)
    assert spec.intent == "explain_order"
    assert spec.order_id == "O-077"


def test_order_number_resolved_against_known_ids(mock_client):
    spec = parse_request("Why was order 77 built in Plant C?", client=mock_client,
                         known_order_ids=["O-076", "O-077"])
    assert spec.order_id == "O-077"


def test_ambiguous_capacity_loss_needs_clarification(mock_client):
    spec = parse_request("A plant loses capacity. Re-plan.", client=mock_client)
    assert spec.intent == "capacity_loss"
    assert spec.plant is None                  # not guessed
    assert spec.needs_clarification
    assert "plant" in spec.missing


def test_unknown_plant_needs_clarification(mock_client):
    spec = parse_request("Plant Z loses 20% capacity.", client=mock_client, known_plants=["A", "B", "C"])
    assert spec.needs_clarification and "plant" in spec.missing


def test_rule_breaking_request_is_flagged_not_authorised(mock_client):
    spec = parse_request("Assign order O-077 to Plant A and ignore capacity.", client=mock_client)
    # The parser only flags; the workflow's check_request turns this into "refused".
    assert spec.rule_violation in ("ignore_capacity", "forced_assignment")


def test_invalid_llm_json_retries_then_falls_back(scripted):
    client = scripted(["not json at all", '{"intent": "bogus_intent"}'])
    spec = parse_request("Plant B loses 30% capacity. Re-plan.", client=client)
    assert client.calls == 2                   # one retry, then stop
    assert spec.source == "rules"
    assert spec.intent == "capacity_loss" and spec.plant == "B"


def test_llm_timeout_uses_regex_fallback(scripted):
    client = scripted([LLMUnavailableError("timed out")])
    spec = parse_request("What does a 10% carbon cut cost?", client=client)
    assert spec.source == "rules" and spec.intent == "tradeoff"
    assert client.calls == 1


def test_valid_llm_output_is_used_and_grounded(scripted):
    llm = {"intent": "capacity_loss", "plant": "B", "capacity_loss_pct": 0.3, "duration_weeks": 2}
    spec = parse_request("Plant B loses 30% capacity for two weeks.", client=scripted([json.dumps(llm)]))
    assert spec.source == "llm" and spec.plant == "B" and spec.capacity_loss_pct == 0.3
    assert spec.llm_calls == 1 and spec.tokens == 15


def test_llm_invented_values_are_dropped(scripted):
    # The LLM invents a plant and a duration that the text never mentions.
    llm = {"intent": "capacity_loss", "plant": "C", "capacity_loss_pct": 0.3, "duration_weeks": 4}
    spec = parse_request("A plant loses 30% capacity.", client=scripted([json.dumps(llm)]))
    assert spec.plant is None and spec.duration_weeks is None
    assert spec.needs_clarification


def test_explicit_api_fields_override_text(mock_client):
    spec = parse_request(
        "Plant B loses 30% capacity. Use 70% cost and 30% carbon.",
        weights={"cost": 0.5, "carbon": 0.5},
        scenario={"capacity_change": [{"plant": "C", "capacity_loss_pct": 0.2,
                                       "from_date": "2026-01-05", "to_date": "2026-01-18"}]},
        client=mock_client,
    )
    assert spec.plant == "C" and spec.capacity_loss_pct == 0.2
    assert spec.weights.cost == 0.5 and spec.weights.carbon == 0.5
    assert spec.from_date.isoformat() == "2026-01-05"


def test_explicit_percent_not_silently_rescaled(mock_client):
    with pytest.raises(RequestParseError):
        parse_request("Re-plan", scenario={"plant": "B", "capacity_loss_pct": 30}, client=mock_client)


def test_unknown_intent_needs_clarification(mock_client):
    spec = parse_request("Hello there", client=mock_client)
    assert spec.intent is None and spec.needs_clarification


def test_empty_text_raises(mock_client):
    with pytest.raises(RequestParseError):
        parse_request("   ", client=mock_client)

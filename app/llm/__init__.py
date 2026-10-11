"""LLM module for CarbonWise Allocator.

Public API (import from here; internals may change):

    parse_request(text, weights=None, scenario=None, *, client=None,
                  known_plants=None, known_order_ids=None) -> RequestSpec
    write_text(facts, *, client=None, user_request=None) -> WriterResult
    fact_check(answer, facts) -> FactCheckResult
    render_template(facts) -> str
    build_recommendations(facts) -> list[Recommendation]
    LLMClient, LLMSettings, get_client
    build_facts_sheet(intent=..., result=..., base=..., comparison=...,
                      explanation=..., weight_points=..., scenario_events=...) -> dict
    to_capacity_events(spec, default_start=None) -> list[dict]   # CapacityEvent-shaped
    to_objective_weights(spec) -> dict | None                      # for solve_allocation

The LLM never produces allocations or KPI values. It only (1) turns request
text into structured parameters and (2) words results that code computed.
"""

from app.llm.client import (
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMSettings,
    LLMUnavailableError,
    get_client,
)
from app.llm.models import (
    FactCheckResult,
    Recommendation,
    RequestSpec,
    RequestWeights,
    WriterResult,
)
from app.llm.facts import build_facts_sheet
from app.llm.parser import (
    RequestParseError,
    ScenarioIncompleteError,
    parse_request,
    rule_parse,
    to_capacity_events,
    to_objective_weights,
)
from app.llm.writer import build_recommendations, fact_check, render_template, write_text

__all__ = [
    "LLMClient", "LLMError", "LLMResponseError", "LLMSettings", "LLMUnavailableError",
    "get_client", "FactCheckResult", "Recommendation", "RequestSpec", "RequestWeights",
    "WriterResult", "RequestParseError", "ScenarioIncompleteError", "parse_request",
    "rule_parse", "to_capacity_events", "to_objective_weights", "build_facts_sheet",
    "build_recommendations", "fact_check", "render_template", "write_text",
]

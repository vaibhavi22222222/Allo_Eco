"""Typed data structures used by the LLM module (parser, writer, client).

WHY THIS FILE EXISTS
--------------------
The build plan says shared models should live in ``app/contracts.py``. When this
module was written, that file did not exist yet. To avoid inventing a competing
"official" contract, the models the LLM module needs live here, in one small
file. When the team creates ``app/contracts.py``, these classes can be moved
there unchanged (or re-exported) and the rest of the module keeps working.

Relationship to Vibhu's optimizer contracts
-------------------------------------------
``carbonwise-data/optimization/contracts.py`` defines the optimizer boundary
(``PlanningConfig``, ``ObjectiveWeights``, ``SolvePolicy``, ``CapacityEvent``,
``SolveResult``). Nothing here duplicates those:

* ``RequestSpec`` is the parsed USER REQUEST, which the optimizer has no model
  for. ``parser.to_capacity_events`` turns it into ``CapacityEvent``-shaped
  dicts.
* ``RequestWeights`` holds only the weights the user actually typed (all
  optional). ``parser.to_objective_weights`` passes them on; the optimizer's
  ``ObjectiveWeights`` then applies its own defaults (0.7 / 0.3 / 0).
* ``Recommendation`` validates the guide-6.1 dicts that ``SolveResult``
  already produces (the optimizer stores them as plain dicts).

UNIT CONVENTIONS (verified against the optimizer code)
------------------------------------------------------
* ``capacity_loss_pct`` is a FRACTION from 0 to 1 (0.30 = 30 %), the same as
  ``CapacityEvent.capacity_loss_pct`` (``le=1``) and
  ``PlanningConfig.percentage_convention = "fraction_0_1"``.
* ``carbon_cut_pct`` is in PERCENTAGE POINTS (10 = a 10 % cut), the same as
  ``optimization.scenarios.solve_carbon_target(..., carbon_cut_pct)`` which
  requires 0-100 and divides by 100 itself.
* Weights are non-negative floats with the optimizer's field names
  (cost, carbon, lateness).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# The six intents the workflow knows how to route.
Intent = Literal[
    "allocate",
    "tradeoff",
    "cheapest_on_time",
    "capacity_loss",
    "explain_order",
    "brief",
]

# Rule-breaking requests the parser can DETECT. The parser never authorises
# them; it only flags them so the workflow's check_request node can refuse.
RuleViolation = Literal[
    "ignore_capacity",      # "ignore capacity", "go over capacity", ...
    "forced_assignment",    # "assign order X to plant Y" (bypasses the solver)
    "ineligible_plant",     # "even though plant A can't make it"
    "fabricate_numbers",    # "make up / invent / guess a number"
]


class RequestWeights(BaseModel):
    """Optional objective weights supplied by the user or parsed from text."""

    model_config = ConfigDict(extra="forbid")

    cost: Optional[float] = Field(default=None, ge=0)
    carbon: Optional[float] = Field(default=None, ge=0)
    lateness: Optional[float] = Field(default=None, ge=0)


class RequestSpec(BaseModel):
    """Structured, validated version of a natural-language request.

    This is the output of ``parse_request``. Fields that could not be resolved
    stay ``None``; they are NEVER filled with guesses. When something essential
    is missing, ``needs_clarification`` is True and ``clarification_question``
    explains what to ask the user.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Optional[Intent] = None
    plant: Optional[str] = Field(default=None, description="Plant code, e.g. 'B'")
    capacity_loss_pct: Optional[float] = Field(
        default=None, ge=0, le=1, description="Fraction 0-1 (0.30 = 30 %)"
    )
    duration_weeks: Optional[float] = Field(default=None, gt=0)
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    carbon_cut_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Percentage points (10 = 10 % cut)"
    )
    order_id: Optional[str] = None
    weights: Optional[RequestWeights] = None
    scenario: Optional[dict[str, Any]] = None

    # Guard-rail information for the workflow.
    rule_violation: Optional[RuleViolation] = None
    missing: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    # Provenance, useful for the trace.
    source: Literal["llm", "rules"] = "rules"
    raw_text: str = ""
    llm_calls: int = 0
    tokens: int = 0


class LLMResponse(BaseModel):
    """Raw text returned by one LLM call, plus token usage when available."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ClaimCheck(BaseModel):
    """One numeric claim found in generated text, and whether it is supported."""

    raw: str                      # the exact text matched, e.g. "₹5,00,000"
    value: Optional[float] = None  # parsed numeric value (None for dates)
    unit: str                     # currency / percent / tonnes / kg / count / days / weeks / date / plain
    supported: bool
    matched_fact: Optional[str] = None  # dotted path in the facts sheet, e.g. "kpis.total_cost"
    reason: str = ""


class FactCheckResult(BaseModel):
    """Outcome of checking every number in a text against the facts sheet."""

    passed: bool
    verified: list[ClaimCheck] = Field(default_factory=list)
    unsupported: list[ClaimCheck] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Recommendation shape required by Guide Section 6.1."""

    action: str
    rationale: str
    evidence_refs: list[str]
    confidence: Literal["high", "medium", "low"]
    estimated_impact: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool


class WriterResult(BaseModel):
    """Output of ``write_text``: final text plus how it was produced."""

    text: str
    source: Literal["llm", "template"]
    attempts: int                 # LLM attempts made (0 in mock mode)
    fact_check: FactCheckResult
    recommendations: list[Recommendation] = Field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    fallback_reason: Optional[str] = None

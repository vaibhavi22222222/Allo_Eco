"""HTTP request/response models for the FastAPI layer.

Only API-boundary models live here. They do not duplicate anything:

* Optimizer models (``PlanningConfig``, ``ObjectiveWeights``, ``SolvePolicy``,
  ``CapacityEvent``, ``SolveResult``) stay in ``carbonwise-data/optimization``.
* LLM models (``RequestSpec``, ``RequestWeights``, ``Recommendation``, ...)
  stay in ``app/llm/models.py``. ``RequestWeights`` is reused here.

``AnalyzeResponse`` follows Guide Section 6.1 (run_id, answer_type, summary,
allocation, kpis, tradeoff, weights_used, validator, assumptions, limitations,
recommendations). Fields after ``recommendations`` are documented EXTENSIONS,
each filled only from real engine or LLM-module output.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.llm.models import RequestWeights

# Guide 6.2 answer types plus one documented extension, "error", used when the
# engine or validator fails so a failed plan is never shown as an allocation.
AnswerType = Literal[
    "allocation", "scenario", "brief", "infeasible",
    "clarification_needed", "refused", "error",
]

IngestTable = Literal["orders", "plants", "transport_duties", "capacity_events"]


class AnalyzeRequest(BaseModel):
    """Body of POST /analyze (Guide 6: request, optional weights, optional scenario)."""

    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=2000)
    weights: Optional[RequestWeights] = None
    # Free-form JSON, validated by the parser (e.g. capacity_loss_pct must be 0-1).
    scenario: Optional[dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    """Body returned by POST /analyze."""

    run_id: str
    answer_type: AnswerType
    summary: str
    allocation: list[dict[str, Any]] = Field(default_factory=list)
    kpis: Optional[dict[str, Any]] = None
    tradeoff: list[dict[str, Any]] = Field(default_factory=list)
    weights_used: Optional[dict[str, float]] = None
    validator: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Extensions (all optional; never invented) -------------------------
    intent: Optional[str] = None
    dataset_version_id: Optional[str] = None
    solver: Optional[dict[str, Any]] = None            # status, termination_reason, solve_seconds
    base_kpis: Optional[dict[str, Any]] = None         # baseline for scenarios
    comparison: Optional[dict[str, Any]] = None        # compare_plans() summary
    explanation: Optional[dict[str, Any]] = None       # explain_order() facts
    unassigned_orders: list[dict[str, Any]] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    summary_source: Optional[Literal["llm", "template"]] = None
    fact_check: Optional[dict[str, Any]] = None        # passed + unsupported claims
    warnings: list[str] = Field(default_factory=list)
    trace_available: bool = False


class IngestResponse(BaseModel):
    """Body returned by POST /ingest (Guide 6: rows_loaded, rows_rejected, status)."""

    rows_loaded: int
    rows_rejected: int
    status: Literal["accepted", "rejected"]
    table: IngestTable
    dataset_version_id: Optional[str] = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Body returned by GET /health. Never contains secrets."""

    api: Literal["ok"] = "ok"
    data_store: Literal["ok", "empty", "error"]
    dataset_version_id: Optional[str] = None
    solver: Literal["ok", "unavailable"]
    llm: dict[str, Any]

"""The shared state that flows through the LangGraph workflow.

How workflow state works
------------------------
LangGraph passes ONE dictionary (this ``WorkflowState``) from node to node.
Each node returns only the keys it changed; LangGraph merges them in.

Most keys are simply overwritten. ``trace`` is special: it is annotated with
``operator.add``, so when a node returns ``{"trace": [step]}`` LangGraph
APPENDS the step instead of replacing the list. That is how every node adds
itself to the audit trail served by GET /trace/{run_id}.

The state holds real Python objects (the optimizer's ``PreparedProblem`` and
``SolveResult``, the LLM client). That is fine: the graph runs in memory and
the API only returns the JSON assembled by the final node.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from app.llm.client import LLMClient
from app.llm.models import RequestSpec, WriterResult


class WorkflowError(TypedDict):
    """An error recorded by a node. ``code`` decides the HTTP status later."""

    code: str      # invalid_request | no_dataset | validation_failed | engine_input | internal
    message: str   # safe, user-facing text (never a stack trace)


class WorkflowState(TypedDict, total=False):
    # ---- request (set by the API) ---------------------------------------
    run_id: str
    request: str
    weights: Optional[dict[str, Any]]
    scenario: Optional[dict[str, Any]]
    llm: LLMClient                      # one client per run: counts calls/tokens

    # ---- data (load_data) ------------------------------------------------
    dataset_version_id: str
    tables: dict[str, Any]
    config: Any                         # optimization.PlanningConfig

    # ---- understanding the request (parse_request, check_request) -------
    spec: RequestSpec
    route: str                          # solve | refused | clarify
    refusal_reason: Optional[str]
    clarification_question: Optional[str]
    events: list[dict[str, Any]]        # CapacityEvent-shaped, fractions 0-1
    solver_weights: Optional[dict[str, float]]
    notes: list[str]                    # workflow assumptions shown to the user

    # ---- optimizer outputs (real engine objects) -------------------------
    problem: Any                        # optimization.PreparedProblem
    base: Any                           # optimization.SolveResult (baseline)
    result: Any                         # optimization.SolveResult (plan to report)
    comparison: Optional[dict[str, Any]]
    explanation: Optional[dict[str, Any]]
    weight_points: Optional[list[dict[str, Any]]]
    verification: dict[str, Any]

    # ---- explanation (build_facts, write_summary) ------------------------
    facts: dict[str, Any]
    writer: WriterResult

    # ---- output ----------------------------------------------------------
    error: Optional[WorkflowError]
    response: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]

"""Turn the final workflow state into the API response, and store the run.

How optimizer results reach the dashboard
-----------------------------------------
``SolveResult.model_dump(mode="json")`` (Vibhu's engine) is copied field by
field into the response: ``allocation``, ``kpis``, ``weights_used``,
``validator``, ``assumptions``, ``limitations``, ``warnings``. The text summary
and recommendations come from Abhijeet's ``write_text`` (already fact-checked).
Nothing numeric is computed here; Streamlit only displays these fields.

Storage
-------
Runs are stored with the optimizer's existing ``save_optimization_run`` into
the same SQLite file, table ``optimization_runs``. The workflow trace is kept
inside the stored payload under ``workflow_trace`` and is served by
GET /trace/{run_id}. That table requires an accepted dataset version, so a run
that failed before any data was loaded is answered but not stored.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.bootstrap import ENGINE_DIR  # noqa: F401
from app.graph.state import WorkflowState
from storage.optimizer_repository import save_optimization_run

logger = logging.getLogger(__name__)

# Keep stored payloads small: per-order change lists can be long.
_MAX_CHANGES = 200


def _base_fields(state: WorkflowState) -> dict[str, Any]:
    spec = state.get("spec")
    return {
        "run_id": state["run_id"],
        "intent": spec.intent if spec else None,
        "dataset_version_id": state.get("dataset_version_id"),
    }


def _error_response(state: WorkflowState) -> dict[str, Any]:
    """A failed run is reported as answer_type "error", never as an allocation."""
    error = state["error"]
    return {
        **_base_fields(state),
        "answer_type": "error",
        "summary": error["message"],
        "limitations": ["No plan is reported because the run did not complete successfully."],
    }


def _plan_fields(state: WorkflowState) -> dict[str, Any]:
    """Real optimizer fields for the plan being reported (empty if none)."""
    result = state.get("result")
    if result is None:
        return {}
    r = result.model_dump(mode="json")
    facts = state["facts"]
    usable = "kpis" in facts  # facts builder drops KPIs when there is no usable plan
    verification = state.get("verification") or r.get("validator") or {}
    out = {
        "allocation": r["allocation"] if usable else [],
        "kpis": r["kpis"] if usable else None,
        "weights_used": r.get("weights_used"),
        "validator": {k: v for k, v in verification.items() if k != "recomputed_kpis"},
        "assumptions": list(state.get("notes") or []) + r.get("assumptions", []),
        "limitations": r.get("limitations", []),
        "warnings": r.get("warnings", []),
        "unassigned_orders": r.get("unassigned_orders", []),
        "solver": {k: r["solver"].get(k) for k in
                   ("status", "termination_reason", "optimality_proven", "solve_seconds")},
    }
    if state["spec"].intent in ("capacity_loss", "tradeoff") and state.get("base") is not None:
        out["base_kpis"] = state["base"].model_dump(mode="json")["kpis"]
    return out


def assemble_response(state: WorkflowState) -> dict[str, Any]:
    """Build an ``AnalyzeResponse``-shaped dict from the final state."""
    if state.get("error"):
        return _error_response(state)

    facts, writer = state["facts"], state["writer"]
    response: dict[str, Any] = {
        **_base_fields(state),
        "answer_type": facts["answer_type"],
        "summary": writer.text,
        "tradeoff": facts.get("tradeoff") or facts.get("weight_tradeoff") or [],
        "recommendations": [rec.model_dump() for rec in writer.recommendations],
        "clarification_question": facts.get("clarification_question"),
        "explanation": facts.get("order_explanation"),
        "summary_source": writer.source,
        "fact_check": {
            "passed": writer.fact_check.passed,
            "verified_claims": len(writer.fact_check.verified),
            "unsupported_claims": [c.raw for c in writer.fact_check.unsupported],
        },
        **_plan_fields(state),
    }
    if facts.get("comparison"):
        engine_changes = (state.get("comparison") or {}).get("changes", [])
        response["comparison"] = {**facts["comparison"], "changes": engine_changes[:_MAX_CHANGES]}
    return response


def persist_run(
    state: WorkflowState,
    response: dict[str, Any],
    steps: list[dict[str, Any]],
    db_path: Optional[Path],
) -> bool:
    """Store the run with the optimizer's repository. Returns True if stored."""
    version = state.get("dataset_version_id")
    if not version:
        return False
    llm = state.get("llm")
    engine_trace = state["result"].trace if state.get("result") is not None else []
    payload = {
        **response,
        "workflow_trace": {
            "run_id": state["run_id"],
            "request": state["request"],
            "dataset_version_id": version,
            "answer_type": response["answer_type"],
            "steps": steps,
            "llm_calls": llm.calls if llm else 0,
            "tokens": llm.total_tokens if llm else 0,
            "engine_trace": engine_trace,
        },
    }
    request_record = {"request": state["request"], "weights": state.get("weights"),
                      "scenario": state.get("scenario"),
                      "intent": response.get("intent")}
    try:
        save_optimization_run(version, request_record, payload, db_path, run_id=state["run_id"])
        return True
    except (ValueError, TypeError, sqlite3.Error) as exc:
        # The answer is still returned; we log loudly that it was not stored.
        logger.error("Could not store run %s: %s", state["run_id"], exc)
        return False

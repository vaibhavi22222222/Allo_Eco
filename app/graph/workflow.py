"""CarbonWise LangGraph workflow: request -> real optimizer -> checked explanation.

Graph (one workflow, deterministic routing; the LLM never chooses the route):

    load_data -> parse_request -> check_request --refused/clarify--> build_facts
                                       |
                                     solve
                                       v
                                  solve_base --by intent--> replan | carbon_target
                                       |                    | weight_tradeoff | explain
                                       v                    v
                                  verify_plan <-------------+
                                       v
                                  build_facts -> write_summary -> finalise

Why conditional edges
---------------------
Different intents need different engine calls (re-plan, carbon cap,
explanation). Conditional edges make that routing explicit and testable, and
every node also has an "error exit": if a node records ``state["error"]`` the
next edge jumps straight to ``finalise``. A failed or infeasible plan can
therefore never flow on and be reported as a successful allocation.

Who does what
-------------
* Data:      services.ingestion / storage.repository (via app.services.datasets)
* Numbers:   Vibhu's optimizer (prepare_problem, solve_allocation, replan_capacity,
             solve_carbon_target, compare_weight_settings, explain_order, validate_plan)
* Language:  Abhijeet's LLM module (parse_request, to_capacity_events,
             to_objective_weights, build_facts_sheet, write_text with fact-check)
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph

from app.bootstrap import ENGINE_DIR  # noqa: F401  (makes the engine importable)
from app.config import AppSettings
from app.graph.state import WorkflowState
from app.llm import (
    RequestParseError,
    ScenarioIncompleteError,
    build_facts_sheet,
    get_client,
    parse_request,
    to_capacity_events,
    to_objective_weights,
    write_text,
)
from app.llm.client import LLMClient
from app.services.datasets import NoDatasetError, active_dataset
from app.graph.response import assemble_response, persist_run
from optimization import (
    OptimizationInputError,
    compare_weight_settings,
    explain_order,
    prepare_problem,
    replan_capacity,
    solve_allocation,
    solve_carbon_target,
    validate_plan,
)
from services.ingestion import DatasetIngestionError
from services.optimization import load_planning_config

logger = logging.getLogger(__name__)

# Expected, user-explainable failures -> error code. Anything else is "internal".
_KNOWN_ERRORS: list[tuple[type, str]] = [
    (RequestParseError, "invalid_request"),
    (NoDatasetError, "no_dataset"),
    (DatasetIngestionError, "no_dataset"),
    (OptimizationInputError, "engine_input"),
]


def _summarise(value: Any) -> Any:
    """Short, JSON-safe description of a node output for the trace."""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return value


def traced(agent: str, tool: str) -> Callable:
    """Decorator for nodes: time them, add a trace step, convert errors.

    Errors are never swallowed: known ones become ``state["error"]`` with a
    clear message; unknown ones are logged with their stack trace (server log
    only) and reported to the client as a generic internal error.
    """
    def decorator(node: Callable[["CarbonWiseWorkflow", WorkflowState], dict]) -> Callable:
        @wraps(node)
        def wrapper(self: "CarbonWiseWorkflow", state: WorkflowState) -> dict:
            started = time.perf_counter()
            step: dict[str, Any] = {"node": node.__name__, "agent": agent, "tool": tool}
            try:
                update = node(self, state) or {}
                step["output"] = _summarise(update.pop("_trace_output", {}))
            except Exception as exc:  # noqa: BLE001 - converted, logged, never hidden
                code = next((c for t, c in _KNOWN_ERRORS if isinstance(exc, t)), "internal")
                if code == "internal":
                    logger.exception("Unexpected error in node %s", node.__name__)
                    message = "An internal error occurred. See the server log for run " + state.get("run_id", "")
                else:
                    message = str(exc)
                update = {"error": {"code": code, "message": message}}
                step["error"] = {"code": code, "message": message}
            step["ms"] = round((time.perf_counter() - started) * 1000, 1)
            update["trace"] = [step]
            return update
        return wrapper
    return decorator


class CarbonWiseWorkflow:
    """Builds and runs the LangGraph graph. One instance serves many requests."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        llm_factory: Callable[[], LLMClient] = get_client,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self.llm_factory = llm_factory
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ run
    def run(
        self,
        request: str,
        weights: Optional[dict[str, Any]] = None,
        scenario: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute the graph and return the final state's ``response`` dict.

        The returned dict always has ``response`` (an AnalyzeResponse-shaped
        dict) and ``error`` (None or {code, message}).
        """
        initial: WorkflowState = {
            "run_id": "al-" + uuid.uuid4().hex[:12],
            "request": request,
            "weights": weights,
            "scenario": scenario,
            "llm": self.llm_factory(),
            "trace": [],
        }
        final = self.graph.invoke(initial)
        return {"response": final["response"], "error": final.get("error")}

    # ---------------------------------------------------------------- graph
    def _build_graph(self):
        g = StateGraph(WorkflowState)
        for name in ("load_data", "parse_request", "check_request", "solve_base", "replan",
                     "carbon_target", "weight_tradeoff", "explain", "verify_plan",
                     "build_facts", "write_summary", "finalise"):
            g.add_node(name, getattr(self, name))

        g.add_edge(START, "load_data")
        g.add_conditional_edges("load_data", self._ok_or_finalise("parse_request"))
        g.add_conditional_edges("parse_request", self._ok_or_finalise("check_request"))
        g.add_conditional_edges("check_request", self._route_after_check)
        g.add_conditional_edges("solve_base", self._route_by_intent)
        for node in ("replan", "carbon_target", "weight_tradeoff", "explain"):
            g.add_conditional_edges(node, self._ok_or_finalise("verify_plan"))
        g.add_conditional_edges("verify_plan", self._ok_or_finalise("build_facts"))
        g.add_conditional_edges("build_facts", self._ok_or_finalise("write_summary"))
        g.add_edge("write_summary", "finalise")
        g.add_edge("finalise", END)
        return g.compile()

    @staticmethod
    def _ok_or_finalise(next_node: str) -> Callable[[WorkflowState], str]:
        """Conditional edge: continue normally, or jump to finalise on error."""
        def route(state: WorkflowState) -> str:
            return "finalise" if state.get("error") else next_node
        return route

    @staticmethod
    def _route_after_check(state: WorkflowState) -> str:
        if state.get("error"):
            return "finalise"
        # Refusals and clarifications skip the solver entirely.
        return "solve_base" if state["route"] == "solve" else "build_facts"

    @staticmethod
    def _route_by_intent(state: WorkflowState) -> str:
        if state.get("error"):
            return "finalise"
        spec = state["spec"]
        if spec.intent == "capacity_loss":
            return "replan"
        if spec.intent == "tradeoff":
            return "carbon_target" if spec.carbon_cut_pct is not None else "weight_tradeoff"
        if spec.intent == "explain_order":
            return "explain"
        return "verify_plan"   # allocate, cheapest_on_time, brief

    # ---------------------------------------------------------------- nodes
    @traced("loader", "active_dataset")
    def load_data(self, state: WorkflowState) -> dict:
        """Load the newest accepted dataset and Vibhu's planning config."""
        s = self.settings
        data = active_dataset(s.db_path, s.seed_data_dir)
        config = load_planning_config(s.planning_config_path)
        return {"dataset_version_id": data.dataset_version_id, "tables": data.tables,
                "config": config,
                "_trace_output": {"dataset_version_id": data.dataset_version_id,
                                  "orders": len(data.tables["orders"])}}

    @traced("parser", "parse_request")
    def parse_request(self, state: WorkflowState) -> dict:
        """Abhijeet's parser: text (+ explicit API fields) -> RequestSpec."""
        tables = state["tables"]
        spec = parse_request(
            state["request"], state.get("weights"), state.get("scenario"),
            client=state["llm"],
            known_plants=tables["plants"]["plant"].astype(str).unique().tolist(),
            known_order_ids=tables["orders"]["order_id"].astype(str).tolist(),
        )
        return {"spec": spec, "_trace_output": {
            "intent": spec.intent, "source": spec.source, "plant": spec.plant,
            "capacity_loss_pct": spec.capacity_loss_pct, "carbon_cut_pct": spec.carbon_cut_pct,
            "order_id": spec.order_id, "needs_clarification": spec.needs_clarification,
            "rule_violation": spec.rule_violation}}

    @traced("checker", "check_request")
    def check_request(self, state: WorkflowState) -> dict:
        """Guard rails: refuse rule breaks, ask for missing info, build engine inputs."""
        spec = state["spec"]
        if spec.rule_violation:
            reason = {
                "ignore_capacity": "it asks to ignore or exceed plant capacity",
                "forced_assignment": "it asks to force an allocation instead of letting the solver decide",
                "ineligible_plant": "it asks to use a plant that cannot make the product",
                "fabricate_numbers": "it asks for numbers that were not computed by the optimizer",
            }[spec.rule_violation]
            return {"route": "refused", "refusal_reason": reason,
                    "_trace_output": {"route": "refused", "rule": spec.rule_violation}}
        if spec.needs_clarification:
            return {"route": "clarify", "clarification_question": spec.clarification_question,
                    "_trace_output": {"route": "clarify", "missing": ",".join(spec.missing)}}

        events: list[dict] = []
        notes: list[str] = []
        if spec.intent == "capacity_loss":
            try:
                # Fraction 0-1 passes through unchanged (no second /100).
                events = to_capacity_events(spec, default_start=state["config"].planning_start)
            except ScenarioIncompleteError as exc:
                return {"route": "clarify",
                        "clarification_question": f"Please provide: {', '.join(exc.missing)}.",
                        "_trace_output": {"route": "clarify"}}
            if spec.from_date is None:
                # Make the default visible instead of silently assuming it.
                notes.append(f"No start date was given, so the capacity loss starts at the "
                             f"planning start ({events[0]['from_date']}).")

        weights = to_objective_weights(spec)
        if spec.intent == "cheapest_on_time":
            # Cheapest plan = pure cost objective; the solver's strict stage
            # already requires every order on time.
            weights = {"cost": 1.0, "carbon": 0.0, "lateness": 0.0}
        return {"route": "solve", "events": events, "solver_weights": weights, "notes": notes,
                "_trace_output": {"route": "solve", "events": len(events)}}

    @traced("optimiser", "solve_allocation")
    def solve_base(self, state: WorkflowState) -> dict:
        """Build the problem and solve the baseline plan with Vibhu's engine."""
        problem = prepare_problem(state["tables"], state["config"], None, state["dataset_version_id"])
        base = solve_allocation(problem, state.get("solver_weights"), self.settings.solve_policy())
        return {"problem": problem, "base": base, "result": base,
                "_trace_output": {"status": base.solver.get("status"), "answer_type": base.answer_type}}

    @traced("scenario", "replan_capacity")
    def replan(self, state: WorkflowState) -> dict:
        """Capacity loss: re-solve with the extra events and compare to the base."""
        out = replan_capacity(state["problem"], state["base"], state["events"],
                              state.get("solver_weights"), self.settings.solve_policy())
        comparison = out["comparison"]
        return {"result": out["revised"], "comparison": comparison,
                "_trace_output": {"status": out["revised"].solver.get("status"),
                                  "comparison_available": comparison.get("available"),
                                  "plant_assignment_changes": comparison.get("plant_assignment_changes")}}

    @traced("scenario", "solve_carbon_target")
    def carbon_target(self, state: WorkflowState) -> dict:
        """Carbon cut: percentage points (0-100) go straight to the engine."""
        result = solve_carbon_target(state["problem"], state["base"], state["spec"].carbon_cut_pct,
                                     self.settings.solve_policy())
        return {"result": result, "_trace_output": {"status": result.solver.get("status")}}

    @traced("scenario", "compare_weight_settings")
    def weight_tradeoff(self, state: WorkflowState) -> dict:
        """General trade-off: the engine's sampled cost/carbon weight settings."""
        out = compare_weight_settings(state["problem"], None, self.settings.solve_policy())
        return {"weight_points": out["points"], "_trace_output": {"points": len(out["points"])}}

    @traced("explainer", "explain_order")
    def explain(self, state: WorkflowState) -> dict:
        """Deterministic evidence for one order from the engine (no counterfactual solves)."""
        explanation = explain_order(state["problem"], state["base"], state["spec"].order_id,
                                    self.settings.solve_policy())
        return {"explanation": explanation, "_trace_output": {"order_id": explanation["order_id"]}}

    @traced("checker", "validate_plan")
    def verify_plan(self, state: WorkflowState) -> dict:
        """Independent re-check of the reported plan with Vibhu's validate_plan.

        The engine already checks inside solve_allocation; we call the checker
        again here so the trace proves it ran for THIS response. For a re-plan
        the revised problem is rebuilt exactly as replan_capacity builds it.
        """
        result = state["result"]
        if str(result.solver.get("status", "")).lower() == "error":
            detail = "; ".join(result.solver.get("validation_violations", [])) or \
                result.solver.get("termination_reason", "solver error")
            return {"error": {"code": "validation_failed",
                              "message": f"The optimizer could not produce a valid plan: {detail}"},
                    "_trace_output": {"valid": False}}
        if result.kpis is None:
            # Honest "no plan" (e.g. infeasible carbon cap): nothing to check.
            return {"verification": {"applicable": False},
                    "_trace_output": {"applicable": False}}
        problem = state["problem"]
        if state.get("events"):
            problem = prepare_problem(
                problem.tables, problem.config,
                {"capacity_events": [e.model_dump(mode="json") for e in problem.events] + state["events"]},
                problem.dataset_version_id,
            )
        report = validate_plan(problem, result)
        if not report.get("valid"):
            return {"error": {"code": "validation_failed",
                              "message": "Independent validator rejected the plan: "
                                         + "; ".join(report.get("violations", [])[:5])},
                    "verification": report, "_trace_output": {"valid": False}}
        return {"verification": report,
                "_trace_output": {k: report.get(k) for k in ("valid", "capacity_ok", "eligibility_ok", "delivery_ok")}}

    @traced("writer", "build_facts_sheet")
    def build_facts(self, state: WorkflowState) -> dict:
        """Abhijeet's facts builder: copies engine numbers, never recalculates."""
        route = state.get("route")
        if route == "refused":
            facts = build_facts_sheet(intent=state["spec"].intent, answer_type="refused",
                                      refusal_reason=state.get("refusal_reason"))
        elif route == "clarify":
            facts = build_facts_sheet(intent=state["spec"].intent, answer_type="clarification_needed",
                                      clarification_question=state.get("clarification_question"))
        else:
            facts = self._facts_for_plan(state)
        return {"facts": facts, "_trace_output": {"answer_type": facts.get("answer_type")}}

    @staticmethod
    def _facts_for_plan(state: WorkflowState) -> dict:
        intent = state["spec"].intent
        result_json = state["result"].model_dump(mode="json")
        is_scenario = intent in ("capacity_loss", "tradeoff")
        facts = build_facts_sheet(
            intent=intent,
            result=result_json,
            base=state["base"].model_dump(mode="json") if is_scenario else None,
            comparison=state.get("comparison"),
            explanation=state.get("explanation"),
            weight_points=state.get("weight_points"),
            scenario_events=state.get("events") or None,
        )
        if intent == "brief" and facts["answer_type"] == "allocation":
            facts["answer_type"] = "brief"
        if intent == "tradeoff" and state.get("weight_points"):
            facts["answer_type"] = "scenario"
        return facts

    @traced("writer", "write_text")
    def write_summary(self, state: WorkflowState) -> dict:
        """Abhijeet's writer: LLM text, fact-checked; template on failure or mock."""
        out = write_text(state["facts"], client=state["llm"], user_request=state["request"])
        return {"writer": out, "_trace_output": {
            "source": out.source, "attempts": out.attempts, "fact_check_passed": out.fact_check.passed,
            "llm_calls": out.llm_calls}}

    def finalise(self, state: WorkflowState) -> dict:
        """Assemble the API response and store the run (not traced: it IS the trace)."""
        step = {"node": "finalise", "agent": "api", "tool": "assemble_response"}
        response = assemble_response(state)
        persisted = persist_run(state, response, state["trace"] + [step], self.settings.db_path)
        response["trace_available"] = persisted
        return {"response": response, "trace": [step]}

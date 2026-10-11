"""Build the writer's facts sheet from Vibhu's optimizer outputs.

WHY THIS FILE EXISTS
--------------------
The optimizer (``carbonwise-data/optimization``) returns rich JSON:
``SolveResult.model_dump(mode="json")``, ``compare_plans(...)`` and
``explain_order(...)``. The writer needs a small, flat "facts sheet" whose key
names tell the fact-checker what each number means (cost, tonnes, kg, counts).

This module only COPIES and RENAMES values the optimizer already computed. It
never solves, recalculates totals or changes an allocation. The only "new"
numbers are trivial counts (e.g. how many orders changed plant) and the
removal of floating-point noise (e.g. -2.2e-16 shown as 0).

It deliberately does NOT import the optimizer package: it accepts plain dicts,
so the LLM module stays independently testable and has no circular imports.

Typical use from the workflow
-----------------------------
    result = solve_allocation(problem, weights, policy)
    facts = build_facts_sheet(intent="allocate", result=result.model_dump(mode="json"))

    rp = replan_capacity(problem, base, events, weights, policy)
    facts = build_facts_sheet(intent="capacity_loss",
                              result=rp["revised"].model_dump(mode="json"),
                              base=rp["base"].model_dump(mode="json"),
                              comparison=rp["comparison"], scenario_events=events)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# Statuses for which the optimizer has NO usable plan. Vibhu's handoff notes:
# "When solver.status is error/no_solution, do not present an empty zero-cost
# plan as success."
_NO_PLAN_STATUSES = {"error", "no_solution", "not_solved", "undefined"}

# Values smaller than this are floating-point noise (e.g. -2.22e-16 t).
_NOISE = 1e-6


def _clean(value: Any) -> Any:
    """Turn tiny float noise into exact 0 so text never says 'a decrease of 0.0 t'."""
    if isinstance(value, float) and abs(value) < _NOISE:
        return 0.0
    return value


def _kpis(result: Optional[dict]) -> Optional[dict]:
    """Copy the headline KPIs a reader needs (drops the long per-week table)."""
    if not result or not result.get("kpis"):
        return None
    k = result["kpis"]
    keep = ("total_cost", "total_carbon_t", "late_orders", "total_lateness_days",
            "unassigned_orders", "unassigned_units", "fulfilled_units")
    out = {key: _clean(k[key]) for key in keep if k.get(key) is not None}
    util = {p: v for p, v in (k.get("plant_utilisation") or {}).items() if v is not None}
    if util:
        out["plant_utilisation"] = util
    return out


def _has_plan(result: Optional[dict]) -> bool:
    """True only if the optimizer produced a checked plan that serves some orders."""
    if not result or not result.get("kpis"):
        return False
    if str(result.get("solver", {}).get("status", "")).lower() in _NO_PLAN_STATUSES:
        return False
    if not (result.get("validator") or {}).get("valid", False):
        return False
    return (result["kpis"].get("fulfilled_units") or 0) > 0


def _validator_flags(result: Optional[dict]) -> dict:
    """Keep only the boolean checks (the recomputed numbers are not for text)."""
    v = (result or {}).get("validator") or {}
    return {k: v[k] for k in ("valid", "capacity_ok", "eligibility_ok", "delivery_ok") if k in v}


def _comparison(comparison: Optional[dict]) -> Optional[dict]:
    """Rename compare_plans() output into the writer's comparison block."""
    if not comparison or not comparison.get("available"):
        return None
    delta = comparison.get("delta") or {}
    moved = [
        c["order_id"] for c in comparison.get("changes", [])
        if c.get("before") and c.get("after") and c["before"]["plant"] != c["after"]["plant"]
    ]
    return {
        "delta_cost": _clean(delta.get("total_cost")),
        "delta_carbon_t": _clean(delta.get("total_carbon_t")),
        "delta_late": delta.get("late_orders"),
        "delta_unassigned_orders": delta.get("unassigned_orders"),
        "moved_orders": moved,
        "dispatch_changes": comparison.get("dispatch_changes"),
        "production_bucket_changes": comparison.get("production_bucket_changes"),
        # Vibhu's flag: when fulfilment changed, lower totals are NOT savings.
        "comparable_fulfillment": comparison.get("comparable_fulfillment"),
        "interpretation": comparison.get("interpretation"),
    }


def _explanation(explanation: Optional[dict]) -> Optional[dict]:
    """Rename explain_order() output into the writer's order_explanation block."""
    if not explanation:
        return None
    chosen = explanation.get("chosen") or {}
    options, reasons = [], []
    for cand in explanation.get("candidates", []):
        plant, status = cand.get("plant"), cand.get("status")
        option: dict[str, Any] = {"plant": plant, "status": status}
        if status == "ineligible":
            option["eligible"] = False
            reasons.append(f"PLANT_{plant}_INELIGIBLE")
        elif plant == chosen.get("plant"):
            # For the chosen plant, report the plan's actual numbers.
            option.update({"landed_cost": chosen.get("cost"), "carbon_kg": chosen.get("carbon_kg"),
                           "arrival": chosen.get("arrival"), "on_time": chosen.get("on_time")})
        elif status == "no_on_time_capacity_even_in_isolation":
            option["on_time"] = False
            reasons.append(f"PLANT_{plant}_CANNOT_MEET_DUE_DATE")
        elif cand.get("standalone_estimate"):
            est = cand["standalone_estimate"]
            option.update({"standalone_cost": est.get("cost"), "standalone_carbon_kg": est.get("carbon_kg")})
        if cand.get("counterfactual"):
            cf = cand["counterfactual"]
            option["counterfactual_delta_cost"] = cf.get("cost_delta")
            option["counterfactual_delta_carbon_kg"] = cf.get("carbon_delta_kg")
        options.append(option)
    return {
        "order_id": explanation.get("order_id"),
        "chosen_plant": chosen.get("plant"),
        "options": options,
        "reason_codes": reasons,
        "caveat": "Standalone estimates ignore competition with other orders.",
    }


def _carbon_target(result: Optional[dict], base: Optional[dict]) -> Optional[list]:
    """Two-point trade-off table for a carbon-cut request (base vs capped plan)."""
    target = ((result or {}).get("problem_context") or {}).get("carbon_target")
    if not target:
        return None
    points = []
    base_k = _kpis(base)
    if base_k:
        points.append({"carbon_cut_pct": 0, "cost": base_k.get("total_cost"),
                       "carbon_t": base_k.get("total_carbon_t"), "late_orders": base_k.get("late_orders")})
    capped = {"carbon_cut_pct": target.get("requested_cut_pct"),
              "carbon_cap_kg": target.get("carbon_cap_kg"),
              "baseline_carbon_kg": target.get("baseline_carbon_kg")}
    if _has_plan(result) and not result.get("alternative"):
        k = _kpis(result)
        capped.update({"cost": k.get("total_cost"), "carbon_t": k.get("total_carbon_t"),
                       "late_orders": k.get("late_orders"), "achieved_cut_pct": target.get("actual_cut_pct"),
                       "feasible": True})
    else:
        capped["feasible"] = False
    points.append(capped)
    return points


def _weight_points(points: Optional[Iterable[dict]]) -> Optional[list]:
    """Rows from compare_weight_settings()['points'] (already computed by the solver)."""
    if not points:
        return None
    return [{"weights": p.get("weights"), "cost": p.get("cost"), "carbon_t": p.get("carbon_t"),
             "late_orders": p.get("late_orders"), "dominated": p.get("dominated")} for p in points]


def build_facts_sheet(
    *,
    intent: Optional[str],
    result: Optional[dict] = None,
    base: Optional[dict] = None,
    comparison: Optional[dict] = None,
    explanation: Optional[dict] = None,
    weight_points: Optional[Iterable[dict]] = None,
    scenario_events: Optional[list[dict]] = None,
    answer_type: Optional[str] = None,
    clarification_question: Optional[str] = None,
    refusal_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Create the writer's facts sheet from optimizer JSON (no recalculation).

    Args:
        intent: parsed intent ("allocate", "capacity_loss", ...).
        result: ``SolveResult.model_dump(mode="json")`` for the plan to describe.
        base: baseline ``SolveResult`` dump, for scenarios and carbon targets.
        comparison: ``compare_plans(...)`` output (or ``replan_capacity(...)["comparison"]``).
        explanation: ``explain_order(...)`` output.
        weight_points: ``compare_weight_settings(...)["points"]``.
        scenario_events: capacity events applied (fractions 0-1), for the text.
        answer_type: override; otherwise taken from ``result`` (guide values).
        clarification_question / refusal_reason: for those answer types.

    Returns:
        Facts-sheet dict documented in ``app/llm/writer.py``.
    """
    kind = answer_type or (result or {}).get("answer_type") or "allocation"
    facts: dict[str, Any] = {"answer_type": kind, "intent": intent}

    if kind in ("clarification_needed", "refused"):
        facts["clarification_question"] = clarification_question
        facts["refusal_reason"] = refusal_reason
        return facts

    if result is not None:
        status = str((result.get("solver") or {}).get("status", "")).lower()
        facts.update({
            "currency": (result.get("kpis") or {}).get("currency"),
            "solver_status": status,
            "validator": _validator_flags(result),
            "assumptions": list(result.get("assumptions") or []),
            "limitations": list(result.get("limitations") or []),
            "data_warnings": list(result.get("warnings") or []),
            "recommendations": list(result.get("recommendations") or []),
        })
        if _has_plan(result):
            facts["kpis"] = _kpis(result)
        else:
            # No usable plan: say so instead of describing a zero-cost "plan".
            facts["results_unavailable"] = True
            if intent != "tradeoff":
                facts["answer_type"] = "infeasible"
        if result.get("alternative") or kind == "infeasible":
            facts["answer_type"] = "infeasible" if intent != "tradeoff" else kind
            facts["closest_alternative"] = bool(result.get("alternative"))
        unassigned = result.get("unassigned_orders") or []
        late = [r for r in result.get("allocation", []) if not r.get("on_time", True)]
        if unassigned or late:
            facts["infeasible_orders"] = (
                [{"order_id": u["order_id"], "reason": u.get("reason_code", "unassigned")} for u in unassigned]
                + [{"order_id": r["order_id"], "reason": f"late by {r['late_days']} days"} for r in late]
            )

    if base is not None and _kpis(base):
        facts["base_kpis"] = _kpis(base)
        if not facts.get("currency"):
            facts["currency"] = base["kpis"].get("currency")
    comp = _comparison(comparison)
    if comp:
        facts["comparison"] = comp
    if scenario_events:
        ev = scenario_events[0]
        facts["scenario"] = {k: ev.get(k) for k in ("plant", "capacity_loss_pct", "from_date", "to_date")}
    expl = _explanation(explanation)
    if expl:
        facts["order_explanation"] = expl
    tradeoff = _carbon_target(result, base)
    if tradeoff:
        facts["tradeoff"] = tradeoff
        facts["answer_type"] = "scenario"
    points = _weight_points(weight_points)
    if points:
        facts["weight_tradeoff"] = points
    return facts

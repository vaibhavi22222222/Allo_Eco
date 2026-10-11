"""Ground explanations in a controlled language rendered from engine facts.

Arbitrary prose cannot be verified reliably with a number-membership regex.
Instead, the model arranges complete verified statements. The public checker
fails closed on unfamiliar wording, including numbers written as words.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from app.llm.client import LLMClient, LLMError


class FactsError(ValueError):
    """Malformed facts sheet; workflow must fix its adapter rather than guess."""


@dataclass(frozen=True)
class CheckResult:
    """Detailed verification result with unsupported statements for retry feedback."""

    valid: bool
    unsupported: tuple[str, ...]


def _number(value: object, *, integer: bool = False, nonnegative: bool = False) -> str:
    """Format supplied values only; rounding is display formatting, not a KPI."""
    if isinstance(value, bool):
        raise FactsError("A boolean cannot be a numerical metric")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or (nonnegative and number < 0):
            raise ValueError
        if integer and number != number.to_integral_value():
            raise ValueError
        quantum = Decimal("1") if integer else Decimal("0.01")
        rounded = number.quantize(quantum, rounding=ROUND_HALF_UP)
        return format(rounded, ",.0f" if integer else ",.2f")
    except (InvalidOperation, ValueError):
        raise FactsError("Expected a finite metric in its documented unit") from None


def _identifier(value: object) -> str:
    """Restrict IDs so newlines or sentence fragments cannot become claims."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise FactsError("Expected an alphanumeric identifier")
    return value


def _mapping(value: object) -> Mapping:
    """Validate containers before traversing an engine-provided facts sheet."""
    if not isinstance(value, Mapping):
        raise FactsError("Expected a facts object")
    return value


def _statements(facts: Mapping) -> list[str]:
    """Build labelled claims; never solve, compare totals, or infer causal reasons."""
    facts = _mapping(facts)
    mode = facts.get("intent", "allocate")
    headings = {
        "allocate": "Allocation results from the supplied facts.",
        "capacity_loss": "Capacity-loss replanning results from the supplied facts.",
        "tradeoff": "Cost and carbon trade-off results from the supplied facts.",
        "cheapest_on_time": "Cheapest-on-time results from the supplied facts.",
        "explain_order": "Order explanation from the supplied facts.",
        "brief": "Management brief from the supplied facts.",
    }
    if mode not in headings:
        raise FactsError("Unsupported facts intent")
    status = facts.get("status")
    if status not in {"OPTIMAL", "FEASIBLE", "INFEASIBLE", "clarification_needed", "refused"}:
        raise FactsError("Provide the engine status or clarification_needed/refused")
    lines = [headings[mode]]
    status_text = {
        "OPTIMAL": "The solver reports an optimal plan.",
        "FEASIBLE": "The solver reports a feasible plan; optimality is not established.",
        "INFEASIBLE": "The requested plan is infeasible; supplied metrics may describe a closest alternative.",
        "clarification_needed": "More information is required before planning.",
        "refused": "The request cannot proceed under the supported constraints.",
    }
    lines.append(status_text[status])
    kpis = _mapping(facts.get("kpis", {}))
    fields = {
        "total_cost": ("Total landed cost", "INR ", False),
        "total_carbon_t": ("Total carbon emissions", "", False),
        "late_orders": ("Late orders", "", True),
        "unmet_orders": ("Unmet orders", "", True),
    }
    for key, (label, prefix, integer) in fields.items():
        if key in kpis:
            unit = " t" if key == "total_carbon_t" else ""
            lines.append(f"{label}: {prefix}{_number(kpis[key], integer=integer, nonnegative=True)}{unit}.")
    for plant, value in _mapping(kpis.get("plant_utilisation", {})).items():
        # The agreed API stores utilization as a ratio, not a percentage.
        lines.append(f"Plant {_identifier(plant)} utilisation ratio: {_number(value, nonnegative=True)}.")
    comparison = _mapping(facts.get("comparison", {}))
    for key, label, prefix, unit, integer in (
        ("delta_cost", "Cost change (new minus baseline)", "INR ", "", False),
        ("delta_carbon_t", "Carbon change (new minus baseline)", "", " t", False),
        ("delta_late", "Late-order change (new minus baseline)", "", "", True),
        ("moved_orders", "Moved orders", "", "", True),
    ):
        if key in comparison:
            value = _number(comparison[key], integer=integer, nonnegative=key == "moved_orders")
            lines.append(f"{label}: {prefix}{value}{unit}.")
    allocation = facts.get("allocation", [])
    if not isinstance(allocation, list):
        raise FactsError("allocation must be a list")
    for raw_row in allocation:
        row = _mapping(raw_row)
        order, plant = _identifier(row.get("order_id")), _identifier(row.get("plant"))
        scope = f"Order {order} at Plant {plant}"
        lines.append(f"{scope}: assigned by the solver.")
        for key, label, prefix, unit in (
            ("cost", "landed cost", "INR ", ""),
            ("carbon_kg", "carbon emissions", "", " kg"),
        ):
            if key in row:
                lines.append(f"{scope}, {label}: {prefix}{_number(row[key], nonnegative=True)}{unit}.")
        if "arrival" in row:
            try:
                arrival = date.fromisoformat(str(row["arrival"]))
            except ValueError:
                raise FactsError("arrival must be an ISO date") from None
            lines.append(f"{scope}, arrival: {arrival.isoformat()}.")
        if "on_time" in row:
            if not isinstance(row["on_time"], bool):
                raise FactsError("on_time must be boolean")
            lines.append(f"{scope}, delivery: {'on time' if row['on_time'] else 'late'}.")
    frontier = facts.get("tradeoff", [])
    if not isinstance(frontier, list):
        raise FactsError("tradeoff must be a list")
    for raw_point in frontier:
        point = _mapping(raw_point)
        cut = _number(point.get("carbon_cut_pct"), nonnegative=True)
        if Decimal(str(point["carbon_cut_pct"])) > 100:
            raise FactsError("carbon_cut_pct must be at most 100")
        if "cost" in point:
            lines.append(f"Carbon cut target {cut}%, total landed cost: INR {_number(point['cost'], nonnegative=True)}.")
        if "late_orders" in point:
            lines.append(f"Carbon cut target {cut}%, late orders: {_number(point['late_orders'], integer=True, nonnegative=True)}.")
    # Notes must be deterministic upstream evidence, never user text or LLM output.
    # JSON quoting preserves their boundary and prevents injected extra lines.
    for key, label in (("reasons", "Engine reason"), ("assumptions", "Assumption"),
                       ("limitations", "Limitation"), ("missing_fields", "Required input")):
        notes = facts.get(key, [])
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
            raise FactsError(f"{key} must be a list of trusted engine strings")
        lines.extend(f"{label}: {json.dumps(note, ensure_ascii=False)}." for note in notes)
    return lines


def _normalise(statement: str) -> str:
    """Allow separators and equivalent decimal spellings, preserving metric/unit/sign.

    Numeric comparisons are exact after the documented two-decimal rendering.
    No broad tolerance can mistakenly turn an increase into a decrease.
    """
    statement = statement.strip().replace("₹", "INR ")
    statement = re.sub(r"\s+", " ", statement)

    def decimal_text(match: re.Match) -> str:
        raw = match[0]
        # Only accept standard Western or Indian digit grouping.
        if "," in raw and not re.fullmatch(
            r"-?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?", raw
        ):
            return raw
        return str(Decimal(raw.replace(",", "")).normalize())

    # Dates and IDs remain verbatim by requiring token boundaries.
    return re.sub(r"(?<![\w-])-?\d[\d,]*(?:\.\d+)?(?![\w-])", decimal_text, statement)


def check_answer(answer: str, facts: Mapping) -> CheckResult:
    """Verify complete statement meaning, scope and numbers against trusted facts.

    Accept newline-delimited controlled statements. Unknown prose, numerical
    words, invented causal links, changed dates and swapped metric values fail.
    """
    allowed = {_normalise(line) for line in _statements(facts)}
    if not isinstance(answer, str) or not answer.strip():
        return CheckResult(False, ("Empty answer",))
    unsupported = tuple(line for line in answer.splitlines()
                        if line.strip() and _normalise(line) not in allowed)
    return CheckResult(not unsupported, unsupported)


def fact_check(answer: str, facts: Mapping) -> bool:
    """Return True only when every statement is grounded in the facts sheet."""
    return check_answer(answer, facts).valid


def write_text(facts: Mapping, *, client: LLMClient | None = None) -> str:
    """Render verified facts; retry invalid model output once, then use a template.

    The model may reorder statements, but cannot omit any supplied statement or
    add a new claim. This deliberate constraint makes numerical grounding and
    non-numerical fidelity independently checkable.
    """
    statements = _statements(facts)
    fallback = "\n".join(statements)
    llm = client or LLMClient()
    if llm.mock:
        return fallback
    feedback = ""
    for _ in range(2):
        try:
            result = llm.complete_json(
                'Arrange the supplied verified statements into a readable explanation. '
                'Return JSON {"answer": "newline-separated statements"}. '
                'Copy every statement exactly once. Do not add prose or calculate values.',
                json.dumps({"statements": statements, "feedback": feedback}),
            )
            answer = result.get("answer")
            checked = check_answer(answer, facts)
            if checked.valid and sorted(answer.splitlines()) == sorted(statements):
                return answer
            feedback = "Use all supplied statements exactly once; no unsupported claims."
        except (LLMError, ValueError, TypeError):
            feedback = "Return valid JSON with an answer string containing the supplied statements."
    return fallback


def write_answer(facts: Mapping, *, client: LLMClient | None = None) -> str:
    """Alias used by the workflow's write_answer node."""
    return write_text(facts, client=client)

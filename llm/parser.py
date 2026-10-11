"""Interpret requests without making allocation decisions or inventing inputs."""

import json
import re
from collections.abc import Mapping

from pydantic import ValidationError

from app.contracts import RequestSpec, Scenario, Weights
from app.llm.client import LLMClient, LLMError


class RequestError(ValueError):
    """Invalid explicit API fields; workflow should return a validation error."""


def _fields(value: Mapping | Scenario | Weights | None) -> dict:
    """Preserve only explicitly set fields, including explicit nulls."""
    if value is None:
        return {}
    if isinstance(value, (Scenario, Weights)):
        return value.model_dump(exclude_unset=True)
    return dict(value)


def _rules(text: str) -> dict:
    """Conservative lexical extraction; unknown parameters remain missing."""
    lower = text.lower()
    data: dict = {"scenario": {}, "weights": {}}
    if re.search(r"\b(ignore|bypass|violate|disregard|override)\b", lower):
        data.update(route="refused", refusal_reason="Hard constraints cannot be overridden.")
        return data
    # Forced assignments are outside this module's supported solver modes.
    if re.search(r"\b(assign|put|force)\b.*\border\b.*\bplant\b", lower):
        data.update(route="refused", refusal_reason="Forced assignments are unsupported.")
        return data
    if "capacity" in lower or re.search(r"re-?plan", lower):
        data["intent"] = "capacity_loss"
    elif re.search(r"\b(why|explain)\b.*\border\b", lower):
        data["intent"] = "explain_order"
    elif "carbon" in lower and re.search(r"cut|reduc|trade|cost", lower):
        data["intent"] = "tradeoff"
    elif "cheapest" in lower and re.search(r"on[ -]time", lower):
        data["intent"] = "cheapest_on_time"
    elif re.search(r"\b(brief|leadership|management summary)\b", lower):
        data["intent"] = "brief"
    elif re.search(r"\b(allocate|allocation)\b", lower):
        data["intent"] = "allocate"
    scenario = data["scenario"]
    plants = re.findall(r"\b[Pp]lant\s+([A-Z0-9][A-Za-z0-9_-]*)\b", text)
    if len(set(plants)) == 1:
        scenario["plant"] = plants[0]
    percentages = re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", lower)
    if len(percentages) == 1:
        # A percentage alone does not establish a reduction. In particular,
        # remaining capacity and capacity gains must never become loss amounts.
        reduction = re.search(r"\b(lose[st]?|lost|loss|cut|reduc\w*|down|drop\w*)\b", lower)
        opposite = re.search(r"\b(gain\w*|increas\w*|retain\w*|remain\w*)\b", lower)
        if reduction and not opposite:
            if data.get("intent") == "capacity_loss":
                scenario["capacity_loss_pct"] = float(percentages[0])
            elif data.get("intent") == "tradeoff":
                scenario["carbon_cut_pct"] = float(percentages[0])
    durations = re.findall(r"\b(\d+|one|two|three|four)\s+weeks?\b", lower)
    if len(durations) == 1:
        words = {"one": 1, "two": 2, "three": 3, "four": 4}
        scenario["duration_weeks"] = words.get(durations[0]) or int(durations[0])
    interval = re.search(r"\bfrom (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", lower)
    if interval:
        scenario.update(from_date=interval[1], to_date=interval[2])
    order = re.search(r"\border\s+(O-[A-Za-z0-9-]+|\d+)\b", text, re.I)
    if order:
        data["order_id"] = order[1].upper()
    for name in ("cost", "carbon", "lateness"):
        weight = re.search(rf"\b{name}\s+weight\s*(?:of|=|:)?\s*(0(?:\.\d+)?|1(?:\.0+)?)\b", lower)
        if weight:
            data["weights"][name] = float(weight[1])
    return data


def parse_request(
    text: str,
    weights: Mapping | Weights | None = None,
    scenario: Mapping | Scenario | None = None,
    *, client: LLMClient | None = None,
) -> RequestSpec:
    """Parse into RequestSpec; explicit fields win, ambiguity routes to clarify.

    Inject one client per workflow run to collect usage. Dataset membership and
    planning-date resolution remain the check_request node's responsibility.
    """
    if not isinstance(text, str) or not text.strip():
        return RequestSpec(route="clarify", clarification=["Provide a request."])
    try:
        explicit_weights, explicit_scenario = _fields(weights), _fields(scenario)
        Weights.model_validate(explicit_weights)
        Scenario.model_validate(explicit_scenario)
    except (ValidationError, TypeError, ValueError) as exc:
        raise RequestError("Invalid explicit weights or scenario") from exc
    data = _rules(text)
    llm = client or LLMClient()
    # Rules retain authority over refusal and parameter provenance. A model can
    # classify unfamiliar phrasing, but cannot add unsupported numeric values.
    if not llm.mock and data.get("route") != "refused":
        feedback = ""
        for _ in range(2):
            try:
                candidate = llm.complete_json(
                    "Classify this request using the JSON schema. Never allocate, "
                    "calculate KPIs or invent parameters. " + json.dumps(RequestSpec.model_json_schema()),
                    json.dumps({"text": text, "feedback": feedback}),
                )
                parsed = RequestSpec.model_validate(candidate)
                # Only intent classification is model-owned. Parameters are
                # extracted from source text to prevent fabricated values.
                if not data.get("intent") and parsed.intent:
                    data["intent"] = parsed.intent
                break
            except (LLMError, ValidationError, ValueError, TypeError):
                feedback = "Invalid output. Return one schema-valid JSON object."
    data["weights"].update(explicit_weights)
    data["scenario"].update(explicit_scenario)
    try:
        spec = RequestSpec.model_validate(data)
    except ValidationError:
        return RequestSpec(route="clarify", clarification=["Check percentages, duration and dates."])
    if spec.route == "refused":
        return spec
    missing = []
    if spec.intent is None:
        missing.append("Specify allocation, tradeoff, capacity loss, order explanation or brief.")
    if spec.intent == "capacity_loss":
        for key in ("plant", "capacity_loss_pct"):
            if getattr(spec.scenario, key) is None:
                missing.append(f"Provide {key}.")
        if spec.scenario.duration_weeks is None and not (
            spec.scenario.from_date and spec.scenario.to_date
        ):
            missing.append("Provide duration_weeks or from_date and to_date.")
    if spec.intent == "tradeoff" and spec.scenario.carbon_cut_pct is None:
        missing.append("Provide carbon_cut_pct.")
    if spec.intent == "explain_order" and not spec.order_id:
        missing.append("Provide order_id.")
    if missing:
        spec.route, spec.clarification = "clarify", missing
    return spec

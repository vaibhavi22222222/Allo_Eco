"""Request parser: natural-language request -> validated ``RequestSpec``.

How it works (in order)
-----------------------
1. A deterministic rule parser (regular expressions) always runs. It is fast,
   needs no model, and is the fallback when the LLM is off or misbehaves.
2. If the LLM is enabled, it is asked for a JSON object matching
   ``RequestSpec``. The JSON is validated with Pydantic. If validation fails,
   the LLM gets ONE retry with the error message. If that also fails (or the
   LLM times out), the rule-parser result is used.
3. Grounding check: every parameter the LLM returns (plant, order id,
   percentages, duration, dates) must actually appear in the user's text. A
   value that does not appear is dropped. The LLM is never allowed to invent a
   plant, order, duration or scenario value.
4. Explicit API fields (``weights``, ``scenario``) override parsed values.
5. Completeness check: if the intent or a required parameter is missing, the
   spec is marked ``needs_clarification`` with a question. Nothing is guessed.

What the parser does NOT do
---------------------------
* It never chooses an allocation, a plant for an order, or any KPI.
* It never authorises rule-breaking requests. It only sets ``rule_violation``
  so the workflow's ``check_request`` node can return ``refused``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Iterable, Optional, Union

from pydantic import ValidationError

from app.llm.client import LLMClient, LLMError, get_client
from app.llm.models import RequestSpec, RequestWeights

logger = logging.getLogger(__name__)


class RequestParseError(ValueError):
    """Raised when EXPLICIT API input is invalid (the API should answer 422).

    Problems in the free text never raise; they produce a clarification state.
    """


# --------------------------------------------------------------------------
# Small vocabularies used by the rule parser
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1,
}

_NUM = r"\d+(?:\.\d+)?"
_PCT_RE = re.compile(rf"({_NUM})\s*(?:%|percent\b|per\s*cent\b|pct\b)", re.I)
# Plant codes are written in capitals ("Plant B"); requiring that avoids
# reading "a plant is down" as plant "IS".
_PLANT_RE = re.compile(r"\b[Pp]lant\s+([A-Z][A-Z0-9]{0,2})\b")
# Order ids such as O-077, ORD-0001, or "order 77".
_ORDER_ID_RE = re.compile(r"\b([A-Za-z]{1,5}-\d{1,6})\b")
_ORDER_NUM_RE = re.compile(r"\border\s+#?(\d{1,6})\b", re.I)
_DURATION_RE = re.compile(
    rf"\b({_NUM}|{'|'.join(NUMBER_WORDS)})\s+(weeks?|days?)\b", re.I
)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Intent keyword patterns, checked in priority order (first match wins).
_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("brief", re.compile(r"\bbrief\b|\b(summary|report|update)\s+for\s+(the\s+)?(leadership|management|board)", re.I)),
    ("explain_order", re.compile(r"\bwhy\b.*\border\b|\bexplain\b.*\border\b|\border\b.*\bwhy\b", re.I)),
    ("capacity_loss", re.compile(r"\b(los[et]s?|loss|lost|down|outage|shut|reduc\w*|cut)\b.*\bcapacity\b|\bcapacity\b.*\b(loss|lost|drop|reduc\w*)|\bre-?plan\b", re.I)),
    ("cheapest_on_time", re.compile(r"\bcheapest\b.*\b(on[\s-]?time|due|deadline|late)\b|\b(on[\s-]?time|no\s+late)\b.*\bcheapest\b", re.I)),
    ("tradeoff", re.compile(r"\bcarbon\b.*\b(cut|reduc\w*|lower\w*|decreas\w*|less)\b|\b(cut|reduc\w*|lower\w*)\b.*\b(carbon|emissions?|co2)\b|\btrade-?off\b|\bfrontier\b|\bgreener\b", re.I)),
    ("allocate", re.compile(r"\balloca\w*\b|\bassign\w*\b|\b(make|build|create)\s+(a\s+)?plan\b|\bplan\b.*\borders\b", re.I)),
]

# Requests that try to break a hard rule. Detected, never executed.
_VIOLATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_capacity", re.compile(r"\b(ignore|exceed|over|beyond|regardless\s+of|bypass|break)\b.{0,20}\bcapacit", re.I)),
    ("ineligible_plant", re.compile(r"\b(even\s+though|even\s+if|although)\b.{0,40}\b(can'?t|cannot|can\s+not|unable|not\s+able)\b.{0,15}\bmake\b", re.I)),
    ("fabricate_numbers", re.compile(r"\b(make\s+up|invent|fabricate|guess|estimate)\b.{0,30}\b(number|figure|saving|cost|carbon|value)s?\b|\bsay\s+we\s+saved\b", re.I)),
    ("forced_assignment", re.compile(r"\b(assign|put|move|force|send)\b\s+(order\s+)?#?[A-Za-z]{0,5}-?\d{1,6}\s+(to|in|into|at)\s+plant\s+[A-Za-z0-9]{1,3}\b", re.I)),
]

_SYSTEM_PROMPT = """You convert a manufacturing planning request into JSON.
Return ONE JSON object with exactly these optional keys:
  intent: one of "allocate","tradeoff","cheapest_on_time","capacity_loss","explain_order","brief", or null
  plant: plant code exactly as written (e.g. "B"), or null
  capacity_loss_pct: capacity loss as a FRACTION between 0 and 1 (30% -> 0.3), or null
  duration_weeks: number of weeks, or null
  from_date, to_date: "YYYY-MM-DD" or null
  carbon_cut_pct: carbon reduction in PERCENTAGE POINTS (10% -> 10), or null
  order_id: order identifier exactly as written, or null
Rules:
- Only copy values that are explicitly written in the request. If a value is not written, use null.
- Never choose a plant for an order, never compute costs, carbon or any result.
- Do not add any other keys and do not add explanations."""

# Keys the LLM is allowed to fill. Everything else is computed by code.
_LLM_KEYS = (
    "intent", "plant", "capacity_loss_pct", "duration_weeks",
    "from_date", "to_date", "carbon_cut_pct", "order_id",
)


# --------------------------------------------------------------------------
# Rule-based extraction
# --------------------------------------------------------------------------
def _to_number(token: str) -> Optional[float]:
    token = token.lower()
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])
    try:
        return float(token)
    except ValueError:
        return None


def _detect_intent(text: str) -> Optional[str]:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return None


def _detect_violation(text: str) -> Optional[str]:
    for name, pattern in _VIOLATION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _extract_duration_weeks(text: str) -> Optional[float]:
    if re.search(r"\bfortnight\b", text, re.I):
        return 2.0
    match = _DURATION_RE.search(text)
    if not match:
        return None
    amount = _to_number(match.group(1))
    if amount is None:
        return None
    # Days are converted to weeks so the workflow has one representation.
    return amount if match.group(2).lower().startswith("week") else round(amount / 7, 4)


def _extract_weights(text: str) -> Optional[RequestWeights]:
    """Parse explicit weights such as '70% cost, 30% carbon'. No defaults."""
    cost = re.search(rf"({_NUM})\s*%\s*(?:on\s+|weight\s+(?:on\s+)?)?cost", text, re.I)
    carbon = re.search(rf"({_NUM})\s*%\s*(?:on\s+|weight\s+(?:on\s+)?)?carbon", text, re.I)
    # "10% carbon cut" is a target, not a weight. Only treat percentages as
    # weights when both are given ("70% cost, 30% carbon") or the word
    # "weight" is used.
    if not ((cost and carbon) or (re.search(r"\bweight", text, re.I) and (cost or carbon))):
        return None
    return RequestWeights(
        cost=float(cost.group(1)) / 100 if cost else None,
        carbon=float(carbon.group(1)) / 100 if carbon else None,
    )


def rule_parse(text: str) -> RequestSpec:
    """Deterministic parse of ``text`` using regular expressions only.

    This is the fallback path, and it is also used in MOCK_LLM mode.
    """
    intent = _detect_intent(text)
    violation = _detect_violation(text)
    plant_match = _PLANT_RE.search(text)
    pct_match = _PCT_RE.search(text)
    pct_value = float(pct_match.group(1)) if pct_match else None

    order_id: Optional[str] = None
    id_match = _ORDER_ID_RE.search(text)
    if id_match:
        order_id = id_match.group(1).upper()
    else:
        num_match = _ORDER_NUM_RE.search(text)
        if num_match:
            order_id = num_match.group(1)  # resolved against known ids later

    dates = _ISO_DATE_RE.findall(text)
    spec = RequestSpec(
        intent=intent,
        plant=plant_match.group(1).upper() if plant_match else None,
        order_id=order_id,
        duration_weeks=_extract_duration_weeks(text),
        from_date=date.fromisoformat(dates[0]) if dates else None,
        to_date=date.fromisoformat(dates[1]) if len(dates) > 1 else None,
        weights=_extract_weights(text),
        rule_violation=violation,
        source="rules",
        raw_text=text,
    )
    # A percentage means different things for different intents:
    # capacity loss -> fraction (data-module convention), carbon cut -> points.
    if pct_value is not None:
        if intent == "capacity_loss" and pct_value <= 100:
            spec.capacity_loss_pct = round(pct_value / 100, 6)
        elif intent == "tradeoff" and pct_value <= 100:
            spec.carbon_cut_pct = pct_value
    return spec


# --------------------------------------------------------------------------
# LLM path
# --------------------------------------------------------------------------
def _numbers_in(text: str) -> list[float]:
    """All numbers written in the text, including number words."""
    values = [float(n) for n in re.findall(_NUM, text)]
    values += [float(v) for w, v in NUMBER_WORDS.items() if len(w) > 2 and re.search(rf"\b{w}\b", text, re.I)]
    return values


def _ground(spec: RequestSpec, text: str) -> RequestSpec:
    """Drop any LLM-supplied parameter that is not literally supported by ``text``."""
    numbers = _numbers_in(text)

    def appears(value: float) -> bool:
        return any(abs(value - n) < 1e-6 for n in numbers)

    dropped: list[str] = []
    if spec.plant and not re.search(rf"\bplant\s+{re.escape(spec.plant)}\b", text, re.I):
        dropped.append("plant"); spec.plant = None
    if spec.order_id and spec.order_id.lower() not in text.lower():
        dropped.append("order_id"); spec.order_id = None
    # The LLM returns a fraction (0.3); the text says 30 (%). Either form counts.
    if spec.capacity_loss_pct is not None and not (
        appears(spec.capacity_loss_pct * 100) or appears(spec.capacity_loss_pct)
    ):
        dropped.append("capacity_loss_pct"); spec.capacity_loss_pct = None
    if spec.carbon_cut_pct is not None and not appears(spec.carbon_cut_pct):
        dropped.append("carbon_cut_pct"); spec.carbon_cut_pct = None
    if spec.duration_weeks is not None:
        rule_duration = _extract_duration_weeks(text)
        if rule_duration is None or abs(rule_duration - spec.duration_weeks) > 1e-3:
            dropped.append("duration_weeks"); spec.duration_weeks = None
    for field in ("from_date", "to_date"):
        value = getattr(spec, field)
        if value is not None and value.isoformat() not in text:
            dropped.append(field); setattr(spec, field, None)
    if dropped:
        logger.warning("Dropped ungrounded LLM fields: %s", dropped)
    return spec


def _llm_parse(text: str, client: LLMClient) -> Optional[RequestSpec]:
    """Ask the LLM for a RequestSpec. Returns None if it fails twice.

    Flow: call -> validate -> (on failure) one retry with the error -> None.
    """
    user_prompt = f"Request: {text}"
    last_error = ""
    for attempt in range(2):
        prompt = user_prompt if attempt == 0 else (
            f"{user_prompt}\n\nYour previous answer was invalid: {last_error}\n"
            "Return a corrected JSON object."
        )
        try:
            response = client.complete(_SYSTEM_PROMPT, prompt, json_mode=True)
        except LLMError as exc:
            # Timeout / unreachable: retrying immediately rarely helps, so we
            # go straight to the deterministic fallback.
            logger.warning("LLM parse unavailable, using rules: %s", exc)
            return None
        try:
            raw = json.loads(response.text)
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object")
            # Keep only the keys the LLM is allowed to set.
            clean = {k: raw.get(k) for k in _LLM_KEYS if raw.get(k) not in (None, "")}
            return RequestSpec(**clean, source="llm", raw_text=text)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            last_error = str(exc)[:300]
            logger.warning("Invalid LLM parse (attempt %d): %s", attempt + 1, last_error)
    return None


# --------------------------------------------------------------------------
# Explicit API overrides and completeness checks
# --------------------------------------------------------------------------
def _apply_weights(spec: RequestSpec, weights: Union[None, dict, RequestWeights]) -> None:
    if weights is None:
        return
    try:
        explicit = weights if isinstance(weights, RequestWeights) else RequestWeights(**weights)
    except ValidationError as exc:
        raise RequestParseError(f"Invalid weights: {exc.errors()[0]['msg']}") from exc
    base = spec.weights.model_dump() if spec.weights else {}
    # Field-level precedence: an explicit value wins, missing ones keep the parsed value.
    base.update({k: v for k, v in explicit.model_dump().items() if v is not None})
    spec.weights = RequestWeights(**base)


def _apply_scenario(spec: RequestSpec, scenario: Optional[dict[str, Any]]) -> None:
    """Merge an explicit scenario JSON. Accepts a flat dict or {'capacity_change': [ {...} ]}."""
    if not scenario:
        return
    if not isinstance(scenario, dict):
        raise RequestParseError("scenario must be a JSON object")
    spec.scenario = scenario
    change = scenario
    if isinstance(scenario.get("capacity_change"), list) and scenario["capacity_change"]:
        change = scenario["capacity_change"][0]

    if "plant" in change and change["plant"]:
        spec.plant = str(change["plant"]).strip().upper()
    if "capacity_loss_pct" in change and change["capacity_loss_pct"] is not None:
        value = float(change["capacity_loss_pct"])
        # Agreed representation is a fraction 0-1. We do NOT rescale 30 -> 0.30
        # silently, because that could hide a real data error.
        if not 0 <= value <= 1:
            raise RequestParseError(
                "scenario.capacity_loss_pct must be a fraction between 0 and 1 (e.g. 0.3 for 30%)"
            )
        spec.capacity_loss_pct = value
    for field in ("from_date", "to_date"):
        if change.get(field):
            try:
                setattr(spec, field, date.fromisoformat(str(change[field])))
            except ValueError as exc:
                raise RequestParseError(f"scenario.{field} must be YYYY-MM-DD") from exc
    if "carbon_cut_pct" in scenario and scenario["carbon_cut_pct"] is not None:
        spec.carbon_cut_pct = float(scenario["carbon_cut_pct"])
    # A scenario with a capacity change implies a capacity-loss request.
    if spec.intent is None and spec.plant and spec.capacity_loss_pct is not None:
        spec.intent = "capacity_loss"


def _resolve_order_id(order_id: str, known: Iterable[str]) -> Optional[str]:
    """Map 'order 77' to a known id like 'O-077' only if exactly one id matches."""
    known = list(known)
    if order_id in known:
        return order_id
    lowered = {k.lower(): k for k in known}
    if order_id.lower() in lowered:
        return lowered[order_id.lower()]
    if order_id.isdigit():
        hits = [k for k in known if re.sub(r"\D", "", k).lstrip("0") == order_id.lstrip("0")]
        if len(hits) == 1:
            return hits[0]
    return None


def _check_completeness(
    spec: RequestSpec,
    known_plants: Optional[Iterable[str]],
    known_order_ids: Optional[Iterable[str]],
) -> None:
    """Fill ``missing`` and the clarification question. Never fills values."""
    missing: list[str] = []
    question: Optional[str] = None

    if spec.intent is None and spec.rule_violation is None:
        missing.append("intent")
        question = ("What would you like to do: allocate orders, compare a carbon cut, "
                    "find the cheapest on-time plan, re-plan after a capacity loss, "
                    "explain an order, or write a brief?")
    if spec.intent == "capacity_loss":
        if not spec.plant:
            missing.append("plant")
        if spec.capacity_loss_pct is None:
            missing.append("capacity_loss_pct")
        if missing:
            question = "Which plant lost capacity, and by what percentage?"
    if spec.intent == "explain_order" and not spec.order_id:
        missing.append("order_id")
        question = "Which order should I explain? Please give its order ID."

    # Validate against the loaded data when the workflow provides it.
    if spec.plant and known_plants is not None:
        plants = {p.upper() for p in known_plants}
        if spec.plant.upper() not in plants:
            missing.append("plant")
            question = f"Plant {spec.plant} is not in the loaded data. Known plants: {', '.join(sorted(plants))}."
    if spec.order_id and known_order_ids is not None:
        resolved = _resolve_order_id(spec.order_id, known_order_ids)
        if resolved is None:
            missing.append("order_id")
            question = f"Order {spec.order_id} was not found in the loaded data."
        else:
            spec.order_id = resolved

    spec.missing = missing
    spec.needs_clarification = bool(missing)
    spec.clarification_question = question


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def parse_request(
    text: str,
    weights: Union[None, dict, RequestWeights] = None,
    scenario: Optional[dict[str, Any]] = None,
    *,
    client: Optional[LLMClient] = None,
    known_plants: Optional[Iterable[str]] = None,
    known_order_ids: Optional[Iterable[str]] = None,
) -> RequestSpec:
    """Convert a request into a validated ``RequestSpec``.

    Args:
        text: the user's request, e.g. "Plant B loses 30% capacity. Re-plan."
        weights: explicit API weights {cost, carbon, lateness}; override parsed ones.
        scenario: explicit API scenario JSON; overrides parsed values.
        client: LLM client to use. Defaults to one built from environment settings.
        known_plants: plant codes in the loaded dataset, for validation (optional).
        known_order_ids: order ids in the loaded dataset, for validation (optional).

    Returns:
        RequestSpec. If ``needs_clarification`` is True the workflow should
        answer ``clarification_needed``; if ``rule_violation`` is set it should
        answer ``refused``.

    Raises:
        RequestParseError: explicit ``weights``/``scenario`` values are invalid.
    """
    if not isinstance(text, str) or not text.strip():
        raise RequestParseError("request text must be a non-empty string")
    text = text.strip()
    client = client or get_client()

    rules = rule_parse(text)
    spec = rules
    if not client.is_mock:
        calls_before, tokens_before = client.calls, client.total_tokens
        llm_spec = _llm_parse(text, client)
        if llm_spec is not None:
            spec = _ground(llm_spec, text)
            # Safety signals and weights always come from deterministic code.
            spec.rule_violation = rules.rule_violation
            spec.weights = rules.weights
            # If grounding removed a value the rules found, use the rule value.
            for field in ("plant", "order_id", "capacity_loss_pct", "carbon_cut_pct",
                          "duration_weeks", "from_date", "to_date"):
                if getattr(spec, field) is None and getattr(rules, field) is not None:
                    setattr(spec, field, getattr(rules, field))
            if spec.intent is None:
                spec.intent = rules.intent
        spec.llm_calls = client.calls - calls_before
        spec.tokens = client.total_tokens - tokens_before

    _apply_weights(spec, weights)
    _apply_scenario(spec, scenario)
    _check_completeness(spec, known_plants, known_order_ids)
    return spec


# --------------------------------------------------------------------------
# Hand-off helpers: RequestSpec -> inputs of Vibhu's optimizer
# --------------------------------------------------------------------------
class ScenarioIncompleteError(ValueError):
    """The request names a capacity loss but the dates cannot be resolved.

    The workflow should answer ``clarification_needed`` (see ``.missing``)
    instead of guessing dates.
    """

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Capacity scenario is missing: {', '.join(missing)}")


def to_capacity_events(spec: RequestSpec, default_start: Optional[date] = None) -> list[dict[str, Any]]:
    """Build the ``capacity_events`` list for ``replan_capacity`` / ``prepare_problem``.

    The dict keys match ``optimization.contracts.CapacityEvent``:
    ``event_id, plant, from_date, to_date, capacity_loss_pct`` (fraction 0-1,
    passed through unchanged, never divided by 100 again).

    Date rules (no guessing):
      * explicit from/to dates win;
      * otherwise ``duration_weeks`` is counted from ``from_date`` or, if that
        is missing, from ``default_start`` (normally ``PlanningConfig.planning_start``);
      * if neither gives two dates, ``ScenarioIncompleteError`` is raised.
    """
    from datetime import timedelta  # local import: only needed here

    missing = [name for name, value in (("plant", spec.plant), ("capacity_loss_pct", spec.capacity_loss_pct))
               if value is None]
    start = spec.from_date or default_start
    end = spec.to_date
    if end is None and start is not None and spec.duration_weeks:
        # "two weeks" from 12 Jan = 12 Jan .. 25 Jan inclusive (14 days).
        end = start + timedelta(days=round(spec.duration_weeks * 7) - 1)
    if start is None:
        missing.append("from_date")
    if end is None:
        missing.append("to_date or duration")
    if missing:
        raise ScenarioIncompleteError(missing)
    pct_label = f"{spec.capacity_loss_pct * 100:g}".replace(".", "_")
    return [{
        "event_id": f"REQ-{spec.plant}-{pct_label}-{start.isoformat()}",
        "plant": spec.plant,
        "from_date": start.isoformat(),
        "to_date": end.isoformat(),
        "capacity_loss_pct": spec.capacity_loss_pct,
    }]


def to_objective_weights(spec: RequestSpec) -> Optional[dict[str, float]]:
    """Weights to pass as ``weights=`` to ``solve_allocation``.

    Only weights the user actually gave are returned. ``None`` means "use the
    optimizer's defaults" (``ObjectiveWeights``: cost 0.7, carbon 0.3,
    lateness 0); the parser never invents weights.
    """
    if spec.weights is None:
        return None
    given = spec.weights.model_dump(exclude_none=True)
    return given or None

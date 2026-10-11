"""Explanation writer and numeric fact-checker.

Responsibility
--------------
Turn a FACTS SHEET (a dict built by the deterministic workflow and optimisation
engine) into readable text, and prove that every number in that text comes from
the facts sheet.

    write_text(facts)          -> WriterResult   (text + check result + recommendations)
    fact_check(answer, facts)  -> FactCheckResult
    render_template(facts)     -> str            (deterministic text, no LLM)
    build_recommendations(facts) -> list[Recommendation]

Flow of ``write_text``
----------------------
1. Derived facts (e.g. "cost difference vs the 0 % cut point") are computed by
   CODE here, so the LLM never has to do arithmetic.
2. MOCK_LLM, or a request type that must be deterministic (refused /
   clarification_needed) -> template.
3. Otherwise the LLM writes from the facts. ``fact_check`` verifies every
   number. On failure: one retry, telling the LLM exactly which claims were
   unsupported. On a second failure (or if the LLM is unavailable) -> template.
4. The template output is fact-checked as well, so a bug in a template is
   caught by the same rule.

Expected facts-sheet shape
--------------------------
Build it with ``app.llm.facts.build_facts_sheet(...)`` from the optimizer's
JSON; do not hand-assemble it. All keys are optional except ``answer_type``.

    {
      "answer_type": "allocation" | "scenario" | "brief" | "infeasible"
                     | "clarification_needed" | "refused",
      "intent": "allocate" | "tradeoff" | "cheapest_on_time" | "capacity_loss"
                | "explain_order" | "brief",
      "currency": "INR",                     # SolveResult.kpis.currency
      "solver_status": "optimal",            # SolveResult.solver.status (lowercase)
      "validator": {"valid": bool, "capacity_ok": bool, "eligibility_ok": bool, "delivery_ok": bool},
      "results_unavailable": true,           # set when there is no usable plan
      "kpis": {"total_cost", "total_carbon_t", "late_orders", "total_lateness_days",
               "unassigned_orders", "unassigned_units", "fulfilled_units",
               "plant_utilisation": {"A": 0.79, ...}},       # fractions 0-1
      "base_kpis": {...same keys...},
      "comparison": {"delta_cost", "delta_carbon_t", "delta_late", "delta_unassigned_orders",
                     "moved_orders": [ids], "dispatch_changes", "production_bucket_changes",
                     "comparable_fulfillment": bool, "interpretation": str},  # revised minus base
      "scenario": {"plant", "capacity_loss_pct" (fraction), "from_date", "to_date"},
      "tradeoff": [{"carbon_cut_pct" (points), "cost", "carbon_t", "late_orders",
                    "feasible", "carbon_cap_kg", "baseline_carbon_kg", "achieved_cut_pct"}],
      "weight_tradeoff": [{"weights": {...}, "cost", "carbon_t", "late_orders", "dominated"}],
      "order_explanation": {"order_id", "chosen_plant", "reason_codes": [...],
                    "options": [{"plant", "status", "eligible", "landed_cost", "carbon_kg",
                                 "arrival", "on_time", "standalone_cost", "standalone_carbon_kg",
                                 "counterfactual_delta_cost", "counterfactual_delta_carbon_kg"}]},
      "infeasible_orders": [{"order_id", "reason"}], "closest_alternative": bool,
      "recommendations": [...SolveResult.recommendations...],
      "assumptions": [...], "limitations": [...], "data_warnings": [...],
      "clarification_question": str, "refusal_reason": str
    }

Key naming drives the fact-checker's unit detection: ``*_cost`` = currency,
``*_t`` = tonnes, ``*_kg`` = kilograms, ``*_pct`` = percentage points (except
``capacity_loss_pct``, a fraction), ``utilisation`` = fraction, ``late`` /
``moved`` / ``orders`` = counts.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterator, Optional

from app.llm.client import LLMClient, LLMError, get_client
from app.llm.models import ClaimCheck, FactCheckResult, Recommendation, WriterResult

logger = logging.getLogger(__name__)

# ==========================================================================
# 1. Facts: flatten the facts sheet into typed, addressable values
# ==========================================================================


@dataclass(frozen=True)
class Fact:
    """One numeric (or date) leaf of the facts sheet."""

    path: str            # dotted path, e.g. "comparison.delta_cost"
    value: Any           # float for numbers, ISO string for dates
    unit: str            # currency / fraction / percent_points / tonnes / kg / count / days / weeks / date / plain
    topics: frozenset    # meaning tags used to match claims to the right field
    is_delta: bool       # True for changes, where the sign matters


_TOPIC_WORDS = {
    "cost": ("cost", "price", "spend", "saving", "expens"),
    "carbon": ("carbon", "emission", "co2"),
    "late": ("late", "delay"),
    "moved": ("moved", "move", "reassign", "shift"),
    "utilisation": ("utilis", "utiliz", "load", "full"),
    "capacity": ("capacity",),
    "orders": ("order",),
    "cut": ("cut", "reduc", "lower"),
    "weight": ("weight",),
}

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _topics_for(text: str) -> frozenset:
    text = text.lower()
    return frozenset(t for t, words in _TOPIC_WORDS.items() if any(w in text for w in words))


def _unit_for(path: str) -> str:
    """Infer the unit of a facts-sheet value from its key name."""
    key = path.lower()
    last = key.rsplit(".", 1)[-1]
    if "weights" in key or last in ("capacity_loss_pct", "duty_pct"):
        return "fraction"
    if "utilis" in key or "utiliz" in key:
        return "fraction"
    if last.endswith("_pct") or last == "pct":
        return "percent_points"
    if "cost" in last or "saving" in last or "price" in last:
        return "currency"
    if last.endswith("_kg") or last == "carbon_kg":
        return "kg"
    if last.endswith("_t") or "tonne" in last:
        return "tonnes"
    if "days" in last:
        return "days"
    if "weeks" in last:
        return "weeks"
    if any(w in last for w in ("late", "moved", "orders", "count", "qty", "units", "changes")):
        return "count"
    return "plain"


def iter_facts(facts: dict[str, Any], prefix: str = "") -> Iterator[Fact]:
    """Yield every number/date in ``facts`` with its path, unit and topics.

    Lists also produce a ``.count`` fact (e.g. number of moved orders), because
    "3 orders moved" is a count claim about the list.
    """
    items = facts.items() if isinstance(facts, dict) else enumerate(facts)
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool) or value is None:
            continue  # booleans are flags, not numeric claims
        if isinstance(value, dict):
            yield from iter_facts(value, path)
        elif isinstance(value, list):
            yield Fact(f"{path}.count", float(len(value)), "count",
                       _topics_for(path) | {"orders"}, False)
            yield from iter_facts(value, path)
        elif isinstance(value, (int, float)):
            unit = _unit_for(path)
            yield Fact(path, float(value), unit, _topics_for(path),
                       any(w in path.lower() for w in ("delta", "change", "extra", "diff")))
        elif isinstance(value, (date, datetime)):
            yield Fact(path, value.isoformat()[:10], "date", _topics_for(path), False)
        elif isinstance(value, str) and _ISO_DATE.match(value):
            yield Fact(path, value, "date", _topics_for(path), False)


def derive_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``facts`` with deterministic derived values added.

    Code, not the LLM, does this arithmetic so that any number the text might
    need is already in the facts sheet.
    """
    out = copy.deepcopy(facts)
    points = out.get("tradeoff") or []
    base = next((p for p in points if p.get("carbon_cut_pct") == 0 and p.get("cost") is not None), None)
    if base:
        for p in points:
            if p.get("cost") is not None:
                p["delta_cost_vs_base"] = round(p["cost"] - base["cost"], 2)
            if p.get("late_orders") is not None and base.get("late_orders") is not None:
                p["delta_late_vs_base"] = p["late_orders"] - base["late_orders"]
    comp = out.get("comparison")
    if isinstance(comp, dict) and isinstance(comp.get("moved_orders"), list):
        comp["moved_orders_count"] = len(comp["moved_orders"])
    if isinstance(out.get("infeasible_orders"), list):
        out["infeasible_orders_count"] = len(out["infeasible_orders"])
    return out


# ==========================================================================
# 2. Claims: find every number written in a text
# ==========================================================================

_SMALL_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
_SCALES = {
    "k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "m": 1e6, "mn": 1e6,
    "million": 1e6, "crore": 1e7, "crores": 1e7, "bn": 1e9, "billion": 1e9,
}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_UP_WORDS = ("increase", "rise", "rose", "up ", "more", "added", "adds", "extra",
             "higher", "grow", "additional", "worse")
_DOWN_WORDS = ("decrease", "fall", "fell", "down ", "less", "lower", "reduc", "sav",
               "drop", "cut", "fewer", "better")

_CLAIM_RE = re.compile(
    r"(?P<sign>[-−+])?\s?"
    r"(?P<cur>₹|\$|€|£|(?:Rs\.?|INR|USD|EUR)\s?)?"
    r"(?P<num>\d(?:[\d,]*\d)?(?:\.\d+)?)"
    r"(?:\s?(?P<scale>k|K|M|mn|thousand|lakhs?|crores?|bn|million|billion)\b)?"
    r"(?:\s?(?P<unit>%|percent\b|per\s?cent\b|percentage\s+points?\b|pp\b|"
    r"tCO2e?\b|tonnes?\b|tons?\b|t\b|kg(?:CO2e?)?\b|kilograms?\b|"
    r"(?:late\s+)?orders?\b|days?\b|weeks?\b|units?\b))?",
)
# Tokens that contain digits but are identifiers, not quantities: O-077, ORD-0001, P1, D12.
_IDENT_RE = re.compile(r"\b(?!(?:Rs|INR|USD|EUR|tCO2e?|kgCO2e?|CO2e?)\b)[A-Za-z]{1,6}-?\d+[A-Za-z0-9-]*\b")
_ISO_IN_TEXT = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_IN_TEXT = re.compile(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})\b", re.I)
_MDY_IN_TEXT = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I)


@dataclass
class _Claim:
    raw: str
    value: Optional[float]
    unit: str
    decimals: int
    scale: float
    explicit_sign: int     # -1 / +1 / 0
    direction: int         # from words like "increase"/"saving": -1 / +1 / 0
    topics: frozenset
    date_iso: Optional[str] = None


def _mask(text: str, start: int, end: int) -> str:
    """Blank out a span so later regexes do not re-read its digits."""
    return text[:start] + " " * (end - start) + text[end:]


def _replace_number_words(text: str) -> str:
    """'two late orders' -> '2 late orders' (keeps positions roughly aligned)."""
    def repl(m: re.Match) -> str:
        return str(_SMALL_NUMBER_WORDS[m.group(0).lower()])
    pattern = r"\b(" + "|".join(_SMALL_NUMBER_WORDS) + r")\b(?=\s+(?:late\s+)?(?:orders?|plants?|days?|weeks?|%|percent))"
    return re.sub(pattern, repl, text, flags=re.I)


def _direction(window: str) -> int:
    window = window.lower() + " "
    up = any(w in window for w in _UP_WORDS)
    down = any(w in window for w in _DOWN_WORDS)
    if up and not down:
        return 1
    if down and not up:
        return -1
    return 0


def extract_claims(text: str) -> list[_Claim]:
    """Find every numeric or date claim in ``text``."""
    claims: list[_Claim] = []
    work = _replace_number_words(text)

    # Dates first, then mask them so "2026" is not read as a quantity.
    for pattern, kind in ((_ISO_IN_TEXT, "iso"), (_DMY_IN_TEXT, "dmy"), (_MDY_IN_TEXT, "mdy")):
        for m in pattern.finditer(work):
            try:
                if kind == "iso":
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                elif kind == "dmy":
                    d = date(int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
                else:
                    d = date(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
            except ValueError:
                d = None
            claims.append(_Claim(m.group(0), None, "date", 0, 1, 0, 0, frozenset(),
                                 d.isoformat() if d else "invalid"))
            work = _mask(work, m.start(), m.end())

    for m in _IDENT_RE.finditer(work):
        work = _mask(work, m.start(), m.end())

    for m in _CLAIM_RE.finditer(work):
        num = m.group("num").replace(",", "")
        if not num:
            continue
        decimals = len(num.split(".")[1]) if "." in num else 0
        scale = _SCALES.get((m.group("scale") or "").lower(), 1.0)
        if m.group("scale") == "M":
            scale = 1e6
        unit_word = (m.group("unit") or "").lower()
        if m.group("cur"):
            unit = "currency"
        elif unit_word.startswith(("%", "percent", "per")) or unit_word == "pp":
            unit = "percent"
        elif unit_word.startswith(("t", "ton")):
            unit = "tonnes"
        elif unit_word.startswith(("kg", "kilo")):
            unit = "kg"
        elif "order" in unit_word:
            unit = "count"
        elif unit_word.startswith("day"):
            unit = "days"
        elif unit_word.startswith("week"):
            unit = "weeks"
        elif unit_word.startswith("unit"):
            unit = "count"
        else:
            unit = "plain"
        sign_char = m.group("sign") or ""
        explicit = -1 if sign_char in "-−" and sign_char else (1 if sign_char == "+" else 0)
        lo, hi = max(0, m.start() - 45), min(len(work), m.end() + 30)
        # Context words around the number tell us what it is about.
        window = work[lo:hi]
        # Meaning comes first from the words BEFORE the number ("cost of 5,000");
        # only if there are none do we look after it ("5,000 in cost").
        # Where the meaning comes from, in order of reliability:
        #  1. the unit phrase itself and the words right after it
        #     ("2 late orders", "3 orders moved plant", "10% carbon cut");
        #  2. the words BEFORE the number ("cost of 5,000");
        #  3. the words after it ("5,000 in cost").
        topics = _topics_for(unit_word + " " + work[m.end():m.end() + 15]) if unit in ("count", "percent") else frozenset()
        topics = topics or _topics_for(work[lo:m.start()]) or _topics_for(work[m.end():hi])
        # A unit carries meaning by itself: money is cost, kg/tonnes is carbon.
        if unit == "currency":
            topics = topics | {"cost"}
        elif unit in ("kg", "tonnes"):
            topics = topics | {"carbon"}
        claims.append(_Claim(
            raw=m.group(0).strip(), value=float(num) * scale, unit=unit,
            decimals=decimals, scale=scale, explicit_sign=explicit,
            direction=_direction(window), topics=topics,
        ))
    return claims


# ==========================================================================
# 3. Fact check
# ==========================================================================

def _candidates(claim: _Claim, fact: Fact) -> list[float]:
    """Values of ``fact`` expressed in the claim's unit (empty = incompatible)."""
    v = fact.value
    pairs = {
        "currency": {"currency": [v]},
        "percent": {"fraction": [v * 100], "percent_points": [v]},
        "tonnes": {"tonnes": [v], "kg": [v / 1000]},
        "kg": {"kg": [v], "tonnes": [v * 1000]},
        "count": {"count": [v]},
        "days": {"days": [v]},
        "weeks": {"weeks": [v]},
        # A bare number is only acceptable for these kinds, and only when its
        # surrounding words name the same topic (checked by the caller).
        "plain": {"currency": [v], "count": [v], "plain": [v], "days": [v], "weeks": [v],
                  "percent_points": [v]},
    }
    return pairs.get(claim.unit, {}).get(fact.unit, [])


def _close(claim_value: float, fact_value: float, claim: _Claim) -> bool:
    """Equal within the precision the claim was written with.

    "1.84M" may stand for 1,835,000-1,845,000; "₹50,000" may not stand for
    5,00,000. Tolerance = half of the last written digit, times the scale.
    """
    tolerance = 0.5 * (10 ** -claim.decimals) * claim.scale + 1e-9
    return abs(claim_value - fact_value) <= tolerance


def _check_one(claim: _Claim, facts: list[Fact]) -> ClaimCheck:
    if claim.unit == "date":
        for f in facts:
            if f.unit == "date" and f.value == claim.date_iso:
                return ClaimCheck(raw=claim.raw, unit="date", supported=True, matched_fact=f.path)
        return ClaimCheck(raw=claim.raw, unit="date", supported=False,
                          reason="date not present in facts")

    pool = facts
    # Meaning check: if the sentence names a topic (cost, carbon, late...),
    # only facts about that topic may support the number.
    if claim.topics:
        pool = [f for f in facts if f.topics & claim.topics] or []
    elif claim.unit == "plain":
        return ClaimCheck(raw=claim.raw, value=claim.value, unit=claim.unit, supported=False,
                          reason="bare number with no stated meaning cannot be verified")

    reasons: list[str] = []
    for fact in pool:
        for target in _candidates(claim, fact):
            signed_claim = claim.value * (claim.explicit_sign or 1)
            if claim.explicit_sign == -1:
                if not _close(signed_claim, target, claim):
                    reasons.append(f"sign/magnitude differs from {fact.path}")
                    continue
            elif not _close(claim.value, abs(target), claim):
                continue
            # For changes, words like "saving"/"increase" must agree with the sign.
            if fact.is_delta and claim.direction and target != 0 and (target > 0) != (claim.direction > 0):
                reasons.append(f"direction contradicts sign of {fact.path}")
                continue
            if not fact.is_delta and target < 0 and claim.explicit_sign != -1:
                continue
            return ClaimCheck(raw=claim.raw, value=claim.value, unit=claim.unit,
                              supported=True, matched_fact=fact.path)
    return ClaimCheck(raw=claim.raw, value=claim.value, unit=claim.unit, supported=False,
                      reason="; ".join(dict.fromkeys(reasons)) or "no matching fact with this unit and meaning")


def _quoted_strings(facts: Any) -> list[str]:
    """Longer text values in the facts (assumptions, reasons, questions).

    These sentences were written by code, so quoting them verbatim is allowed.
    Numbers inside them are removed before checking; numbers NOT inside an
    exact quote are still checked.
    """
    found: list[str] = []
    if isinstance(facts, dict):
        for v in facts.values():
            found += _quoted_strings(v)
    elif isinstance(facts, list):
        for v in facts:
            found += _quoted_strings(v)
    elif isinstance(facts, str) and len(facts) >= 8 and re.search(r"\d", facts) and not _ISO_DATE.match(facts):
        found.append(facts)
    return found


def fact_check(answer: str, facts: dict[str, Any]) -> FactCheckResult:
    """Verify every number and date in ``answer`` against ``facts``.

    A claim is supported only if a fact exists with a compatible unit, a
    matching topic (cost vs carbon vs late orders...), a value equal within the
    written precision, and, for changes, a matching direction/sign.
    """
    fact_list = list(iter_facts(derive_facts(facts)))
    # Longest first, so a sentence is removed before any shorter part of it.
    for quote in sorted(_quoted_strings(facts), key=len, reverse=True):
        answer = answer.replace(quote, " " * len(quote))
    verified, unsupported = [], []
    for claim in extract_claims(answer):
        result = _check_one(claim, fact_list)
        (verified if result.supported else unsupported).append(result)
    return FactCheckResult(passed=not unsupported, verified=verified, unsupported=unsupported)


# ==========================================================================
# 4. Deterministic templates (work with no LLM at all)
# ==========================================================================

def _money(v: float, facts: dict) -> str:
    """Whole currency units with thousands separators, e.g. 'INR 531,088'."""
    cur = facts.get("currency")
    text = f"{abs(v):,.0f}"
    return f"{cur} {text}" if cur else text


def _tonnes(v: float) -> str:
    # Plan totals can be small (about 1.6 t), so keep two decimals.
    return f"{abs(v):,.2f} t"


def _kg(v: float) -> str:
    return f"{abs(v):,.1f} kg"


def _pct_from_fraction(v: float) -> str:
    return f"{v * 100:.0f}%"


def _change(v: Optional[float], fmt) -> str:
    """'an increase of X' / 'a decrease of X' / 'no change'.

    If the value rounds to zero at the displayed precision we say "no change",
    so float noise such as -2.2e-16 is never reported as a decrease.
    """
    if v is None:
        return "not available"
    shown = fmt(v)
    if not re.search(r"[1-9]", shown):
        return "no change"
    return f"{'an increase' if v > 0 else 'a decrease'} of {shown}"


def _kpi_sentence(k: dict, facts: dict, label: str = "The plan") -> str:
    parts = []
    if k.get("total_cost") is not None:
        parts.append(f"total landed cost {_money(k['total_cost'], facts)}")
    if k.get("total_carbon_t") is not None:
        parts.append(f"total carbon {_tonnes(k['total_carbon_t'])}")
    if k.get("late_orders") is not None:
        parts.append(f"{int(k['late_orders'])} late orders")
    if k.get("unassigned_orders"):
        parts.append(f"{int(k['unassigned_orders'])} unassigned orders")
    sentence = f"{label} has " + ", ".join(parts) + "." if parts else ""
    util = k.get("plant_utilisation") or {}
    if util:
        sentence += " Plant utilisation: " + ", ".join(
            f"Plant {p} {_pct_from_fraction(u)}" for p, u in sorted(util.items())) + "."
    return sentence


def _unavailable(f: dict) -> str:
    status = f.get("solver_status") or "unknown"
    return (f"No verified plan is available (solver status: {status}). "
            "No cost, carbon or allocation results are reported.")


def _tpl_allocation(f: dict) -> str:
    if f.get("results_unavailable") or not f.get("kpis"):
        return _unavailable(f)
    return _kpi_sentence(f["kpis"], f)


def _tpl_capacity_loss(f: dict) -> str:
    sc, comp = f.get("scenario") or {}, f.get("comparison") or {}
    lines = []
    if sc.get("plant"):
        desc = f"Plant {sc['plant']}"
        if sc.get("capacity_loss_pct") is not None:
            desc += f" losing {_pct_from_fraction(sc['capacity_loss_pct'])} of capacity"
        if sc.get("from_date") and sc.get("to_date"):
            desc += f" from {sc['from_date']} to {sc['to_date']}"
        lines.append(f"Scenario: {desc}.")
    if not comp:
        lines.append("A comparison with the base plan is not available.")
    else:
        lines.append(f"Cost change: {_change(comp.get('delta_cost'), lambda v: _money(v, f))}.")
        lines.append(f"Carbon change: {_change(comp.get('delta_carbon_t'), _tonnes)}.")
        lines.append(f"Late orders change: {_change(comp.get('delta_late'), lambda v: f'{abs(int(v))} orders')}.")
        moved = comp.get("moved_orders") or []
        lines.append(f"{len(moved)} orders moved plant" + (f": {', '.join(moved[:10])}." if moved else "."))
        if comp.get("dispatch_changes"):
            lines.append(f"{int(comp['dispatch_changes'])} orders changed dispatch date.")
        if comp.get("comparable_fulfillment") is False and comp.get("interpretation"):
            lines.append(comp["interpretation"])
    if f.get("kpis"):
        lines.append(_kpi_sentence(f["kpis"], f, "The re-plan"))
    elif f.get("results_unavailable"):
        lines.append(_unavailable(f))
    return " ".join(lines)


def _tpl_tradeoff(f: dict) -> str:
    lines = []
    for p in derive_facts(f).get("tradeoff") or []:
        cut = p.get("carbon_cut_pct")
        if p.get("feasible") is False:
            s = f"A {cut:g}% carbon cut is not achievable with current capacity and due dates"
            if p.get("carbon_cap_kg") is not None and p.get("baseline_carbon_kg") is not None:
                s += f" (carbon cap {_kg(p['carbon_cap_kg'])} against a baseline of {_kg(p['baseline_carbon_kg'])})"
            lines.append(s + ".")
            continue
        label = "The base plan" if cut == 0 else f"A {cut:g}% carbon cut"
        s = f"{label}: cost {_money(p['cost'], f)}"
        if p.get("carbon_t") is not None:
            s += f", carbon {_tonnes(p['carbon_t'])}"
        if cut != 0 and p.get("delta_cost_vs_base") is not None:
            s += f" ({_change(p['delta_cost_vs_base'], lambda v: _money(v, f))} in cost vs the base plan)"
        if p.get("late_orders") is not None:
            s += f", {int(p['late_orders'])} late orders"
        lines.append(s + ".")
    for p in f.get("weight_tradeoff") or []:
        w = p.get("weights") or {}
        lines.append(
            f"Weights cost {_pct_from_fraction(w.get('cost', 0))} / carbon {_pct_from_fraction(w.get('carbon', 0))}: "
            f"cost {_money(p['cost'], f)}, carbon {_tonnes(p['carbon_t'])}, {int(p['late_orders'])} late orders."
        )
    return " ".join(lines) or "No trade-off points were computed."


def _tpl_explain(f: dict) -> str:
    ex = f.get("order_explanation") or {}
    oid, chosen = ex.get("order_id", "the order"), ex.get("chosen_plant")
    lines = [f"Order {oid} is assigned to Plant {chosen}." if chosen else f"Order {oid} is not assigned in this plan."]
    for o in ex.get("options") or []:
        if o.get("eligible") is False:
            lines.append(f"Plant {o['plant']} cannot make this product.")
        elif o.get("landed_cost") is not None:
            s = f"Plant {o['plant']} (chosen): landed cost {_money(o['landed_cost'], f)}"
            if o.get("carbon_kg") is not None:
                s += f", carbon {_kg(o['carbon_kg'])}"
            if o.get("arrival"):
                s += f", arrives {o['arrival']}" + (" on time" if o.get("on_time") else " late")
            lines.append(s + ".")
        elif o.get("standalone_cost") is not None:
            lines.append(f"Plant {o['plant']}: standalone estimate landed cost {_money(o['standalone_cost'], f)}, "
                         f"carbon {_kg(o['standalone_carbon_kg'])}.")
        elif o.get("on_time") is False:
            lines.append(f"Plant {o['plant']} cannot deliver this order on time even on its own.")
    if any(o.get("standalone_cost") is not None for o in ex.get("options") or []):
        lines.append("The plan balances cost and carbon across all orders; standalone estimates "
                     "ignore competition with other orders for capacity.")
    return " ".join(lines)


def _tpl_infeasible(f: dict) -> str:
    orders = f.get("infeasible_orders") or []
    if f.get("results_unavailable") and not orders:
        return _unavailable(f)
    lines = [f"{len(orders)} orders cannot be met within capacity, eligibility and delivery rules."]
    for o in orders[:10]:
        lines.append(f"Order {o.get('order_id')}: {o.get('reason', 'no feasible option')}.")
    if f.get("kpis"):
        lines.append(_kpi_sentence(f["kpis"], f, "The closest computed alternative"))
    elif f.get("closest_alternative"):
        lines.append("The closest computed alternative assigns no orders.")
    return " ".join(lines)


def _tpl_brief(f: dict) -> str:
    sections = ["Management brief."]
    if f.get("base_kpis"):
        sections.append(_kpi_sentence(f["base_kpis"], f, "The base plan"))
    if f.get("kpis"):
        sections.append(_kpi_sentence(f["kpis"], f, "The current plan"))
    elif f.get("results_unavailable"):
        sections.append(_unavailable(f))
    if f.get("tradeoff") or f.get("weight_tradeoff"):
        sections.append("Trade-off: " + _tpl_tradeoff(f))
    if f.get("comparison"):
        sections.append("Disruption impact: " + _tpl_capacity_loss({**f, "kpis": None, "results_unavailable": False}))
    if f.get("infeasible_orders"):
        sections.append("At risk: " + _tpl_infeasible({**f, "kpis": None}))
    for title, key in (("Assumptions", "assumptions"), ("Limitations", "limitations")):
        if f.get(key):
            sections.append(f"{title}: " + "; ".join(f[key][:5]) + ".")
    return " ".join(s for s in sections if s)


def render_template(facts: dict[str, Any]) -> str:
    """Deterministic text for any answer type. Uses only values in ``facts``."""
    kind = facts.get("answer_type")
    intent = facts.get("intent")
    if kind == "clarification_needed":
        return facts.get("clarification_question") or "Please clarify the request (which plant, order or value?)."
    if kind == "refused":
        reason = facts.get("refusal_reason") or "it would break a capacity or eligibility rule"
        return f"This request cannot be carried out because {reason}. Plans must respect capacity and eligibility."
    if kind == "brief":
        return _tpl_brief(facts)
    if intent == "explain_order" or facts.get("order_explanation"):
        return _tpl_explain(facts)
    if facts.get("tradeoff") or facts.get("weight_tradeoff"):
        return _tpl_tradeoff(facts)
    if kind == "infeasible":
        return _tpl_infeasible(facts)
    if kind == "scenario":
        return _tpl_capacity_loss(facts)
    return _tpl_allocation(facts)


# ==========================================================================
# 5. Recommendations (deterministic, grounded in facts)
# ==========================================================================

def _confidence(facts: dict) -> str:
    if facts.get("answer_type") == "infeasible" or facts.get("results_unavailable"):
        return "low"
    validator = facts.get("validator") or {}
    valid = validator.get("valid", all(v for v in validator.values() if isinstance(v, bool))) if validator else False
    # Vibhu's solver reports lowercase statuses ("optimal", "feasible", ...).
    if str(facts.get("solver_status", "")).lower() == "optimal" and valid and not facts.get("data_warnings"):
        return "high"
    return "medium"


def build_recommendations(facts: dict[str, Any]) -> list[Recommendation]:
    """Recommendations in the guide's 6.1 shape, from facts only.

    1. The optimizer's own recommendations (``SolveResult.recommendations``)
       are reused as-is after validation, never rewritten.
    2. Scenario, trade-off and at-risk recommendations are added with impacts
       copied from the facts sheet. Nothing is estimated here.
    """
    recs: list[Recommendation] = []
    for raw in facts.get("recommendations") or []:
        try:
            recs.append(Recommendation.model_validate(raw))
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            logger.warning("Skipping optimizer recommendation with unexpected shape: %s", exc)

    kind, conf = facts.get("answer_type"), _confidence(facts)
    comp = facts.get("comparison") or {}
    if comp:
        plant = (facts.get("scenario") or {}).get("plant")
        recs.append(Recommendation(
            action=f"Approve the re-plan for the Plant {plant} capacity loss" if plant else "Approve the re-plan",
            rationale="The re-plan was solved under the reduced capacity and passed the independent checker.",
            evidence_refs=[f"comparison.{k}" for k in ("delta_cost", "delta_carbon_t", "delta_late", "moved_orders")
                           if comp.get(k) is not None] + ["validator"],
            confidence=conf,
            estimated_impact={k: comp[k] for k in ("delta_cost", "delta_carbon_t", "delta_late") if comp.get(k) is not None},
            requires_approval=True,
        ))
    if facts.get("tradeoff") or facts.get("weight_tradeoff"):
        recs.append(Recommendation(
            action="Choose a cost/carbon option from the trade-off results",
            rationale="Each option is a separate solver run; infeasible targets are reported as such.",
            evidence_refs=["tradeoff" if facts.get("tradeoff") else "weight_tradeoff"],
            confidence=conf, estimated_impact={}, requires_approval=True,
        ))
    if facts.get("infeasible_orders"):
        recs.append(Recommendation(
            action="Review orders that cannot be met (add capacity or agree new due dates)",
            rationale="No plan satisfies capacity, eligibility and delivery for these orders.",
            evidence_refs=["infeasible_orders"], confidence="low",
            estimated_impact={"orders_at_risk": len(facts["infeasible_orders"])},
            requires_approval=True,
        ))
    if kind == "allocation" and not recs and facts.get("kpis"):
        recs.append(Recommendation(
            action="Review and release the proposed allocation",
            rationale="Plan produced by the solver and checked by the independent validator.",
            evidence_refs=["kpis", "validator"], confidence=conf,
            estimated_impact={k: facts["kpis"][k] for k in ("total_cost", "total_carbon_t", "late_orders")
                              if k in facts["kpis"]},
            requires_approval=True,
        ))
    return recs


# ==========================================================================
# 6. LLM writing with verification and fallback
# ==========================================================================

_WRITER_SYSTEM = """You write short, plain-English explanations for a manufacturing planning tool.
STRICT RULES:
- Use ONLY numbers and dates that appear in the FACTS JSON. Copy them exactly (you may add thousands separators).
- Never calculate, estimate, round differently, compare or invent any number, saving, ranking or emission.
- Do not mention any number that is not in FACTS, even if the user asks for it.
- Say what each number is (cost, carbon in tonnes, late orders, utilisation %).
- Carbon totals ending in _t are tonnes; _kg are kilograms. Fractions such as utilisation 0.88 may be written as 88%.
- 3 to 6 sentences. No headings, no bullet lists, no markdown."""

_KIND_HINT = {
    "allocation": "Summarise the allocation result.",
    "scenario": "Explain what changed in the scenario versus the base plan.",
    "brief": "Write a one-paragraph management brief for leadership.",
    "infeasible": "Explain which orders cannot be met and what the closest plan is.",
}


def _llm_prompt(facts: dict, feedback: Optional[str]) -> str:
    hint = _KIND_HINT.get(facts.get("answer_type"), "Explain the result.")
    if facts.get("intent") == "explain_order":
        hint = "Explain why the order was assigned to its plant, comparing the options."
    prompt = f"TASK: {hint}\n\nFACTS:\n{json.dumps(facts, default=str, indent=1)}"
    if feedback:
        prompt += f"\n\nYour previous answer contained numbers that are NOT in FACTS: {feedback}. Remove or correct them."
    return prompt


def write_text(
    facts: dict[str, Any],
    *,
    client: Optional[LLMClient] = None,
    user_request: Optional[str] = None,
) -> WriterResult:
    """Write a fact-checked summary, explanation or brief from ``facts``.

    Args:
        facts: facts sheet built by the workflow (see module docstring).
        client: LLM client; defaults to one built from environment settings.
        user_request: optional original request. It is NOT shown to the LLM as
            an instruction (so "say we saved 2 million" cannot steer the text);
            it is accepted only so callers have one stable signature.

    Returns:
        WriterResult with ``source`` = "llm" or "template". The text always
        passed ``fact_check`` unless the template itself contains an
        unsupported number, which is reported in ``fact_check`` and logged.
    """
    del user_request  # deliberately unused: requests never steer numeric content
    client = client or get_client()
    enriched = derive_facts(facts)
    recs = build_recommendations(enriched)

    def template(reason: str, attempts: int, calls: int, tokens: int) -> WriterResult:
        text = render_template(enriched)
        check = fact_check(text, enriched)
        if not check.passed:
            logger.error("Template produced unsupported numbers: %s", [c.raw for c in check.unsupported])
        return WriterResult(text=text, source="template", attempts=attempts, fact_check=check,
                            recommendations=recs, llm_calls=calls, tokens=tokens,
                            fallback_reason=reason)

    # These answer types are fixed messages; an LLM adds nothing but risk.
    if facts.get("answer_type") in ("refused", "clarification_needed"):
        return template("deterministic answer type", 0, 0, 0)
    if client.is_mock:
        return template("MOCK_LLM enabled", 0, 0, 0)

    calls0, tokens0 = client.calls, client.total_tokens
    feedback: Optional[str] = None
    for attempt in (1, 2):
        try:
            response = client.complete(_WRITER_SYSTEM, _llm_prompt(enriched, feedback))
        except LLMError as exc:
            logger.warning("Writer LLM unavailable, using template: %s", exc)
            return template(f"LLM unavailable: {exc}", attempt,
                            client.calls - calls0, client.total_tokens - tokens0)
        text = response.text.strip()
        check = fact_check(text, enriched)
        if text and check.passed:
            return WriterResult(text=text, source="llm", attempts=attempt, fact_check=check,
                                recommendations=recs, llm_calls=client.calls - calls0,
                                tokens=client.total_tokens - tokens0)
        feedback = ", ".join(c.raw for c in check.unsupported) or "empty answer"
        logger.warning("Writer attempt %d failed fact-check: %s", attempt, feedback)

    return template(f"fact-check failed twice ({feedback})", 2,
                    client.calls - calls0, client.total_tokens - tokens0)

"""CarbonWise Allocator - Streamlit dashboard.

Run from the project root:   streamlit run ui/app.py

What this page does
-------------------
1. Sends the user's request to FastAPI (POST /analyze) through ``api_client``.
2. Shows ONLY what the API returned: KPI cards, allocation table, Plotly
   charts, the fact-checked explanation, validation results and the trace.
3. Never runs the optimizer or the LLM itself, and never shows a missing
   metric as zero (missing values are simply not displayed).

Per-plant chart totals are sums of the allocation rows the optimizer returned,
labelled as such; the authoritative totals are the KPI cards from the engine.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client

st.set_page_config(page_title="CarbonWise Allocator", page_icon="🏭", layout="wide")

EXAMPLES = [
    "Allocate this week's orders.",
    "What does a 10% carbon cut cost?",
    "Show the cost and carbon trade-off.",
    "Plant B loses 30% capacity for two weeks. Re-plan.",
    "Why was order ORD-0001 assigned to Plant C?",
    "Write the brief for the leadership team.",
]
BADGE = {"allocation": "✅ Allocation", "scenario": "🔁 Scenario", "brief": "📝 Brief",
         "infeasible": "⚠️ Infeasible", "clarification_needed": "❓ Clarification needed",
         "refused": "⛔ Refused", "error": "❌ Error"}


# --------------------------------------------------------------------- helpers
def fmt_inr(value: Optional[float]) -> str:
    return "—" if value is None else f"INR {value:,.0f}"


def fmt_signed(value: Optional[float], unit: str, digits: int = 0) -> Optional[str]:
    if value is None:
        return None
    if abs(value) < 10 ** (-digits) / 2:
        return "No change"
    return f"{value:+,.{digits}f} {unit}"


# ---------------------------------------------------------------------- header
def render_header() -> None:
    st.title("CarbonWise Allocator")
    st.caption("Multi-Plant Order Allocation with Cost and Carbon · solver-built plans, "
               "independently validated, explained with fact-checked text")
    health = api_client.health()
    cols = st.columns(4)
    if not health.ok:
        st.error(f"Backend not reachable. {health.error}")
        return
    h = health.data
    cols[0].success("API connected")
    cols[1].write(f"**Data store:** {h['data_store']}")
    cols[2].write(f"**Solver (CBC):** {h['solver']}")
    llm = h.get("llm", {})
    cols[3].write(f"**LLM:** {llm.get('provider')} · {llm.get('status')}")
    if h.get("dataset_version_id"):
        st.caption(f"Active dataset version: `{h['dataset_version_id']}`")


# --------------------------------------------------------------------- sidebar
def render_sidebar() -> tuple[Optional[dict], Optional[dict]]:
    """Optional weights / scenario JSON, plus CSV upload. Returns (weights, scenario)."""
    st.sidebar.header("Options")
    weights = None
    if st.sidebar.checkbox("Set objective weights", value=False):
        cost = st.sidebar.slider("Cost weight", 0.0, 1.0, 0.7, 0.05)
        weights = {"cost": cost, "carbon": round(1 - cost, 2), "lateness": 0.0}
        st.sidebar.caption("Without this, the optimizer's defaults are used (0.7 cost / 0.3 carbon).")
    scenario = None
    raw = st.sidebar.text_area("Scenario JSON (optional)", placeholder='{"plant": "B", "capacity_loss_pct": 0.3, '
                               '"from_date": "2026-01-12", "to_date": "2026-01-25"}')
    if raw.strip():
        try:
            scenario = json.loads(raw)
            st.sidebar.caption("capacity_loss_pct is a fraction: 0.3 = 30%.")
        except json.JSONDecodeError:
            st.sidebar.error("Scenario is not valid JSON; it will be ignored.")

    st.sidebar.header("Load data (POST /ingest)")
    table = st.sidebar.selectbox("Table", ["orders", "plants", "transport_duties", "capacity_events"])
    upload = st.sidebar.file_uploader("CSV file", type=["csv"])
    if upload is not None and st.sidebar.button("Upload and validate"):
        with st.sidebar.spinner("Validating with the data module..."):
            res = api_client.ingest(table, upload.name, upload.getvalue())
        data = res.data or {}
        if res.ok:
            st.sidebar.success(f"Accepted: {data.get('rows_loaded')} rows. New version {data.get('dataset_version_id')}")
        else:
            st.sidebar.error(f"Rejected ({data.get('rows_rejected', '?')} rows). {res.error or ''}")
            for issue in data.get("errors", [])[:10]:
                st.sidebar.write(f"- {issue.get('rule')}: {issue.get('message')}")
    return weights, scenario


# ---------------------------------------------------------------- KPI section
def render_kpis(r: dict[str, Any]) -> None:
    """Show only metrics present in the response (never fill missing with 0)."""
    kpis, comp = r.get("kpis") or {}, r.get("comparison") or {}
    if not kpis and r.get("base_kpis"):
        # e.g. an unreachable carbon target: there is no new plan, so show the
        # baseline plan's real KPIs and say so.
        st.caption("No feasible plan for the requested change. Showing the BASELINE plan:")
        kpis = r["base_kpis"]
    cards: list[tuple[str, str]] = []
    if "total_cost" in kpis:
        cards.append(("Total landed cost", fmt_inr(kpis["total_cost"])))
    if "total_carbon_t" in kpis:
        cards.append(("Total carbon", f"{kpis['total_carbon_t']:,.2f} t CO2e"))
    if r.get("allocation"):
        cards.append(("Allocated orders", f"{len(r['allocation'])}"))
    if "late_orders" in kpis:
        cards.append(("Late orders", f"{kpis['late_orders']}"))
    if kpis.get("unassigned_orders"):
        cards.append(("Unassigned orders", f"{kpis['unassigned_orders']}"))
    if cards:
        for col, (label, value) in zip(st.columns(len(cards)), cards):
            col.metric(label, value)

    # Changes vs the baseline get their own row, only for scenarios that have
    # a comparison. "No change" is shown in words, not as an arrow.
    if comp:
        changes = [
            ("Cost change", fmt_signed(comp.get("delta_cost"), "INR")),
            ("Carbon change", fmt_signed(comp.get("delta_carbon_t"), "t", 2)),
            ("Late-order change", fmt_signed(comp.get("delta_late"), "orders")),
            ("Orders moved plant", str(len(comp["moved_orders"])) if comp.get("moved_orders") is not None else None),
            ("Dispatch-date changes", str(comp["dispatch_changes"]) if comp.get("dispatch_changes") is not None else None),
        ]
        changes = [(label, value) for label, value in changes if value is not None]
        st.caption("Change versus the baseline plan (revised minus baseline)")
        for col, (label, value) in zip(st.columns(len(changes)), changes):
            col.metric(label, value)
        if comp.get("interpretation"):
            st.caption(comp["interpretation"])

    util = {p: u for p, u in (kpis.get("plant_utilisation") or {}).items() if u is not None}
    if util:
        st.caption("Plant utilisation over the horizon: " +
                   " · ".join(f"Plant {p}: {u:.0%}" for p, u in sorted(util.items())))


# -------------------------------------------------------------------- charts
def render_charts(r: dict[str, Any]) -> None:
    alloc = pd.DataFrame(r.get("allocation") or [])
    kpis = r.get("kpis") or {}
    if alloc.empty and not r.get("tradeoff"):
        st.info("No plan data to chart for this answer.")
        return
    if not alloc.empty:
        by_plant = alloc.groupby("plant", as_index=False).agg(cost=("cost", "sum"), carbon_kg=("carbon_kg", "sum"),
                                                              orders=("order_id", "count"))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(by_plant, x="plant", y="cost", text_auto=".3s",
                               title="Landed cost by plant (INR, sum of allocation rows)"), use_container_width=True)
        c2.plotly_chart(px.bar(by_plant, x="plant", y="carbon_kg", text_auto=".3s",
                               title="Carbon by plant (kg CO2e, sum of allocation rows)"), use_container_width=True)
    weekly = pd.DataFrame(kpis.get("plant_utilisation_by_week") or [])
    if not weekly.empty and "utilisation" in weekly:
        weekly = weekly.dropna(subset=["utilisation"])
        fig = px.line(weekly, x="week_start", y="utilisation", color="plant", markers=True,
                      title="Weekly plant utilisation (share of effective capacity)")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    base = r.get("base_kpis")
    if base and kpis:
        cmp = go.Figure()
        cmp.add_bar(name="Baseline", x=["Cost (INR)"], y=[base["total_cost"]])
        cmp.add_bar(name="Revised", x=["Cost (INR)"], y=[kpis["total_cost"]])
        cmp.update_layout(title="Baseline vs revised plan: landed cost", barmode="group")
        st.plotly_chart(cmp, use_container_width=True)


def render_tradeoff(r: dict[str, Any]) -> None:
    points = pd.DataFrame(r.get("tradeoff") or [])
    if points.empty:
        st.info("No trade-off was requested. Try “What does a 10% carbon cut cost?” "
                "or “Show the cost and carbon trade-off.”")
        return
    if "weights" in points:
        points["label"] = points["weights"].apply(lambda w: f"cost {w.get('cost', 0):.0%} / carbon {w.get('carbon', 0):.0%}")
    elif "carbon_cut_pct" in points:
        points["label"] = points["carbon_cut_pct"].apply(lambda c: f"{c:g}% carbon cut")
    plotted = points.dropna(subset=[c for c in ("cost", "carbon_t") if c in points]) \
        if {"cost", "carbon_t"} <= set(points) else pd.DataFrame()
    if not plotted.empty:
        fig = px.scatter(plotted, x="carbon_t", y="cost", text="label",
                         size=[12] * len(plotted), color="late_orders" if "late_orders" in plotted else None,
                         labels={"carbon_t": "Total carbon (t CO2e)", "cost": "Total landed cost (INR)",
                                 "late_orders": "Late orders"},
                         title="Cost vs carbon for each computed plan")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    if "feasible" in points and (points["feasible"] == False).any():  # noqa: E712
        st.warning("At least one requested target is not achievable with current capacity and due dates.")
    st.dataframe(points.drop(columns=["label"], errors="ignore"), use_container_width=True)


# ------------------------------------------------------------ result section
def render_result(r: dict[str, Any]) -> None:
    kind = r.get("answer_type", "error")
    st.subheader(BADGE.get(kind, kind))
    if kind == "clarification_needed":
        st.info(r.get("clarification_question") or r.get("summary"))
        return
    if kind == "refused":
        st.error(r.get("summary"))
        return
    if kind == "error":
        st.error(r.get("summary"))
        return
    if kind == "infeasible":
        st.warning("The optimizer could not satisfy every order. Nothing below is presented as a successful plan.")

    render_kpis(r)
    tabs = st.tabs(["Explanation", "Allocation", "Charts", "Trade-off", "Reliability"])
    with tabs[0]:
        render_explanation(r)
    with tabs[1]:
        render_allocation(r)
    with tabs[2]:
        render_charts(r)
    with tabs[3]:
        render_tradeoff(r)
    with tabs[4]:
        render_reliability(r)


def render_explanation(r: dict[str, Any]) -> None:
    fc = r.get("fact_check") or {}
    source = "LLM (fact-checked)" if r.get("summary_source") == "llm" else "deterministic template"
    st.write(r.get("summary", ""))
    if fc.get("passed"):
        st.caption(f"Written by: {source} · all {fc.get('verified_claims', 0)} numeric claims match optimizer output")
    else:
        st.warning(f"Unverified numbers were found: {fc.get('unsupported_claims')}. Trust the tables, not the text.")
    expl = r.get("explanation")
    if expl:
        st.markdown(f"**Order {expl.get('order_id')} → Plant {expl.get('chosen_plant')}**")
        st.dataframe(pd.DataFrame(expl.get("options", [])), use_container_width=True)
        st.caption(expl.get("caveat", ""))
    recs = r.get("recommendations") or []
    if recs:
        st.markdown("**Recommendations**")
        for rec in recs:
            approval = " · requires approval" if rec.get("requires_approval") else ""
            st.write(f"- {rec['action']} ({rec['confidence']} confidence{approval}). {rec['rationale']}")


def render_allocation(r: dict[str, Any]) -> None:
    alloc = pd.DataFrame(r.get("allocation") or [])
    if alloc.empty:
        st.info("No allocation is reported for this answer.")
    else:
        cols = [c for c in ("order_id", "plant", "qty", "dispatch", "arrival", "on_time", "late_days",
                            "cost", "carbon_kg") if c in alloc]
        st.caption("cost in INR · carbon in kg CO2e per order")
        st.dataframe(alloc[cols], use_container_width=True, hide_index=True)
    comp = r.get("comparison") or {}
    if comp.get("changes"):
        st.markdown("**Orders whose plant or dispatch date changed**")
        rows = [{"order_id": c["order_id"],
                 "plant before": (c.get("before") or {}).get("plant"), "plant after": (c.get("after") or {}).get("plant"),
                 "dispatch before": (c.get("before") or {}).get("dispatch"),
                 "dispatch after": (c.get("after") or {}).get("dispatch")} for c in comp["changes"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if r.get("unassigned_orders"):
        st.markdown("**Unassigned orders**")
        st.dataframe(pd.DataFrame(r["unassigned_orders"]), use_container_width=True, hide_index=True)


def render_reliability(r: dict[str, Any]) -> None:
    v = r.get("validator") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Independent validator**")
        for key in ("valid", "capacity_ok", "eligibility_ok", "delivery_ok"):
            if key in v:
                st.write(f"{'✅' if v[key] else '❌'} {key}")
        solver = r.get("solver") or {}
        if solver:
            st.write(f"Solver status: `{solver.get('status')}` ({solver.get('termination_reason')})")
    with c2:
        for title, key in (("Assumptions", "assumptions"), ("Limitations", "limitations"), ("Data warnings", "warnings")):
            if r.get(key):
                with st.expander(title):
                    for line in r[key]:
                        st.write(f"- {line}")
    if st.button("Load execution trace"):
        res = api_client.trace(r["run_id"])
        if res.ok:
            st.caption(f"LLM calls: {res.data.get('llm_calls')} · tokens: {res.data.get('tokens')}")
            st.dataframe(pd.DataFrame(res.data.get("steps", [])), use_container_width=True)
        else:
            st.error(res.error)


# ------------------------------------------------------------------------ main
def main() -> None:
    render_header()
    weights, scenario = render_sidebar()

    st.markdown("#### Ask a question")
    example_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if example_cols[i % 3].button(example, key=f"ex{i}", use_container_width=True):
            st.session_state["request_text"] = example
    text = st.text_area("Request", key="request_text", height=80,
                        placeholder="e.g. Plant B loses 30% capacity for two weeks. Re-plan.")
    run_col, rerun_col = st.columns([1, 1])
    run = run_col.button("Run analysis", type="primary", disabled=not text.strip())
    rerun = rerun_col.button("Rerun last analysis", disabled="last_request" not in st.session_state)

    if run or rerun:
        payload = (text, weights, scenario) if run else st.session_state["last_request"]
        with st.spinner("Solving with PuLP/CBC and checking the plan..."):
            res = api_client.analyze(*payload)
        st.session_state["last_request"] = payload
        st.session_state["last_result"] = res

    res = st.session_state.get("last_result")
    if res is None:
        st.info("Choose an example or type a request, then press **Run analysis**.")
    elif not res.ok:
        st.error(f"Request failed ({res.status}): {res.error}")
    else:
        st.caption(f"run_id `{res.data['run_id']}`")
        render_result(res.data)


main()

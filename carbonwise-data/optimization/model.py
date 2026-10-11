"""Pure linear integer model. No access to expected-results files."""

from __future__ import annotations
from dataclasses import dataclass
import pulp

from .contracts import SolvePolicy
from .preprocessing import PreparedProblem, arrival_date, eligible_plants


@dataclass
class ModelBundle:
    model: pulp.LpProblem
    x: dict
    z: dict
    u: dict
    cost: pulp.LpAffineExpression
    carbon: pulp.LpAffineExpression
    lateness: pulp.LpAffineExpression
    missed: pulp.LpAffineExpression
    unassigned: pulp.LpAffineExpression
    stability: pulp.LpAffineExpression


def build_model(
    p: PreparedProblem, policy: SolvePolicy, diagnostic=False, base_plan=None
) -> ModelBundle:
    model = pulp.LpProblem("carbonwise_allocation", pulp.LpMinimize)
    x, z, u = {}, {}, {}
    costs, carbons, lateness_terms, late_terms = [], [], [], []
    base = {r["order_id"]: r for r in base_plan.allocation} if base_plan else {}
    stability_terms = []
    for index, (oid, order) in enumerate(p.orders.items()):
        all_order_x = []
        for plant in eligible_plants(p, oid):
            if (
                oid in policy.forced_assignments
                and policy.forced_assignments[oid] != plant
            ):
                continue
            cap = p.capabilities[plant, order["product"]]
            route = p.routes[plant, order["destination"]]
            pair_x = []
            dispatch_vars = {}
            for ti, t in enumerate(p.dates):
                arrival = arrival_date(p, oid, plant, t)
                late = max(0, (arrival - order["due"]).days)
                if not diagnostic and late:
                    continue
                # No production can occur until a working date with positive capacity.
                if not any(
                    d <= t and p.daily_capacity[plant, d] > 0
                    for d in p.production_dates
                ):
                    continue
                key = oid, plant, t
                var = pulp.LpVariable(
                    f"x_{index}_{p.plants.index(plant)}_{ti}", cat="Binary"
                )
                x[key] = var
                dispatch_vars[t] = var
                pair_x.append(var)
                all_order_x.append(var)
                lateness_terms.append(late * var)
                if late:
                    late_terms.append(var)
                old = base.get(oid)
                if old and (old["plant"] != plant or old["dispatch"] != t.isoformat()):
                    stability_terms.append(var)
            if not pair_x:
                continue
            y = pulp.lpSum(pair_x)
            q = order["qty"]
            costs.append(
                q
                * (
                    cap["unit_cost"] * (1 + route["duty_pct"])
                    + route["transport_cost_per_unit"]
                )
                * y
            )
            carbons.append(q * route["carbon_kg_per_unit"] * y)
            production = []
            for di, d in enumerate(p.production_dates):
                available_dispatch = [v for t, v in dispatch_vars.items() if t >= d]
                if not available_dispatch or p.daily_capacity[plant, d] == 0:
                    continue
                var = pulp.LpVariable(
                    f"z_{index}_{p.plants.index(plant)}_{di}",
                    lowBound=0,
                    upBound=q,
                    cat="Integer",
                )
                z[oid, plant, d] = var
                production.append(var)
                model += var <= q * pulp.lpSum(available_dispatch)
                costs.append(
                    var
                    * cap["energy_kwh_per_unit"]
                    * p.prices[plant, d]
                    * (1 + route["duty_pct"])
                )
                carbons.append(
                    var
                    * cap["energy_kwh_per_unit"]
                    * p.carbon_factors[plant, order["product"], d]
                )
            model += pulp.lpSum(production) == q * y
        if diagnostic and oid not in policy.forced_assignments:
            u[oid] = pulp.LpVariable(f"u_{index}", cat="Binary")
            model += pulp.lpSum(all_order_x) + u[oid] == 1
            if oid in base:
                stability_terms.append(u[oid])
        else:
            model += pulp.lpSum(all_order_x) == 1
    for plant in p.plants:
        for d in p.production_dates:
            model += (
                pulp.lpSum(v for (i, pp, dd), v in z.items() if pp == plant and dd == d)
                <= p.daily_capacity[plant, d]
            )
    cost = pulp.lpSum(costs)
    carbon = pulp.lpSum(carbons)
    if policy.carbon_budget_kg is not None:
        model += carbon <= policy.carbon_budget_kg
    return ModelBundle(
        model,
        x,
        z,
        u,
        cost,
        carbon,
        pulp.lpSum(lateness_terms),
        pulp.lpSum(late_terms) + pulp.lpSum(u.values()),
        pulp.lpSum(u.values()),
        pulp.lpSum(stability_terms),
    )


def weighted_objective(bundle, weights, scales):
    w = weights.normalized()
    return (
        w["cost"] * bundle.cost / scales["cost"]
        + w["carbon"] * bundle.carbon / scales["carbon"]
        + w["lateness"] * bundle.lateness / scales["lateness"]
    )

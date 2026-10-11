"""Numerical reporting calculations; model coefficients are linear equivalents."""

from __future__ import annotations
from datetime import date
from .preprocessing import PreparedProblem


def cost_carbon_for_order(
    problem: PreparedProblem, order_id: str, plant: str, production: list[dict]
) -> dict:
    order = problem.orders[order_id]
    cap = problem.capabilities[plant, order["product"]]
    route = problem.routes[plant, order["destination"]]
    production_cost = order["qty"] * cap["unit_cost"]
    energy = carbon = 0.0
    for item in production:
        d = date.fromisoformat(str(item["date"]))
        qty = item["qty"]
        energy += qty * cap["energy_kwh_per_unit"] * problem.prices[plant, d]
        carbon += (
            qty
            * cap["energy_kwh_per_unit"]
            * problem.carbon_factors[plant, order["product"], d]
        )
    transport = order["qty"] * route["transport_cost_per_unit"]
    duty = (production_cost + energy) * route["duty_pct"]  # input fractions, NOT /100
    carbon += order["qty"] * route["carbon_kg_per_unit"]
    return {
        "cost": production_cost + energy + transport + duty,
        "carbon_kg": carbon,
        "cost_breakdown": {
            "production": production_cost,
            "energy": energy,
            "transport": transport,
            "duty": duty,
        },
    }

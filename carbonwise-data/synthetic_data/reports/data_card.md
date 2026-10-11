# CarbonWise Allocator synthetic data card

## Purpose
This synthetic dataset is created for development testing and demo support for the CarbonWise Allocator challenge. It intentionally contains realistic variation in plant capability, transport cost, duty, and capacity disruption scenarios while remaining deterministic under a fixed seed.

## Deterministic settings
- seed = 42
- planning window = 2026-01-01 to 2026-03-31
- three plants: A, B, C
- 10 destinations: D1 through D10
- synthetic products: P1 to P7

## Assumptions and conventions
- The official schema from Annex B is treated as the source-of-truth contract.
- orders.csv uses order_id, product, qty, destination, due.
- plants.csv stores one row per plant-product eligibility entry with capacity, cost, energy and carbon values.
- transport_duties.csv stores one row per plant-destination pair with cost, transit days, and duty share.
- capacity_events.csv stores plant disruption intervals and reduction percentages.
- Nomenclature: `weekly_capacity` is units available per production week, expressed as a positive integer.
- `carbon_kg_per_kwh` is the plant carbon intensity of purchased energy in kg CO2 per kWh.
- `carbon_kg_per_unit` is the transport carbon factor in kg CO2 per shipped unit.
- We do not infer carbon from energy price; we use the explicit carbon coefficients in the official schema.
- Import duty is applied as a percentage of the landed cost base and is represented in `duty_pct` in the transport table.

## Known limitations
- Energy price is not included in the official annex schema and therefore is not used in the generated calculations.
- Carbon and cost trade-offs are synthetic and designed for challenge testing, not for real-world plant operations.

# CarbonWise Allocator synthetic data package

This package generates deterministic synthetic training and demo data for the CarbonWise Allocator challenge. It follows the official Annex B schema and includes validation checks, scenario fixtures, and exact-answer expectations.

## Structure

- `scripts/generate_synthetic_data.py` creates and validates all CSV outputs.
- `synthetic_data/main/` contains the clean demonstration dataset.
- `synthetic_data/fixtures/` contains focused scenario and invalid-data examples.
- `synthetic_data/expected_results/exact_answer.json` stores the expected exact-answer fixture result.
- `synthetic_data/reports/` contains schema, validation, and data-card reports.
- `tests/` contains automated verification tests.

## Official schema used

The generator preserves the required field names and ordering from the Annnex B data schema:

- `orders.csv`: `order_id, product, qty, destination, due`
- `plants.csv`: `plant, product, weekly_capacity, unit_cost, energy_kwh_per_unit, carbon_kg_per_kwh`
- `transport_duties.csv`: `plant, destination, transport_cost_per_unit, transit_days, duty_pct, carbon_kg_per_unit`
- `capacity_events.csv`: `event_id, plant, from_date, to_date, capacity_loss_pct`

## Regenerating the data

```bash
python scripts/generate_synthetic_data.py
```

This uses a fixed seed of 42 and generates the same CSVs every run.

## Running the tests

```bash
pytest -q
```

## Notes

- The package keeps synthetic data deterministic and replayable.
- The main dataset contains a realistic mix of normal, edge, and disruption scenarios.
- The separate exact-answer fixture is intentionally small enough to reason about manually.

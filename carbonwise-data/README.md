# CarbonWise Allocator synthetic data package

This package generates deterministic synthetic training and demo data for the CarbonWise Allocator challenge. It follows the official Annex B schema and includes validation checks, scenario fixtures, exact-answer expectations, CSV ingestion, and versioned SQLite storage.

## Structure

- `scripts/generate_synthetic_data.py` creates and validates all CSV outputs.
- `scripts/ingest_dataset.py` validates and imports the four CSVs into SQLite.
- `services/ingestion.py` and `services/validation.py` implement CSV loading, normalization, schema validation, and cross-table checks.
- `storage/repository.py` initializes SQLite and persists versioned snapshots and ingestion logs.
- `database/schema.sql` defines the SQL tables and constraints.
- `synthetic_data/main/` contains the clean demonstration dataset.
- `synthetic_data/fixtures/` contains focused scenario and invalid-data examples.
- `synthetic_data/expected_results/exact_answer.json` stores the expected exact-answer fixture result.
- `synthetic_data/reports/` contains schema, validation, and data-card reports.
- `tests/` contains automated verification tests.
- `docs/ingestion_and_database.md` documents ingestion, versioning, SQL inspection, validation semantics, and the downstream interface.

## Official schema used

The generator and ingestion layer preserve the required field names and ordering from the Annex B data schema:

- `orders.csv`: `order_id, product, qty, destination, due`
- `plants.csv`: `plant, product, weekly_capacity, unit_cost, energy_kwh_per_unit, carbon_kg_per_kwh`
- `transport_duties.csv`: `plant, destination, transport_cost_per_unit, transit_days, duty_pct, carbon_kg_per_unit`
- `capacity_events.csv`: `event_id, plant, from_date, to_date, capacity_loss_pct`

## Regenerating the data

```bash
python scripts/generate_synthetic_data.py
```

This uses a fixed seed of 42 and generates the same CSVs every run.

## Ingesting a dataset

Install requirements and ingest the checked-in main dataset:

```powershell
python -m pip install -r requirements.txt
python scripts/ingest_dataset.py
```

The default persistent SQLite database is `data/carbonwise.sqlite3`. Override it with
`--db-path` or the `CARBONWISE_DB_PATH` environment variable. To ingest another
four-file dataset, pass its directory using `--data-dir`. A successful ingestion
prints a real dataset version ID, row counts, and validation report. A rejected
dataset also gets a rejected version with persisted issue logs.

The downstream calculation engine can call:

```python
from services.ingestion import ingest_all

result = ingest_all("synthetic_data/main", db_path="data/carbonwise.sqlite3")
tables = result["tables"]
dataset_version_id = result["dataset_version_id"]
validation_report = result["validation_report"]
```

See [docs/ingestion_and_database.md](docs/ingestion_and_database.md) for schema
details, validation rules, SQL queries, rollback behavior, and how to reload a saved
version.

## Running the tests

```powershell
python -m pytest -q
```

## Notes

- The package keeps synthetic data deterministic and replayable.
- The main dataset contains a realistic mix of normal, edge, and disruption scenarios.
- The separate exact-answer fixture is intentionally small enough to reason about manually.

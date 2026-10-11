# CSV ingestion and versioned SQLite storage

## Architecture and source schema

`services/ingestion.py` reads the four CSVs explicitly, normalizes header whitespace,
validates and canonicalizes the values, and asks `storage/repository.py` to persist a
versioned snapshot. `services/validation.py` owns the shared schema and validation
rules. SQL definitions live in `database/schema.sql`.

The official fields are:

| CSV | Columns |
|---|---|
| `orders.csv` | `order_id, product, qty, destination, due` |
| `plants.csv` | `plant, product, weekly_capacity, unit_cost, energy_kwh_per_unit, carbon_kg_per_kwh` |
| `transport_duties.csv` | `plant, destination, transport_cost_per_unit, transit_days, duty_pct, carbon_kg_per_unit` |
| `capacity_events.csv` | `event_id, plant, from_date, to_date, capacity_loss_pct` |

Extra columns are accepted, reported as warnings, and omitted from the canonical
DataFrames and SQL snapshot. Header whitespace is trimmed; names are otherwise
case-sensitive and not renamed. Duplicate headers or names that collide after trimming
are errors. Energy price is not part of this schema and is not invented.

## SQL tables and keys

- `dataset_versions`: UUID-based `dataset_version_id`, UTC timestamp, status, source,
  row counts, validation status, summary, and JSON metadata.
- `orders`: primary key `(dataset_version_id, order_id)`.
- `plants`: primary key `(dataset_version_id, plant, product)`.
- `transport_duties`: primary key `(dataset_version_id, plant, destination)`.
- `capacity_events`: primary key `(dataset_version_id, event_id)`.
- `ingestion_logs`: structured error, warning, and info records.

Each data table has a foreign key to `dataset_versions`. Version-scoped natural keys
are also database-enforced. Cross-table business references are checked by validation:
the source schema has no standalone plant or destination master table, and products
are intentionally not foreign-key constrained so a product with no eligible plant can
be reported as a valid-but-infeasible order rather than malformed input.
SQL tables also store an internal `row_order` value (not a CSV/source field) so a
round-trip reload preserves the source DataFrame's row order.

## Validation semantics

Errors reject ingestion; warnings do not:

- Errors include missing required fields, conversion failures, null or blank required
  values, duplicate natural keys, invalid numeric ranges, unknown referenced plants,
  missing plant/destination route combinations, and reversed capacity-event dates.
- Warnings include unsupported extra columns, overlapping capacity events, and an
  order product with no eligible plant. Such an order is reported as business-
  infeasible; ingestion does not invent eligibility.
- Duty and capacity-loss percentages are stored as fractions from `0` to `1`, matching
  the generated data (for example `0.30` means 30%). Values are not silently rescaled.
- All dates must parse as ISO `YYYY-MM-DD` values and are stored in that format.
  `planning_start` is
  optional because the four-file schema has no planning-start field; if provided,
  earlier order due dates are validation errors.
- Capacity-event overlaps are reported, not combined. Effective capacity and event
  combination are solver/business rules outside this ingestion phase.
- A parsed due date that cannot be reached by any plant is not a schema error; delivery
  feasibility is left to the solver.

Reports contain `valid`, `errors`, `warnings`, and a summary. Each issue includes
severity, table, row or identifier where available, rule name, and explanation.

## Versioning and transaction behavior

Each ingestion creates a new UUID-based snapshot; accepted versions are immutable
through this API. Ingesting again never updates or replaces earlier records. The
default database is persistent at `data/carbonwise.sqlite3`; set
`CARBONWISE_DB_PATH` or pass `--db-path` to choose another location.

The version row starts as `processing`. All four table writes, their final row counts,
the accepted status, and info/warning logs are committed in one transaction. If the
write fails, SQLite rolls back every input row. A separate transaction then marks the
version `rejected` and persists failure diagnostics, so failures remain inspectable.
Validation failures create a rejected version and log issues without writing input
rows. The failure record is deliberately separate from the rolled-back data
transaction.

`ingest_dataset(table_name, df, db_path=None)` validates and stores one table as a
`partial` version. It is useful for administrative/test ingestion but is not returned
by `load_dataset_from_db` and must not be passed to the solver as a complete dataset.

## Running ingestion

From the `carbonwise-data` directory:

```powershell
python -m pip install -r requirements.txt
python scripts/ingest_dataset.py
```

Specify another CSV directory or database path:

```powershell
python scripts/ingest_dataset.py --data-dir "synthetic_data/main" --db-path "data/demo.sqlite3"
```

Optionally enforce a planning start:

```powershell
python scripts/ingest_dataset.py --planning-start 2026-01-01
```

In Python:

```python
from services.ingestion import ingest_all

result = ingest_all("synthetic_data/main", db_path="data/carbonwise.sqlite3")
tables = result["tables"]
dataset_version_id = result["dataset_version_id"]
validation_report = result["validation_report"]
row_counts = result["row_counts"]
```

On rejected input, `ingest_all` raises `DatasetIngestionError`; its
`dataset_version_id` references the persisted rejected version and its `report`
contains actual diagnostics. Successful DataFrames use the official field names;
strings are trimmed, integer/float fields have pandas numeric dtypes, and date fields
are `datetime64[ns]`.

To reload an accepted snapshot:

```python
from storage.repository import load_dataset_from_db

tables = load_dataset_from_db(dataset_version_id, "data/carbonwise.sqlite3")
```

The calculation engine should consume the `tables` dictionary above and retain the
version identifier alongside each run. It should not re-read or mutate the source CSVs.
This phase does not calculate landed cost/carbon or solve allocations.

## Inspecting SQLite

The database is a persistent file, not an in-memory database. Inspect with Python's
standard-library `sqlite3` module or a SQLite browser. Example queries (replace the
placeholder with a real version identifier):

```sql
-- Versions and outcomes
SELECT dataset_version_id, created_at, status, validation_status,
       orders_row_count, plants_row_count, transport_duties_row_count,
       capacity_events_row_count
FROM dataset_versions
ORDER BY created_at DESC;

-- Counts by input table for one version
SELECT 'orders' AS table_name, COUNT(*) AS row_count
FROM orders WHERE dataset_version_id = :version_id
UNION ALL
SELECT 'plants', COUNT(*) FROM plants WHERE dataset_version_id = :version_id
UNION ALL
SELECT 'transport_duties', COUNT(*) FROM transport_duties
WHERE dataset_version_id = :version_id
UNION ALL
SELECT 'capacity_events', COUNT(*) FROM capacity_events
WHERE dataset_version_id = :version_id;

-- Orders in a selected version
SELECT order_id, product, qty, destination, due
FROM orders WHERE dataset_version_id = :version_id
ORDER BY due, order_id;

-- Orders whose product has no eligible plant (these are business-infeasible)
SELECT o.order_id, o.product
FROM orders AS o
WHERE o.dataset_version_id = :version_id
  AND NOT EXISTS (
      SELECT 1 FROM plants AS p
      WHERE p.dataset_version_id = o.dataset_version_id
        AND p.product = o.product
  );

-- Missing plant/destination routes for destinations in this version's orders
SELECT DISTINCT p.plant, d.destination
FROM plants AS p
CROSS JOIN (
    SELECT DISTINCT destination FROM orders
    WHERE dataset_version_id = :version_id
) AS d
WHERE p.dataset_version_id = :version_id
  AND NOT EXISTS (
      SELECT 1 FROM transport_duties AS t
      WHERE t.dataset_version_id = p.dataset_version_id
        AND t.plant = p.plant AND t.destination = d.destination
  );

-- Warnings and errors for one version
SELECT severity, table_name, row_id, rule_name, message
FROM ingestion_logs
WHERE dataset_version_id = :version_id
  AND severity IN ('warning', 'error')
ORDER BY ingestion_log_id;
```

Runtime SQLite files are ignored by Git and should not be committed.

# Detailed working record

## 2026-10-11

### Objective
Build the synthetic data generation package for the CarbonWise Allocator challenge using the official Annex B schema and requirement list.

### Actions completed
1. Reviewed the attached participant guide and extracted the core criteria for the challenge.
2. Confirmed the canonical CSV schema for the four main files and the required validation expectations.
3. Created the project structure for a reproducible synthetic-data package.
4. Implemented a deterministic generator that creates the main datasets and scenario fixtures.
5. Added validation checks for schema correctness, referential integrity, and scenario coverage.
6. Created a small exact-answer fixture with expected assignments and a verification test.
7. Generated reports and documentation describing generation logic, assumptions, and limitations.
8. Ran the project tests to verify the package works end-to-end.

### Files created
- `carbonwise-data/README.md`
- `carbonwise-data/requirements.txt`
- `carbonwise-data/scripts/generate_synthetic_data.py`
- `carbonwise-data/synthetic_data/...` generated datasets and reports
- `carbonwise-data/tests/...` automated validation tests

### Important design decisions
- Used a fixed seed of 42 to guarantee repeatability.
- Kept the output schemas identical to the official annex specification.
- Kept the data generation deterministic while embedding required business scenarios.
- Split validation from generation to make the package easier to debug and extend.
- Used a small exact-answer fixture independent from the larger main dataset so expected results are traceable and manually checkable.

### Remaining assumptions or limitations
- The annex data definitions are exact about the schema but intentionally leave some operational formulas open-ended, so the generation package documents the assumption-driven conventions used.
- Energy price is not present in the official schema and is therefore not included in cost calculations.
- The package is designed for challenge demo and validation, not for production-grade physical planning.

## 2026-10-11 — Phase 1 ingestion and database

### Objective
Implement robust four-file CSV ingestion, schema and cross-table validation, persistent
SQLite storage, immutable dataset versions, and ingestion diagnostics for the CarbonWise
calculation-engine teammate.

### Repository and data checks
1. Confirmed the package is under the existing Git root and working on the
   `data-creation` branch.
2. Left the unrelated untracked participant-guide DOCX at the repository root untouched.
3. Read the current generator, README, data card, and prior work record.
4. Verified actual main CSV headers and counts: 125 orders, 13 plant/product rows,
   30 plant/destination routes, and 3 capacity events.
5. Did not edit or regenerate any source CSV or modify the synthetic-data generator.

### Implementation completed
1. Added explicit CSV loading with missing/unreadable/duplicate-header errors and
   deterministic mapping from filenames to input tables.
2. Added whitespace-only header normalization, collision detection, schema/type checks,
   canonical pandas data types, and warnings for accepted-but-omitted extra columns.
3. Added row and cross-table checks for natural keys, numeric limits, plant references,
   route coverage, capacity-event intervals, optional planning start, and business
   infeasibility warnings.
4. Added `database/schema.sql` and repository functions for initialization, versioned
   writes, reloads, logs, and version listing.
5. Made complete dataset writes transactional; database-write failures roll back all
   input rows and persist the rejected version and diagnostics separately.
6. Added UUID-based version identifiers, version-scoped keys, SQL constraints, and an
   internal row-order field so a SQL reload preserves the input DataFrame's order.
7. Added one-table `ingest_dataset` behavior as a `partial` snapshot that cannot be
   loaded as a complete solver-ready dataset.
8. Added the ingestion CLI and detailed database/validation documentation; updated the
   README and this work log.
9. Added `.gitignore` rules for runtime SQLite databases and Python/test caches.

### Downstream interface
Call `ingest_all(data_dir, db_path=None, planning_start=None)`. On success it returns
`tables`, `dataset_version_id`, `row_counts`, and `validation_report`. The version ID
refers to a complete persisted snapshot. Validation rejection raises
`DatasetIngestionError`, whose `dataset_version_id` and `report` identify a persisted
rejected attempt.

### Assumptions and boundaries
- `duty_pct` and `capacity_loss_pct` are fractional values from 0 through 1; no
  automatic percent rescaling is performed.
- Dates must be ISO `YYYY-MM-DD`. A planning start is optional because the CSV schema
  has no planning-start field.
- Missing product eligibility is reported as a business-infeasible order warning.
- Overlapping capacity events are reported but not combined by ingestion.
- Cross-table references are application-validated; each row table is constrained to a
  real dataset version in SQLite. The source schema does not provide plant or
  destination dimension tables.
- Energy price is absent and is not invented. Cost/carbon calculations and allocation
  remain outside this phase.

### Verification
- Ran the full pre-existing and new pytest suite: **33 passed**.
- Ran the documented CLI on the unchanged main CSV directory: ingestion was accepted,
  with 125/13/30/3 rows and zero validation errors or warnings.
- Tested SQL round-trip row order, immutable versions, rejection logs, and atomic
  rollback after an injected database write failure.
- The runtime database is local and ignored by Git; it must not be committed.

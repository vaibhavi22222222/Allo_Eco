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

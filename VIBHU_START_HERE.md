# Vibhu — optimization implementation ready for integration

This ZIP contains your teammate's original data-creation branch plus Vibhu's completed optimization module. The input CSVs, data loader, validation, SQL schema, generator and original tests have not been edited.

Start in the carbonwise-data directory. This contribution is the optimizer and its integration tools; the other teammates still own the full API, LangGraph/LLM and frontend.

## Run on Windows

Open a terminal in carbonwise-data:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-optimizer.txt
.\.venv\Scripts\python.exe scripts/run_optimizer.py
```

The program ingests the existing synthetic files through the original loader, solves the model, independently checks it, stores a version-linked run in SQLite, and writes data/optimization_result.json. No API key is required.

If you already activated a virtual environment, ordinary python commands are sufficient:

```bash
python -m pip install -r requirements-optimizer.txt
python scripts/run_optimizer.py
python -m pytest -q --junitxml=data/optimizer_tests.xml
```

## What is implemented

- PuLP/CBC integer allocation with one plant and dispatch date per order.
- Daily production quantities, shared capacity and product eligibility.
- Production, electricity, transport and duty cost.
- Production and transport carbon using dated factor overrides.
- Normalized user-adjustable cost/carbon objective.
- Dated capacity-event replanning and baseline comparisons.
- Cost-focused/balanced/carbon-focused scenarios.
- Explicit carbon-budget constraints for requests such as a 10% reduction.
- Honest infeasible results, closest alternatives and time-limit reporting.
- Independent plan checker, greedy baseline and order-choice evidence.
- Existing SQLite version integration and additive optimization-run persistence.
- Standalone CLI, tests, measured evaluation and three sample traces.
- Optimizer Docker smoke configuration; Docker itself was unavailable here.

## Important integration decisions

- The teammate's CSVs/SQLite use FRACTIONS: 0.30 = 30%; 0.05 duty = 5%. This replaces the earlier provisional 0–100 convention. Do not divide these input values by 100 again.
- Existing product rows disagree about shared plant capacity. config/optimizer.yaml explicitly uses A=850, B=760 and C=640 units/week. The source rows remain unchanged. These conservative shared values need team/organizer agreement.
- Electricity prices in that config (A=8, B=6, C=7 INR/kWh) are declared synthetic assumptions because the CSV schema does not contain prices.
- The supplied pack actually has 125 orders and 10,677 units. The default horizon covers its dated orders, January 1 through April 23, 2026.
- The original exact-answer generator fixture is not used as optimizer ground truth. New independent reference cases verify the solver.
- The 30% B-loss example may leave order-level cost/plant assignments unchanged when spare capacity is sufficient. Production-day changes are reported; the program does not manufacture changes for the demo.

See carbonwise-data/docs/optimization.md for formulas, all functions, scenario commands, caveats and API integration details.

## Backend teammate: simplest call

From carbonwise-data:

```python
from services.ingestion import ingest_all
from services.optimization import load_planning_config, optimize_tables

loaded = ingest_all("synthetic_data/main", "data/carbonwise.sqlite3")
config = load_planning_config("config/optimizer.yaml")
result = optimize_tables(
    loaded["tables"],
    config,
    dataset_version_id=loaded["dataset_version_id"],
    weights={"cost": 0.7, "carbon": 0.3, "lateness": 0},
)
response = result.model_dump(mode="json")
```

For an accepted dataset already in SQLite, use services.optimization.optimize_dataset_version. It returns a JSON-ready dictionary with a persisted run_id.

The backend adds the natural-language summary/LLM reporting, maps domain errors, wraps the computation in LangGraph tools, and exposes the required HTTP routes. Preserve solver.status separately from answer_type.

## Test and evaluation evidence

The implementation run passed 74 tests: 33 original data tests plus 41 optimizer tests. See carbonwise-data/evaluation/test_cases.json and optimizer_report.json for actual recorded results.

The full supplied dataset produced a checked on-time allocation. Tests also cover failed delivery promises, insufficient capacity, shutdowns, carbon constraints and time-limit behavior.

Docker configuration is included but not runtime-tested here. Python/CBC execution and SQLite integration were actually tested.

## Add the changes to your GitHub branch

1. In your existing Git clone of Allo_Eco, switch to vibhu-optimization.
2. Copy the CONTENTS of this ZIP's Allo_Eco-data-creation folder into that clone, merging directories. Keep the clone's existing .git directory.
3. Review status. This package adds optimizer-related files; original teammate files should show no content changes.
4. Run tests from carbonwise-data.
5. Commit and push from the clone.

```bash
git switch vibhu-optimization
git status
git add carbonwise-data/optimization carbonwise-data/config carbonwise-data/docs/optimization.md carbonwise-data/requirements-optimizer.txt
git add carbonwise-data/services/optimization.py carbonwise-data/storage/optimizer_repository.py
git add carbonwise-data/scripts/run_optimizer.py carbonwise-data/scripts/evaluate_optimizer.py
git add carbonwise-data/tests/__init__.py carbonwise-data/tests/optimization carbonwise-data/evaluation
git add carbonwise-data/Dockerfile.optimizer carbonwise-data/compose.optimizer.yaml carbonwise-data/.dockerignore
git add VIBHU_START_HERE.md VIBHU_FILE_MANIFEST.json
git diff --cached --stat
git commit -m "Implement Vibhu optimization solver with scenarios and validation"
git push -u origin vibhu-optimization
```

If you have not created your local branch yet, do this before copying:

```bash
git fetch origin
git switch --no-track -c vibhu-optimization origin/data-creation
```

No GitHub commit or push was performed on your behalf. The ZIP is ready for your review and integration.

## AI-use and contribution note

Vibhu's optimizer was implemented with ChatGPT/Codex assistance and checked with the included tests and actual solver executions. Review the model and assumptions before the individual judging discussion. The original data module remains your teammate's contribution.


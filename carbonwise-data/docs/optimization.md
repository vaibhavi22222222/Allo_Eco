# Vibhu's optimization module

This implementation extends the data-creation branch. The original schemas, CSVs, generator, validation, ingestion and SQLite input-version code are retained. New code lives in optimization/, with an additive service facade and optional SQLite run storage.

## Start

From carbonwise-data:

```bash
python -m pip install -r requirements-optimizer.txt
python scripts/run_optimizer.py
python -m pytest -q --junitxml=data/optimizer_tests.xml
python scripts/evaluate_optimizer.py --repeat 3
```

No LLM key is needed. The default CLI ingests all four original CSV files using services.ingestion.ingest_all, obtains a real accepted dataset version, solves, checks and saves the result. Generated SQLite and runtime output files go under data/, already ignored by Git.

Default output: data/optimization_result.json. It contains every order assignment, daily production, KPIs, independent validation and solver traces. An understood infeasible request has exit code 0 and answer_type=infeasible; invalid configuration exits 1; solver error/no incumbent exits 2.

## Critical reconciliations with the teammate branch

1. **Percentages are fractions.** Existing CSV validation/SQLite require duty_pct and capacity_loss_pct in [0,1]. Use 0.30 for 30%, not 30. This integrated implementation supersedes the provisional 0–100 convention in the earlier design text. A hypothetical scenario sent from a 0–100 UI slider must be divided by 100 exactly once in the API adapter.
2. **Shared capacities require explicit interpretation.** Plant A rows say 850/865, B 760/775 and C 640/655. config/optimizer.yaml explicitly sets shared budgets A=850, B=760, C=640, using the conservative lower figures. No source row is rewritten; no capacities are summed. Without an override, inconsistent rows are rejected. These are demo assumptions to confirm with the team/organizer.
3. **Energy prices are missing from the fixed CSV schema.** The supplied config explicitly uses synthetic INR/kWh prices A=8, B=6, C=7. Missing prices cause an error. The solver does not treat them as organizer data.
4. **The actual input pack is 125 orders, not 150.** It has 13 capability rows, 30 routes and three capacity events. The default horizon is 1 January–23 April 2026, covering the actual latest due date; the old data-card quarter ends earlier.
5. **Existing exact-answer expectations are generator fixtures, not solver ground truth.** Their generator returns prewritten assignment dictionaries and omits the integrated energy/date model; its unused landed-cost helper also uses a different duty formula. They remain untouched for source preservation. The runtime optimizer never imports the generator or expected_results. New solver tests use independent small numerical references.

## Module map

| File | Responsibility |
|---|---|
| optimization/contracts.py | Pydantic config, events, weights, policy, result |
| optimization/preprocessing.py | Canonical-data adapter, dates, capacities, factors, hashes |
| optimization/costs.py | Numerical landed cost and carbon |
| optimization/model.py | Linear integer variables, constraints and expressions |
| optimization/solver_adapter.py | CBC invocation, incumbent checks, honest statuses |
| optimization/solve.py | Strict solve, diagnostic priorities, stability tie-break |
| optimization/checker.py | Independent source-row arithmetic and constraint checks |
| optimization/scenarios.py | Replanning, comparison, carbon targets |
| optimization/explanations.py | Deterministic order-choice/counterfactual evidence |
| optimization/baseline.py | Due-date-ordered cheapest-feasible heuristic |
| optimization/results.py | JSON-compatible reporting and validation handoff |
| services/optimization.py | Entry points for the backend owner |
| storage/optimizer_repository.py | Additive optimization_runs table |
| scripts/run_optimizer.py | Standalone command-line interface |
| scripts/evaluate_optimizer.py | Measured report and sample traces |

## Model

Binary x[i,p,t] selects one plant and dispatch day for order i.
Integer z[i,p,d] is the quantity manufactured on production day d.

For each order, sum x = 1. For each order/plant pair, sum z = quantity * sum x.
z[i,p,d] <= quantity * sum(x[i,p,t] for t >= d), so production cannot follow dispatch.
For every plant/day, total z <= available shared daily capacity.
Eligible variables are created only for supported products and routes.
Strict mode excludes dispatch dates that would arrive after the due date.

Production may span days at the one selected plant. All orders are available at planning start. Dispatch is at end of day; transit is calendar days. Early inventory carries no cost. No machine setups/routing or work-in-progress are supplied, so this is a daily quantity-capacity approximation.

Capacity is shared across products and spread over configured working days. The largest active event loss applies to each affected date. Integer quotas are floored, which is conservative if weekly capacity is not divisible by working-day count. The checker separately reconstructs capacities and weekly utilization.

Cost = production + energy + transport + duty.
Energy and production emissions are attributed to actual production dates.
Duty = fraction * (production + energy), under the explicit config policy.
Carbon = production energy * date-specific factor + transport emissions.

Objective = normalized cost weight * cost / fixed cost scale
          + normalized carbon weight * kg carbon / fixed carbon scale.

Scales are disclosed sums of each order's minimum candidate coefficient estimates, with positive floors for zero totals. They are not jointly feasible reference plans. All weight scenarios use identical scales. Explicit config scales can be supplied. Default weights: 0.7 cost and 0.3 carbon. Lateness-only strict objectives are rejected.

## Infeasibility and status

Strict delivery is tried first. Direct necessary conditions may prove it impossible before CBC. Otherwise CBC distinguishes:
- optimal: verified integer incumbent and proven optimality;
- feasible: verified incumbent, proof unavailable;
- infeasible: proof of infeasibility;
- no_solution: no verified incumbent before the limit, not proof;
- error: missing solver, failed verification or execution error.

PuLP can expose status=Optimal for a time-limited incumbent. The adapter also checks sol_status, termination logs, integrality and constraint residuals; it never trusts the coarse status alone.

If strict delivery is proven impossible, the optional diagnostic model allows unassigned/late orders while preserving capacity and eligibility:
1. Minimize missed commitments (unassigned + late orders).
2. Fix that optimum and minimize unassigned orders.
3. Fix that optimum and minimize weighted cost/carbon/lateness.
4. Optional stability tie-break against a baseline.

A timed-out stage cannot be called a proven lexicographic optimum. The result retains strict_solver_status=infeasible and answer_type=infeasible even if the alternative itself is optimal. Delivery flags remain honest. The finite horizon limits every feasibility claim.

## Public Python interface

Use the existing ingestion result directly:

```python
from services.ingestion import ingest_all
from services.optimization import load_planning_config
from optimization import prepare_problem, solve_allocation

loaded = ingest_all("synthetic_data/main", "data/carbonwise.sqlite3")
config = load_planning_config("config/optimizer.yaml")
problem = prepare_problem(
    loaded["tables"], config,
    dataset_version_id=loaded["dataset_version_id"],
)
result = solve_allocation(
    problem,
    weights={"cost": 0.7, "carbon": 0.3, "lateness": 0},
    policy={"time_limit_seconds": 15, "total_time_limit_seconds": 60},
)
payload = result.model_dump(mode="json")
```

Or solve an accepted version already stored by the data teammate:

```python
from services.optimization import optimize_dataset_version
payload = optimize_dataset_version(
    dataset_version_id,
    config,
    db_path="data/carbonwise.sqlite3",
    weights={"cost": 0.7, "carbon": 0.3},
)
```

This facade reloads through storage.repository.load_dataset_from_db and adds a run_id when persisted. Partial/rejected versions cannot be used.

Pydantic models are optimizer-specific boundary models; import or adapt them into the backend's shared contract. No unrelated HTTP routes or LLM logic have been introduced.

## Scenarios

```python
from optimization import (
    compare_weight_settings, replan_capacity, solve_carbon_target,
    solve_greedy_baseline, compare_plans, explain_order,
)

tradeoffs = compare_weight_settings(problem)
scenario = replan_capacity(problem, result, [{
    "event_id": "WHATIF-B-30",
    "plant": "B",
    "from_date": "2026-01-12",
    "to_date": "2026-01-25",
    "capacity_loss_pct": 0.30,
}])
comparison = scenario["comparison"]
greener = solve_carbon_target(problem, result, carbon_cut_pct=10)
evidence = explain_order(problem, result, "ORD-0001", counterfactual=True)
baseline = solve_greedy_baseline(problem)
comparison_with_heuristic = compare_plans(baseline, result)
```

The carbon-target baseline must be fully on time, verified and from the same problem. A 10% cut adds carbon <= 0.9 * baseline_carbon and minimizes cost; weight changes alone are not sufficient. No order may be dropped to make a strict carbon target appear successful.

Event overlays do not mutate original tables, baseline objects or SQLite snapshots. Duplicate identical IDs are deduplicated; conflicting event IDs fail clearly. Comparisons include changes in plant, dispatch and daily production. A disruption may only move production days, or have no effect when spare capacity suffices; no fake changes are manufactured.

Same-workload comparison uses an allocation-input hash that excludes event tables, allowing comparisons after an event-only ingestion. Other order/capability/route changes require a new clearly specified comparison policy.

To replan after a newly ingested event snapshot, prepare the new accepted dataset, retain the old result, preserve scales, solve and compare:

```python
from dataclasses import replace
new_problem = prepare_problem(new_tables, config, dataset_version_id=new_version)
new_problem = replace(new_problem, scales=dict(problem.scales))
new_result = solve_allocation(new_problem, base_plan=result)
changes = compare_plans(result, new_result)
```

Data/config compatibility is checked before stability is applied. For weekly carbon scenarios create a validated new PlanningConfig with weekly_carbon_overrides, prepare/solve separately, retain reference scales and compare unchanged order data. Override entries use {"plant":"B","week_start":"2026-01-19","value":0.12}; week_start must be Monday. Counterfactual explanations must use the exact problem for their result.

## Commands

```bash
# Base solve with the checked-in data and declared config.
python scripts/run_optimizer.py

# Three comparable settings.
python scripts/run_optimizer.py --mode tradeoffs --output data/tradeoffs.json

# Dated hypothetical loss. JSON uses fractions.
python scripts/run_optimizer.py --mode replan --events-json config/capacity_scenario.example.json --output data/replan.json

# Explicit carbon cap; infeasible is a valid, honest result.
python scripts/run_optimizer.py --mode carbon-target --carbon-cut 10 --output data/carbon_target.json

# Deterministic evidence, including bounded forced-plant re-solves.
python scripts/run_optimizer.py --mode explain --order-id ORD-0001 --output data/explanation.json

# Existing input version; use the actual id from ingestion output.
python scripts/run_optimizer.py --dataset-version ACTUAL_DATASET_VERSION
```

A policy's total budget is shared across solves in a workflow; CBC receives remaining time. Model construction, operating-system startup and result checking add overhead, so it is a best-effort wall-clock budget rather than hard process preemption. Use a background worker if the API owner later requires hard cancellation. Batch trade-off points each carry their actual status.

## Backend/LangGraph handoff

Keep the required routes /analyze, /ingest, /health and /trace/{run_id}. This contribution intentionally supplies computation tools and a facade, not a competing full backend.

- /ingest continues to use the teammate's loader/versioning.
- /analyze parses intent and calls these tools. Request weights keep cost/carbon/lateness field names.
- /health can call optimization.solver_available() as one readiness signal; it is not a provider health check.
- /trace can expose the result trace, hashes, config and checker records stored in optimization_runs.
- Backend adds the grounded summary, resolves optional scenario JSON, maps solver errors to suitable HTTP status and preserves the guide answer_type values.
- The LLM only receives checked tool results and may not alter allocation/production.
- Frontend consumes result.kpis and allocation; it must not independently recompute authoritative totals.
- result.validator.valid refers to operational/arithmetic checks. A diagnostic alternative can have valid=true and delivery_ok=false; original_request_met remains false.
- When solver.status is error/no_solution, do not present an empty zero-cost plan as success.
- Null gap/bound means unavailable, not zero.
- Numerical recommendations require approval and include action/rationale/evidence/confidence/impact.

## Testing and evaluation

```bash
python -m pytest -q --junitxml=data/optimizer_tests.xml
python scripts/evaluate_optimizer.py --repeat 3
```

New tests cover independent small optima, cost/carbon arithmetic, eligibility, due-date boundaries, multi-day production, shared capacity, differing product rows, 30/50/100% losses, partial weeks/overlap, carbon targets, dated carbon, baseline immutability, event deduplication, fulfillment-aware comparisons, invalid inputs, result tampering, row permutations, timeout/status behavior, counterfactuals and SQLite integration.

Original tests are retained. New tests do not read synthetic_data/expected_results.

Evaluation files include measured local timings, plan validity, actual dataset/config hashes, comparison with the feasible heuristic, sampled trade-offs, individual test outcomes and sample allocation/scenario/infeasible traces. Small timing samples are explicitly labelled. No LLM, full API/UI or organizer hidden-score results are claimed.

## Docker

```bash
docker compose -f compose.optimizer.yaml up --build
```

This is an optimizer smoke container with persistent data volume, not the whole team's API/UI compose. Its build checks CBC availability. Docker was not installed in the implementation environment, so the Docker configuration is supplied but not runtime-verified. The Python tests and CBC solves were run locally. The integrator can copy its dependency setup into the full application's image.

## Remaining domain assumptions

Confirm energy prices, shared capacity interpretation, duty base and the planning calendar with organizers. No factual price/carbon forecasts, order priority, machine sequencing or actual completed production is inferred. Changing agreed assumptions should change configuration/contracts with tests, not silent constants scattered across modules.

AI-use disclosure: this optimizer contribution was generated with ChatGPT/Codex assistance and verified through the included tests and actual CBC runs. Vibhu must review and understand it before presenting the individual contribution.


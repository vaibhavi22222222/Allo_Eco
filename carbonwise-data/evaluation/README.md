# Optimizer evaluation — measured results

Scope: Vibhu's deterministic optimization module, not the full application.

- Tests recorded: 74; passed: 74.
- Computed plans checked: 9; validity rate: 100%.
- Timing sample: 3 runs; p50 3.237s; p95 3.260s.
- Timing includes model preparation inside solve, solving, extraction and independent checking; excludes CSV ingestion.
- Small local sample; not a runtime guarantee.

| Plan | Cost | Carbon t | Late | Unassigned | Status |
|---|---:|---:|---:|---:|---|
| Balanced optimizer | 531087.84 | 1.613534 | 0 | 0 | optimal |
| Greedy feasible heuristic | 501550.62 | 2.747825 | 0 | 0 | feasible |

Different fulfilled workloads are not like-for-like savings.
See optimizer_report.json for deltas, trade-offs, actual configuration identifiers and all qualifications.
sample_allocation.json, sample_scenario.json and sample_infeasible.json contain solver traces and verification.
test_cases.json lists individual regression/evaluation checks.
No LLM faithfulness or full API/UI results are claimed. Docker was not run in this environment.

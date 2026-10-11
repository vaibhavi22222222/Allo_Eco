# CarbonWise Allocator — how to run

All commands are run from the **project root** (the folder that contains `app/`,
`ui/` and `carbonwise-data/`). No machine-specific paths are needed.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The CBC solver ships inside PuLP. To check it:

```bash
python -c "import pulp; print(bool(pulp.PULP_CBC_CMD(msg=False).available()))"
```

## 2. Configure

```bash
cp .env.example .env                 # edit if needed; .env is git-ignored
set -a; source .env; set +a          # export the variables into this shell
```

`MOCK_LLM=true` (the default in `.env.example`) needs no API key, no network and
no local model: the parser uses rules and the writer uses fact-checked templates.
To use Ollama instead: `export MOCK_LLM=false LLM_PROVIDER=ollama` and run
`ollama pull qwen2.5:7b-instruct`.

## 3. Start the backend (terminal 1)

```bash
python -m uvicorn app.api.main:app --port 8000
```

On first start the API loads `carbonwise-data/synthetic_data/main` into SQLite
(`carbonwise-data/data/carbonwise.sqlite3`) using the existing data module.

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"request": "Plant B loses 30% capacity for two weeks. Re-plan."}'
curl -s http://localhost:8000/trace/<run_id from the previous answer>
curl -s -X POST http://localhost:8000/ingest \
     -F "file=@carbonwise-data/synthetic_data/main/capacity_events.csv" \
     -F 'meta={"table":"capacity_events"}'
```

Interactive API docs: http://localhost:8000/docs

## 4. Start the dashboard (terminal 2)

```bash
python -m streamlit run ui/app.py
```

Open http://localhost:8501. The UI only talks to the API at `CARBONWISE_API_URL`.

## 5. Tests

```bash
python -m pytest -q app/tests carbonwise-data/tests     # everything
python -m pytest -q app/tests                           # app + LLM module only
(cd carbonwise-data && python -m pytest -q)             # optimizer + data module only
```

## Architecture in one paragraph (for judges)

Streamlit calls FastAPI. FastAPI runs one LangGraph workflow:
`load_data → parse_request → check_request → solve_base → (replan | carbon_target |
weight_tradeoff | explain) → verify_plan → build_facts → write_summary → finalise`.
The data module validates and versions the CSVs in SQLite. Vibhu's PuLP/CBC optimizer
builds every plan and computes every KPI. The independent `validate_plan` checker
re-verifies the plan. Abhijeet's LLM module only parses the request and writes the
explanation from a facts sheet, and every number in that text is checked against the
optimizer output. Rule-breaking requests are refused before any solve, and infeasible
results are reported as infeasible. Each run is stored in the optimizer's
`optimization_runs` table and can be read at `/trace/{run_id}`.

## Units

| Quantity | Unit |
|---|---|
| `capacity_loss_pct` | fraction 0–1 (0.3 = 30%) |
| `carbon_cut_pct` | percentage points 0–100 |
| Costs | INR |
| Plan carbon total (`total_carbon_t`) | tonnes CO2e |
| Per-order carbon (`carbon_kg`) | kg CO2e |

## Docker

```bash
cp .env.example .env            # optional; MOCK_LLM=true needs no keys
docker compose up --build       # first build takes a few minutes
```

- UI: http://localhost:8501 · API: http://localhost:8000 (docs at /docs)
- SQLite is stored in the `carbonwise-db` volume (`docker compose down -v` deletes it).
- Ollama, if used, runs on the host: set `MOCK_LLM=false` in `.env` and run `ollama serve` on the host.
- Tests inside the image: `docker compose run --rm api python -m pytest -q app/tests carbonwise-data/tests`

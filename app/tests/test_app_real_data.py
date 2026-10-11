"""End-to-end check on the team's real synthetic data pack (125 orders).

Slower than the mini tests (one full CBC solve, a few seconds) but proves the
whole chain works on the data the demo uses: data module -> Vibhu's optimizer
-> validator -> facts sheet -> fact-checked text -> API response.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.bootstrap import DEFAULT_PLANNING_CONFIG, DEFAULT_SEED_DATA_DIR
from app.config import AppSettings
from app.graph import CarbonWiseWorkflow
from app.tests.app_helpers import mock_llm


def test_real_dataset_allocation_end_to_end(tmp_path):
    settings = AppSettings(db_path=tmp_path / "real.sqlite3", planning_config_path=DEFAULT_PLANNING_CONFIG,
                           seed_data_dir=DEFAULT_SEED_DATA_DIR, solver_time_limit_s=30,
                           solver_total_limit_s=90)
    client = TestClient(create_app(settings, CarbonWiseWorkflow(settings, mock_llm)))
    body = client.post("/analyze", json={"request": "Allocate this week's orders."}).json()

    assert body["answer_type"] == "allocation"
    assert len(body["allocation"]) == 125            # every order in the pack is allocated
    assert body["validator"]["valid"] and body["validator"]["capacity_ok"]
    assert body["solver"]["status"] in ("optimal", "feasible")
    assert body["fact_check"]["passed"]
    # Units as returned by the engine: INR totals, tonnes for the plan total.
    assert body["kpis"]["currency"] == "INR" and body["kpis"]["total_carbon_t"] < 100

"""FastAPI tests with TestClient (no network, mock LLM, temporary database)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.tests.app_helpers import MINI_DATA, make_workflow, mini_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    settings = mini_settings(tmp_path)
    app = create_app(settings, make_workflow(tmp_path))
    # raise_server_exceptions=False lets us see the real 500 response.
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    body = client.get("/health").json()
    assert body["api"] == "ok" and body["solver"] == "ok"
    assert body["llm"]["mock_llm"] is True
    assert "key" not in str(body).lower()             # no secrets of any kind


def test_analyze_success_preserves_engine_results(client):
    r = client.post("/analyze", json={"request": "Allocate this week's orders."})
    assert r.status_code == 200
    body = r.json()
    assert body["answer_type"] == "allocation"
    for field in ("run_id", "summary", "allocation", "kpis", "weights_used", "validator",
                  "assumptions", "limitations", "recommendations"):
        assert field in body
    assert sum(a["cost"] for a in body["allocation"]) == pytest.approx(body["kpis"]["total_cost"])


def test_analyze_with_explicit_weights(client):
    body = client.post("/analyze", json={"request": "Allocate this week's orders.",
                                         "weights": {"cost": 1.0, "carbon": 0.0}}).json()
    assert body["weights_used"]["cost"] == pytest.approx(1.0)


@pytest.mark.parametrize("payload", [
    {"request": ""},
    {"text": "missing request field"},
    {"request": "ok", "unexpected": 1},
    {"request": "ok", "weights": {"cost": -1}},
])
def test_analyze_invalid_body_is_422(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422


def test_analyze_invalid_scenario_fraction_is_422(client):
    r = client.post("/analyze", json={"request": "Re-plan",
                                      "scenario": {"plant": "B", "capacity_loss_pct": 30}})
    assert r.status_code == 422 and "fraction" in r.json()["detail"]


def test_ingest_accepted(client):
    csv = (MINI_DATA / "capacity_events.csv").read_bytes()
    r = client.post("/ingest", files={"file": ("capacity_events.csv", csv, "text/csv")},
                    data={"table": "capacity_events"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"


def test_ingest_with_meta_json(client):
    csv = (MINI_DATA / "orders.csv").read_bytes()
    r = client.post("/ingest", files={"file": ("orders.csv", csv, "text/csv")},
                    data={"meta": '{"table": "orders"}'})
    assert r.status_code == 200 and r.json()["rows_loaded"] == 6


def test_ingest_invalid_rows_rejected(client):
    bad = b"event_id,plant,from_date,to_date,capacity_loss_pct\nX1,Z,2026-01-05,2026-01-06,30\n"
    r = client.post("/ingest", files={"file": ("e.csv", bad, "text/csv")}, data={"table": "capacity_events"})
    body = r.json()
    assert r.status_code == 422 and body["status"] == "rejected" and body["rows_rejected"] == 1
    assert {e["rule"] for e in body["errors"]} >= {"plant_reference"}


@pytest.mark.parametrize("data", [{"table": "not_a_table"}, {}, {"meta": "{bad json"}])
def test_ingest_invalid_request_is_422(client, data):
    csv = (MINI_DATA / "orders.csv").read_bytes()
    assert client.post("/ingest", files={"file": ("o.csv", csv, "text/csv")}, data=data).status_code == 422


def test_trace_roundtrip_and_unknown(client):
    run_id = client.post("/analyze", json={"request": "Allocate this week's orders."}).json()["run_id"]
    trace = client.get(f"/trace/{run_id}").json()
    assert trace["run_id"] == run_id and any(s["tool"] == "validate_plan" for s in trace["steps"])
    assert client.get("/trace/al-doesnotexist").status_code == 404


def test_internal_failure_is_500_without_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("database password=hunter2")
    monkeypatch.setattr(client.app.state.workflow, "run", boom)
    r = client.post("/analyze", json={"request": "Allocate this week's orders."})
    assert r.status_code == 500 and "hunter2" not in r.text

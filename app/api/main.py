"""FastAPI backend: the four endpoints required by the guide (Section 6).

    GET  /health            readiness of API, data store, solver and LLM (no secrets)
    POST /analyze           run the LangGraph workflow on a request
    POST /ingest            upload one CSV table; re-validated with the data module
    GET  /trace/{run_id}    stored audit record for one /analyze run

How the backend invokes the workflow
------------------------------------
``create_app`` builds ONE ``CarbonWiseWorkflow`` (the compiled LangGraph
graph) and stores it on ``app.state``. ``/analyze`` validates the body with
Pydantic, calls ``workflow.run(...)`` and turns the workflow's error code (if
any) into an HTTP status. The handlers contain no business logic.

Status codes
------------
200  answer produced (including infeasible / refused / clarification_needed,
     and answer_type "error" when the optimizer or validator rejects a plan)
409  no accepted dataset is loaded
422  invalid input (Pydantic body errors, invalid scenario values, bad CSV)
404  unknown run_id
500  unexpected internal error (message only; stack trace stays in the log)

Run:  uvicorn app.api.main:app --reload   (from the project root)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.bootstrap import ENGINE_DIR  # noqa: F401
from app.config import AppSettings
from app.contracts import AnalyzeRequest, AnalyzeResponse, HealthResponse, IngestResponse
from app.graph import CarbonWiseWorkflow
from app.llm import get_client
from app.services.datasets import ensure_seeded, ingest_table, latest_accepted_version
from optimization import solver_available
from storage.optimizer_repository import load_optimization_run

logger = logging.getLogger(__name__)

_STATUS_FOR_ERROR = {"invalid_request": 422, "no_dataset": 409, "internal": 500}
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _llm_status() -> dict[str, Any]:
    """Describe the LLM without secrets; ping Ollama briefly if it is the provider."""
    client = get_client()
    info: dict[str, Any] = client.describe()
    if client.is_mock:
        info["status"] = "mock"
    elif client.settings.provider == "ollama":
        try:
            with urllib.request.urlopen(f"{client.settings.ollama_base_url}/api/tags", timeout=2):
                info["status"] = "ok"
        except OSError:
            # The app still answers with deterministic templates.
            info["status"] = "unreachable (template fallback active)"
    else:
        info["status"] = "configured"
    return info


def create_app(
    settings: Optional[AppSettings] = None,
    workflow: Optional[CarbonWiseWorkflow] = None,
) -> FastAPI:
    """Application factory (tests pass their own settings / workflow)."""
    settings = settings or AppSettings.from_env()
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Load the seed data pack once at startup so /health shows a dataset
        # immediately. Failure is logged, not fatal: /ingest can still load data.
        try:
            ensure_seeded(settings.db_path, settings.seed_data_dir)
        except Exception:  # noqa: BLE001 - logged with stack trace
            logger.exception("Could not seed the database at startup")
        yield

    app = FastAPI(title="CarbonWise Allocator API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.workflow = workflow or CarbonWiseWorkflow(settings)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            version = latest_accepted_version(settings.db_path)
            store = "ok" if version else "empty"
        except Exception:  # noqa: BLE001 - reported as a status, and logged
            logger.exception("Health check could not read the data store")
            version, store = None, "error"
        return HealthResponse(
            data_store=store, dataset_version_id=version,
            solver="ok" if solver_available() else "unavailable",
            llm=_llm_status(),
        )

    @app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
    def analyze(body: AnalyzeRequest) -> Any:
        weights = body.weights.model_dump(exclude_none=True) if body.weights else None
        out = app.state.workflow.run(body.request, weights, body.scenario)
        error = out["error"]
        status = _STATUS_FOR_ERROR.get(error["code"]) if error else None
        if status:
            return JSONResponse(status_code=status, content={
                "detail": error["message"], "code": error["code"],
                "run_id": out["response"]["run_id"]})
        return out["response"]

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        file: UploadFile = File(..., description="CSV file in the Annex B schema"),
        table: Optional[str] = Form(None, description="orders | plants | transport_duties | capacity_events"),
        meta: Optional[str] = Form(None, description='Alternative: JSON such as {"table": "orders"}'),
    ) -> Any:
        table_name = table or _table_from_meta(meta)
        data = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=422, detail="File larger than 5 MB")
        try:
            result = ingest_table(table_name, data, settings.db_path, settings.seed_data_dir)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        body = IngestResponse(**result)
        # A rejected upload is an input problem: 422 with the full validation report.
        if body.status == "rejected":
            return JSONResponse(status_code=422, content=body.model_dump(mode="json"))
        return body

    @app.get("/trace/{run_id}")
    def trace(run_id: str) -> dict[str, Any]:
        if not _RUN_ID.match(run_id):
            raise HTTPException(status_code=422, detail="Invalid run_id format")
        try:
            payload = load_optimization_run(run_id, settings.db_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
        return payload.get("workflow_trace") or {"run_id": run_id, "steps": [],
                                                 "note": "Stored by the optimizer CLI; no workflow trace."}

    return app


def _table_from_meta(meta: Optional[str]) -> str:
    """Read the table name from the guide's JSON part, e.g. {"table": "orders"}."""
    if not meta:
        raise HTTPException(status_code=422, detail="Provide the table name as 'table' or 'meta'")
    try:
        value = json.loads(meta).get("table")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="'meta' must be JSON like {\"table\": \"orders\"}") from exc
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail="'meta.table' must be a string")
    return value


# Module-level app for `uvicorn app.api.main:app`.
app = create_app()

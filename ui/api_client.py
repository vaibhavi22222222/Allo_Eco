"""Tiny HTTP client the Streamlit UI uses to talk to FastAPI.

How the frontend calls the API
------------------------------
The UI never imports the optimizer or the LLM module. Every action is an HTTP
call to the backend (``CARBONWISE_API_URL``, default http://localhost:8000).
Each function returns ``ApiResult(ok, status, data, error)`` so the page can
show a clear message instead of crashing when the backend is down or returns
an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

API_URL = os.getenv("CARBONWISE_API_URL", "http://localhost:8000").rstrip("/")


@dataclass
class ApiResult:
    ok: bool
    status: Optional[int]
    data: Any = None
    error: Optional[str] = None


def _call(method: str, path: str, timeout: float, **kwargs: Any) -> ApiResult:
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return ApiResult(False, None, error=f"Cannot reach the backend at {API_URL}: {exc.__class__.__name__}")
    try:
        data = resp.json()
    except ValueError:
        data = None
    if resp.ok:
        return ApiResult(True, resp.status_code, data)
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, list):  # Pydantic 422 errors
        detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
    return ApiResult(False, resp.status_code, data, error=detail or f"HTTP {resp.status_code}")


def health() -> ApiResult:
    return _call("GET", "/health", timeout=5)


def analyze(request: str, weights: Optional[dict] = None, scenario: Optional[dict] = None) -> ApiResult:
    body: dict[str, Any] = {"request": request}
    if weights:
        body["weights"] = weights
    if scenario:
        body["scenario"] = scenario
    # Optimisation can take tens of seconds (several CBC solves).
    return _call("POST", "/analyze", timeout=300, json=body)


def ingest(table: str, filename: str, content: bytes) -> ApiResult:
    return _call("POST", "/ingest", timeout=120,
                 files={"file": (filename, content, "text/csv")}, data={"table": table})


def trace(run_id: str) -> ApiResult:
    return _call("GET", f"/trace/{run_id}", timeout=10)

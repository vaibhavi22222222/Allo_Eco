"""Shared test helpers: scripted fake LLM clients and synthetic facts sheets.

No test calls a real LLM. ``ScriptedClient`` replays prepared answers (or
raises prepared errors) so we can test retries, timeouts and fallbacks.
All numbers below are SYNTHETIC fixtures, not hackathon results.
"""

from __future__ import annotations

from typing import Union

import pytest

from app.llm.client import LLMClient, LLMSettings
from app.llm.models import LLMResponse


class ScriptedClient(LLMClient):
    """LLM client that returns scripted responses in order."""

    def __init__(self, script: list[Union[str, Exception]]):
        super().__init__(settings=LLMSettings(provider="ollama", mock=False))
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        self.calls += 1
        self.prompts.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.prompt_tokens += 10
        self.completion_tokens += 5
        return LLMResponse(text=item, prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def mock_client() -> LLMClient:
    return LLMClient(settings=LLMSettings(mock=True))


@pytest.fixture
def scripted():
    return ScriptedClient


@pytest.fixture
def allocation_facts() -> dict:
    return {
        "answer_type": "allocation",
        "intent": "allocate",
        "kpis": {"total_cost": 500000, "total_carbon_t": 12.4, "late_orders": 1,
                 "plant_utilisation": {"A": 0.9, "B": 0.5}},
        "solver_status": "OPTIMAL",
        "validator": {"capacity_ok": True, "eligibility_ok": True},
    }


@pytest.fixture
def scenario_facts() -> dict:
    return {
        "answer_type": "scenario",
        "intent": "capacity_loss",
        "scenario": {"plant": "B", "capacity_loss_pct": 0.3,
                     "from_date": "2026-01-05", "to_date": "2026-01-18"},
        "base_kpis": {"total_cost": 500000, "total_carbon_t": 12.4, "late_orders": 1},
        "kpis": {"total_cost": 521000, "total_carbon_t": 13.1, "late_orders": 2},
        "comparison": {"delta_cost": 21000, "delta_carbon_t": 0.7, "delta_late": 1,
                       "moved_orders": ["ORD-0003", "ORD-0009"]},
        "solver_status": "OPTIMAL",
        "validator": {"capacity_ok": True, "eligibility_ok": True},
    }


@pytest.fixture
def infeasible_facts() -> dict:
    return {
        "answer_type": "infeasible",
        "intent": "capacity_loss",
        "infeasible_orders": [{"order_id": "ORD-0004", "reason": "only Plant C makes P3 and it has no capacity"}],
        "kpis": {"total_cost": 480000, "total_carbon_t": 11.0, "late_orders": 0},
    }

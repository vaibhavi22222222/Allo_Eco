"""Shared helpers for workflow/API tests.

Every test gets its OWN temporary SQLite file seeded from the small synthetic
pack in ``fixtures/mini_dataset`` (6 orders, plants A/B/C). It runs through
the real data module, the real optimizer (PuLP/CBC) and the real LLM module in
mock mode. No hosted LLM is called and no hackathon answers are hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from app.config import AppSettings
from app.graph import CarbonWiseWorkflow
from app.llm.client import LLMClient, LLMSettings

FIXTURES = Path(__file__).parent / "fixtures"
MINI_DATA = FIXTURES / "mini_dataset"
MINI_CONFIG = FIXTURES / "mini_planning.yaml"


def mini_settings(tmp_path: Path, seed: bool = True) -> AppSettings:
    """Settings pointing at a fresh temporary database and the mini dataset."""
    return AppSettings(
        db_path=tmp_path / "carbonwise_test.sqlite3",
        planning_config_path=MINI_CONFIG,
        seed_data_dir=MINI_DATA if seed else None,
        solver_time_limit_s=10,
        solver_total_limit_s=40,
    )


def mock_llm() -> LLMClient:
    """A client in MOCK_LLM mode: parser uses rules, writer uses templates."""
    return LLMClient(settings=LLMSettings(mock=True))


def make_workflow(tmp_path: Path, llm_factory: Optional[Callable[[], LLMClient]] = None,
                  seed: bool = True) -> CarbonWiseWorkflow:
    return CarbonWiseWorkflow(mini_settings(tmp_path, seed), llm_factory or mock_llm)


def step_names(response_trace: list[dict]) -> list[str]:
    return [s["node"] for s in response_trace]

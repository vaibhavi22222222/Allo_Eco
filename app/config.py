"""Application settings (paths, solver time limits, API URL).

This is NOT a planning model. Planning parameters (energy prices, shared
capacities, horizon, currency) stay in Vibhu's ``PlanningConfig`` loaded from
``carbonwise-data/config/optimizer.yaml``. LLM settings stay in
``app.llm.client.LLMSettings`` (LLM_PROVIDER, LLM_MODEL, MOCK_LLM, ...).

Everything here comes from environment variables so the same code runs on a
laptop, in Docker and in tests:

    CARBONWISE_DB_PATH        SQLite file (default: the data module's default,
                              carbonwise-data/data/carbonwise.sqlite3)
    CARBONWISE_CONFIG         planning YAML (default: carbonwise-data/config/optimizer.yaml)
    CARBONWISE_SEED_DIR       CSV folder used when the database is empty
    SOLVER_TIME_LIMIT_S       seconds per CBC solve (default 20)
    SOLVER_TOTAL_LIMIT_S      seconds for all solves in one request (default 90)
    CARBONWISE_API_URL        used by the Streamlit UI (default http://localhost:8000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.bootstrap import DEFAULT_PLANNING_CONFIG, DEFAULT_SEED_DATA_DIR


def _optional_path(name: str) -> Optional[Path]:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class AppSettings:
    """Immutable settings for one running application (or one test)."""

    # None means "use storage.repository.default_db_path()", which itself
    # honours CARBONWISE_DB_PATH. Keeping None avoids two competing defaults.
    db_path: Optional[Path] = None
    planning_config_path: Path = DEFAULT_PLANNING_CONFIG
    seed_data_dir: Optional[Path] = DEFAULT_SEED_DATA_DIR
    solver_time_limit_s: float = 20.0
    solver_total_limit_s: float = 90.0
    api_url: str = "http://localhost:8000"

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Read settings from environment variables (see module docstring)."""
        return cls(
            db_path=_optional_path("CARBONWISE_DB_PATH"),
            planning_config_path=_optional_path("CARBONWISE_CONFIG") or DEFAULT_PLANNING_CONFIG,
            seed_data_dir=_optional_path("CARBONWISE_SEED_DIR") or DEFAULT_SEED_DATA_DIR,
            solver_time_limit_s=float(os.getenv("SOLVER_TIME_LIMIT_S", "20")),
            solver_total_limit_s=float(os.getenv("SOLVER_TOTAL_LIMIT_S", "90")),
            api_url=os.getenv("CARBONWISE_API_URL", "http://localhost:8000").rstrip("/"),
        )

    def solve_policy(self) -> dict:
        """Policy dict accepted by Vibhu's ``SolvePolicy.model_validate``."""
        return {
            "time_limit_seconds": self.solver_time_limit_s,
            "total_time_limit_seconds": self.solver_total_limit_s,
        }

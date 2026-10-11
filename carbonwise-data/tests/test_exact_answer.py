import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import _solve_fixture


def test_exact_answer_fixture_matches_expected_result():
    expected = json.loads((ROOT / "synthetic_data" / "expected_results" / "exact_answer.json").read_text(encoding="utf-8"))
    actual = _solve_fixture()["assignments"]
    assert actual["cost_only"] == expected["cost_only"]
    assert actual["carbon_only"] == expected["carbon_only"]
    assert actual["default_weight"] == expected["default_weight"]

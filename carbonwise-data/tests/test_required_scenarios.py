import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_required_scenarios_present():
    orders = pd.read_csv(ROOT / "synthetic_data" / "main" / "orders.csv")
    plants = pd.read_csv(ROOT / "synthetic_data" / "main" / "plants.csv")
    events = pd.read_csv(ROOT / "synthetic_data" / "main" / "capacity_events.csv")

    assert len(set(orders["order_id"])) == len(orders)
    assert len(plants["plant"].unique()) == 3
    assert len(orders["destination"].unique()) >= 8
    assert len(events) >= 1
    assert any(code in set(orders["order_id"]) for code in {"SCN-001", "SCN-002", "SCN-003", "SCN-004", "SCN-IMP-001"})
    assert any(product in set(plants["product"]) for product in {"P6", "P7"})

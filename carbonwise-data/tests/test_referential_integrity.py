import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_referential_integrity_for_main_dataset():
    orders = pd.read_csv(ROOT / "synthetic_data" / "main" / "orders.csv")
    plants = pd.read_csv(ROOT / "synthetic_data" / "main" / "plants.csv")
    transport = pd.read_csv(ROOT / "synthetic_data" / "main" / "transport_duties.csv")
    events = pd.read_csv(ROOT / "synthetic_data" / "main" / "capacity_events.csv")

    assert set(orders["product"]).issubset(set(plants["product"]))
    assert set(events["plant"]).issubset(set(plants["plant"]))
    plant_dest_pairs = set(transport[["plant", "destination"]].itertuples(index=False, name=None))
    for plant in plants["plant"].unique():
        for destination in orders["destination"].unique():
            assert (plant, destination) in plant_dest_pairs

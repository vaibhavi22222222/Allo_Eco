import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import load_official_schema


def test_main_schema_matches_contract():
    schema = load_official_schema()
    for key, spec in schema.items():
        path = ROOT / "synthetic_data" / "main" / spec["filename"]
        df = pd.read_csv(path)
        assert list(df.columns) == spec["columns"], f"schema mismatch for {key}"

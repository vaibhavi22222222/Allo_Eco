import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import generate_capacity_events, generate_orders, generate_transport


def test_reproducible_generation():
    o1 = generate_orders()
    t1 = generate_transport()
    e1 = generate_capacity_events()

    o2 = generate_orders()
    t2 = generate_transport()
    e2 = generate_capacity_events()

    assert o1.equals(o2)
    assert t1.equals(t2)
    assert e1.equals(e2)

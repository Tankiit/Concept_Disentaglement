import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# REACH may import `shared`, never RL.
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(rl|l1_l2_stub)\b", re.M)


def test_reach_imports_nothing_from_rl():
    bad = [p.name for p in ROOT.rglob("*.py") if "tests" not in p.parts and FORBIDDEN.search(p.read_text())]
    assert not bad, f"REACH imports RL code: {bad}"

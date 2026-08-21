import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.gauge.group.cem import CBMPartial, D_CEM


def test_gauge_cem_smoke():
    assert D_CEM["A"] == "A'w+ = w+"
    assert D_CEM["A'"] == "A'w- = w-"

    cbm = CBMPartial(k=5, k_supervised=3)
    assert cbm.supervised_fraction == 0.6

    arr = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert arr.shape == (2, 2)

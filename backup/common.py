from dataclasses import dataclass, asdict
import numpy as np
from shared.rng import rng_for  # noqa: F401  (independent streams per redraw/purpose)

ITEM_COLS = ["arm", "redraw", "num_seed", "n", "m", "eta", "test_id", "stratum",
             "W_cls", "W_label", "W_cov", "W_total", "additivity_resid",
             "solver_status", "ball_resolution", "eps_n"]

@dataclass
class Row:
    arm: str; redraw: int; num_seed: int; n: int; m: int; eta: float
    test_id: int; stratum: str
    W_cls: float; W_label: float; W_cov: float; W_total: float
    additivity_resid: float; solver_status: str; ball_resolution: float; eps_n: float

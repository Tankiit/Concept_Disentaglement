import numpy as np, pandas as pd

def per_redraw_effect(items, comp, stratum, n_lo, n_hi, by="n"):
    # TODO: filter arm/stratum; mean over test_id within (redraw, value); return array over redraws
    #       of mean(width[hi]) - mean(width[lo])   (redraw is the unit)
    ...

def redraw_bootstrap_ci(effects, B=2000, alpha=0.05, seed=0):
    # TODO: resample whole redraws (paired trajectories), not test inputs
    ...

def verdict(effect, ci, expected_sign, status, tol):
    # TODO: 'supported' | 'contradicted' | 'inconclusive' | 'invariant_by_construction'
    # TODO: exact zeros handled explicitly; no log floor; log-log slopes only if scaling is claimed
    ...

def build_response_summary(items, cfg):
    # TODO: loop cfg["expected"]; write response_summary.csv
    #       cols: component, acquisition, stratum, effect, ci_lo, ci_hi, expected, status, verdict
    ...

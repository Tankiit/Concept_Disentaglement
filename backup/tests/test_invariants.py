import pandas as pd

def test_label_flat_in_n(items: pd.DataFrame, tol):
    # TODO: per (redraw,test_id) max-min of W_label across n <= tol, for arm R1 only
    ...

def test_additivity(items, tol):
    # TODO: max additivity_resid <= tol; algebra still needs a proposition, not just this
    ...

def test_label_cache_identity(cfg):
    # TODO: assert label_objects hash is identical across n within a redraw
    ...

def test_no_forbidden_terms(paths, banned=("relative_likelihood", "density_ratio", "decalibration", "dre_")):
    # TODO: grep reach_exp sources/drafts; D1 removed from REACH
    ...

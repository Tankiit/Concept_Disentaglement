"""R1: at fixed labels, what changes when more deployment inputs are observed?"""
import pandas as pd
from common import Row, rng_for

def make_redraw(cfg, redraw):
    # TODO(API): generator -> classifier (frozen), labelled set (size m), log POOL, test inputs
    # TODO: order pool once with rng_for(redraw, "log_order"); prefixes are nested
    # TODO: tag each test input with stratum from the generating distribution
    ...

def label_objects(cfg, labelled):
    # TODO(API): everything label-derived (class embeddings, conditionals). Computed ONCE,
    # cached by hash; reused for every n so the invariance check is meaningful.
    ...

def widths(cfg, clf, label_obj, log_prefix, x_test):
    # TODO(API): call native W_cls, W_label, W_unk(->W_cov), total
    # TODO: additivity_resid = |total - (cls+label+cov)|  (definition from the definitions sheet)
    # TODO: ball_resolution per manifest definition; solver status
    ...

def run_redraw(cfg, redraw):
    data = make_redraw(cfg, redraw)
    lo = label_objects(cfg, data.labelled)
    rows = []
    for n in cfg["n_grid"]:
        prefix = data.log_pool[:n]
        # TODO: eps_n from cfg (constant c, confidence) - do NOT tune to get a slope
        out = widths(cfg, data.clf, lo, prefix, data.x_test)
        # TODO: append Row(...) per test input
    return rows

def gate_ball_resolution(rows, cfg):
    # TODO: at smallest n, min resolution >= cfg["gates"]["ball_resolution_min"], else STOP
    ...

def main(cfg_path, mode="pilot"):
    # TODO: load yaml; loop redraws (pilot=3); write item_outputs.parquet; run gate
    ...

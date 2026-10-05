"""R2 (m contrast), R3 (eta), R4 (A6 EMQ+bootstrap), R5 (A12 inclusion), R6 (violation)."""

def r2_m_contrast(cfg, redraw):
    # TODO: fixed log, vary m; refit ONLY label-derived quantities; backbone frozen
    # TODO: do not force other components to stay flat if they share those estimates
    ...

def r3_eta(cfg, redraw):
    # TODO: state corruption mechanism + fixed target; report raw responses, no (1-2eta) law
    ...

def r4_a6_emq_bootstrap(cfg, redraw):
    # TODO: same frozen predictions, same nested prefixes; resample the LOG (document it)
    # TODO: compare prevalence estimates + predictive spread; EMQ does not give 3 parts
    ...

def r5_a12_ball_inclusion(cfg, redraw):
    # TODO: is true (or high-accuracy reference) embedding inside the ball? freq over redraws
    # TODO: log radius + resolution beside inclusion (vacuous ball => trivial inclusion)
    ...

def r6_violation(cfg, redraw, kind, strength):
    # TODO: kind in {mean_shift, cov_change, mixture_contam}; perturb class-conditionals
    #       between labelled set and deployment; sweep strength; log which widths move
    ...

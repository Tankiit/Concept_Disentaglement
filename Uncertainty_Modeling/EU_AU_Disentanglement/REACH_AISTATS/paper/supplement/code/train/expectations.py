"""Pre-registered signs for the response matrix (written 2026-10-02, before any response run).

Key: (acquisition, component, region, world). Value is the predicted sign of the log-log slope
of mean width against that acquisition's budget, all other budgets held at BASE:

    'negative'  width contracts with this acquisition
    'positive'  width grows (only label noise eta, which degrades the label channel)
    '~0'        no response: the slope is indistinguishable from the same-budget placebo null
    'exact'     invariance holds by construction (the component never reads this acquisition);
                reported, but it is NOT evidence of identification and is not counted as a test
    None        exploratory, no claim

Baseline rows use the same keys with the baseline name as the component. Their predictions state
the paper's argument: a scalar epistemic score (ensemble MI / spread) moves with every acquisition
that feeds it, so it cannot say which acquisition will help; EMQ + bootstrap reads only the log.

Freeze this file before confirmatory runs; the manifest records its SHA-256.
"""

ACQUISITIONS = {'unlabelled': 'n', 'labels': 'm', 'noise': 'eta'}
COMPONENTS = ('W_cls', 'W_label', 'W_unk')
BASELINES = ('EMQ_cls', 'EMQ_unk', 'Ens_MI', 'Ens_std')
REGIONS = ('on_support', 'boundary', 'off_support')

# Budgets the non-swept axes are held at. n=4096 sits past the vacuous ceiling (README: n=256 is
# saturated at sep=1); eta=0 is the noiseless label channel.
BASE = dict(n=4096, m=256, eta=0.)
# Log-spaced grids; eta excludes 0 because the slope is taken in log(eta).
GRIDS = dict(n=(1024, 2048, 4096, 8192, 16384, 32768, 65536),
             m=(32, 64, 128, 256, 512, 1024, 2048),
             eta=(.0125, .025, .05, .1, .2))

_main = {
    # Unlabelled log: prevalence and novelty contract where they are supported; labels are untouched.
    ('unlabelled', 'W_cls',   'on_support'):  'negative',
    ('unlabelled', 'W_cls',   'boundary'):    'negative',
    ('unlabelled', 'W_cls',   'off_support'): '~0',       # zero-mass cell: vacuous [0,1] at every n
    ('unlabelled', 'W_label', 'on_support'):  'exact',
    ('unlabelled', 'W_label', 'boundary'):    'exact',
    ('unlabelled', 'W_unk',   'on_support'):  'negative',
    ('unlabelled', 'W_unk',   'boundary'):    'negative',
    ('unlabelled', 'W_unk',   'off_support'): '~0',
    # Labels: only the label channel contracts.
    ('labels', 'W_cls',   'on_support'):  'exact',
    ('labels', 'W_cls',   'boundary'):    'exact',
    ('labels', 'W_label', 'on_support'):  'negative',
    ('labels', 'W_label', 'boundary'):    'negative',
    ('labels', 'W_unk',   'on_support'):  'exact',
    ('labels', 'W_unk',   'boundary'):    'exact',
    # Known label noise: only the label channel widens.
    ('noise', 'W_cls',   'on_support'):  'exact',
    ('noise', 'W_cls',   'boundary'):    'exact',
    ('noise', 'W_label', 'on_support'):  'positive',
    ('noise', 'W_label', 'boundary'):    'positive',
    ('noise', 'W_unk',   'on_support'):  'exact',
    ('noise', 'W_unk',   'boundary'):    'exact',
    # Baselines.
    ('unlabelled', 'EMQ_cls', 'on_support'): 'negative',
    ('unlabelled', 'EMQ_cls', 'boundary'):   'negative',
    ('unlabelled', 'EMQ_unk', 'on_support'): 'negative',
    ('labels',     'EMQ_cls', 'on_support'): 'exact',
    ('noise',      'EMQ_cls', 'on_support'): 'exact',
    ('unlabelled', 'Ens_MI',  'on_support'): 'negative',
    ('unlabelled', 'Ens_MI',  'boundary'):   'negative',
    ('labels',     'Ens_MI',  'on_support'): 'negative',
    ('labels',     'Ens_MI',  'boundary'):   'negative',
    ('noise',      'Ens_MI',  'on_support'): 'positive',
    ('noise',      'Ens_MI',  'boundary'):   'positive',
    ('unlabelled', 'Ens_std', 'on_support'): 'negative',
    ('labels',     'Ens_std', 'on_support'): 'negative',
    ('noise',      'Ens_std', 'on_support'): 'positive',
}

EXPECT = {key+('separated',): sign for key, sign in _main.items()}
# Novel-class arm. With q inside the known convex hull the log cannot identify novelty, so the
# W_unk width converges to a positive constant and its slope must flatten towards zero.
EXPECT.update({
    ('unlabelled', 'W_unk', 'on_support', 'confounded'): '~0',
    ('unlabelled', 'W_unk', 'boundary',   'confounded'): '~0',
    ('unlabelled', 'W_cls', 'on_support', 'confounded'): None,
    # No novel mass at all: prevalence still contracts; W_unk is exploratory (its truth is 0).
    ('unlabelled', 'W_cls', 'on_support', 'absent'): 'negative',
    ('unlabelled', 'W_unk', 'on_support', 'absent'): None,
})

# Family-wise level for the placebo test: Bonferroni over the predicted (non-exact) cells.
FAMILY_ALPHA = .05

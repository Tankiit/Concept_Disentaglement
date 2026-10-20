# Pilot results — 2026-10-19

All 126 original REACH-engine calls completed, producing 1,386 finite cell
records. Bounds, interval ordering, width identities, and pool assignments
passed checks. No warnings were captured in these calls. Class interval
statuses: {'optimal': 916, 'zero_mass_policy': 470}.

## Frozen-model checks

Known-digit CNN accuracy on held-out evaluation images: 98.04%.
Entropy AUROC: letters 0.837, KMNIST 0.907.
Distance AUROC: letters 0.864, KMNIST 0.941.
These are ordinary OOD diagnostics, not REACH guarantees or MC-dropout results.
The LDA outlier cell receives 31.1% of letters and
56.6% of KMNIST evaluation images. Many novel
images therefore still land in known-class cells.

## REACH outcome

At the baseline budgets, mean on-support W_joint and W_unk_joint are approximately
1 in all six scenarios. These intervals contain almost the whole probability
range. Coverage is high but precision is poor. No declared-family incompatibility
was flagged in any of the 126 runs, including KMNIST deployment. This does not
establish that the undeclared novelty was modeled correctly.

Increasing digit references from 200 to 1000 per class reduces mean W_label
from about 0.6404 to 0.4251. In the letters-5% scenario, mean W_unk_joint
changes from 1 to 0.9792. This is only modest tightening.
Increasing log size or letter reference size alone generally leaves the joint
novelty intervals near 1 at the baseline digit-reference budget.

## Worked REACH call

Scenario: 5% EMNIST novelty; seed 0; n=1000; m=200 per digit;
r=50 per letter class (1300 letter references total).

The radius is eps_log + max(reference radii). In this call:

- eps_log = 0.1221.
- Known-reference radius = 0.3255.
- Total radius = 0.4476.

The known-reference term dominates, so adding only letter references cannot
reduce the maximum reference-radius term once it is already smaller. Adding
log data reduces its own term but leaves the known-reference term in place.

At cell 0, the true novel posterior is 0.2059, although global novel mass
is only 0.05. A conditional probability at a cell can differ substantially
from the global mixture weight. The true value is computed by multiplying
each source weight by its probability of that cell and normalizing.
The joint novelty interval is [0.0000, 1.0000].
The width reflects lack of precision; it does not estimate the novel posterior
as 1. Inputs and all cell intervals are in single_run_inputs.npz and single_run.csv.

## Interpretation and next decision

This pilot supports neither precise novelty attribution nor reliable detection
of a mismatched novelty family at these budgets. It does show why reference
uncertainty matters even when the encoder classifies digits well. Smaller
posterior intervals require investigating acquisition bottlenecks and cell
resolution, not merely adding more deployment observations.

A next experiment could increase digit references and log budget together,
then compare to an explicitly labelled oracle-conditional diagnostic to separate
finite-reference conservatism from non-identification. That experiment has not
been run. Fashion-MNIST and medical-image extensions also remain pending.

There is only one trained encoder and three simulation seeds. Cell averages
are descriptive; cells sharing a run are not independent replications.
Declared-letter and undeclared-KMNIST coverage is measured against finite
empirical deployment-pool truth; the latter is outside the assumed family
unless its cell law happens to be representable. No population theorem is
established by this pilot.

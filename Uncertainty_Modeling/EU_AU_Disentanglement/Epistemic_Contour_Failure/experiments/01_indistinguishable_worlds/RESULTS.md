# First construction: different clean contours, identical observations

Date: 2026-10-23. Status: analytic construction with executed numerical checks; no epistemic estimator evaluated yet.

## Result

Two spherical Gaussian sources with unit standard deviation and centers (-0.2, 0) and (0.2, 0) admit distinct contamination distributions at the same known contamination fraction 0.20 such that their observed mixtures are identical. The clean 95%-mass contours have radius 2.447747 and Hausdorff separation 0.4.

Consequently, additional unlabeled mixture observations cannot distinguish the worlds. This is true at every sample size, including exact knowledge of the observed distribution. The construction also holds before partitioning, so the ambiguity is not a discretization artifact.

![Clean contours and shared observations](indistinguishable_worlds.png)

## Continuous construction and proof

Let f_0 and f_1 denote the two Gaussian densities, and let delta be their total variation distance. With equal spherical covariance and center separation d,

    delta = 2 Phi(d / (2 sigma)) - 1.

To see this, the densities cross at the perpendicular bisector between the centers; integration over either half-space reduces the calculation to the one-dimensional Gaussian projection along the displacement.

Set r = (f_0 + f_1)/2 and R = 1 - (1-epsilon)(1+delta). When R >= 0, define

    g(x) = (1-epsilon) max(f_0(x), f_1(x)) + R r(x),
    h_i(x) = [g(x) - (1-epsilon) f_i(x)] / epsilon.

Since the integral of max(f_0,f_1) is 1+delta, g and each h_i integrate to one and are nonnegative. Thus g = (1-epsilon)f_i + epsilon h_i in both worlds. These h_i are valid unrestricted contamination densities; they are not assumed Gaussian and differ across worlds.

Here delta = 0.158519419, and the smallest contamination fraction admitting a shared mixture for this pair is delta/(1+delta) = 0.136829315. At epsilon = 0.20, the maximum admissible equal-variance center separation is

    d_max = 2 sigma Phi^{-1}(1 / (2(1-epsilon))) = 0.637278728.

These are direct derivations, not claims of novelty. Contamination and partial-identification literature must be reviewed before positioning them as a paper contribution.

## What this requires of uncertainty

The contours are translations of the same circle, so their Hausdorff distance is exactly the center separation, 0.4. By the triangle inequality and the identical observation laws, every estimator has expected Hausdorff error at least 0.2 in at least one of these two worlds. This is a worst-case expected-error bound, not a lower bound on every individual realized error.

A confidence set of contours that covers the truth with probability at least 95% in each world must contain both contours with probability at least 90%, under their shared observation law. Hence its Hausdorff diameter must be at least 0.4 with probability at least 90%, regardless of sample size. The 95% confidence requirement is separate from the 95% clean probability mass used to define the contour.

This gives a benchmark for a failure claim: an uncertainty set that collapses below this persistent ambiguity cannot provide that uniform coverage. A scalar disagreement score has no automatic confidence-set interpretation.

## Executed checks

The script computes rectangular cell masses through Gaussian CDF differences, with unbounded tail bins so that the partition covers the entire plane. Every grid places an edge at x=0, the density-crossing line. This ensures that integrating the maximum density equals taking the maximum of the two cell masses. Other grid alignments need not preserve total variation exactly.

| Interior cell width | Total cells, including tails | Cell total variation | Largest mixture reconstruction error |
| --- | ---: | ---: | ---: |
| 1.0 | 196 | 0.158519419 | 4.34e-19 |
| 0.5 | 676 | 0.158519419 | 1.73e-18 |
| 0.25 | 2500 | 0.158519419 | 2.17e-19 |
| 0.125 | 9604 | 0.158519419 | 1.08e-19 |

All clean, contamination, and mixture probability vectors passed nonnegativity and normalization assertions; the largest normalization error was 4.45e-16. The figure omits tail bins for display only. No Monte Carlo experiment is needed to establish equality of the laws, and none is claimed here.

## Interpretation and next step

This establishes unresolved knowledge about the clean source under unknown unrestricted contamination. It does not establish that every epistemic definition fails. A correctly specified method could preserve this ambiguity. Known contamination, source identities, or clean references change the information available and may resolve it.

Next, select and inspect the original definitions of a small number of uncertainty formulations. A contamination-ignoring Gaussian posterior provides an interpretable diagnostic candidate, but its failure may primarily demonstrate misspecification. The stronger conceptual question is whether a quantity interpreted as epistemic uncertainty omits relevant ambiguity even when estimated accurately. We must distinguish those claims before running or interpreting baseline comparisons.

## Reproduce

From the repository root:

```bash
python Uncertainty_Modeling/EU_AU_Disentanglement/Epistemic_Contour_Failure/experiments/01_indistinguishable_worlds/run.py
```

Dependencies used: numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0. The computation is deterministic. Outputs: `summary.json`, `cell_laws.npz`, and `indistinguishable_worlds.png`. The NPZ arrays contain evaluator ground truth; future learners must receive only the declared observations and assumptions.

## Theoretical decision record and sources (added 2026-10-23)

- Chose equal known scale and horizontally displaced centers to obtain an explicit density-crossing line and a calculable contour separation.
- Allowed different unrestricted contamination laws in the two worlds. This assumption is essential; the argument does not establish the same ambiguity for a common known Q.
- Constructed identical continuous densities before binning. This strengthened the initial finite-cell argument and showed that this example's ambiguity is not caused by aggregation.
- Aligned grid edges with the density-crossing line to verify exact preservation of total variation across the tested resolutions. Arbitrarily shifted grids need not preserve it exactly.
- Distinguished a lower bound on worst-case expected error from a realized-error guarantee, and contour-set coverage from pointwise band coverage.

No papers were consulted for this initial construction. The calculations above are local derivations, with no novelty claim. A literature review of contamination neighborhoods and nonidentifiability is still needed; later papers must not be retrospectively represented as sources used to derive this experiment.

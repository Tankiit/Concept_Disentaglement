# Focused REACH interval diagnostic — 2026-10-19

We traced one declared-novelty condition from the MNIST pilot: EMNIST letters
at 5% global novelty, simulation seed 0, cell 0. We reused saved reference
estimates, source distributions, weights, and the original 1000 log samples.
No encoder training, new dataset download, or changes to original REACH code.

## Finding

Two uncertainties contribute: reference bounds and finite deployment samples.
The problem is not explained solely by inaccurate source estimates or a
structurally unidentifiable representation in this example.

The empirical source matrix has full rank (11 of 11), with condition number
about 3.54. With the exact source distributions and exact deployment marginal,
the novelty posterior interval collapses to the truth, 0.20587. This establishes
identification for this finite-pool example, not every source or cell.

| References / deployment information | n | Joint novelty interval |
| --- | ---: | --- |
| Estimated references, original bounds | 1,000 | [0, 1] |
| Exact source distributions, no reference error | 1,000 | [0, 0.98552] |
| Exact source distributions, no reference error | 16,000 | [0, 0.87570] |
| Exact source distributions, no reference error | 64,000 | [0.02263, 0.54268] |
| Estimated references, original bounds | 64,000 | [0, 1] |
| Exact source distributions and exact marginal | No sampling error | [0.20587, 0.20587], within solver tolerance |

Even perfect references do not make the finite-log bound precise at n=1000.
The cell has true mass 0.03283 and only 35 of the original 1000 observations.
The log radius is 0.12211, permitting considerable variation relative to that
cell mass. This comparison helps explain sensitivity; it is not itself a proof
of the tightness or optimality of the radius formula.

## Why the original upper endpoint becomes 1

At n=1000 the log radius is 0.12211 and the dominant reference radius is
0.32551, giving total radius 0.44762. Per-entry reference half-widths are
0.15401. The joint bound's upper calculation uses LOWER known-source entries
and the UPPER novelty entry in its denominator.

For cell 0, the known lower entries are zero except for source 0. A feasible
mixture can put zero weight on source 0 and on the novel source. That makes
this surrogate denominator exactly zero. A saved explicit nonnegative,
unit-sum weight vector has residual about 0.17197, well inside radius 0.44762,
and zero denominator. This verifies the policy trigger without relying only
on the cone solver's tiny numerical mass estimate.

The original solver deliberately returns the vacuous [0,1] policy interval
when the minimum denominator can be zero; the joint upper bound takes its
upper endpoint. This is not a solver crash. The joint lower calculation returns
zero by optimization, not the same zero-denominator shortcut.

The denominator here is the conservative joint-bound expression. We should
not equate it unconditionally with an exactly realizable uncertain-source
model: the per-entry outer bounds need not enforce every joint dependence.
The near-zero plug-in denominator is recorded separately in mass_witnesses.json.

## Controls and interpretation

- Exact conditionals WITH the original reference/entry bounds still give
  [0,1]. Better centres alone do not fix the conservative bound construction.
- Estimated conditionals with reference bounds artificially removed can give
  narrower intervals. This is an uncalibrated ablation, not a valid fix.
- Increasing only log size leaves the fixed reference floor in place.
- The zero-sampling-error oracle endpoint test uses privileged finite-pool
  truth. It is a diagnostic, not deployable performance.

The additional log samples are simulated from the same finite-pool marginal,
with replacement and an independent seed. The first 1000 original observations
are preserved; larger budgets are nested. This does not acquire 64,000 distinct
real images. One cell and one random stream are studied, so there are no
multi-seed conclusions or population coverage claims.

## What should change next

The evidence supports testing joint increases in reference and log budgets,
and reviewing whether a tighter justified region can preserve the dependencies
lost by per-entry bounds. Changing the encoder is not the first supported fix
for this identified example. Do not remove reference uncertainty or lower the
radius arbitrarily to obtain attractive intervals. Validate any new interval
construction against explicit finite-pool truth and repeated samples.

## Reproduce and inspect

From REACH_AISTATS with reach-aistats active:

```bash
python workbook/reach_interval_diagnostic_2026-10-19/diagnose.py
```

Files: intervals.csv (all controls), mass_witnesses.json (minimum-denominator
weights and solver status), zero_denominator_certificate.json (explicit feasible
zero witness), identifiability.json (source matrix spectrum), diagnostic_inputs.npz,
config.json, output.txt, and oracle_comparison.png.

Baseline joint endpoints were checked against the prior saved run. All returned
endpoints were finite and interval ordering passed a 1e-6 numerical tolerance.
A tiny negative width in the exact-marginal limit is solver roundoff and is
retained in raw results rather than silently modified.

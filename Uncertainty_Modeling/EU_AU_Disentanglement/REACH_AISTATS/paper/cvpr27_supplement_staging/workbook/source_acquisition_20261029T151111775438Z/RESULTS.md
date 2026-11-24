# Acquisition experiment results

This is an image-derived finite-cell simulation, not a natural-image benchmark.
6 paired repeats; 2880 target intervals; budget maximum 100000.
Reference and target-label budgets are per known source; their total cost is twice the x-axis value.

## Candidate membership under constructed confounding

- mnist: exact input-only width 0.6732; finite mean width after inputs 0.7071; after source identities 0.1280.
- emnist: exact input-only width 0.6758; finite mean width after inputs 0.7925; after source identities 0.5017.
- fashion: exact input-only width 0.6710; finite mean width after inputs 0.7145; after source identities 0.1273.
- blood: exact input-only width 0.6775; finite mean width after inputs 0.7319; after source identities 0.0820.

## Interpretation limits

Positive exact floors demonstrate input-only non-identification in the declared model. Whether finite identity acquisition crosses them is read from the numbers above; finite reference uncertainty remains.
All target endpoints, inclusion indicators, solver residuals, class roles, source laws and acquisition counts are exported.
Widths concern distinct probabilities and are not an additive decomposition. Noise is specified; aleatoric uncertainty is not estimated.
Inclusion summaries pool dependent settings and are descriptive. Six repeats do not establish precise calibration.
See SOURCES.md for every external theoretical and dataset source, and config.json for assumptions.
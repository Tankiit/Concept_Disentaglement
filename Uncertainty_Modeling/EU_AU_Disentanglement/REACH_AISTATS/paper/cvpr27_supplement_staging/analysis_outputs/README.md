# Figure 9 and Appendix E.4

The records accompany Figure 9 in `_CVPR27_REACH (9).pdf`. The table counts each qualifying ordered pair once, including pairs that share a state or occupy the same plotted location.

| Dataset | Fit-only states | Qualifying pairs | Identity role positive | Label role positive | Both positive | Selected pair (identity, label) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CAMELYON17 | 600 | 39 | 19 | 35 | 19 | (390, 566) |
| FMoW | 600 | 32 | 8 | 20 | 6 | (287, 561) |

## Records and units

- `search_records/camelyon17/` and `search_records/fmow/` contain the archived candidate states, fit observation IDs keyed by state ID (gzip-compressed JSON), protocol, ranked qualifying pairs, original selected states, target definition, population rates and selected-state one-step risks. The source runs are `workbook/camelyon17_counterexample_20261030T092024988890Z/` and `workbook/fmow_counterexample_20261030T094929883846Z/`. The candidate CSV preserves the original state IDs, identity and label budgets, posterior moments, MI, both modeled action values and all three Beta shape pairs in the order W, Theta_0, Theta_1.
- `pair_selection_log.csv` reproduces the fit-only matches, sorted within dataset by relative MI gap, identity state ID, then label state ID. Rank 1 is the original selection. The relative gaps are fractions of the pair average.
- `pairwise_heldout_advantages.csv` contains one row per qualifying pair. Columns `E_I_i` and `E_L_i` give expected one-step squared errors after identity and label acquisition in the identity-preferred state. Columns `E_I_j` and `E_L_j` give the corresponding errors in the label-preferred state. The signed columns are `identity_advantage = E_L_i - E_I_i` and `label_advantage = E_I_j - E_L_j`. All six columns use squared-probability units. The figure multiplies each advantage by one million. A 1 in `both_confirmed` requires both advantages to be strictly positive.
- `pairwise_confirmation_summary.csv` records the counts and the maximum absolute discrepancy from the four archived selected-state/action risks in each dataset.

## Reproduction

From the paper or extracted bundle directory, run the following commands in order.

```sh
python scripts/reproduce_pairwide.py
python scripts/plot_pairwide.py
```

The first command uses the Python standard library. It reapplies the saved action margins and matching tolerances to all candidates, checks every ranked match and the selected states against the archive, and enumerates the one-step risks in Eqs. (68)–(69). The empirical identity law uses the source-1 fraction in the fixed held-out pool. Label acquisition chooses either source with probability one half and uses its empirical event rate. The second command uses Matplotlib to read the advantages CSV and write `figures/supp_pairwide_confirmation.pdf`.

The eight selected risks agree with the archived values to within 3.5e-17 MSE. Both coordinates of every plotted pair and both selected stars were also checked against Figure 9 in PDF (9). These are descriptive counts for the saved searches and evaluation pools.

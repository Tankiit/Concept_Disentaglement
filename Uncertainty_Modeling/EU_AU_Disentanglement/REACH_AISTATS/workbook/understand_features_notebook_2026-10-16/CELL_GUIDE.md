# features.ipynb cell guide

Reviewed/executed bookkeeping: 2026-10-19. Positions are zero-based notebook
cell positions, not the execution numbers displayed by Jupyter.
The original notebook has not been executed or modified.

| Cell | Purpose | Inputs/dependencies | Output or side effect |
| --- | --- | --- | --- |
| 0 | Explain cache assumptions | Historical environment description | Markdown only |
| 1 | Inventory NPZ files and detect unreadable arrays | cache/features/*.npz | intact/broken lists and schema output |
| 2 | Load clean CLIP train/test arrays | real.head.load, cifar10_train/test caches | train/test dictionaries |
| 3 | Inspect available corruption severities | intact list and corruption caches | Shape checks and severity counts |
| 4 | Check feature normalization | train features | Row-norm summary |
| 5 | Describe schemas and Modal commands | Historical cache setup | Markdown; shell commands are not automatically executed |
| 6 | Explain four pools | data.pools conventions | Markdown only |
| 7 | Construct limited reference pools | CLIP train/test, seed 0, 200 references/class | dev/reference/log/eval and disjointness check |
| 8 | Replace log with brightness severity 3 | brightness cache, earlier pools | Check same log IDs and unchanged evaluation features |
| 9 | Export full-reference datasets | CLIP train/test | Writes cache/sets/labeled.npz, unlabeled.npz, test.npz; can overwrite existing files |
| 10 | Visualize labelled/log feature overlap | Exported sets; PCA then t-SNE | Three-panel figure; shared embedding |
| 11 | Quantify original-space distances | Exported sets; seeded subsamples | Nearest-neighbour and empirical threshold summaries |
| 12 | Describe six-known-class novelty setup | Classes 0–5 known | Markdown only |
| 13 | Visualize six-known-class setup | Earlier plotting variables and RNG state | Figure and sampled novel fraction |
| 14 | Quantify known/novel distance differences | Samples from cell 13 | Distance ratios and threshold-based novelty estimate |
| 15 | Describe five-known-class REACH sweep | New regime: classes 0–4 known | Markdown only |
| 16 | Record proposed predictions | hashlib, json | Predictions dictionary and hash, not proof of historical preregistration |
| 17 | Construct audit contexts | CLIP train/test; real.audit.build_context | Frozen head/cells; deployment restricted to test log half; recomputed truth conditionals |
| 18 | Run sweep | Contexts, 3 novelty arms, 3 axes × 3 values × 5 seeds | 135 run_cell calls, 810 cell records; repeated baselines across axes |
| 19 | Aggregate widths and coverage | sweep DataFrame | Heuristic direction checks and novelty-interval summaries |
| 20 | Compare prevalence corrections | Contexts, fresh LDA head, eval-half known images | Accuracy differences labelled excess risk; crossover n* |

## Execution order and interpretation

Cells share state. In particular, `rng` is reused between the geometry cells,
and later cells depend on contexts and frames built earlier. Partial reruns
can change subsamples or leave stale variables. Execute a separate copy with
an explicit environment and input manifest for future reproduction.

The notebook uses three different class regimes: all ten classes for the
first overlap check, six known classes for the novelty visualization, and
five known classes for the REACH sweep. These are not the three-known-class
OSSIM split from our dropout workbook.

Nearest-neighbour thresholds and t-SNE are exploratory diagnostics. They do
not establish a guarantee that feature distributions have disjoint support.
The empirical 95% distance threshold is not a calibrated novel-class probability.

## Underlying audit details verified by source inspection

- `build_context()` fits a PCA/LDA head on the development pool. Source labels
  organize reference pools and simulated deployment draws. Log labels are not
  directly provided as labelled observations to the mixture estimator.
- The notebook restricts deployment pools to the first test half and recomputes
  truth conditionals there. Evaluation uses the second half.
- `_draw_from()` samples with replacement, so n=10,000 does not imply 10,000
  distinct deployment images. Truth refers to the finite empirical source pools.
- `run_cell()` generates a shared maximum-size log and takes prefixes for each n.
  Reference acquisition also uses shared seeded streams, capped by pool size.
- q_mode='oracle' estimates the novel row from held-out classes in the training
  reference pool, with an estimation radius. It is favourable reference access,
  not exact knowledge of every deployment distribution.
- W_cls plugs in estimated source entries; W_joint also varies those entries.
  W_unk_joint similarly accounts for entry uncertainty. Their guarantees must
  be interpreted with the stated family and reference assumptions.
- `run_cell()` takes its set kind from module state/default, not directly from
  the passed context. The default is L2 here; future variants must verify it.
- Baselines include resampling-based ensembles over reference/log histograms;
  these are not independently trained neural-network deep ensembles.

## Performance-analysis issues to retain for later review

The last cell's excess_risk fields are method accuracy minus oracle accuracy,
the opposite sign of method error minus oracle error. Its fixed evaluation
class mixture is not reweighted to each simulated deployment mixture.
True weights combined with estimated conditionals need not give an optimal
classifier. The direct argmax-to-label comparison works for classes 0–4 but
would require explicit remapping for noncontiguous class IDs. The crossover
between distinct width estimands is exploratory, not a validated acquisition rule.

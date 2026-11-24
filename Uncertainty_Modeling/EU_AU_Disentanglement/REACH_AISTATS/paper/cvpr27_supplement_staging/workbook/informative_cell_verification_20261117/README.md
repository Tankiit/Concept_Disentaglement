# Informative-cell verification

From REACH_AISTATS run `python scripts/verify_informative_cell.py`.

This separately archived finite-model calculation verifies PDF Eqs. (3)–(7), Appendices 8.2–8.5 and Table 3. It does not replace or modify the earlier constant-input experiment. All probabilities and risks use exact rational arithmetic; displayed decimals are rounded. Ordered-sequence enumeration independently checks count enumeration for budgets 0–6 at three priors. The affine regret identity is checked on all policy-simplex vertices, with the attaining equal-identity/label policy checked separately.

Figure provenance: supp_exact_worlds.pdf/png is generated directly from the verified one-step values by scripts/verify_informative_cell.py. table3.tex is generated from the rational multi-budget risks. PDF bytes can differ between runs due to creation metadata.

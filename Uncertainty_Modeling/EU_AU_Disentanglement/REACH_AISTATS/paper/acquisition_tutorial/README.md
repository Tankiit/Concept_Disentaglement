# Acquisition experiments: a tutorial from first principles

Start with **acquisition_tutorial.pdf**. The editable source is **acquisition_tutorial.tex**.

The document covers the six requested notebooks and their exact saved runs. It includes probability foundations; concentration and optimization derivations; the exact acquisition counterexample; neural architectures, losses, and metrics; Beta posterior calculations; all 14 original standalone figures; detailed figure interpretation; numerical tables; a glossary; and provenance.

## Compile the document

Keep the `.tex` files and `figures/` directory together. From this directory run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error acquisition_tutorial.tex
pdflatex -interaction=nonstopmode -halt-on-error acquisition_tutorial.tex
pdflatex -interaction=nonstopmode -halt-on-error acquisition_tutorial.tex
```

The included numerical tables and provenance are already generated. Compilation uses local files and requires no experiment rerun or network access. The source uses standard LaTeX packages: fontenc, inputenc, lmodern, amsmath, amssymb, graphicx, booktabs, longtable, array, geometry, xcolor, float, hyperref, and url.

## Regenerate numerical tables and artifact checks

Within the original repository location, run:

```bash
python build_tables.py
```

This standard-library script reads the original workbook CSV files and generates `numerical_tables.tex`, `provenance.tex`, `source_manifest.json`, and `artifact_checks.json`. It verifies interval counts, statuses, source-simulation inclusion, the two held-out confirmation outcomes, original/copied figure hashes, and closed-form means, variances, and acquisition values for all 1,200 searched Beta states.

The original workbook data are required only for table regeneration and artifact auditing. The included `.tex` files suffice for compilation when the document directory is copied elsewhere.

## Files

- `acquisition_tutorial.tex`: full explanatory narrative and mathematical derivations.
- `numerical_tables.tex`: companion tables extracted from saved CSV results.
- `provenance.tex`: exact notebook/run locations and recorded software versions.
- `figures/`: all 14 original result plots, copied byte-for-byte.
- `figure_manifest.json`: source paths and SHA-256 hashes for each figure.
- `source_manifest.json`: hashes of the local inputs inspected or extracted.
- `artifact_checks.json`: machine-readable validation results.
- `build_tables.py`: reproducible numerical extraction and validation.
- `document_checks.json`: final PDF/build checks.

Figures follow their original experiment semantics. In particular, the document distinguishes probability widths, predictive MI, posterior variance, and held-out MSE; explains the zero-denominator policy and population-transfer assumptions; and preserves the failed FMoW counterexample confirmation.

Original experiments and existing draft files remain unchanged. The document reports the saved runs, with additional mathematical explanations and numerical extraction checks.

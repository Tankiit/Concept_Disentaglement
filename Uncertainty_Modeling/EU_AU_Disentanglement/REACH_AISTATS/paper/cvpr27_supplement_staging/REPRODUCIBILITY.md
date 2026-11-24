# Experimental records and reproducibility

Run commands from the supplement root. The original acquisition notebooks, code, saved configurations, run-specific package versions, selection records and numerical outputs are under the run IDs below. The replay commands create a new working directory for each notebook. All paths in the commands are relative to this file unless they refer to external image data.

| Paper result | Primary record |
| --- | --- |
| Figure 1(b), Figure 8, Tables 6 and 9, Appendix D.2 | `four_channel_full_seed20261111` in `TinyImageNet_REACH_Results_v2full/` and `tiny_imagenet_notebook_source/`. Figure 1(b) uses the known-membership input-arm rows at probe cell 1. The original montage image list for Figure 1(a) is unavailable. |
| Section 3, Figure 5, Table 3 | `workbook/informative_cell_verification_20261117/` (informative visual cell). The earlier constant-input construction is `workbook/acquisition_counterexample_20261029T192454214626Z/`. |
| Three-target constructions and Table 4 | `workbook/three_target_characterization_20261030T144711349004Z/` |
| Study 1, Figures 2, 6 and 7, Tables 5 and 8 | `workbook/source_acquisition_20261029T151111775438Z/`. The eight frozen classifier inputs are from `workbook/dappr_epistemic_sets_seed0_20261029T085507170686Z/`. |
| Image pilots, Figure 3, Tables 10 and 14–16 | `workbook/camelyon17_acquisition_20261030T092422959226Z/` and `workbook/fmow_acquisition_20261030T094932286096Z/` |
| Prevalence searches, Figures 4 and 9, Tables 2, 7 and 11–13 | `workbook/camelyon17_counterexample_20261030T092024988890Z/` and `workbook/fmow_counterexample_20261030T094929883846Z/`. The post-hoc Figure 9 data and sign conventions are in [analysis_outputs/README.md](analysis_outputs/README.md). |

Table 1 summarizes the study protocols. The run configurations specify dataset roles, budgets, seeds and targets. The executed notebooks preserve the acquisition and sampler calls. Each population-level interpretation in the paper uses the corresponding frozen source and evaluation pool.

## Where to find the records for Appendices A–H

| Appendix | Supporting material in this folder |
| --- | --- |
| A. Proofs and acquisition calculations | The proofs and Algorithm 1 appear in the manuscript. The [informative-cell calculation](workbook/informative_cell_verification_20261117/) gives the joint laws and one-step and multi-budget values for Figure 5 and Table 3. The [three-target construction](workbook/three_target_characterization_20261030T144711349004Z/) gives the support hulls for Table 4 and the positive controls. The two prevalence runs contain the independent-Beta update and action-value code. |
| B. Confidence sets and optimization | The derivations appear in the manuscript. The [controlled-study solver](workbook/source_acquisition_20261029T151111775438Z/source_acquisition.py), [Tiny ImageNet implementation](tiny_imagenet_notebook_source/tiny_four_channel_executed.py), and each image pilot’s `wilds_vision.py` and `bounds_definition.json` specify the study-specific constraints, error allocations, and denominator rules. |
| C. Datasets, representations, and protocols | Tables 5–7 draw on the [controlled class roles](table_data/controlled_source_roles.csv), [Tiny ImageNet cell counts](TinyImageNet_REACH_Results_v2full/audit.json), and the two prevalence runs’ `population_rates.csv`. The pilot runs include record and group selections, frozen encoders’ weight hashes, heads, cells, and features. Tiny ImageNet includes the reconstructed selections, checkpoint, and feature cache. |
| D. Acquisition results | Figures 6–8 and Tables 8–10 draw on the controlled run’s `acquisition_summary.csv`, `acquisition_intervals.csv`, `plotted_endpoints.csv`, `oracle_intervals.csv`, and `inclusion_and_solver_checks.csv`; the [Tiny ImageNet exports](TinyImageNet_REACH_Results_v2full/) and [diagnostics](tiny_imagenet_diagnostics/); and both pilots’ per-cell intervals and [2,048-field endpoints](table_data/visual_endpoints_2048.csv). |
| E. Fit-only selection and held-out evaluation | Tables 11–13 use the selected states, population rates and exact one-step risks. Both prevalence runs also contain trajectories. Figure 9 uses the [full searches](analysis_outputs/search_records/), [ranked matches](analysis_outputs/pair_selection_log.csv), and [per-pair held-out advantages](analysis_outputs/pairwise_heldout_advantages.csv). |
| F. Transfer, coverage, and numerical diagnostics | Both pilots contain `source_law_drift.csv` (Table 14), `empirical_containment_summary.csv`, and `empirical_target_diagnostics.csv`. Each prevalence run contains `target_credible_enclosures.csv`, `selected_states.csv` (including order-48 MI), and `probly_check.csv`. The [pilot replay](provenance/pilot_replay.json) records the denominator and containment checks. |
| G. Neural prediction and detection diagnostics | Both image pilots contain `accuracy.csv` (Table 15), `ood_scores.csv` (Table 16), and `image_scores.csv` (per-image inputs). The [prediction check](scripts/check_pilot_predictions.py) verifies the reported source accuracies and tie-correct detection metrics. |
| H. Experimental records and reproducibility | The run index above, [figure](figure_data/source_manifest.json) and [table](table_data/source_manifest.json) manifests, [software versions](provenance/software_versions.json), data digests, and replay commands below identify the inputs to each study. |

## Rebuild from frozen records

```bash
python scripts/check_image_pilots.py
python scripts/check_pilot_predictions.py
python scripts/reproduce_pairwide.py
python scripts/plot_pairwide.py
python scripts/plot_recorded_studies.py
python scripts/check_tiny_checkpoint.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/replay_tiny_laws.py --from-cache
python tiny_imagenet_diagnostics/check_widths.py
python tiny_imagenet_diagnostics/plot_acquisition.py
python scripts/plot_supplement_figures.py
```

The [supplement figure script](scripts/plot_supplement_figures.py) draws the revised Figures 5–9 from the archived two-world action values, controlled-study summaries, Tiny ImageNet records, and pairwise advantages. Its five vector PDFs in `figures/` have prefixes `supp5_` through `supp9_`. Figure 8 uses `supp8_tiny_widths.pdf`.

The pilot check uses NumPy, SciPy and pandas. It re-solves the bounds for every budget, compares the denominator statuses, recomputes the two membership targets from evaluation cells and the anchored event from equally weighted known-source reference populations, and checks every containment indicator. It writes the Table 10 rows to `table_data/visual_endpoints_2048.csv`. The summary curves use equal weights across the four cells and five repeats. The corresponding 24 means and fallback counts at budget 2,048 match Table 10 to its displayed precision. The prediction check uses PyTorch, Probly and scikit-learn with the archived features, heads, per-image scores and cell assignments. The drawing commands use Matplotlib. The per-study source manifests name the recorded inputs to the exported tables and curves.

The Tiny ImageNet cache replay uses the original four-channel code and scikit-learn 1.6.1 to refit the cells on known-source fit features. It checks the resulting audit against the saved audit. Running the same command without `--from-cache` starts with the recorded cell laws. It compares the per-repeat intervals, acquisition summaries, transfer table and exact-input benchmarks to the saved CSVs. The diagnostics command also reproduces the input-arm checks in Appendix D.2.

## Rerun the original acquisition notebooks

Use a Python environment with the packages logged for the study plus IPython and Matplotlib for notebook display. Use a fresh output directory for each invocation. The runner executes the code cells from the archived notebook in order, with its own frozen modules and inputs. The informative-cell command executes the archived rational-arithmetic verifier and its five tests.

```bash
python scripts/replay_study.py informative runs/informative
python scripts/replay_study.py constant runs/constant
python scripts/replay_study.py three_target runs/three_target
python scripts/replay_study.py controlled runs/controlled
python scripts/replay_study.py camelyon17_search runs/cam_search --data-root "$CAMELYON_DATA"
python scripts/replay_study.py fmow_search runs/fmow_search --data-root "$FMOW_DATA"
python scripts/replay_study.py camelyon17_pilot runs/cam_pilot --data-root "$CAMELYON_DATA" --weights "$RESNET18_V1"
python scripts/replay_study.py fmow_pilot runs/fmow_pilot --data-root "$FMOW_DATA" --weights "$RESNET18_V1"
```

Set `CAMELYON_DATA` to the directory of CAMELYON17 parquet shards, `FMOW_DATA` to the directory containing the FMoW WILDS parquet shards, and `RESNET18_V1` to the torchvision ImageNet1K V1 ResNet18 weight file. The source revisions and access information are in each run's `SOURCES.md`. The pilot records include the 18,000 selected IDs per dataset, group roles, reference and evaluation cells, acquired record streams, features, scaler, classifier heads and scores. The prevalence searches use the full retained populations. Their records include initial and acquired observation IDs and the fixed-pool trajectories. The acquisition-level verification runs on CPU. The original pilot feature extraction and head training used CUDA.

The controlled acquisition command starts with the archived dropout predictions. The optional upstream command below regenerates the classifier pilot from the six source image files. The preparation script reads the pinned download locations and SHA-256 digests, and accepts matching local files in the same folder.

```bash
python scripts/prepare_classifier_inputs.py data/controlled
python scripts/replay_study.py controlled_classifier runs/classifier --image-cache data/controlled
```

## Tiny ImageNet image pipeline

The original Colab run used Python 3.13.16 and a Tesla T4. The first executed cell records torch 2.14.1+cu130, torchvision 0.29.1+cu130, SciPy 1.16.3 and scikit-learn 1.6.1. Run `tiny_imagenet_notebook_source/source_notebook.ipynb` in cell order for the original training, caching and acquisition commands. The primary acquisition used the code in `tiny_four_channel_executed.py` after the notebook disabled the minimum-20 fit-support check. The optional Bayesian analysis uses a different target.

The 40-epoch checkpoint and feature cache are in [tiny_imagenet_model/](tiny_imagenet_model/). The [artifact manifest](provenance/tiny_model_cache.json) gives file sizes and SHA-256 digests computed on receipt. All seven primary result files supplied with the model matched the staged exports byte for byte. Training and the source assignment use seed 20261111 as their base seed. Cell fitting and acquisition use 20261029. Source assignment and noise depend on the full image path. The notebook used `/content/tiny_reach_cvpr/data/tiny-imagenet-200`. The reconstructed per-image selection retains the order, roles, class IDs and source assignments, and its ordered labels and source IDs agree with every fit and evaluation cache row. To regenerate it from an extracted copy of the archive, run the following command.

```bash
python scripts/tiny_selection.py /path/to/tiny-imagenet-200
```

The checkpoint head reproduces all 30,000 cached top-1 predictions. To repeat the raw-image spot check, run `python scripts/check_tiny_checkpoint.py --images /path/to/tiny-imagenet-200`. This checks checkpoint features and predictions on the first four images of each source in each split (24 images in total). The replay uses the trained checkpoint and does not rerun classifier training. The original notebook records the dataset URL and verifies its MD5 on download. The redownloaded archive matched that digest.

## Verification records

- [Software versions](provenance/software_versions.json) lists the packages actually recorded in each run. Local run versions were checked against the retained interpreters. The Tiny ImageNet versions come from the executed Colab output. These are partial environment records.
- [Run file provenance](provenance/run_files.json) maps distributed run files to source and exported SHA-256 values. Machine-local paths in text, notebook output and compressed CSVs were shortened for portability. The original artifact manifests cover their original run directories. Two subsequently edited README files in the controlled and constant-input runs were omitted.
- [WILDS input verification](provenance/wilds_input_hash_checks.json) compares independently computed digests against all 53 logged mirror shards and the encoder weights. [Controlled raw inputs](provenance/controlled_classifier_raw_inputs.json) records the six checked source image files. The controlled run's `input_manifest.json` records the frozen predictions, scores, assignments and code. [Tiny ImageNet archive verification](provenance/tiny_image_archive.json) records the original MD5 and the newly computed SHA-256.
- [Pilot replay](provenance/pilot_replay.json) gives status counts, empirical containment at the stated identity budget and maximum endpoint deviations. [Notebook replay comparisons](provenance/study_replays.json) identifies the archived numerical tables checked after re-execution. Image-network retraining was outside those comparisons.
- [Figure export manifest](figure_data/source_manifest.json) and [table export manifest](table_data/source_manifest.json) identify sources and hashes. For transformed files, consult the transformation and the original-to-distributed mapping above.

# REACH: supplementary material (anonymous)

Code, machine-checked proofs, pre-registration files and every result file cited in the paper.

## Layout
- `code/` — all experiment code. Run every command from `code/`.
- `lean/` — Lean 4 / Mathlib project with the machine-checked statements (Appendix B).
- `results/` — the result files each number in the paper is read from (`summary.json`, `response_matrix.csv`, `rows.csv.gz`).

## Paper section -> command -> results
| Paper | Command (from `code/`) | Results |
|---|---|---|
| §5.2 synthetic response matrix | `python main.py response --seeds 128 --set l2 --out ../results/response_l2_s128` | `results/response_l2_s128` (box variant: `response_s128`) |
| CLIP / DINOv2 / ResNet-50 features | `modal run real/modal_extract.py`, `::more`, `::natural`, `::resnet` | written to `cache/features/` |
| §5.3 CIFAR-10 audit (label shift, held-out) | `python main.py audit --arms label_shift held_out --grouping super4` (and `classes`; `--backbone dinov2`) | `results/audit_v3_{clip,dinov2}_{super4,classes}_lsho`, `results/audit_{clip,dinov2}_*_held_out_v2` |
| §5.3 CIFAR-10-C | `python main.py audit --arms corruption --grouping super4` (and `classes`) | `results/audit_v3_clip_{super4,classes}_corruption` |
| §5.4 acquisition under shift (v2) | `python -m real.blindspot_v2 --out ../results/blindspot_v2` | `results/blindspot_v2` (superseded v1: `results/blindspot`) |
| §5.5 exchange rate | `python -m real.exchange --setting synthetic --out ...`; `--setting real --backbone resnet50` (and `clip`) | `results/exchange_{synthetic,resnet50,clip}` |
| App. D CIFAR-100 | `python main.py audit --arms held_out --grouping c100_2 --dataset cifar100 ...` | `results/audit_c100g_*` |
| Tests | `python main.py test` | — |
| Lean | `cd lean && lake exe cache get && lake build && lake env lean Axioms.lean` | — |

## Pre-registration files (SHA-256 recorded in each run's summary or manifest)
| File | SHA-256 prefix | Governs |
|---|---|---|
| `code/train/expectations.py` | 7a97650e | §5.2 |
| `code/real/expectations.py` | add1ccd3 (final state; earlier states listed in App. E) | §5.3 |
| `code/real/blindspot_prereg.py` | e699e72d | superseded v1 of §5.4 |
| `code/real/blindspot_prereg_v2.py` | 2fdecf48 | §5.4 |
| `code/real/exchange_prereg.py` | a70683e3 | §5.5 |

## Third-party assets and licences
| Asset | Use | Licence / terms |
|---|---|---|
| CIFAR-10, CIFAR-100 (Krizhevsky, 2009) | data | no formal licence stated by the creators; cite the 2009 technical report |
| CIFAR-10-C (Hendrycks & Dietterich, 2019) | data | Creative Commons Attribution 4.0 International (Zenodo record 2535967) |
| CIFAR-10.1 (Recht et al., 2018) | data | code MIT; images subject to the Tiny Images dataset terms |
| CLIP ViT-B/32 (Radford et al., 2021) via `open_clip` | frozen encoder | OpenAI CLIP: MIT; open_clip: MIT |
| DINOv2 ViT-S/14 (Oquab et al., 2024) | frozen encoder | Apache License 2.0 |
| ResNet-50 (He et al., 2016), torchvision `IMAGENET1K_V2` weights | frozen encoder | torchvision: BSD 3-Clause |
| Mathlib | Lean library | Apache License 2.0 |

No new data were collected; no human subjects were involved.

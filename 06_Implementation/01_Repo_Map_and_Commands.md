---
title: Repo Map and Commands
type: implementation
status: active
tags:
  - implementation
  - repo-map
---

# 🛠️ Repo Map and Commands

## Layout (vault root = git repo)

```
REACH/
├── Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS/   <-- project code root
│   ├── main.py            single entry point (train/separation/matched/conditionals/evaluate/response/audit)
│   ├── gates.py           ordered kill gates — see [[02_Decision_Gates]]
│   ├── README.md          canonical runbook
│   ├── COLLABORATOR_PLAN.md  sharding protocol, timings, owner-work split
│   ├── requirements.txt
│   ├── paper/             AISTATS 2027: main.tex, sections/, refs.bib, builds, supplement zip
│   ├── lean/              Reach.lean, Axioms.lean (machine-checked, standard axioms only)
│   ├── test/              precision, slopes, inference, evaluate
│   ├── original/          predictions, ball, experiments, inference (reference implementations)
│   └── real/              cached-feature pipeline + prereg registry — [[03_Allocation_Dose_and_Preregisters]]
├── latent_subspace.py / attention_prompt.py / wilds_expts.py   adjacent EU/AU thread, NOT REACH evidence
├── backup/                snapshots, do not cite
└── 00_Dashboard.md …      this vault
```

## Environment & Tests

```bash
python -m pip install -r requirements.txt   # in REACH_AISTATS/
python main.py test        # 18+ numerical/integration tests, ~1 s
python gates.py            # ordered kill gates, exit code = verdict
```

Run every command from `REACH_AISTATS/`. Results land as `rows.csv` + `summary.json` per output dir; `train` writes the pilot bundle + a SHA-256 manifest of every source file.

## Sharding Protocol (from COLLABORATOR_PLAN.md)

```bash
python main.py train --seeds 128 --seed-start 0    --out results/A_shard0
python main.py train --seeds 128 --seed-start 1000 --out results/A_shard1   # different machine
```

- **Disjoint seed ranges are mandatory.** Pool = concatenate `rows.csv`, re-analyse from Python (`train.experiments`, `train.predictions`).
- Two shards agreeing on every decision label is the replication check.
- Reference timings (one laptop core): `train --seeds 128` ≈ 80 s; 256 ≈ 2.7 min; `conditionals` ≈ 19 s; `matched` ≈ 16 s; tests ≈ 1 s.

## Vault ↔ Code Conventions

- Notes in this vault reference code by path from `REACH_AISTATS/`.
- Update [[00_Dashboard]] status board after any run that changes a gate or a prereg decision.
- Never edit `original/` — it is the reference implementation baseline.

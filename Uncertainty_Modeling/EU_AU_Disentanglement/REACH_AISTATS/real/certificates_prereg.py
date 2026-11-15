# certificates_prereg.py -- FROZEN 2026-11-10, before any run of the certificate
# pipeline on the datasets below. The confirmation gate of the 10 Nov plan (CVPR2027_PLAN
# §4): the certificate paper is submittable only if P1 holds everywhere in this grid.
#
# This file records the protocol and the predictions. The runner is
# certificates_multidata.py, which refuses to run unless this file's sha256 matches the
# value recorded in results/certificates_multidata/prereg.sha256 (written by
# `python real/certificates_prereg.py --freeze`). Amendments are dated entries appended
# below -- never in-place edits.
#
# PROTOCOL (frozen)
# -----------------
# Backbone: timm wide_resnet50_2 (ImageNet pretrained, num_classes=0), the same model
#   as every existing cache. New extractions follow example_wrn50.extract_features
#   verbatim: timm.data.create_transform(**resolve_model_data_config(model)), per-image
#   RGB PIL, batch 256. Existing npz caches are reused untouched.
# Features: rows L2-normalized after load (matches the CIFAR-10/CLIP protocol).
# Pool per dataset: stratified subsample (np.random.default_rng(0), per-class draw)
#   to 10,000 rows of the official test split if it has >= 10,000 rows, else of all
#   official splits concatenated. Datasets whose existing cache is exactly 10k test
#   rows are used as-is; SVHN's 26k test cache is subsampled to 10k.
# Worlds: make_oss(pool, seed=ws) for ws in 0..4 (70/20/10 split, DynamicOSS).
#   Budgets: kuc=1, uuc_val=2, uuc_test=5 for >=10 classes (kkc = n-8, DISCLOSED:
#   kkc varies: 2 for 10-class sets, 18 EMNIST, 92 CIFAR-100); 9 classes (pathmnist)
#   (1,2,4); 8 classes (bloodmnist) (1,2,3). kkc=2 in both small cases.
# Head: epistemic_sets.train_kp1(seed=0, epochs=300) -- identical to the delivered
#   CIFAR-10/CLIP runs. MC stack: T=30, fixed mask sequence (mc_stack).
# Certificates: level 1/2 on the unknown column; certainly-novel = lower bound >= 1/2,
#   certainly-known = upper bound <= 1/2 (certificate_audit.conventions).
# Near/far: credal_head.near_far_tags (k=5, q=0.95, tau-normalized kNN in the pooled
#   feature space), tags classes, applied to rows.
# Scores reported: P_unknown, set_median, EU_bald, neg_msp(K+1), width_mean; pooled
#   AUROC pos uuc | neg kkc. Absorbed rows: P_unknown <= 0.01.
#
# DATASET GRID (frozen)
# ---------------------
#   mnist, fashionmnist, svhn, cifar100, cifar10  (cached WRN50 test features;
#     cifar10 = backbone-swap arm vs the CLIP reference)
#   bloodmnist (8), pathmnist (9), organamnist (11), organcmnist (11), organsmnist (11)
#     -- ~/.medmnist, all official splits
#   emnist_letters (26) -- torchvision download
#   eurosat (10) -- torchvision download; satellite land-use stand-in for FMoW
#   NOT RUN, with reasons: camelyon17 (not on disk; hospital-source structure is not a
#     class taxonomy for this protocol -- pathmnist is the histopathology stand-in);
#     fmow (not on disk; ~90 GB -- eurosat is the satellite stand-in).
#
# PREDICTIONS (frozen; scored by the runner)
# ------------------------------------------
#   P1 (THE GATE): zero false novelty certificates on kkc test rows in every world of
#      every dataset. Failure anywhere -> soundness does not generalize and, per the
#      10 Nov gate, the certificate paper is not submittable this cycle.
#   P2: at least one world with certified-novel coverage on uuc outside [0.10, 0.80]
#      (the coverage range does not simply reproduce CIFAR-10/CLIP's 0.13-0.72).
#   P3: certified-novel rate on far classes > on near classes in every world where both
#      tags exist.
#   P4: >= 95% of absorbed rows receive the certainly-known certificate in every world.
#   P5: width_mean - EU_bald pooled AUROC <= +0.01 in every world (the universal width
#      negative extends).
#   Stop rule: any P1 failure stops nothing (all worlds still run, disclosed), but the
#   gate verdict is FAIL and no post-hoc re-scoping of P1 is permitted.
PRED = {
    "frozen": "2026-11-10, before any pipeline run on these datasets",
    "P1_gate_zero_false_certificates_everywhere": True,
    "P2_coverage_escapes_cifar10_band": True,
    "P3_far_gt_near_certified_every_world": True,
    "P4_absorbed_certknown_ge_0.95_every_world": True,
    "P5_width_minus_bald_le_0.01_every_world": True,
}

# AMENDMENTS (append-only)
# ------------------------
AMENDMENTS = []

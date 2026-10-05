"""Pre-registered signs for the real-shift audit (written 2026-10-02, before the Dirichlet,
label-noise and delta=0 runs; earlier skew-prevalence runs were seen, so those cells are not
blind and this file says so).

Key: (arm, q_mode, delta, axis, component, region). Values as in train/expectations.py:
'negative', 'positive', '~0', 'exact' (invariance by construction, reported not tested), None.

Differences from the synthetic study, stated in advance:
- On real data labels estimate the source rows the ball is built from, so m and eta CROSS-LOAD
  on W_cls (README: report cross-effects separately). Predicted here as signed, not as ~0.
- W_label never reads the log: exact in n.
- delta = 0 (held-out classes absent): the novel weight's truth is 0, it is identified because
  q is outside the known hull (s > 0), so W_unk contracts on known cells. delta = 0.1: same sign.
- The outlier cell carries little known mass; no prediction there except W_label exact in n.
"""

EXPECT = {}
for region in ('on_support',):
    EXPECT.update({
        # Label shift, Dirichlet prevalence.
        ('label_shift', 'none', 0., 'n', 'W_cls', region): 'negative',
        ('label_shift', 'none', 0., 'n', 'W_label', region): 'exact',
        ('label_shift', 'none', 0., 'n', 'W_joint', region): 'negative',
        ('label_shift', 'none', 0., 'n', 'EMQ_cls', region): 'negative',
        ('label_shift', 'none', 0., 'n', 'Ens_MI', region): 'negative',
        ('label_shift', 'none', 0., 'm', 'W_cls', region): 'negative',     # cross-load
        ('label_shift', 'none', 0., 'm', 'W_label', region): 'negative',
        ('label_shift', 'none', 0., 'm', 'W_joint', region): 'negative',
        ('label_shift', 'none', 0., 'm', 'Ens_MI', region): 'negative',
        ('label_shift', 'none', 0., 'eta', 'W_label', region): 'positive',
        ('label_shift', 'none', 0., 'eta', 'W_joint', region): 'positive',
        ('label_shift', 'none', 0., 'eta', 'W_cls', region): 'positive',   # cross-load
        ('label_shift', 'none', 0., 'eta', 'EMQ_cls', region): None,
        # Held-out classes, correct novel row.
        ('held_out', 'oracle', .1, 'n', 'W_unk', region): 'negative',
        ('held_out', 'oracle', 0., 'n', 'W_unk', region): 'negative',
        ('held_out', 'oracle', .1, 'n', 'W_label', region): 'exact',
        # Declared uniform novel row: exploratory.
        ('held_out', 'uniform', .1, 'n', 'W_unk', region): None,
        ('held_out', 'uniform', 0., 'n', 'W_unk', region): None,
    })
EXPECT[('label_shift', 'none', 0., 'n', 'W_label', 'outlier_cell')] = 'exact'

FAMILY_ALPHA = .05

# --- Added 2026-10-02 evening, before any DINOv2 or CIFAR-100 feature existed. ---------------
# Second backbone (DINOv2 ViT-S/14): judged against the SAME label_shift / held_out cells above,
# unchanged. Corruption arm not run for this backbone.
#
# CIFAR-100 limit of Proposition C. Known: four trees (maple 47, oak 52, palm 56, pine 59) and
# five vehicles (bicycle 8, bus 13, motorcycle 48, pickup truck 58, train 90).
#   c100_in : novel = willow tree (96), a sibling of known classes. Its component should lie near
#             the known hull (small s), so novelty is weakly identified: W_unk stays wide.
#   c100_out: novel = whale (95), unrelated to every known class (larger s): W_unk contracts.
# Prediction for c100_in is NOT "low W_unk": with the novel component supplied but inside the
# hull, the width stays wide (unidentified), as in the synthetic 'confounded' world. Registered:
EXPECT.update({
    ('c100_out', 'oracle', .1, 'n', 'W_unk', 'on_support'): 'negative',
    ('c100_in', 'oracle', .1, 'n', 'W_unk', 'on_support'): None,   # sign exploratory; see CONTRASTS
})
# Cross-arm contrasts, checked by hand from the two response matrices:
CONTRASTS = [
    'separation: s(c100_in) < s(c100_out)',
    'attenuation: slope W_unk(c100_in) > slope W_unk(c100_out)   (less negative)',
    'width: W_unk(c100_in) > W_unk(c100_out) at the largest n',
    'validity: W_unk covers the truth in both arms',
]

# --- DEVIATION, recorded 2026-10-02 after the first CIFAR-100 runs (results seen) ---------------
# With nine singleton known sources and 400 reference images per class, every CIFAR-100 run
# failed the resolution gate (0.67 CLIP, 0.73 DINOv2 at n=500), so those runs are not reported.
# Rerun with grouping 'c100_2': trees (47 52 56 59) and vehicles_1 (8 13 48 58 90) as two
# sources. The registered predictions and CONTRASTS above are unchanged. Also after seeing DINOv2
# held-out coverage of plug-in W_unk (0.875-0.89), a joint novelty interval (W_unk_joint) was added;
# it, not plug-in W_unk, carries the coverage claim. Both changes are disclosed in the paper.

# --- Added 2026-10-02 ~23:25 CEST after the five-seat review, before the baseline reruns. --------
# One-seed smoke test of the new baselines was seen (CLIP super4, n=16000, seed 3: joint 0.59,
# constraint region 0.56, two-stage bootstrap 0.07; joint and region covered, bootstrap 0.8).
# Registered: (i) the constraint-region baseline (after Li, Han & Ma 2026) is valid by construction
# and should cover >= 0.95 on clean label shift; (ii) the two-stage bootstrap has no guarantee, no
# direction registered; (iii) NO direction registered for joint vs region width (the smoke test
# already showed the region narrower on one seed). Corruption coverage is now reported both among
# answered runs and unconditionally (an empty ball counts as abstention, not as coverage).
BASELINE_EXPECT = {'mac_coverage_clean_label_shift': '>=0.95', 'EMQ2_coverage': None,
                   'joint_minus_region_width': None}

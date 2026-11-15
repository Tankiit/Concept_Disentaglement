# certificate_audit.py -- CVPR certificate-pair validation (items 1-5 of the 10 Nov
# proposal), predictions frozen below BEFORE any computation.
#
#   1 (kill test)  certified-novel coverage vs a P_unknown threshold at zero false
#                  positives on kkc VAL rows (threshold discipline: val only);
#                  structural P_unknown >= 0.5 comparator as disclosure.
#   2 (near/far)   certified coverage split by near/far tag of the uuc class.
#   3 (T sweep)    certificates at T = 10/30/100 from nested dropout prefixes.
#   4 (hull vs KL) sampled-hull certificates vs tau-relative-likelihood certificates
#                  (delivered credal_p_lo/hi_unk_t090 columns). Direct novel<->known
#                  conflicts are impossible by construction (p-bar sandwich), so the
#                  empirical content is the undecided<->definite flips.
#   5 (absorbed)   which certificate the absorbed rows (P_unknown <= 0.01) receive.
#
# Replays the delivered epistemic_sets heads deterministically (asserted against the
# delivered rows before any scoring). Reads results/, writes results/certificate_audit/.
# Run from REACH_AISTATS:  OMP_NUM_THREADS=1 python certificate_audit.py
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import epistemic_sets as es

OUT = Path("results/certificate_audit")
DELIVERED = Path("results/credal_head/rows.csv")
FEAT = "cache/features/cifar10_test.npz"

# ---------------------------------------------------------------------------
# PRED -- frozen 2026-11-10, before any certificate was computed.
# ---------------------------------------------------------------------------
PRED = {
    "K1_kill_test": {
        "tie_within_0.01": False,
        "cert_more_conservative": "5/5 worlds: certificate coverage < val-threshold coverage",
        "cert_sound_5of5": True,
        "threshold_slips_on_test": ">=1/5 worlds: val-threshold has FPR>0 on kkc test rows",
        "kill_rule": "if |coverage difference| <= 0.01 in every world, the certificate "
                     "reduces to a calibratable threshold and needs a different "
                     "justification (collaborator rule)",
    },
    "N2_near_far": {
        "far_certified_more": "5/5 worlds: P(cert-novel | far) > P(cert-novel | near)",
        "undecided_enriched_near": "5/5 worlds: near share among undecided uuc rows > "
                                   "near base rate of that world",
    },
    "T3_sweep": {
        "monotone": "exact by nesting (prefixes share the mask sequence): "
                    "coverage(10) >= coverage(30) >= coverage(100)",
        "no_collapse": ">=4/5 worlds: coverage(100) >= 0.4 * coverage(10)",
        "soundness_all_T": "0 false novel certificates on kkc rows at every T (exact "
                           "by nesting: the certificate set only shrinks with T)",
    },
    "C4_hull_vs_tau": {
        "no_direct_conflicts": "exact: hull cert-novel => all passes >= 1/2 => p_bar >= "
                               "1/2, while tau cert-known => p_bar < 1/2 (and vice "
                               "versa); novel<->known flips are impossible",
        "differ": ">=3/5 worlds: certificates differ (incl. undecided flips) on >=5% "
                  "of uuc rows",
        "direction": "5/5 worlds: tau-set cert-novel coverage >= hull coverage "
                     "(disagreement can only remove certificates: hull lower = min "
                     "over passes, tau lower depends only on p_bar)",
        "tau_sound": "5/5 worlds: tau-set also has 0 false novel certificates on kkc",
    },
    "A5_absorbed": {
        "known_cert": "5/5 worlds: >=95% of absorbed rows (P_unknown <= 0.01) receive "
                      "the certainly-known certificate (set_upper <= 0.5)",
        "near_dominate": "5/5 worlds: >=60% of absorbed rows are near-tagged",
    },
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def certificates(lo, hi, level=0.5):
    """novel if every sampled model is above level; known if every one is below."""
    return np.where(lo >= level, "cert_novel", np.where(hi <= level, "cert_known", "undecided"))


def share(mask, arr, val):
    m = mask & (arr == val)
    return float(m.sum() / mask.sum()) if mask.sum() else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pred.json").write_text(json.dumps(PRED, indent=2))
    print("PRED frozen to", OUT / "pred.json", "-- before any certificate computation")

    rows = pd.read_csv(DELIVERED)
    torch.set_num_threads(1)
    per_row, agg = [], []
    for ws in range(5):
        world = es.make_oss(FEAT, kuc=1, uuc_val=2, uuc_test=5, seed=ws)
        head, lut, acc = es.train_kp1(world, seed=0, epochs=300)
        K = len(world["kkc"])
        ref = rows[rows.world_seed == ws].reset_index(drop=True)
        Xte = torch.from_numpy(world["X"][world["idx"]["test"]]).float()
        yte = world["y"][world["idx"]["test"]]
        S100 = es.mc_stack(head, Xte, T=100)                     # prefixes are nested
        pb = S100[:30].mean(0)[:, K]
        assert np.allclose(pb, ref.P_unknown.to_numpy(), rtol=1e-4, atol=1e-6), \
            f"replay drift, world {ws}"
        lo100, hi100 = S100.min(0)[:, K], S100.max(0)[:, K]

        known = np.isin(yte, world["kkc"])
        uuc = np.isin(yte, world["uuc"])
        near = ref.novel_tag.to_numpy() == "near"
        far = ref.novel_tag.to_numpy() == "far"

        # ---- item 3: certificates at T = 10/30/100 (nested prefixes) ------------
        certs = {f"T{t}": certificates(S100[:t].min(0)[:, K], S100[:t].max(0)[:, K])
                 for t in (10, 30, 100)}
        # ---- item 4: tau-set certificates (delivered closed-form bounds) -------
        tau = certificates(ref.credal_p_lo_unk_t090.to_numpy(),
                           ref.credal_p_hi_unk_t090.to_numpy())
        # ---- item 1: kill test --------------------------------------------------
        Xva = torch.from_numpy(world["X"][world["idx"]["val"]]).float()
        yva = world["y"][world["idx"]["val"]]
        pv = es.mc_stack(head, Xva, T=30).mean(0)[:, K]
        thr = float(pv[np.isin(yva, world["kkc"])].max())        # val-knowns only
        thr_cov_uuc = float((pb[uuc] >= thr).mean())
        thr_fpr_test = float((pb[known] >= thr).mean())
        struct_cov = float((pb[uuc] >= 0.5).mean())              # disclosure
        cert_cov = share(uuc, certs["T30"], "cert_novel")
        cert_fpr = share(known, certs["T30"], "cert_novel")

        a = dict(
            world_seed=ws, threshold_val=thr,
            cert_cov_uuc=cert_cov, cert_fpr_kkc=cert_fpr,
            thr_cov_uuc=thr_cov_uuc, thr_fpr_kkc=thr_fpr_test,
            struct05_cov_uuc=struct_cov,
            cov_T10=share(uuc, certs["T10"], "cert_novel"),
            cov_T100=share(uuc, certs["T100"], "cert_novel"),
            tau_cov_uuc=share(uuc, tau, "cert_novel"),
            tau_fpr_kkc=share(known, tau, "cert_novel"),
            tau_known_share_kkc=share(known, tau, "cert_known"),
            flips_hull_tau_uuc=float((certs["T30"] != tau)[uuc].mean()),
            direct_conflicts=int(((certs["T30"] == "cert_novel") & (tau == "cert_known")
                                  | ((certs["T30"] == "cert_known") & (tau == "cert_novel"))).sum()),
            # item 2
            certnovel_far=share(uuc & far, certs["T30"], "cert_novel"),
            certnovel_near=share(uuc & near, certs["T30"], "cert_novel"),
            certknown_far=share(uuc & far, certs["T30"], "cert_known"),
            certknown_near=share(uuc & near, certs["T30"], "cert_known"),
            undecided_near_share=float(near[uuc & (certs["T30"] == "undecided")].mean()),
            near_base=float(near[uuc].mean()),
            # item 5
            absorbed_share=float((pb[uuc] <= 0.01).mean()),
            absorbed_certknown=float((certs["T30"][uuc][pb[uuc] <= 0.01] == "cert_known").mean()),
            absorbed_near=float(near[uuc][pb[uuc] <= 0.01].mean()),
            certknown_share_kkc=share(known, certs["T30"], "cert_known"),
        )
        agg.append(a)
        pr = pd.DataFrame(dict(
            world_seed=ws, row=ref.row, group=np.where(known, "kkc", np.where(
                np.isin(yte, world["kuc"]), "kuc", "uuc")),
            novel_tag=ref.novel_tag, P_unknown=pb, absorbed=pb <= 0.01,
            cert_T10=certs["T10"], cert_T30=certs["T30"], cert_T100=certs["T100"],
            cert_tau=tau))
        per_row.append(pr)
        print(f"world {ws}: cert-novel uuc {cert_cov:.3f} (far {a['certnovel_far']:.3f} / "
              f"near {a['certnovel_near']:.3f}) | threshold {thr:.4f} cov {thr_cov_uuc:.3f} "
              f"fpr {thr_fpr_test:.3f} | T10/100 {a['cov_T10']:.3f}/{a['cov_T100']:.3f} | "
              f"tau cov {a['tau_cov_uuc']:.3f} flips {a['flips_hull_tau_uuc']:.3f} | "
              f"absorbed {a['absorbed_share']:.3f} -> known-cert {a['absorbed_certknown']:.3f}")

    A = pd.DataFrame(agg)
    A.to_csv(OUT / "certificates.csv", index=False)
    pd.concat(per_row, ignore_index=True).to_csv(OUT / "rows_certs.csv", index=False)

    # ---- PRED scorecard --------------------------------------------------------
    v = {
        "K1_kill_test": {
            "tie_within_0.01": bool((A.cert_cov_uuc - A.thr_cov_uuc).abs().max() <= 0.01),
            "cert_more_conservative_5of5": bool((A.cert_cov_uuc < A.thr_cov_uuc).all()),
            "cert_sound_5of5": bool((A.cert_fpr_kkc == 0).all()),
            "threshold_slips_ge1of5": bool((A.thr_fpr_kkc > 0).any()),
        },
        "N2_near_far": {
            "far_certified_more_5of5": bool((A.certnovel_far > A.certnovel_near).all()),
            "undecided_enriched_near_5of5": bool((A.undecided_near_share > A.near_base).all()),
        },
        "T3_sweep": {
            "monotone_5of5": bool((A.cov_T10 >= A.cert_cov_uuc).all()
                                  and (A.cert_cov_uuc >= A.cov_T100).all()),
            "no_collapse_ge4of5": bool((A.cov_T100 >= 0.4 * A.cov_T10).sum() >= 4),
            "sound_all_T": bool(all((pr[pr.group == "kkc"][f"cert_T{t}"] == "cert_novel").sum() == 0
                                     for pr in per_row for t in (10, 30, 100))),
        },
        "C4_hull_vs_tau": {
            "direct_conflicts_total": int(A.direct_conflicts.sum()),
            "differ_ge5pct_ge3of5": bool((A.flips_hull_tau_uuc >= 0.05).sum() >= 3),
            "tau_cov_ge_hull_5of5": bool((A.tau_cov_uuc >= A.cert_cov_uuc).all()),
            "tau_sound_5of5": bool((A.tau_fpr_kkc == 0).all()),
        },
        "A5_absorbed": {
            "known_cert_ge95_5of5": bool((A.absorbed_certknown >= 0.95).all()),
            "near_dominate_ge60_5of5": bool((A.absorbed_near >= 0.60).all()),
        },
    }
    score = {"pred": PRED, "got": v}
    (OUT / "summary.json").write_text(json.dumps(score, indent=2, default=str))
    print("\n=== PRED scorecard ===")
    for item, d in v.items():
        for k, got in d.items():
            print(f"  {item:14s} {k:32s} {got}")

    # ---- figure (item 5's certificate-pair figure) ------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.6), sharey=True)
    for ax, ws in zip(axes, range(5)):
        pr = per_row[ws]
        for gi, (g, tag) in enumerate([("kkc", "known rows"), ("uuc", "uuc near"),
                                       ("uuc", "uuc far")]):
            m = (pr.group == g) if g == "kkc" else (
                (pr.group == "uuc") & (pr.novel_tag == ("near" if tag.endswith("near") else "far")))
            sub = pr[m]
            y0 = gi
            shares = [(sub.cert_T30 == "cert_known").mean(),
                      (sub.cert_T30 == "undecided").mean(),
                      (sub.cert_T30 == "cert_novel").mean()]
            colors = ["#4c72b0", "#c9c9c9", "#c44e52"]
            left = 0
            for s, c, lab in zip(shares, colors,
                                 ["certainly known", "undecided", "certainly novel"]):
                ax.barh(y0, s, left=left, color=c, label=lab if (gi == 0 and ws == 0) else None)
                left += s
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["known rows", "uuc near", "uuc far"])
        ax.set_xlim(0, 1); ax.set_title(f"world {ws}")
        ax.invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.suptitle("Envelope certificates (T=30): certainly known / undecided / certainly novel", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "certificates.png", dpi=150, bbox_inches="tight")

    manifest = dict(script_sha256=sha(Path(__file__)), delivered_rows_sha256=sha(DELIVERED),
                    epistemic_sets_sha256=sha(Path("epistemic_sets.py")),
                    replay="deterministic; P_unknown asserted against delivered rows "
                           "(rtol 1e-4) before scoring",
                    note="T=10/30/100 use nested dropout prefixes of one T=100 stack "
                         "(mc_stack reseeds torch to 0, so prefixes coincide by construction)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()

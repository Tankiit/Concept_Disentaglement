"""One-seed acquisition/accounting smoke test; frozen CNN cells, real parity."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import solver

HERE = Path(__file__).resolve().parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
H = module('identity', HERE.parent/'camouflage_identities_2026-10-21/run.py')
TV = module('tv_solver', HERE.parent/'candidate_tv_2026-10-22/solver.py')
solver.linprog = TV.solve
BUDGETS = [0, 64, 128, 256]
ARMS = ['audit', 'known', 'candidate', 'balanced']
# Eight families for each of sixteen predeclared settings, union bound <= .05.
ALPHA = .05 / (len(BUDGETS)*len(ARMS)*8)

def bounds(counts, alpha=ALPHA):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    rate = np.divide(counts, n, out=np.zeros_like(counts), where=n>0)
    half = np.sqrt(np.log(2*counts.size/alpha)/(2*np.maximum(n, 1)))
    return np.where(n>0, np.clip(rate-half, 0, 1), 0), np.where(n>0, np.clip(rate+half, 0, 1), 1)

def allocation(arm, budget):
    out = dict(audit=0, known=0, candidate=0)
    if arm == 'balanced':
        for i, name in enumerate(out):
            out[name] = budget//3 + (i < budget%3)
    else:
        out[arm] = budget
    return out

def counts(frame):
    out = np.zeros((11, 11), dtype=int)
    np.add.at(out, (frame.source.to_numpy(dtype=int), frame.cell.to_numpy(dtype=int)), 1)
    return out

def ratio(lo, hi, positive):
    """Conservative ratio range; coordinate-box normalization can only tighten."""
    positive = np.asarray(positive, dtype=bool)
    if hi.sum() == 0 or lo.sum() == 0:
        return 0., 1., 'zero_mass_policy'
    a, b = lo[positive].sum(), hi[positive].sum()
    lower = a / (a + hi[~positive].sum())
    upper = b / (b + lo[~positive].sum())
    return float(lower), float(upper), 'box_ratio'

def main():
    path = HERE.parent/'candidate_audit_2026-10-22/image_cells.csv'
    images = pd.read_csv(path)
    images = images[images.dataset.isin(['mnist', 'fashion'])].copy()
    images['source'] = np.where(images.dataset == 'mnist', images.label, 10)
    images['parity'] = np.where(images.source < 10, images.source % 2, -1)
    rng = np.random.default_rng(20261031)
    nk = np.floor(3200*np.arange(1, 11)/55).astype(int)
    nk[-1] += 3200-nk.sum()
    w = np.r_[nk, 800]/4000
    audit_sources = rng.choice(11, size=256, p=w)
    audit_n = np.bincount(audit_sources, minlength=11)
    roles = []
    for source in range(11):
        pool = images[images.source == source]
        pool = pool.iloc[rng.permutation(len(pool))].copy()
        sizes = [('initial', 200 if source < 10 else 512),
                 ('audit', int(audit_n[source])),
                 ('reference', 26 if source < 10 else 256),
                 ('log', int(nk[source]) if source < 10 else 800),
                 ('probe', 100 if source < 10 else 1000)]
        assert sum(n for _, n in sizes) <= len(pool), (source, len(pool))
        offset = 0
        for role, n in sizes:
            sub = pool.iloc[offset:offset+n].copy()
            sub['role'] = role
            sub['order'] = np.arange(n)
            roles.append(sub)
            offset += n
    manifest = pd.concat(roles, ignore_index=True)
    # Preserve iid source sequence, with without-replacement images within source.
    for source in range(11):
        mask = (manifest.source == source)&(manifest.role == 'audit')
        manifest.loc[mask, 'order'] = np.flatnonzero(audit_sources == source)
    mask = (manifest.source < 10)&(manifest.role == 'reference')
    manifest.loc[mask, 'order'] = manifest.loc[mask, 'order']*10+manifest.loc[mask, 'source']
    assert not manifest.duplicated(['dataset', 'index']).any()
    manifest.to_csv(HERE/'image_assignments.csv', index=False)
    # Only cell IDs are provided for the unannotated log.
    log_counts = np.bincount(manifest.loc[manifest.role == 'log', 'cell'], minlength=11)
    pl, ph = bounds(log_counts)
    initial = manifest[manifest.role == 'initial']
    A = counts(images)/counts(images).sum(1, keepdims=True)
    theta = np.arange(10)%2
    cfg = dict(seed=20261031, status='one-seed CNN smoke test', budgets=BUDGETS,
               arms=ARMS, alpha_per_family=ALPHA, families=8, overall_alpha=.05,
               weights=w.tolist(), laws=A.tolist(), theta=theta.tolist(),
               cells_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               cls_source='cell index, with fixed source 0 for outlier',
               representation='existing frozen CNN; pixel/PCA comparison pending',
               baseline_cost=2512, added_cost_unit='verified image; catalog access assumed')
    (HERE/'config.json').write_text(json.dumps(cfg, indent=2))
    records, purchases, evidence, decisions = [], [], [], []
    for arm in ARMS:
        for budget in BUDGETS:
            alloc = allocation(arm, budget)
            acquired = []
            for action, n in alloc.items():
                if action == 'audit':
                    pool = manifest[manifest.role == 'audit']
                else:
                    pool = manifest[(manifest.role == 'reference') &
                                    ((manifest.source < 10) if action == 'known' else (manifest.source == 10))]
                part = pool.sort_values('order').iloc[:n].copy()
                assert len(part) == n
                part['action'] = action
                acquired.append(part)
                for row in part.itertuples():
                    purchases.append(dict(arm=arm, budget=budget, action=action, dataset=row.dataset,
                                          index=row.index, source=row.source, cell=row.cell,
                                          parity=row.parity, cost=1))
            bought = pd.concat(acquired)
            assert len(bought) == budget and not bought.duplicated(['dataset', 'index']).any()
            refs = pd.concat([initial, bought[bought.action != 'audit']])
            audits = bought[bought.action == 'audit']
            rc, ac = counts(refs), counts(audits)
            lo, hi = bounds(rc)  # Family 1: catalog source-cell rows.
            al, ah = bounds(ac)  # Family 2: audit rows, conditional on source counts.
            lo, hi = np.maximum(lo, al), np.minimum(hi, ah)
            wl, wh = bounds(ac.sum(1))  # Family 3: random audited source proportions.
            jl, jh = bounds(ac.reshape(-1))  # Family 4: audited joint probabilities.
            jl, jh = jl.reshape(11, 11), jh.reshape(11, 11)
            # Family 5 is the baseline log; family 6 is a balanced anchor sample.
            m = int(rc[:10].sum(1).min())
            balanced = pd.concat([refs[refs.source == s].iloc[:m] for s in range(10)])
            bc = counts(balanced)
            pair = np.stack([bc[:10][theta == 0].sum(0), bc[:10][theta == 1].sum(0)])
            bl, bh = bounds(pair.reshape(-1))
            bl, bh = bl.reshape(2, 11), bh.reshape(2, 11)
            # Families 7/8 reserved. Dependencies are handled by union bounds.
            evidence.append(dict(arm=arm, budget=budget, allocation=alloc,
                                 reference_counts=rc.tolist(), audit_counts=ac.tolist(),
                                 log_counts=log_counts.tolist(), anchor_counts=pair.tolist()))
            for cell in range(11):
                src = cell if cell < 10 else 0
                joint = w*A[:, cell]
                truths = dict(W_cls=joint[src]/joint[:10].sum(), W_unk=joint[-1]/joint.sum(),
                              W_label=theta@A[:10, cell]/A[:10, cell].sum())
                direct = dict(W_cls=ratio(jl[:10, cell], jh[:10, cell], np.arange(10)==src),
                              W_unk=ratio(jl[:, cell], jh[:, cell], np.arange(11)==10),
                              W_label=ratio(bl[:, cell], bh[:, cell], [False, True]))
                reach = dict(W_cls=solver.interval(lo, hi, pl, ph, wl, wh, cell, src, jl, jh),
                             W_unk=solver.interval(lo, hi, pl, ph, wl, wh, cell, None, jl, jh),
                             W_label=H.label_interval(lo[:10], hi[:10], theta, theta, cell))
                # Use the same available anchor information in modeled intervals.
                l, u, status = reach['W_label']
                dl, du, _ = direct['W_label']
                reach['W_label'] = (max(l, dl), min(u, du), status)
                for method, result in [('reach_direct_law', reach), ('direct_target', direct)]:
                    for target, (lower, upper, status) in result.items():
                        truth = truths[target]
                        records.append(dict(arm=arm, budget=budget, method=method, cell=cell,
                                            source=src, target=target, lower=lower, upper=upper,
                                            width=upper-lower, status=status, truth=truth,
                                            covers=lower-1e-6 <= truth <= upper+1e-6))
                        for threshold in [.05, .1, .2]:
                            choice = 1 if lower >= 1-threshold else (0 if upper <= threshold else -1)
                            # A negative source-membership decision is not a digit prediction.
                            if target == 'W_cls' and choice == 0:
                                choice = -1
                            if status == 'empty_set':
                                choice = -1
                            probe = manifest[(manifest.role == 'probe') & (manifest.cell == cell)]
                            if target != 'W_unk':
                                probe = probe[probe.source < 10]
                            outcome = ((probe.source == 10) if target == 'W_unk' else
                                       ((probe.source == src) if target == 'W_cls' else (probe.parity == 1)))
                            # Probe stratification differs from deployment; use declared population weights.
                            popw = np.r_[np.ones(10)/10, 0.] if target == 'W_label' else w.copy()
                            if target == 'W_cls':
                                popw[-1] = 0; popw /= popw.sum()
                            base_sizes = np.r_[np.full(10, 100), 1000]
                            mass = popw[probe.source.to_numpy()]/base_sizes[probe.source.to_numpy()]
                            decisions.append(dict(arm=arm, budget=budget, method=method, target=target,
                                                  cell=cell, threshold=threshold, choice=choice,
                                                  probe_mass=float(mass.sum()),
                                                  certified_mass=float(mass.sum()) if choice >= 0 else 0.,
                                                  error_mass=float(mass[outcome.to_numpy()!=choice].sum()) if choice >= 0 else 0.,
                                                  false_statement=bool(choice >= 0 and (truth < 1-threshold if choice == 1 else truth > threshold))))
            print(arm, budget, 'complete', flush=True)
    df = pd.DataFrame(records)
    df.to_csv(HERE/'intervals.csv', index=False)
    pd.DataFrame(purchases).to_csv(HERE/'purchases.csv', index=False)
    (HERE/'evidence.json').write_text(json.dumps(evidence, indent=2))
    pd.DataFrame(decisions).to_csv(HERE/'decisions.csv', index=False)
    summary = df.groupby(['arm', 'budget', 'method', 'target']).agg(
        width=('width', 'mean'), coverage=('covers', 'mean'),
        rejected=('status', lambda x: int((x == 'empty_set').sum())))
    summary.to_csv(HERE/'summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained', sharey=True)
    for ax, target in zip(axes, ['W_cls', 'W_label', 'W_unk']):
        for arm in ARMS:
            g = summary.xs((arm, 'reach_direct_law', target), level=['arm', 'method', 'target'])
            ax.plot(g.index, g.width, 'o-', label=arm)
        ax.set(title=target, xlabel='Additional verified images', ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean width, including outlier'); axes[-1].legend()
    fig.suptitle('One-seed accounting smoke test — frozen CNN; real parity')
    fig.savefig(HERE/'smoke_widths.png', dpi=160)
    (HERE/'solver_retries.json').write_text(json.dumps(TV.RETRIES))

if __name__ == '__main__':
    main()

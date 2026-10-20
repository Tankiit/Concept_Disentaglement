"""Finite-pool simulation: same representation, alternative coherent confidence set.

Independent multinomial reference/log draws from saved evaluation histograms.
These are simulated acquisitions, not additional independently collected images.
Run from any directory with the reach-aistats Python environment.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.optimize import linprog
import cvxpy as cp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
from models.ball import MMDBall, embedding_radius
from diagnose import interval


def bounds(counts, alpha):
    """Bonferroni simultaneous Clopper-Pearson bounds for multinomial cells."""
    counts = np.asarray(counts)
    n = counts.sum(axis=-1, keepdims=True)
    tail = alpha / (2 * counts.size)
    lo = np.where(counts == 0, 0, beta.ppf(tail, counts, n-counts+1))
    hi = np.where(counts == n, 1, beta.ppf(1-tail, counts+1, n-counts))
    return lo, hi


def joint_interval(lo, hi, plo, phi, cell):
    """LP over joint masses Z[s,c] = weight[s] * conditional[s,c].

    lo[s,c]*sum_c Z[s,c] <= Z[s,c] <= hi[s,c]*sum_c Z[s,c].
    Column sums obey log confidence bounds. Thus numerator and denominator
    use the SAME source distributions and the SAME deployment marginal.
    Charnes-Cooper: Y=Z/p(cell), t=1/p(cell).
    """
    k, d = lo.shape
    size = k*d
    rows, rhs = [], []
    for s in range(k):
        row_sum = np.zeros(size+1); row_sum[s*d:(s+1)*d] = 1
        for c in range(d):
            unit = np.zeros(size+1); unit[s*d+c] = 1
            rows.extend([unit-hi[s,c]*row_sum, lo[s,c]*row_sum-unit])
            rhs.extend([0., 0.])
    for c in range(d):
        col = np.zeros(size+1); col[c:size:d] = 1
        upper = col.copy(); upper[-1] = -phi[c]
        lower = -col; lower[-1] = plo[c]
        rows.extend([upper, lower]); rhs.extend([0., 0.])
    total = np.r_[np.ones(size), -1.]
    # Check in ordinary probability coordinates before ratio rescaling. An
    # incompatible model can be numerically awkward in homogeneous coordinates.
    matrix = np.asarray(rows)
    feasible = linprog(np.zeros(size), A_ub=matrix[:, :-1], b_ub=-matrix[:, -1],
                       A_eq=np.ones((1,size)), b_eq=[1.], bounds=(0.,None), method='highs')
    if feasible.status == 2:
        return np.nan, np.nan, 'empty_set'
    if not feasible.success:
        z = cp.Variable(size, nonneg=True)
        check = cp.Problem(cp.Minimize(0), [matrix[:,:-1]@z <= -matrix[:,-1], cp.sum(z)==1])
        check.solve(solver=cp.CLARABEL)
        if check.status == 'infeasible':
            return np.nan, np.nan, 'empty_set'
        if check.status != 'optimal':
            raise RuntimeError('Feasibility fallback: '+check.status)
    den = np.zeros(size+1); den[cell:size:d] = 1
    eq = np.stack([total, den])
    cost = np.zeros(size+1); cost[(k-1)*d+cell] = 1
    values = []
    for sign in (1., -1.):
        res = linprog(sign*cost, A_ub=rows, b_ub=rhs, A_eq=eq,
                      b_eq=[0., 1.], bounds=(0., None), method='highs')
        if res.status not in (0, 2):
            res = linprog(sign*cost, A_ub=rows, b_ub=rhs, A_eq=eq,
                          b_eq=[0., 1.], bounds=(0., None), method='highs-ds',
                          options={'presolve': False})
        if res.status == 2:
            return np.nan, np.nan, 'empty_set'
        if not res.success:
            x = cp.Variable(size+1, nonneg=True)
            problem = cp.Problem(cp.Minimize(sign*cost@x), [matrix@x<=rhs, eq@x==[0.,1.]])
            problem.solve(solver=cp.CLARABEL)
            if problem.status != 'optimal':
                raise RuntimeError('Ratio fallback: '+problem.status)
            solution, objective = x.value, problem.value
        else:
            solution, objective = res.x, res.fun
        assert np.max(matrix@solution-np.asarray(rhs)) < 1e-6
        assert np.max(np.abs(eq@solution-[0., 1.])) < 1e-6
        values.append(float(np.clip(sign*objective, 0, 1)))
    # We use cells with positive log lower bounds in this experiment.
    # For a zero-mass-compatible cell retain the existing vacuous policy.
    if plo[cell] == 0:
        return 0., 1., 'zero_lower_log_bound_policy'
    return *values, 'optimal'


def hist(frame, domain):
    sub = frame[frame.dataset == domain]
    return np.stack([np.bincount(g.cell, minlength=11)/len(g)
                     for _, g in sub.groupby('label')])


def main():
    base = ROOT/'workbook/mnist_reach_novelty_2026-10-19'
    ext = ROOT/'workbook/fashion_blood_reach_2026-10-19'
    old = pd.read_csv(base/'evaluation_scores.csv')
    new = pd.read_csv(ext/'evaluation_scores.csv')
    known = hist(new, 'mnist')
    letters = hist(old, 'letters').mean(axis=0)
    truths = {'letters': letters, **{s: hist(new, s).mean(axis=0) for s in ('fashion', 'blood')}}
    records = []
    for domain, novel in truths.items():
        truth = np.vstack([known, novel])
        for n, m in [(1000, 200), (4000, 1000)]:
            for delta in (.05, .2):
                for seed in range(10):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, 812]))
                    weights = np.r_[(1-delta)*rng.dirichlet(np.ones(10)), delta]
                    marginal = weights@truth
                    counts = rng.multinomial(n, marginal)
                    refs = np.stack([rng.multinomial(m, a) for a in known])
                    plo, phi = bounds(counts, .05/2)
                    klo, khi = bounds(refs, .05/4)
                    matching_counts = rng.multinomial(1300, novel)
                    letter_counts = rng.multinomial(1300, letters)
                    for mode in ('matching', 'letters_reference', 'unrestricted_novel'):
                        qcounts = matching_counts if mode == 'matching' else letter_counts
                        qlo, qhi = bounds(qcounts, .05/4)
                        if mode == 'unrestricted_novel':
                            qlo, qhi = np.zeros(11), np.ones(11)
                        lo, hi = np.vstack([klo, qlo]), np.vstack([khi, qhi])
                        for cell in (0, 10):
                            low, high, status = joint_interval(lo, hi, plo, phi, cell)
                            target = weights[-1]*novel[cell]/marginal[cell]
                            records.append(dict(domain=domain, n=n, m=m, delta=delta,
                                seed=seed, mode=mode, cell=cell, lower=low, upper=high,
                                width=high-low, status=status, truth=target,
                                covers=bool(low-1e-7 <= target <= high+1e-7)))
                    # Current REACH, matching references: isolate construction from mismatch.
                    qcounts = matching_counts
                    a = np.vstack([refs/m, qcounts/1300])
                    rr = embedding_radius(m, .05/33)
                    qr = embedding_radius(1300, .05/33)
                    er = np.sqrt(np.log(2/(.05/(3*10*11)))/(2*m))
                    ball = MMDBall(a, counts/n, embedding_radius(n, .05/3), max(rr, qr), 'l2')
                    for cell in (0, 10):
                        out, _ = interval(ball, np.clip(a[:-1]-er, 0, 1),
                                          np.clip(a[:-1]+er, 0, 1), qr, cell)
                        low, high = out['joint_lower'], out['joint_upper']
                        target = weights[-1]*novel[cell]/marginal[cell]
                        records.append(dict(domain=domain, n=n, m=m, delta=delta,
                            seed=seed, mode='original_matching', cell=cell, lower=low,
                            upper=high, width=high-low, status='original', truth=target,
                            covers=bool(low-1e-7 <= target <= high+1e-7)))
        print('Finished', domain, flush=True)
    df = pd.DataFrame(records)
    df.to_csv(HERE/'intervals.csv', index=False)
    summary = df.groupby(['domain', 'n', 'm', 'delta', 'mode', 'cell']).agg(
        width=('width', 'mean'), lower=('lower', 'mean'), upper=('upper', 'mean'),
        truth=('truth', 'mean'), coverage=('covers', 'mean'),
        empty_fraction=('status', lambda x: (x == 'empty_set').mean()))
    summary.to_csv(HERE/'summary.csv')
    (HERE/'config.json').write_text(json.dumps(dict(seeds=list(range(10)),
        budgets=[[1000,200],[4000,1000]], novel_reference_draws=1300,
        delta=[.05,.2], cells=[0,10], alpha=.05,
        scope='iid finite-pool simulation using saved evaluation histograms; no new images'), indent=2))
    assert (df.width.dropna() >= -1e-7).all()
    print(summary.to_string())


if __name__ == '__main__':
    main()

"""Response matrix: which width moves with which acquisition, with signed log-log slopes.

For each world, acquisition axis (n, m, eta) and component (REACH widths plus baselines), the
budget on that axis sweeps a log-spaced grid while the other two stay at BASE. Every row of the
matrix is one log-log slope of mean width against budget, with a 95% seed-bootstrap CI, and a
decision against the same-budget placebo null (test/slopes.py). Expectations are pre-registered
in train/expectations.py, whose SHA-256 the manifest records.

Also written: per-(world, n) ball coverage with a Clopper-Pearson lower bound (A12), per-probe
interval coverage for REACH and EMQ + bootstrap (A6), the ball_resolution gate, the LaTeX table,
and Fig 2 (W_cls against n).
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from data.worlds import make_world
from data.acquisitions import draw_log, draw_labelled_set
from models.ball import MMDBall
from models.widths import label_intervals, credal_widths
from models.baselines import emq_probe_intervals, ensemble_epistemic
import test.inference as inf
import test.slopes as sl
import train.expectations as E

PLACEBO_OFFSET = 10**6  # placebo seeds are disjoint from every grid seed
RESOLUTION_PAIR = ([.5, .5, 0], [.9, .1, 0])


def truths(world, cell):
    a = world.conditionals[:, cell]
    joint = world.weights*a
    known = joint[:2].sum()
    return dict(true_cls=float(joint[0]/known) if known > 0 else np.nan,
                true_unk=float(joint[-1]/joint.sum()) if joint.sum() > 0 else np.nan,
                true_target=world.target(cell))


def run_cell(world, n, m, eta, seed, alpha=.05, n_boot=200, members=20, set_kind='box'):
    """One replicate at one budget: REACH widths and baselines at every probe cell.

    Draws are prefixes of a per-seed stream (draw_log, draw_labelled_set), so one seed is paired
    across the whole grid. Alpha is split between the log ball and the label channel.
    """
    # Draw at the largest budget and take prefixes: a (2, m) label draw is not a prefix of a
    # (2, 2m) one, so drawing at m directly would silently unpair the grid.
    log = draw_log(world, max(E.GRIDS['n']+(n,)), seed)[:n]
    labels = draw_labelled_set(world, max(E.GRIDS['m']+(m,)), eta, seed)[:, :m]
    ball = MMDBall.from_log(world.conditionals, log, alpha/2, set_kind=set_kind)
    lo, hi = label_intervals(labels, eta, alpha/2)
    counts = np.bincount(log, minlength=world.conditionals.shape[1])
    cells = range(len(world.regions))
    emq = emq_probe_intervals(world.conditionals, counts, cells, n_boot, alpha, seed+7)
    ens = ensemble_epistemic(world.conditionals, counts, labels, eta, cells, members, seed+11)
    shared = dict(world_id=world.name, seed=int(seed), n=int(n), m=int(m), eta=float(eta),
                  eps_n=ball.eps_n, ball_resolution=ball.resolution(*RESOLUTION_PAIR),
                  ball_covers_truth=ball.covers_truth(world.marginal),
                  outer_covers_truth=ball.outer_covers(world.weights),
                  labels_cover_truth=bool(np.all(lo <= world.theta[:2]) and np.all(world.theta[:2] <= hi)))
    rows = []
    for cell, region in zip(cells, world.regions):
        row = dict(shared, probe_id=cell, region=region, **truths(world, cell))
        row.update(credal_widths(ball, cell, lo, hi))
        row.update(emq[cell]); row.update(ens[cell])
        for name, truth in (('cls', 'true_cls'), ('unk', 'true_unk'), ('total', 'true_target'),
                            ('EMQ_cls', 'true_cls'), ('EMQ_unk', 'true_unk')):
            lower, upper, t = row[f'{name}_lower'], row[f'{name}_upper'], row[truth]
            row[f'{name}_covers'] = (bool(lower-1e-9 <= t <= upper+1e-9)
                                     if np.isfinite(t) and np.isfinite(lower) else None)
        rows.append(row)
    return rows


def _tasks(worlds, seeds, set_kind='box'):
    """(world, axis, n, m, eta, seed); axis='placebo' rows sit at BASE on disjoint seeds."""
    tasks = []
    for name in worlds:
        for axis, grid in E.GRIDS.items():
            for g in grid:
                budget = dict(E.BASE, **{axis: g})
                tasks += [(name, axis, budget['n'], budget['m'], budget['eta'], s, set_kind) for s in seeds]
        placebo = range(PLACEBO_OFFSET+seeds.start,
                        PLACEBO_OFFSET+seeds.start+len(seeds)*max(map(len, E.GRIDS.values())))
        tasks += [(name, 'placebo', E.BASE['n'], E.BASE['m'], E.BASE['eta'], s, set_kind) for s in placebo]
    return tasks


def _run(task):
    name, axis, n, m, eta, seed, set_kind = task
    return [dict(row, axis=axis) for row in run_cell(make_world(name), n, m, eta, seed, set_kind=set_kind)]


def simulate(worlds, seeds, workers=None, set_kind='box'):
    tasks = _tasks(worlds, seeds, set_kind)
    with ProcessPoolExecutor(workers or os.cpu_count()) as pool:
        chunks = list(pool.map(_run, tasks, chunksize=16))
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def ball_resolution_gate(df):
    """Every world must resolve the declared weight change at its smallest swept n."""
    sub = df[df.axis == 'n']
    smallest = sub[sub.n == sub.n.min()]
    worst = float(smallest.ball_resolution.min())
    if not worst > 1:
        raise RuntimeError(f'ball_resolution={worst:.3f} <= 1 at n={sub.n.min()}: the ball cannot '
                           'resolve the declared weight change, so no slope here is interpretable')
    return dict(n=int(sub.n.min()), min_ball_resolution=worst, passed=True)


def response_matrix(df, n_boot=2000, n_null=2000):
    components = E.COMPONENTS+E.BASELINES
    tested = sum(1 for v in E.EXPECT.values() if v not in (None, 'exact'))
    alpha = E.FAMILY_ALPHA/max(tested, 1)
    out = []
    for world in df.world_id.unique():
        wdf = df[df.world_id == world]
        placebo = wdf[wdf.axis == 'placebo']
        for acq, axis in E.ACQUISITIONS.items():
            grid = E.GRIDS[axis]
            sweep = wdf[wdf.axis == axis]
            for comp in components:
                for region in E.REGIONS:
                    cell = sweep[sweep.region == region].pivot_table(
                        index='seed', columns=axis, values=comp, aggfunc='first', dropna=False)
                    if cell.empty:
                        continue
                    widths = cell.reindex(columns=list(grid)).to_numpy(dtype=float)
                    res = sl.signed_slope(grid, widths, n_boot)
                    pool = placebo[placebo.region == region][comp].to_numpy(dtype=float)
                    null = (sl.placebo_slopes(grid, pool, res['seeds'], n_null)
                            if res['status'] == 'ok' and np.isfinite(pool).sum() >= len(grid)*res['seeds']
                            else np.array([]))
                    dec = sl.decide(res, null, alpha)
                    expected = E.EXPECT.get((acq, comp, region, world))
                    means = np.nanmean(widths, 0) if np.isfinite(widths).any() else [np.nan]*len(grid)
                    out.append(dict(world_id=world, acquisition=acq, axis=axis, component=comp,
                                    region=region, **res, **dec, expected=expected,
                                    agrees=sl.agrees(expected, dec['decision']),
                                    width_at_min=float(means[0]), width_at_max=float(means[-1])))
    return pd.DataFrame(out), alpha


def coverage_tables(df):
    """A12: ball coverage per (world, n), one event per seed; probe coverage for REACH vs EMQ."""
    sweep = df[df.axis == 'n']
    ball = []
    for (world, n), block in sweep.drop_duplicates(['world_id', 'n', 'seed']).groupby(['world_id', 'n']):
        ball.append(dict(world_id=world, n=int(n),
                         **inf.coverage_at_least_nominal(block.ball_covers_truth, .975)))
    probes = []
    for (world, region, n), block in sweep.groupby(['world_id', 'region', 'n']):
        row = dict(world_id=world, region=region, n=int(n))
        for name in ('cls', 'unk', 'total', 'EMQ_cls', 'EMQ_unk'):
            col = block[f'{name}_covers'].dropna()
            row[f'{name}_coverage'] = float(col.astype(bool).mean()) if len(col) else None
        probes.append(row)
    return ball, probes


def latex_table(matrix, world='separated'):
    sub = matrix[(matrix.world_id == world) & matrix.expected.notna()]
    symbol = {'negative': '$-$', 'positive': '$+$', '~0': '$\\approx 0$',
              'exact_constant': '$0^{\\ast}$', 'undefined': 'n/a'}
    lines = ['\\begin{tabular}{lllrrc}', '\\toprule',
             'Acquisition & Component & Region & Slope [95\\% CI] & $\\tau$ & Decision \\\\', '\\midrule']
    for r in sub.itertuples():
        ci = (f'{r.slope:+.3f} [{r.lower:+.3f}, {r.upper:+.3f}]' if r.status == 'ok' else 'n/a')
        mark = '' if r.agrees else ' \\dag'
        lines.append(f"{r.acquisition} & {r.component.replace('_', chr(92)+'_')} & "
                     f"{r.region.replace('_', ' ')} & {ci} & {r.tau:.3f} & "
                     f"{symbol.get(r.decision, r.decision)}{mark} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def plot_fig2(df, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sweep = df[(df.axis == 'n') & (df.world_id == 'separated')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    styles = {'on_support': ('C0', 'o'), 'boundary': ('C1', 's')}
    for region, (colour, marker) in styles.items():
        block = sweep[sweep.region == region].groupby('n')
        for comp, ls, label in (('W_cls', '-', 'REACH $W_{cls}$'), ('EMQ_cls', ':', 'EMQ+bootstrap')):
            mean = block[comp].mean()
            axes[0].plot(mean.index, mean.values, ls=ls, marker=marker, color=colour, ms=4,
                         label=f'{label}, {region.replace("_", " ")}')
    n = np.array(sorted(sweep.n.unique()), dtype=float)
    ref = sweep[(sweep.region == 'on_support') & (sweep.n == n[0])].W_cls.mean()
    axes[0].plot(n, ref*np.sqrt(n[0]/n), color='grey', lw=.8, ls='--', label='$n^{-1/2}$')
    axes[0].set(xscale='log', yscale='log', xlabel='unlabelled log size $n$', ylabel='width')
    axes[0].legend(fontsize=6)
    res = sweep.drop_duplicates(['n', 'seed']).groupby('n').ball_resolution.mean()
    axes[1].plot(res.index, res.values, marker='o', color='C2')
    axes[1].axhline(1, color='k', lw=.8, ls='--')
    axes[1].set(xscale='log', yscale='log', xlabel='unlabelled log size $n$', ylabel='ball resolution')
    fig.tight_layout()
    fig.savefig(out/'fig2_wcls_vs_n.pdf'); fig.savefig(out/'fig2_wcls_vs_n.png', dpi=160)
    plt.close(fig)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/response')
    parser.add_argument('--seeds', type=int, default=64)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--worlds', nargs='+', default=['separated', 'confounded', 'absent'])
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--set', choices=('box', 'l2'), default='box',
                        help='box: coordinate outer LP; l2: exact MMD ball (SOCP), as in the real audit')
    args = parser.parse_args(argv)
    if args.seeds < 2 or args.seed_start < 0:
        parser.error('At least two seeds and a nonnegative seed start required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = dict(config=vars(args), base=E.BASE, grids=E.GRIDS,
                    expectations_sha256=_sha(E.__file__),
                    source_sha256={str(f.relative_to(root)): _sha(f) for folder in
                                   ('models', 'data', 'train', 'test') for f in sorted((root/folder).glob('*.py'))})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    df = simulate(args.worlds, range(args.seed_start, args.seed_start+args.seeds), args.workers, args.set)
    gate = ball_resolution_gate(df)
    df.to_csv(out/'rows.csv.gz', index=False)
    matrix, alpha = response_matrix(df, args.bootstrap)
    matrix.to_csv(out/'response_matrix.csv', index=False)
    ball, probes = coverage_tables(df)
    tested = matrix[matrix.expected.notna() & (matrix.expected != 'exact')]
    summary = dict(ball_resolution_gate=gate, placebo_alpha_per_cell=alpha,
                   predicted_cells=int(len(tested)), agreeing=int(tested.agrees.sum()),
                   disagreeing=tested[~tested.agrees.astype(bool)][
                       ['world_id', 'acquisition', 'component', 'region', 'expected', 'decision',
                        'slope', 'lower', 'upper', 'tau']].to_dict('records'),
                   exact_by_construction=int((matrix.expected == 'exact').sum()),
                   max_tau=float(matrix.tau[np.isfinite(matrix.tau)].max()),
                   ball_coverage=ball, probe_coverage=probes, config=vars(args))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    (out/'response_matrix.tex').write_text(latex_table(matrix))
    plot_fig2(df, out)
    print(f'{len(df)} rows; {summary["agreeing"]}/{summary["predicted_cells"]} predicted cells agree; '
          f'ball_resolution at n={gate["n"]}: {gate["min_ball_resolution"]:.2f}')


if __name__ == '__main__':
    main()

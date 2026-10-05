"""Signed log-log response slopes with a seed bootstrap and a same-budget placebo null.

A slope is the OLS fit of log(mean width over seeds) against log(budget). The bootstrap unit is
the seed (one independent replicate shares its draws across the grid), never the probe or cell.

What counts as "~0". The null for one matrix cell is the slope you get when every grid level is
evaluated at the SAME base budget on fresh, disjoint seeds: any slope there is pure Monte Carlo
noise of that estimator at that seed count. A cell is called nonzero when its point slope falls
outside the central (1 - alpha_family/K) mass of its placebo distribution (Bonferroni over the K
tested cells), and '~0' otherwise. The tolerance tau is therefore that cell's placebo quantile,
not a chosen margin; it is reported next to every decision so the minimum detectable slope is
visible. Placebo levels are unpaired while the real grid is paired by seed, so the null is wider
than the paired sampling noise: conservative for 'negative'/'positive', liberal for '~0'.
"""
import numpy as np

FLOOR = 1e-6  # log of an exactly-zero width; only reached by identified point intervals


def _slope(log_grid, mean_widths):
    y = np.log(np.maximum(mean_widths, FLOOR))
    x = log_grid - log_grid.mean()
    return float(x @ (y - y.mean()) / (x @ x))


def signed_slope(grid, widths, n_boot=2000, level=.95, seed=0):
    """widths: seeds x grid array. Returns slope, bootstrap CI over seeds, and constancy flag."""
    widths = np.asarray(widths, dtype=float)
    grid = np.log(np.asarray(grid, dtype=float))
    if widths.ndim != 2 or widths.shape[1] != len(grid) or len(grid) < 2:
        raise ValueError('widths must be seeds x grid with at least two grid levels')
    keep = np.all(np.isfinite(widths), axis=1)
    if keep.sum() < 2:
        return dict(slope=np.nan, lower=np.nan, upper=np.nan, seeds=int(keep.sum()),
                    constant=False, status='undefined')
    widths = widths[keep]
    # Invariant within every replicate across the grid (seeds may differ from each other).
    constant = bool(np.all(np.ptp(widths, axis=1) <= 1e-12))
    slope = _slope(grid, widths.mean(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(widths), (n_boot, len(widths)))
    boot = np.array([_slope(grid, widths[i].mean(0)) for i in idx])
    lo, hi = np.quantile(boot, [(1-level)/2, (1+level)/2])
    return dict(slope=slope, lower=float(lo), upper=float(hi), seeds=int(len(widths)),
                constant=constant, status='ok')


def placebo_slopes(grid, pool, seeds_per_level, n_draws=2000, seed=0):
    """Null slopes: partition a pool of same-budget widths into len(grid) disjoint seed groups."""
    pool = np.asarray(pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    g = len(grid)
    if len(pool) < g*seeds_per_level:
        raise ValueError('placebo pool needs len(grid) * seeds_per_level replicates')
    log_grid = np.log(np.asarray(grid, dtype=float))
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for b in range(n_draws):
        groups = rng.permutation(pool)[:g*seeds_per_level].reshape(g, seeds_per_level)
        out[b] = _slope(log_grid, groups.mean(1))
    return out


def decide(result, null, alpha):
    """Decision for one cell against its placebo null at two-sided level alpha."""
    if result['status'] != 'ok':
        return dict(decision='undefined', tau=np.nan)
    if result['constant']:
        return dict(decision='exact_constant', tau=0.)
    tau = float(np.quantile(np.abs(null), 1-alpha)) if len(null) else np.nan
    s = result['slope']
    if not np.isfinite(tau):
        return dict(decision='undefined', tau=tau)
    if abs(s) <= tau:
        return dict(decision='~0', tau=tau)
    return dict(decision='negative' if s < 0 else 'positive', tau=tau)


def agrees(expected, decision):
    """Whether a decision matches a pre-registered sign. 'exact' cells are reported, not tested."""
    if expected is None:
        return None
    if expected == 'exact':
        return decision == 'exact_constant'
    if expected == '~0':
        return decision in ('~0', 'exact_constant')
    return decision == expected

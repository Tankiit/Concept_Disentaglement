import numpy as np
import pytest
import torch
from example_wrn50 import (make_plain_ood, build_head, Config, run, knn_kth_dist,
                           mix_path, path_metrics, fit_knn_reference,
                           trace_paths, write_path_outputs, parse_args, credal_intervals,
                           rank_disagreement, sampling_stability, choose_control_partners,
                           stratified_tercile_auroc, divergent_hardness_concentration,
                           plot_credal_examples, load_test_dataset, dataset_pixels,
                           SUPPORTED_DATASETS, auroc_fast)


def test_auroc_fast_matches_example():
    from example import auroc as auroc_ref
    rng = np.random.default_rng(0)
    for trial in range(5):
        k = rng.normal(0, 1, 40)
        o = rng.normal(.4, 1, 60)
        if trial == 3:                       # heavy ties
            k = np.round(k)/2
            o = np.round(o)/2
        if trial == 4:                       # all tied
            k = np.zeros(40)
            o = np.zeros(60)
        np.testing.assert_allclose(auroc_fast(k, o), auroc_ref(k, o), atol=1e-12)
    with pytest.raises(ValueError):
        auroc_fast(np.zeros(0), np.ones(3))


def test_dataset_registry():
    assert set(SUPPORTED_DATASETS) >= {'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn'}
    with pytest.raises(ValueError):
        load_test_dataset('imagenet-21k')


def test_dataset_registry_ctor_wiring(tmp_path):
    # download=False + empty root: construction is attempted (no network) and fails loudly
    with pytest.raises(Exception):
        load_test_dataset('mnist', root=str(tmp_path), download=False)


class _FakeArrayDS:
    def __init__(self, data):
        self.data = data


def test_dataset_pixels_layouts():
    n, h, w = 3, 2, 2
    nhwc = np.full((n, h, w, 3), 255, np.uint8)                    # CIFAR-style
    gray = torch.full((n, h, w), 128, dtype=torch.uint8)           # MNIST-style tensor
    nchw = np.full((n, 1, h, w), 64, np.uint8)                     # SVHN-grayscale-style
    for ds in (_FakeArrayDS(nhwc), _FakeArrayDS(gray), _FakeArrayDS(nchw)):
        px = dataset_pixels(ds, [0, 2])
        assert px.shape == (2, h, w, 3) and px.dtype == np.float32
        assert 0 <= px.min() and px.max() <= 1


def test_plot_credal_examples(tmp_path):
    rng = np.random.default_rng(0)
    stacks = {'known': rng.dirichlet(np.ones(3), size=6).astype(np.float32),
              'ood': rng.dirichlet(np.ones(3)*.3, size=6).astype(np.float32)}
    out = plot_credal_examples(stacks, tmp_path/'credal.png', class_labels=['a', 'b', 'c'])
    assert out['lower'].shape == (2, 3) and out['upper'].shape == (2, 3)
    assert (out['lower'] <= out['upper']).all()
    assert (tmp_path/'credal.png').exists() and (tmp_path/'credal.png').stat().st_size > 1000


def test_rank_disagreement():
    a = np.arange(10, dtype=float)
    same = rank_disagreement(a, a.copy())
    assert same['frac'] == 0
    b = a.copy()
    b[[0, 9]] = a[[9, 0]]                  # two examples swap percentile extremes
    out = rank_disagreement(a, b, tol=.1)
    assert out['frac'] > 0 and set(out['worst'][:2]) == {0, 9}
    assert out['max_gap'] > .8


def test_credal_width_nesting_monotone():
    rng = np.random.default_rng(1)
    stacked = rng.uniform(.02, .95, (100, 5, 3)).astype(np.float32)
    _, _, w_full = credal_intervals(stacked)
    _, _, w_sub = credal_intervals(stacked[:10])          # nested subset of the same passes
    assert (w_sub <= w_full + 1e-6).all(), 'min/max envelope can only widen with more passes'


def test_choose_control_partners():
    labels = np.repeat(np.arange(4), 5)
    starts = np.array([0, 5, 10, 15])
    partners, relation = choose_control_partners(starts, labels, n_same=2, seed=0)
    assert list(relation) == ['same_class', 'same_class', 'diff_class', 'diff_class']
    for p, s, rel in zip(partners, starts, relation):
        assert p != s
        assert (labels[p] == labels[s]) == (rel == 'same_class')
    with pytest.raises(ValueError):
        choose_control_partners(np.array([0]), np.zeros(5, dtype=int), n_same=0, seed=0)  # no other class exists



def test_cli_flags_map_to_config():
    go_paths, config = parse_args(['--paths', '--n-paths', '3', '--device', 'mps'])
    assert go_paths and config.n_paths == 3 and config.device == 'mps'
    go_paths, config = parse_args([])
    assert not go_paths and config.mode == 'oss' and len(config.t_grid) == 11


def write_cache(path, classes, rows, d, shift=0., seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(classes*rows, d))+shift).astype(np.float32)
    y = np.repeat(np.arange(classes), rows)
    np.savez(path, X=X, y=y)
    return path


def test_plain_world_cross_dataset(tmp_path):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=1)
    ood_p = write_cache(tmp_path/'ood.npz', 10, 20, 16, shift=50., seed=2)
    world = make_plain_ood(id_p, ood_path=ood_p, seed=0)
    assert world['kkc'] == list(range(10)) and world['kuc'] == []
    assert world['uuc'] == list(range(1000, 1010))
    assert world['uuc_val'] == [] and world['uuc_test'] == world['uuc']
    idx = world['idx']
    assert set(idx) == {'train', 'val', 'test'}
    assert not (set(idx['train']) & set(idx['val']))
    assert not (set(idx['train']) & set(idx['test']))
    assert not (set(idx['val']) & set(idx['test']))
    assert len(world['X']) == len(world['y']) == 600
    for split in ('train', 'val'):
        assert (world['y'][idx[split]] < 1000).all()
    assert (world['y'][idx['test']] >= 1000).sum() == 200


def test_plain_world_heldout_classes(tmp_path):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=3)
    world = make_plain_ood(id_p, n_known=5, seed=0)
    assert len(world['kkc']) == 5 and len(world['uuc']) == 5
    assert world['kkc'] == sorted(world['kkc']) and world['kuc'] == []
    assert world['uuc_val'] == [] and world['uuc_test'] == world['uuc']
    assert set(world['kkc']).isdisjoint(world['uuc'])
    known = np.isin(world['y'], world['kkc'])
    novel = ~known
    idx = world['idx']
    assert not (set(idx['train']) & set(idx['val']))
    assert not (set(idx['train']) & set(idx['test']))
    assert not (set(idx['val']) & set(idx['test']))
    assert novel[idx['train']].sum() == 0 and novel[idx['val']].sum() == 0
    assert novel[idx['test']].sum() == novel.sum()
    assert known[idx['train']].sum() == 28*5  # 70% of 40 rows per known class


def test_plain_world_explicit_known(tmp_path):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=4)
    world = make_plain_ood(id_p, known=[2, 5], seed=0)
    assert world['kkc'] == [2, 5]
    assert world['uuc'] == [0, 1, 3, 4, 6, 7, 8, 9]


def test_plain_world_requires_ood(tmp_path):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=5)
    with pytest.raises(ValueError):
        make_plain_ood(id_p, known=list(range(10)), seed=0)


@pytest.mark.parametrize('name', ['mlp', 'etf'])
def test_head_accepts_wrn50_dim(name):
    head = build_head(name, d_in=2048, k=5)
    assert head(torch.randn(4, 2048)).shape == (4, 5)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='mps unavailable'))])
def test_run_plain_end_to_end(tmp_path, device):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=6)
    ood_p = write_cache(tmp_path/'ood.npz', 10, 20, 16, shift=50., seed=7)
    config = Config(mode='plain', id_cache=str(id_p), ood_cache=str(ood_p),
                    epochs=1, device=device, out_dir=str(tmp_path/'out'))
    result = run(config)
    assert result['device'] == device
    assert result['mode'] == 'plain'
    assert np.isfinite(result['auroc']['kkc_vs_uuc'])
    assert 0 <= result['auroc']['kkc_vs_uuc'] <= 1
    assert np.isfinite(result['auroc']['kkc_vs_uuc_cw'])
    assert 'frac' in result['rank_disagreement']
    assert 0 <= result['rank_disagreement']['frac'] <= 1
    assert len(result['terciles']) == 3
    assert (tmp_path/'out'/'wrn50_eval.json').exists()
    assert (tmp_path/'out'/'budget_info_wrn50.json').exists()


def test_sampling_stability_structure(tmp_path):
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-3, .3, (20, 8)), rng.normal(3, .3, (20, 8))].astype(np.float32)
    y = np.repeat([0, 1], 20)
    head = build_head('mlp', d_in=8, k=2)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(150):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(head(xt), yt).backward()
        opt.step()
    out = sampling_stability(head, x, is_ood=np.r_[np.zeros(20), np.ones(20)].astype(bool),
                             Ts=(5, 10), repeats=3)
    for t, stats in out.items():
        for key in ('eu_auroc_mean', 'eu_auroc_std', 'cw_auroc_mean', 'cw_auroc_std',
                    'eu_mean', 'cw_mean'):
            assert np.isfinite(stats[key])
        assert 0 <= stats['eu_auroc_mean'] <= 1 and 0 <= stats['cw_auroc_mean'] <= 1


def test_stratified_tercile_auroc():
    rng = np.random.default_rng(0)
    n = 90
    hardness = rng.uniform(0, 1, n)
    is_ood = rng.uniform(0, 1, n) < .5
    easy, hard = hardness < 1/3, hardness >= 2/3
    eu = np.full(n, .5)
    cw = np.full(n, .5)
    eu[easy & is_ood], eu[easy & ~is_ood] = .9, .1   # EU separates only among easy rows
    cw[hard & is_ood], cw[hard & ~is_ood] = .9, .1   # width separates only among hard rows
    rows = stratified_tercile_auroc(eu, cw, hardness, is_ood, n_boot=40, seed=0)
    assert len(rows) == 3
    for r in rows:
        assert {'tercile', 'n', 'auroc_eu', 'auroc_cw', 'diff', 'ci_lo', 'ci_hi'} <= set(r)
        assert r['ci_lo'] <= r['diff'] <= r['ci_hi']
    assert rows[0]['auroc_eu'] > .9 > rows[0]['auroc_cw']   # easy tercile: EU wins
    assert rows[2]['auroc_cw'] > .9 > rows[2]['auroc_eu']   # hard tercile: width wins


def test_divergent_hardness_concentration():
    hardness = np.arange(10)*0.1                       # rows sorted easy -> hard, top tercile >= 0.6
    eu = np.arange(10)*0.1
    cw = eu.copy()
    cw[7], cw[8] = 0.3, 0.4                            # two hard rows drop far in width rank
    out = divergent_hardness_concentration(eu, cw, hardness, tol=.1)
    assert out['n_divergent'] == 4                     # rows 4, 6, 7, 8 under stable tie-breaking
    assert out['frac_in_hard_tercile'] == 0.75         # row 4 sits in the middle tercile
    assert out['base_rate_hard_tercile'] == pytest.approx(0.4, abs=.01)


def test_run_oss_mode_smoke(tmp_path):
    id_p = write_cache(tmp_path/'id.npz', 10, 40, 16, seed=8)
    config = Config(mode='oss', id_cache=str(id_p), epochs=1, device='cpu',
                    out_dir=str(tmp_path/'out'))
    result = run(config)
    assert result['mode'] == 'oss'
    assert set(world_keys := result['world_keys']) >= {'X', 'y', 'idx', 'kkc', 'kuc', 'uuc'}


def test_knn_kth_dist_matches_bruteforce():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(30, 8)).astype(np.float32)
    q = ref[:5]
    dist = np.sqrt(((q[:, None]-ref[None])**2).sum(-1))
    np.testing.assert_allclose(knn_kth_dist(ref, q, 4), np.sort(dist, 1)[:, 3], rtol=1e-4)
    np.testing.assert_allclose(knn_kth_dist(ref, q, 4, exclude_self=True), np.sort(dist, 1)[:, 4], rtol=1e-4)
    with pytest.raises(AssertionError):
        knn_kth_dist(ref[:3], q, 5)


def test_mix_path_endpoints_and_range():
    x0 = np.full((2, 4, 4, 3), .2, dtype=np.float32)
    x1 = np.full_like(x0, .8)
    np.testing.assert_allclose(mix_path(x0, x1, 0.), x0)
    np.testing.assert_allclose(mix_path(x0, x1, 1.), x1)
    np.testing.assert_allclose(mix_path(x0, x1, .5), .5, atol=1e-6)
    for t in (0., .3, 1.):
        out = mix_path(x0, x1, t)
        assert out.shape == x0.shape and 0 <= out.min() and out.max() <= 1
    with pytest.raises(ValueError):
        mix_path(x0, x1, 1.5)


def test_credal_intervals_match_bounds():
    rng = np.random.default_rng(0)
    stacked = rng.uniform(.05, .9, (7, 6, 4)).astype(np.float32)
    lower, upper, width = credal_intervals(stacked)
    np.testing.assert_allclose(lower, stacked.min(0), rtol=1e-6)
    np.testing.assert_allclose(upper, stacked.max(0), rtol=1e-6)
    np.testing.assert_allclose(width, stacked.max(0)-stacked.min(0), rtol=1e-6)
    np.testing.assert_allclose(width.mean(-1), (stacked.max(0)-stacked.min(0)).mean(-1), rtol=1e-6)


def test_path_metrics_on_synthetic_features():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-3, .3, (40, 8)), rng.normal(3, .3, (40, 8))].astype(np.float32)
    y = np.repeat([0, 1], 40)
    head = build_head('mlp', d_in=8, k=2)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(xt), yt)
        loss.backward()
        opt.step()
    world = {'X': x, 'y': y, 'idx': {'train': np.arange(80), 'val': np.arange(0), 'test': np.arange(0)},
             'kkc': [0, 1], 'kuc': [], 'uuc': [], 'uuc_val': [], 'uuc_test': []}
    ref = fit_knn_reference(world, q=.95, k=5)
    rows = path_metrics(x[:10], y[:10], head, ref, k=5, T=5)
    assert set(rows) >= {'eu', 'margin_top12', 'margin_true', 'credal_width', 'dist_k',
                         'dist_norm', 'pred_class'}
    n = len(x[:10])
    for key in ('eu', 'margin_top12', 'credal_width', 'dist_k', 'dist_norm'):
        assert rows[key].shape == (n,) and np.isfinite(rows[key]).all()
    assert (rows['eu'] >= -1e-6).all(), 'BALD is nonnegative'
    assert rows['eu'].max() <= np.log(2) + 1e-6, 'BALD is bounded by the mean-predictive entropy'
    from example import eu_bald
    torch.manual_seed(0)
    expected = eu_bald(head, torch.from_numpy(x[:10]), T=5).numpy()
    torch.manual_seed(0)
    again = path_metrics(x[:10], y[:10], head, ref, k=5, T=5)
    np.testing.assert_allclose(again['eu'], expected, atol=1e-6)
    torch.manual_seed(0)
    head.train()
    with torch.no_grad():
        stacked = torch.stack([torch.softmax(head(torch.from_numpy(x[:10])), dim=1)
                               for _ in range(5)]).numpy()
    np.testing.assert_allclose(again['credal_width'],
                               (stacked.max(0)-stacked.min(0)).mean(-1), atol=1e-6)
    assert (rows['dist_k'] >= 0).all() and (rows['dist_norm'] >= 0).all()
    assert (rows['pred_class'] == y[:10]).mean() > .8
    rows_ood = path_metrics(x[:10], np.full(n, np.nan), head, ref, k=5, T=5)
    assert np.isnan(rows_ood['margin_true']).all()


def test_trace_and_write_paths(tmp_path):
    rng = np.random.default_rng(11)
    start = rng.uniform(.2, .8, (2, 4, 4, 3)).astype(np.float32)
    partner = np.zeros_like(start)           # path walks brightness (and the feature) to a corner
    ref_x = np.r_[rng.normal(0, .1, (30, 12)), rng.normal(4, .1, (30, 12))].astype(np.float32)
    world = {'X': ref_x, 'y': np.repeat([0, 1], 30),
             'idx': {'train': np.arange(60)}, 'kkc': [0, 1]}
    reference = fit_knn_reference(world, q=.95, k=5)
    head = build_head('mlp', d_in=12, k=2)
    def stub_extractor(pixels):
        # feature drifts with the pixel mix: mean brightness moves along dim 0
        feats = np.full((len(pixels), 12), .1, np.float32)
        feats[:, 0] = pixels.mean(axis=(1, 2, 3), dtype=np.float32)*4.
        return feats
    rows = []
    for kind, rel in (('cifar_to_mnist', 'ood'), ('cifar_to_cifar', 'same_class')):
        traced = trace_paths(start, partner, (0., .5, 1.), stub_extractor, head, reference,
                             true_labels=np.array([0, 1]), k=5, T=3)
        rows.extend(dict(kind=kind, partner_relation=rel, **r) for r in traced)
    assert len(rows) == 2*2*3
    for key in ('eu', 'margin_top12', 'credal_width', 'dist_k', 'dist_norm', 't', 'kind',
                'path_id', 'partner_relation'):
        assert key in rows[0]
    result = write_path_outputs(rows, tmp_path/'out')
    assert result['summary']['cifar_to_mnist'][0.0]['eu'] is not None
    import csv, json
    with open(tmp_path/'out'/'wrn50_paths.csv') as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(rows)
    assert json.loads((tmp_path/'out'/'wrn50_paths_summary.json').read_text())['cifar_to_cifar']
    assert (tmp_path/'out'/'wrn50_paths.png').exists()


def test_budget_report_matches_example_fields(tmp_path):
    import json
    from example_wrn50 import tag_and_report_budget_np
    id_p = write_cache(tmp_path/'id.npz', 6, 30, 16, seed=9)
    world = make_plain_ood(id_p, n_known=4, seed=0)
    report = tag_and_report_budget_np(world, k=5, q=.95, save_path=str(tmp_path/'budget.json'))
    saved = json.loads((tmp_path/'budget.json').read_text())
    for key in ('n_kkc', 'n_kuc', 'n_uuc', 'classes', 'novel_classes', 'tau', 'mode'):
        assert key in saved
    assert saved['mode'] == 'plain'
    assert len(report['novel_classes']) == 2
    assert all(item['tag'] in ('near', 'far') for item in report['novel_classes'])
    assert len(report['tau']) == 4 and all(t > 0 for t in report['tau'])

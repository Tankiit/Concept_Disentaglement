import numpy as np
import pytest
from novel_variations import (make_novel_world, _sample_world_rows,
    _build_known_val_scores, categorize_detection, run)
from data.worlds import make_world


def test_endpoints_and_constant_known_laws():
    for a, name in [(0., 'confounded'), (1., 'separated')]:
        w = make_novel_world(novelty=a)
        np.testing.assert_allclose(w.conditionals, make_world(name).conditionals)
    w = make_novel_world(novelty=.3)
    np.testing.assert_allclose(w.conditionals[2], .7*make_world('confounded').conditionals[2]
                               + .3*make_world('separated').conditionals[2])
    np.testing.assert_allclose(w.weights, make_world().weights)


@pytest.mark.parametrize('kind', ['separated','confounded','interpolate','near_known0','near_known1','absent'])
def test_probability_laws(kind):
    w = make_novel_world(kind)
    np.testing.assert_allclose(w.conditionals.sum(1), 1)
    assert (w.conditionals >= 0).all()


def test_known_anchor_and_custom():
    for i in (0, 1):
        w = make_novel_world('near_known'+str(i), novelty=0)
        np.testing.assert_allclose(w.conditionals[2], w.conditionals[i])
    w = make_novel_world('custom', custom_q=[0,0,.5,.5])
    np.testing.assert_allclose(w.conditionals[2], [0,0,.5,.5])
    with pytest.raises(ValueError):
        make_novel_world('custom', custom_q=[-1,0,1,1])


def test_pairing_prefixes_and_source_ids():
    a = _sample_world_rows(make_novel_world(novelty=0), 500, seed=3)
    b = _sample_world_rows(make_novel_world(novelty=1), 500, seed=3)
    np.testing.assert_array_equal(a['source_id'], b['source_id'])
    known = a['source_id'].numpy() != 2
    for k in ('cell_id','logits','ood_score'):
        np.testing.assert_array_equal(a[k][known], b[k][known])
    small = _sample_world_rows(make_novel_world(novelty=0), 100, seed=3)
    for k in a:
        np.testing.assert_array_equal(small[k], a[k][:100])
    np.testing.assert_array_equal(a['class_id'], a['source_id'])
    np.testing.assert_array_equal(a['is_ood'], a['source_id'] == 2)


def test_calibration_is_known_examples_and_frozen_across_variations():
    a = _build_known_val_scores(make_novel_world(novelty=0), m=512)
    b = _build_known_val_scores(make_novel_world(novelty=1), m=512)
    assert a.shape == (512,)
    np.testing.assert_array_equal(a, b)
    eval_scores = _sample_world_rows(make_novel_world(), 20000, seed=90,
                                    known_only=True)['ood_score'].numpy()
    assert .01 < (eval_scores > np.quantile(a, .95, method='higher')).mean() < .1


def test_threshold_tie_is_accepted():
    df, threshold = categorize_detection([[1.,0.]], [1.], [1.,1.], [True], [2])
    assert not df.detected.iloc[0]
    assert df.category.iloc[0] == 'weakly_accepted'


def test_oracle_mode_is_explicit():
    w = make_novel_world()
    a = _sample_world_rows(w, score_mode='known_surprisal')
    b = _sample_world_rows(w, score_mode='oracle_ratio')
    assert not np.allclose(a['ood_score'], b['ood_score'])
    with pytest.raises(ValueError):
        run(seed=1, calibration_seed=1, show=False)


def test_end_to_end():
    out = run(n_samples=80, calibration_size=60, show=False)
    assert len(out['rows']) == 80
    assert len(out['known_val_scores']) == 60
    assert set(out['rows'].class_id) <= {0,1,2}

import numpy as np
import pytest
from wrn_compare.core import (test_pools as make_test_pools, pair_indices, transform_images,
    GaussianDensity, BootstrapHeads, score_bundle, thresholds_from_known,
    paired_auc_difference, shift_auc)


def test_disjoint_pools_and_pairs():
    y = np.repeat(np.arange(3),30)
    pools = make_test_pools(y,5,0)
    assert len(set(np.concatenate(list(pools.values())))) == 45
    for kind in ['same_class_mix','cross_class_mix']:
        partners = pair_indices(y,pools['evaluation'],pools['partner'],kind,0)
        assert len(np.unique(partners)) == 15
        assert np.all((y[partners]==y[pools['evaluation']]) == (kind=='same_class_mix'))


def test_transformation_endpoints():
    x = np.full((3,4,4,3),.2,dtype='float32')
    other = np.full_like(x,.8)
    np.testing.assert_array_equal(transform_images(x,0,'cross_class_mix',other),x)
    np.testing.assert_array_equal(transform_images(x,1,'cross_class_mix',other),other)
    np.testing.assert_allclose(transform_images(x,.5,'cross_class_mix',other),.5)
    np.testing.assert_array_equal(transform_images(x,0,'gaussian_noise',noise=np.ones_like(x)),x)
    assert transform_images(x,1,'gaussian_noise',noise=np.ones_like(x)*100).max() == 1


def test_density_and_bootstrap_scores():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-3,.2,(30,5)),rng.normal(3,.2,(30,5))]
    y = np.repeat([0,1],30)
    density = GaussianDensity(3).fit(x,y)
    assert density.score(x+100).mean() > density.score(x).mean()
    heads = BootstrapHeads(3,seed=0).fit(x,y)
    heads2 = BootstrapHeads(3,seed=0).fit(x,y)
    np.testing.assert_allclose(heads.probabilities(x),heads2.probabilities(x))
    scores, pred = score_bundle(x,np.column_stack([-x[:,0],x[:,0]]),density,heads)
    assert (pred==y).mean() > .9
    assert (scores['EU'] >= 0).all()
    assert (scores['EU'] <= scores['ensemble_entropy']+1e-10).all()
    np.testing.assert_allclose(scores['EU'],scores['ensemble_entropy']-scores['mean_member_entropy'],atol=1e-10)


def test_thresholds_and_auc():
    s = np.arange(100)
    thresholds = thresholds_from_known({'s':s},.05)
    assert (s>thresholds['s']).mean() <= .05
    assert shift_auc(s,s) == .5
    contrast = paired_auc_difference(s,s+200,s,s+200,repeats=10)
    assert contrast == dict(difference=0.,lower=0.,upper=0.)


@pytest.mark.parametrize('level', [-1,2,np.nan])
def test_invalid_strength(level):
    with pytest.raises(ValueError):
        transform_images(np.zeros((1,4,4,3)),level,'gaussian_noise',noise=np.zeros((1,4,4,3)))

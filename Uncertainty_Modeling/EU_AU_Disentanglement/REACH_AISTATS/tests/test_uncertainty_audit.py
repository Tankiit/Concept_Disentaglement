import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from real.uncertainty_audit import (mc_probs, class_vs_known, low_eu_subset,
    paired_auc_difference, pgd_l2, linear_boundary_path, cluster_mean_difference)


class TestAudit(unittest.TestCase):
    def test_example_evaluation_and_optional_neighbor_backend(self):
        import example
        head=example.Head(d=4,k=2)
        loader=example.make_loader(np.ones((8,4),np.float32),np.array([0,1]*4),
                                   np.arange(8),batch_size=2)
        loss,eu=example.evaluate(head,loader,torch.tensor([0,1]),T=4)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(eu),8)
        self.assertFalse(head.training)
        with patch.object(example,'faiss',None):
            ix=example.fit_class_indices(np.array([[0.,0.],[3.,4.]]),np.array([0,0]),1)[0]
            np.testing.assert_allclose(example.kth_dist(ix,np.array([[0.,0.]]),2),[5.])

    def test_mc_restores_modes_and_batchnorm_statistics(self):
        head = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5), torch.nn.Linear(3,2))
        head.train(); head[0].eval()
        modes = [m.training for m in head.modules()]
        mean = head[0].running_mean.clone()
        p = mc_probs(head, torch.randn(12,3), 4)
        self.assertEqual(modes, [m.training for m in head.modules()])
        torch.testing.assert_close(mean, head[0].running_mean)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)

    def test_class_and_low_eu_masks(self):
        novel = np.array([False,False,True,True,True,True])
        labels = np.array([1,1,0,0,3,3])
        select, y = class_vs_known(novel, labels, 0)
        np.testing.assert_array_equal(select,[1,1,1,1,0,0])
        np.testing.assert_array_equal(y,[0,0,1,1])
        select, y = low_eu_subset(novel, np.arange(6), 3)
        np.testing.assert_array_equal(select,[1,1,1,0,0,0])

    def test_paired_comparison_identical_scores(self):
        y = np.array([0,0,1,1]); s = np.array([0.,.4,.3,1.])
        result = paired_auc_difference(y,s,s,n_boot=20,pairs=True)
        self.assertEqual(result,dict(difference=0.,lower=0.,upper=0.))

    def test_cluster_bootstrap_does_not_count_repeats_as_new_images(self):
        v=np.array([0.,.1,.8,.9]); group=np.array([0,0,1,1],dtype=bool); ids=np.arange(4)
        a=cluster_mean_difference(v,group,ids,n_boot=50)
        b=cluster_mean_difference(np.tile(v,3),np.tile(group,3),np.tile(ids,3),n_boot=50)
        for key in a:self.assertAlmostEqual(a[key],b[key])

    def test_attack_budget_and_mode(self):
        h = torch.nn.Sequential(torch.nn.Dropout(.5),torch.nn.Linear(3,2)).eval()
        x = torch.randn(8,3); y = h(x).argmax(1)
        for eps in (0.,.03,.2):
            adv, delta = pgd_l2(h,x,y,eps,steps=5)
            self.assertLessEqual(float(delta.norm(dim=1).max()),eps+1e-6)
            torch.testing.assert_close(adv-x,delta)
            self.assertFalse(h.training)

    def test_exact_boundary_and_deep_wrong(self):
        w=np.array([[1.,0.],[-1.,0.]]); b=np.zeros(2)
        x=np.array([[2.,1.],[-2.,1.]]); y=np.array([0,1])
        at, info = linear_boundary_path(w,b,x,y,1.)
        np.testing.assert_allclose(at@w.T,0.)
        deep,_=linear_boundary_path(w,b,x,y,2.)
        self.assertTrue(np.all((deep@w.T).argmax(1)!=y))
        np.testing.assert_allclose(info['distance'],2.)

    def test_notebook_five_epoch_optimizer_and_no_stale_outputs(self):
        p=Path(__file__).resolve().parents[1]/'eu_blindspot.ipynb'
        nb=json.loads(p.read_text())
        cell=next(''.join(c['source']) for c in nb['cells'] if 'o5 = torch.optim.Adam' in ''.join(c['source']))
        self.assertIn('l.backward(); o5.step()',cell)
        self.assertNotIn('l.backward(); o.step()',cell)
        for c in nb['cells']:
            if c['cell_type']=='code':
                self.assertFalse(c.get('outputs'))
                # Colab's pip magic is the only non-Python line.
                code=''.join(line for line in c['source'] if not line.lstrip().startswith('%'))
                compile(code,'notebook','exec')

    def test_notebook_five_epoch_loop_changes_its_own_parameters(self):
        p=Path(__file__).resolve().parents[1]/'eu_blindspot.ipynb'
        nb=json.loads(p.read_text())
        code=next(''.join(c['source']) for c in nb['cells'] if 'o5 = torch.optim.Adam' in ''.join(c['source']))
        start=code.index('torch.manual_seed(0)\nh5 =')
        end=code.index('# etf head',start)
        created=[]
        def factory(d,k_total):
            h=torch.nn.Linear(d,k_total)
            created.append((h,[p.detach().clone() for p in h.parameters()]))
            return h
        scope=dict(torch=torch,F=torch.nn.functional,HeadDrop=factory,
                   X=np.zeros((8,3)),K=1,
                   train_ld=[(torch.ones(8,3),torch.zeros(8,dtype=torch.long))])
        exec(code[start:end],scope)
        h,initial=created[0]
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(h.parameters(),initial)))


if __name__=='__main__':
    unittest.main()

"""Synthetic fixtures validate mechanics only; they are not empirical results."""
import unittest
import numpy as np
import pandas as pd
from real.wilds_records import group_partition, image_sample, location_groups
from real.wilds_vision import label_bounds, dappr_loss
from real.wilds_counterexample import target_interval
import torch
from pathlib import Path
import tempfile
from real.wilds_records import read_records

class TestWildsAcquisition(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(source=s,group=f'{s}_{g}',target=(i+g)%2,record_id=f'{s}_{g}_{i}')
                             for s in range(3) for g in range(10) for i in range(5)])

    def test_groups_do_not_leak_and_labels_cannot_select_roles(self):
        f=self.fixture();a=group_partition(f)
        f['target']=1-f.target;b=group_partition(f)
        self.assertTrue((a.role==b.role).all())
        self.assertEqual(a.groupby('group').role.nunique().max(),1)
        self.assertEqual(set(a.role),{'fit','reference','evaluation'})

    def test_small_group_count_rejected(self):
        with self.assertRaises(ValueError):group_partition(self.fixture().query("group == '0_0'"))

    def test_sample_reproducible_and_unique(self):
        f=group_partition(self.fixture())
        a=image_sample(f,7);b=image_sample(f,7)
        self.assertEqual(a.record_id.tolist(),b.record_id.tolist())
        self.assertFalse(a.record_id.duplicated().any())
        self.assertLessEqual(a.groupby(['role','source']).size().max(),7)

    def test_location_repeats_grouped(self):
        f=pd.DataFrame(dict(lat=[10.001,10.002],lon=[20.001,20.002]))
        self.assertEqual(location_groups(f).nunique(),1)
        with self.assertRaises(ValueError):location_groups(pd.DataFrame(dict(lat=[1])))

    def test_label_bounds_use_cell_conditional_rates(self):
        laws=np.array([[.8,.2],[.2,.8],[.5,.5]])
        theta=np.array([[.1,.9],[.8,.2]])
        first=label_bounds(laws,laws,theta,theta,0)
        second=label_bounds(laws,laws,theta,theta,1)
        self.assertAlmostEqual(first['lower'],.8*.1+.2*.8)
        self.assertAlmostEqual(second['lower'],.2*.9+.8*.2)
        self.assertAlmostEqual(first['width'],0)

    def test_credible_enclosure_contains_parameter_box_samples(self):
        shapes=np.array([[2.,4.],[3.,5.],[6.,2.]])
        lo,hi=target_interval(shapes)
        self.assertTrue(0<=lo<hi<=1)
        from scipy.stats import beta
        lower=beta.ppf(.05/6,shapes[:,0],shapes[:,1])
        upper=beta.ppf(1-.05/6,shapes[:,0],shapes[:,1])
        points=np.random.default_rng(0).uniform(lower,upper,(1000,3))
        target=(1-points[:,0])*points[:,1]+points[:,0]*points[:,2]
        self.assertGreaterEqual(target.min(),lo);self.assertLessEqual(target.max(),hi)

    def test_dappr_finite_gradient(self):
        logits=torch.tensor([[0.,1.],[-1.,2.]],requires_grad=True)
        loss=dappr_loss(logits,torch.tensor([0,1]),2e-4);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())

    def test_official_fmow_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame({'alpha-3':['USA','FRA','JPN'],'region':['Americas','Europe','Asia']}).to_csv(root/'country_code_mapping.csv',index=False)
            pd.DataFrame(dict(country_code=['USA','FRA','JPN'],category=['airport','zoo','airport'],
                              split=['train','val','test'],timestamp=['2010-01-01','2012-01-01','2017-01-01'],
                              lat=[1.,2.,3.],lon=[4.,5.,6.])).to_csv(root/'rgb_metadata.csv',index=False)
            f=read_records('fmow',root)
            self.assertEqual(f.source.tolist(),[0,1,2])
            self.assertEqual(f.temporal_ood.tolist(),[False,False,True])
            self.assertEqual(f.target.tolist(),[0,1,0])
            self.assertTrue(f.image_file.iloc[2].endswith('images/rgb_img_2.png'))

    def test_mirror_fmow_adapter(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            f=pd.DataFrame(dict(label=[3,2,1],domain_labels_readable=[{'region':s} for s in ['Americas','Europe','Asia']],
                                lat=[1.,2.,3.],lon=[4.,5.,6.],timestamp=['2010-01-01']*3))
            pq.write_table(pa.Table.from_pandas(f),Path(tmp)/'train-00000-of-00001.parquet')
            actual=read_records('fmow',tmp)
            self.assertEqual(actual.source.tolist(),[0,1,2])
            self.assertEqual(actual.target.tolist(),[3,2,1])
            self.assertEqual(actual.image_row.tolist(),[0,1,2])

if __name__=='__main__':unittest.main()

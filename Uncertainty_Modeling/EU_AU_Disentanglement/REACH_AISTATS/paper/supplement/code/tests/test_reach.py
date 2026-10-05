import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from data.worlds import make_world,make_separation_world,separation_gap,tilted_conditionals
from data.acquisitions import draw_log,draw_labelled_set
from models.ball import MMDBall,embedding_radius
from models.widths import label_intervals,credal_widths
from models.baselines import em_prevalence,feature_match
from test.inference import paired_contrast,seed_bootstrap_ci,equivalence_flat
from test.precision import contrast_precision
from train.cache import run_cache
from train.conditionals import condition_rows,summarise as conditional_summary
from train.matched import select_pair
from train.separation import separation_rows,summarise as separation_summary

class TestReach(unittest.TestCase):
    def test_exact_identified_weights(self):
        world=make_world(); ball=MMDBall(world.conditionals,world.marginal,0.)
        interval=ball.ratio_interval(world.conditionals[:,0]*world.theta,
                                     world.conditionals[:,0]*world.theta,world.conditionals[:,0])
        self.assertEqual(interval['status'],'optimal')
        self.assertAlmostEqual(interval['lower'],world.target(0),places=7)
        self.assertAlmostEqual(interval['upper'],world.target(0),places=7)
    def test_confounded_novel_mass_persists(self):
        world=make_world('confounded');ball=MMDBall(world.conditionals,world.marginal,0.)
        result=credal_widths(ball,0,world.theta[:2],world.theta[:2])
        self.assertGreater(result['W_unk'],.1)
        self.assertLessEqual(result['total_lower'],world.target(0)+1e-8)
        self.assertGreaterEqual(result['total_upper'],world.target(0)-1e-8)
    def test_zero_support_policy(self):
        world=make_world();ball=MMDBall(world.conditionals,world.marginal,0.)
        result=credal_widths(ball,3,world.theta[:2],world.theta[:2])
        self.assertEqual(result['total_status'],'zero_mass_policy')
        self.assertEqual(result['W_total'],1.)
        self.assertTrue(np.isnan(world.target(3)))
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): embedding_radius(0)
        with self.assertRaises(ValueError): draw_labelled_set(make_world(),10,.5,0)
        with self.assertRaises(ValueError): MMDBall.from_log(make_world().conditionals,np.array([-1]),.05)
    def test_empty_set(self):
        world=make_world();ball=MMDBall(world.conditionals,np.array([0,0,0,1.]),0.)
        self.assertFalse(ball.feasible())
        self.assertEqual(credal_widths(ball,0,world.theta[:2],world.theta[:2])['total_status'],'empty_set')
    def test_labels_ignore_log(self):
        world=make_world();lo,hi=label_intervals(draw_labelled_set(world,256,.2,2),.2)
        outputs=[]
        for n in [64,1024]:
            ball=MMDBall.from_log(world.conditionals,draw_log(world,n,2))
            outputs.append(credal_widths(ball,0,lo,hi)['W_label'])
        self.assertEqual(outputs[0],outputs[1])
    def test_matched_contrast_holds_nuisances(self):
        frame=pd.DataFrame([dict(world_id='w',probe_id=0,seed=0,n=n,m=m,eta=0,W_cls=n+m)
                            for n in [10,20] for m in [2,4]])
        result=paired_contrast(frame,'W_cls','n',10,20)
        self.assertEqual(len(result),2)
        np.testing.assert_array_equal(result.difference,[10,10])
    def test_bootstrap_groups(self):
        result=seed_bootstrap_ci([1,1,-1,-1],[0,0,1,1],100)
        self.assertEqual(result['independent_replicates'],2)
        self.assertEqual(result['estimate'],0)
        self.assertFalse(equivalence_flat(result,.01))
    def test_baseline_misspecification(self):
        world=make_world()
        self.assertIsNone(em_prevalence(world.conditionals[:2],np.array([2])))
        fit=feature_match(world.conditionals,world.marginal)
        self.assertLess(fit['residual'],1e-6)
    def test_joint_truth_inclusion(self):
        world=make_world();count=0
        for seed in range(30):
            ball=MMDBall.from_log(world.conditionals,draw_log(world,256,seed),.025)
            lo,hi=label_intervals(draw_labelled_set(world,256,.2,seed),.2,.025)
            for cell in [0,1,2]:
                result=credal_widths(ball,cell,lo,hi)
                count+=result['total_lower']-1e-8<=world.target(cell)<=result['total_upper']+1e-8
        self.assertGreaterEqual(count,85)  # numerical integration check, not proof of coverage
    def test_cached_feature_adapter(self):
        rng=np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=np.repeat([0,1],100)
            ref=rng.normal(size=(200,2))+source[:,None]*3
            np.savez(root/'cache.npz',X_partition=rng.normal(size=(100,2)),X_reference=ref,
                     source_reference=source,target_reference=(rng.random(200)<np.where(source==0,.2,.8)).astype(int),
                     X_log=ref[:50],X_eval=ref[50:60],target_eval=np.zeros(10,dtype=int))
            run_cache(root/'cache.npz',root/'out',cells=3)
            self.assertTrue((root/'out/rows.csv').exists())

class TestFailureWorldsAndPlanning(unittest.TestCase):
    """Experiments 3b, 5 and 6, plus replication planning. Grids are deliberately tiny:
    these check the machinery and the direction of each claim, not the reported evidence."""
    def test_separation_gap_matches_construction(self):
        for sep in (1.,.25):
            self.assertAlmostEqual(separation_gap(make_separation_world(sep)),.4*sep*np.sqrt(2))
        with self.assertRaises(ValueError): make_separation_world(0.)
    def test_small_gap_resolves_slower_without_losing_coverage(self):
        summary=separation_summary(separation_rows([1.,.1],[512],range(6)),.2)
        self.assertEqual(summary['best_case']['sep'],1.)
        self.assertEqual(summary['worst_case']['sep'],.1)
        # A near-singular embedding must NOT report the narrower interval.
        self.assertGreater(summary['worst_case']['W_cls_at_max_n'],
                           summary['best_case']['W_cls_at_max_n'])
        self.assertTrue(summary['validity_preserved_at_all_separations'])
    def test_tilt_leaves_declared_family(self):
        world=make_world();tilted=tilted_conditionals(world,.5)
        np.testing.assert_allclose(tilted[:2,-1],.5*world.conditionals[:2,0])
        moved=float(world.weights[:2]@tilted[:2,-1])
        residual=feature_match(world.conditionals,world.weights@tilted)['residual']
        self.assertGreaterEqual(residual,moved-1e-9)  # unsupported cell bounds the residual below
        with self.assertRaises(ValueError): tilted_conditionals(world,1.5)
    def test_dropping_reference_radius_loses_coverage(self):
        world=make_world()
        frame=pd.concat([condition_rows(world,condition,[32],[512],range(6),m=64)
                         for condition in ('positive_control','ignored_reference')],ignore_index=True)
        table={row['condition']:row for row in conditional_summary(frame)['table']}
        self.assertGreater(table['positive_control']['mean_radius'],
                           table['ignored_reference']['mean_radius'])
        self.assertGreaterEqual(table['positive_control']['weight_coverage_rate'],
                                table['ignored_reference']['weight_coverage_rate'])
    def test_shifted_deployment_is_flagged_incompatible(self):
        world=make_world()
        frame=condition_rows(world,'conditional_shift',[2048],[2048],range(4),shift=.5,m=64)
        self.assertTrue(frame.incompatible.all())
        self.assertTrue((frame.declared_family_residual>frame.radius).all())
    def test_matched_selection_respects_band_and_ceiling(self):
        arms=[(.5,.3,.05),(.505,.05,.3),(.95,.9,0.)]  # third arm sits above the vacuous ceiling
        frame=pd.DataFrame([dict(arm_id=i,arm=f'a{i}',seed=seed,baseline=base,
                                 gain_log=gl,gain_label=gb)
                            for i,(base,gl,gb) in enumerate(arms) for seed in range(3)])
        pair=select_pair(frame,band=.02,max_baseline=.92)
        self.assertEqual((pair['log_favoured_arm'],pair['label_favoured_arm']),(0,1))
        self.assertIsNone(select_pair(frame,band=1e-6,max_baseline=.92))
    def test_seed_shards_are_disjoint_replications(self):
        """Two shards must not recompute the same draws, or 'more seeds' buys nothing."""
        world=make_world()
        self.assertFalse(np.array_equal(draw_log(world,512,3),draw_log(world,512,4)))
        # n=4096 is above the vacuous ceiling; at small n every width pins to 1 for
        # structural reasons and would hide a shared-draw bug.
        first=separation_rows([1.],[4096],range(0,2))
        second=separation_rows([1.],[4096],range(2,4))
        self.assertEqual(set(first.seed)&set(second.seed),set())
        probe=lambda frame:frame[frame.region=='on_support'].W_cls.to_numpy()
        self.assertLess(probe(first).max(),1.)
        self.assertFalse(np.allclose(probe(first),probe(second)))
    def test_precision_planning_scales_with_target(self):
        frame=pd.DataFrame([dict(world_id='w',probe_id=0,region='on_support',seed=seed,n=n,m=4,
                                 eta=0.,W_cls=.1*n+(.05*seed if n==20 else 0.))
                            for n in (10,20) for seed in range(5)])
        predicted={('W_cls','n','w'):1}
        loose=contrast_precision(frame,predicted,.5,n_boot=200)[0]
        tight=contrast_precision(frame,predicted,.01,n_boot=200)[0]
        self.assertEqual(loose['independent_seeds'],5)
        self.assertGreater(tight['seeds_for_target'],loose['seeds_for_target'])
        self.assertTrue(loose['target_met'])
        self.assertFalse(tight['target_met'])


class TestResponseMatrix(unittest.TestCase):
    """Slopes, the placebo null, the exact L2 ball, and the resolution gate."""
    def test_slope_recovers_rate_and_flags_constants(self):
        from test.slopes import signed_slope
        grid=np.array([1e3,2e3,4e3,8e3]); rng=np.random.default_rng(0)
        widths=grid[None,:]**-.5*(1+.01*rng.standard_normal((20,4)))
        res=signed_slope(grid,widths,n_boot=200)
        self.assertAlmostEqual(res['slope'],-.5,delta=.02)
        self.assertLess(res['lower'],res['slope']); self.assertGreater(res['upper'],res['slope'])
        self.assertFalse(res['constant'])
        # Per-seed constant across the grid but different between seeds: invariance by construction.
        self.assertTrue(signed_slope(grid,np.repeat(rng.random((20,1)),4,1),n_boot=50)['constant'])
    def test_placebo_null_calls_noise_flat(self):
        from test.slopes import signed_slope,placebo_slopes,decide,agrees
        grid=np.array([1.,2.,4.,8.]); rng=np.random.default_rng(1)
        res=signed_slope(grid,1+.05*rng.standard_normal((30,4)),n_boot=200)
        null=placebo_slopes(grid,1+.05*rng.standard_normal(30*4),30,500)
        self.assertEqual(decide(res,null,.01)['decision'],'~0')
        self.assertTrue(agrees('~0','exact_constant')); self.assertFalse(agrees('exact','~0'))
    def test_l2_ball_is_tighter_and_still_covers(self):
        world=make_world(); log=draw_log(world,2048,3)
        d=world.conditionals[:,0]; num=np.zeros(3); num[0]=d[0]; den=d.copy(); den[-1]=0
        truth=world.weights[0]*d[0]/(world.weights[:2]@d[:2])
        box=MMDBall.from_log(world.conditionals,log,.025).ratio_interval(num,num,den)
        l2=MMDBall.from_log(world.conditionals,log,.025,set_kind='l2').ratio_interval(num,num,den)
        self.assertLessEqual(box['lower'],l2['lower']+1e-7); self.assertGreaterEqual(box['upper'],l2['upper']-1e-7)
        self.assertLessEqual(l2['lower'],truth); self.assertGreaterEqual(l2['upper'],truth)
        zero=MMDBall.from_log(world.conditionals,log,.025,set_kind='l2').ratio_interval(
            np.zeros(3),np.zeros(3),world.conditionals[:,3])
        self.assertEqual(zero['status'],'zero_mass_policy')
    def test_batched_em_matches_single(self):
        from models.baselines import em_batch,em_from_counts
        world=make_world(); counts=np.random.default_rng(2).multinomial(4096,world.marginal,size=3)
        batch=em_batch(world.conditionals,counts)
        for row,c in zip(batch,counts):
            np.testing.assert_allclose(row,em_from_counts(world.conditionals,c),atol=1e-6)
        self.assertEqual(len(em_batch(world.conditionals[:2],np.array([[0,0,1,0]]))),0)
    def test_run_cell_is_paired_across_budgets(self):
        from train.response import run_cell
        world=make_world()
        small=pd.DataFrame(run_cell(world,1024,64,0.,5,n_boot=20,members=5))
        large=pd.DataFrame(run_cell(world,4096,64,0.,5,n_boot=20,members=5))
        # Same labels at both n: the label channel is identical, and widths are in [0, 1].
        np.testing.assert_array_equal(small.W_label.fillna(-1),large.W_label.fillna(-1))
        on=lambda f:f[f.region=='on_support'].W_cls.item()
        self.assertGreater(on(small),on(large))
        self.assertTrue(((large[['W_cls','W_unk']]>=0)&(large[['W_cls','W_unk']]<=1)).all().all())
    def test_resolution_gate_refuses_unresolved_ball(self):
        from train.response import ball_resolution_gate
        frame=pd.DataFrame(dict(axis=['n','n'],n=[64,128],ball_resolution=[.8,1.5]))
        with self.assertRaises(RuntimeError): ball_resolution_gate(frame)
        self.assertTrue(ball_resolution_gate(frame.assign(ball_resolution=[1.2,1.5]))['passed'])


class TestRealAuditPieces(unittest.TestCase):
    """Label-noise debiasing, simplex projection, and per-seed Dirichlet weights (no feature cache)."""
    def _ctx(self, k=4, d=6, size=4000):
        rng=np.random.default_rng(0)
        rows=rng.dirichlet(np.ones(d),size=k)
        groups=[(j,) for j in range(k)]
        return dict(groups=groups,d=d,delta=0.,prevalence='dirichlet',dirichlet_alpha=1.,
                    ref_cells={g:[rng.choice(d,size,p=rows[j])] for j,g in enumerate(groups)}),rows
    def test_simplex_projection(self):
        from real.audit import _simplex_projection
        for v in (np.array([.2,.3,.5]),np.array([1.4,-.2,-.1]),np.array([.5,.5,.5])):
            w=_simplex_projection(v)
            self.assertAlmostEqual(w.sum(),1.); self.assertTrue(np.all(w>=0))
        np.testing.assert_allclose(_simplex_projection(np.array([.2,.3,.5])),[.2,.3,.5])
    def test_noisy_reference_is_debiased_and_inflated(self):
        import real.audit as A
        ctx,rows=self._ctx()
        clean,_,r0,e0=A._reference(ctx,4000,1,0.)
        noisy,_,r1,e1=A._reference(ctx,4000,1,.2)
        _,_,rs,_=A._reference(ctx,4000,1,.0125)
        self.assertGreater(r1,r0); self.assertGreater(e1,e0)
        self.assertLess(rs,r1)  # composition term shrinks with eta: the axis starts near r0
        # Debiased rows are close to the truth, not to the noise-mixed rows.
        M=.8*np.eye(4)+.2/3*(np.ones((4,4))-np.eye(4))
        self.assertLess(np.abs(noisy-rows).max(),.05)
        self.assertGreater(np.abs(M@rows-rows).max(),np.abs(noisy-rows).max())
        # Same flips across eta: a larger eta flips a superset of labels, so the grid stays paired.
        a=A._reference(ctx,1000,2,.1)[0]; b=A._reference(ctx,1000,2,.1)[0]
        np.testing.assert_array_equal(a,b)
    def test_dirichlet_weights_are_per_seed_and_shared_across_budgets(self):
        import real.audit as A
        ctx,_=self._ctx()
        w0,w0b,w1=A.true_weights(ctx,0),A.true_weights(ctx,0),A.true_weights(ctx,1)
        np.testing.assert_array_equal(w0,w0b); self.assertFalse(np.allclose(w0,w1))
        self.assertAlmostEqual(w0.sum(),1.)

if __name__=='__main__': unittest.main()

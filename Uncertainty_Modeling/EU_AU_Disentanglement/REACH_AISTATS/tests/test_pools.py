"""data.pools: disjoint dev / reference / log / evaluation pools over cached features."""
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from data import pools
from real.head import load

CORRUPTION = 'brightness'  # one of the locally intact CIFAR-10-C files


class TestSplit(unittest.TestCase):
    def test_log_and_eval_halves_are_disjoint_and_exhaustive(self):
        log_idx, eval_idx = pools.split_test()
        self.assertEqual(len(log_idx) + len(eval_idx), 10000)
        self.assertEqual(len(np.intersect1d(log_idx, eval_idx, assume_unique=True)), 0)
        self.assertTrue(np.array_equal(np.r_[log_idx, eval_idx], np.arange(10000)))


class TestPools(unittest.TestCase):
    def test_pools_are_disjoint_reproducible_and_row_aligned(self):
        a = pools.load_pools(seed=0)
        a2 = pools.load_pools(seed=0)
        b = pools.load_pools(seed=1)
        pools.check_disjoint(a)
        for key in ('dev', 'reference', 'log', 'eval'):
            self.assertEqual(len(a[key]['X']), len(a[key]['y']))
            self.assertEqual(len(a[key]['X']), len(a[key]['index']))
            # Same seed: identical split. Across seeds: dev/reference move; the test
            # split is contiguous, so log/eval indices are fixed by construction.
            for field in ('X', 'y', 'index'):
                self.assertTrue(np.array_equal(a[key][field], a2[key][field]))
        self.assertFalse(np.array_equal(a['dev']['index'], b['dev']['index']))
        self.assertTrue(np.array_equal(a['log']['index'], b['log']['index']))
        # Eval rows are exactly the clean test rows at their indices.
        test = load('cifar10_test')
        self.assertTrue(np.array_equal(a['eval']['X'], test['X'][a['eval']['index']]))
        self.assertTrue(np.array_equal(a['eval']['y'], test['y'][a['eval']['index']]))

    def test_reference_stratification_takes_m_per_class(self):
        p = pools.load_pools(seed=0, m_per_class=50)
        counts = np.bincount(p['reference']['y'], minlength=10)
        self.assertEqual(counts.tolist(), [50] * 10)
        self.assertEqual(len(p['reference']['index']), 500)

    def test_corruption_log_matches_severity_block_and_stays_disjoint(self):
        p = pools.load_pools(seed=0, corruption=CORRUPTION, severity=3)
        pool = load(f'cifar10c_{CORRUPTION}')
        test = load('cifar10_test')
        rows = (3 - 1) * 10000 + p['log']['index']  # severity s lives in 0-based block s-1
        self.assertTrue(np.array_equal(p['log']['X'], pool['X'][rows]))
        self.assertTrue(np.array_equal(p['log']['y'], test['y'][p['log']['index']]))
        pools.check_disjoint(p)

    def test_misaligned_corruption_file_is_rejected(self):
        genuine = load(f'cifar10c_{CORRUPTION}')
        test = load('cifar10_test')
        genuine['y'][:100] = (test['y'][:100] + 1) % 10  # break label alignment on the log half
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / f'cifar10c_{CORRUPTION}.npz', **genuine)
            for name in ('cifar10_train', 'cifar10_test'):
                shutil.copy(Path('cache/features') / f'{name}.npz', Path(tmp) / f'{name}.npz')
            with self.assertRaises(ValueError):
                pools.load_pools(seed=0, corruption=CORRUPTION, severity=1, root=tmp)

    def test_check_disjoint_flags_overlapping_log_and_eval(self):
        p = pools.load_pools(seed=0)
        p['log']['index'] = p['eval']['index'].copy()
        with self.assertRaises(ValueError):
            pools.check_disjoint(p)


if __name__ == '__main__':
    unittest.main()

"""oss_from_cache.make_loader: dict-batch DataLoader over cached feature rows."""
import unittest
import numpy as np
import torch

from oss_from_cache import group_loaders, make_loader, make_oss

CACHE = 'cache/features/cifar10_test.npz'


class TestMakeLoader(unittest.TestCase):
    def test_batches_are_dicts_with_keys_and_values(self):
        X = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        y = np.arange(10) % 3
        batches = list(make_loader(X, y, batch_size=4))
        self.assertEqual([len(b['index']) for b in batches], [4, 4, 2])
        for b in batches:
            self.assertEqual(set(b.keys()), {'X', 'y', 'index'})
            self.assertEqual(b['X'].dtype, torch.float32)
            self.assertEqual(b['y'].dtype, torch.int64)
            # every value is traceable to its original row via 'index'
            for j in range(len(b['index'])):
                self.assertTrue(np.array_equal(b['X'][j].numpy(), X[b['index'][j]]))
                self.assertEqual(int(b['y'][j]), y[int(b['index'][j])])

    def test_indices_select_rows_in_order(self):
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = np.zeros(10, dtype=np.int64)
        b = next(iter(make_loader(X, y, indices=[5, 3, 1], batch_size=8)))
        self.assertTrue(np.array_equal(b['index'].numpy(), [5, 3, 1]))
        self.assertTrue(np.array_equal(b['X'].numpy(), X[[5, 3, 1]]))

    def test_shuffle_is_seed_reproducible(self):
        X = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)
        y = np.zeros(64, dtype=np.int64)
        order = lambda seed: [int(i) for b in make_loader(X, y, batch_size=16, shuffle=True, seed=seed)
                              for i in b['index'].tolist()]
        self.assertEqual(order(7), order(7))
        self.assertNotEqual(order(7), order(8))

    def test_extra_arrays_flow_through_by_key(self):
        X = np.zeros((6, 2), dtype=np.float32)
        y = np.arange(6) % 2
        cells = np.arange(6) % 3
        b = next(iter(make_loader(X, y, indices=[4, 2], batch_size=8, cell=cells)))
        self.assertIn('cell', b)
        self.assertTrue(np.array_equal(b['cell'].numpy(), [1, 2]))

    def test_integration_with_oss_split(self):
        world = make_oss(CACHE, seed=0)
        loader = make_loader(world['X'], world['y'], indices=world['idx']['val'],
                             batch_size=512, shuffle=True, seed=0)
        seen = set()
        for b in loader:
            self.assertEqual(set(b.keys()), {'X', 'y', 'index'})
            seen.update(b['index'].tolist())
        self.assertEqual(sorted(seen), sorted(world['idx']['val'].tolist()))


class TestGroupLoaders(unittest.TestCase):
    def test_groups_partition_rows_and_are_class_pure(self):
        gl = group_loaders(CACHE, batch_size=4096, shuffle=False, seed=0)
        world = make_oss(CACHE, seed=0)
        for split in ('train', 'val', 'test'):
            seen = []
            for g, classes_ in (('kkc', world['kkc']), ('kuc', world['kuc']), ('uuc', world['uuc'])):
                rows, ys = [], []
                for b in gl[split][g]:
                    self.assertEqual(set(b.keys()), {'X', 'y', 'index', 'group'})
                    self.assertTrue((b['group'] == {'kkc': 0, 'kuc': 1, 'uuc': 2}[g]).all())
                    rows += b['index'].tolist()
                    ys += b['y'].tolist()
                self.assertTrue(set(ys) <= {int(c) for c in classes_} if classes_ else not ys,
                                f'{split}/{g}: classes leak across groups')
                seen += rows
            self.assertEqual(sorted(seen), sorted(world['idx'][split].tolist()),
                             f'{split}: group loaders do not partition the split')

    def test_kuc_loaders_are_empty_under_default_budget(self):
        gl = group_loaders(CACHE, seed=0)     # kuc=0 -> no known-unknown rows anywhere
        for split in ('train', 'val', 'test'):
            self.assertEqual(list(gl[split]['kuc']), [])

    def test_group_shuffle_is_seed_reproducible(self):
        first = [int(i) for b in group_loaders(CACHE, batch_size=512, shuffle=True, seed=3)['val']['uuc']
                 for i in b['index'].tolist()]
        again = [int(i) for b in group_loaders(CACHE, batch_size=512, shuffle=True, seed=3)['val']['uuc']
                 for i in b['index'].tolist()]
        self.assertEqual(first, again)


if __name__ == '__main__':
    unittest.main()

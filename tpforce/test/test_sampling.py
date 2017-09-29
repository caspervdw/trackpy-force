from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np
import pandas as pd

from tpforce.sampling import _find_pairs, extract_pairs

import unittest
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_equal, assert_array_less)


class RandomWalker(object):
    def __init__(self, N, length, ds, shape, pos_columns=None):
        self.shape = shape
        self._len = length
        self.N = N
        self.ds = ds
        if pos_columns is None:
            self.pos_columns = ['z', 'y', 'x'][-len(shape):]
        else:
            assert len(pos_columns) == len(shape)
            self.pos_columns = pos_columns
        self.generate()

    def __len__(self):
        return self._len

    def generate(self):
        startpos = np.array([np.random.random(self.N) * s for s in self.shape]).T
        steps = (np.random.random((self._len, self.N, len(self.shape))) - 0.5) * self.ds
        self.pos = steps.cumsum(axis=0) + np.array(startpos)[np.newaxis, :, :]

    def f(self, noise=0):
        result = self.pos + np.random.random(self.pos.shape) * noise
        result = pd.DataFrame(result.reshape((result.shape[0] * result.shape[1], result.shape[2])),
                              columns=self.pos_columns)
        result['frame'] = np.repeat(np.arange(self._len), self.N)
        result['particle'] = np.tile(np.arange(self.N), self._len)
        return result


class TestFindPairs(unittest.TestCase):
    def test_two_particles(self):
        pairs, _ = _find_pairs([[2, 2], [2, 4]], 1, 0.5)
        assert_array_equal(np.sort(pairs), np.empty((0, 2)))

        pairs, _ = _find_pairs([[2, 2], [2, 3]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[0, 1]])

        pairs, _ = _find_pairs([[2, 2], [2, 2.5]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[0, 1]])

        pairs, _ = _find_pairs([[2, 2], [2, 4]], 0.1, 0.5)
        assert_array_equal(np.sort(pairs), np.empty((0, 2)))

        pairs, _ = _find_pairs([[2, 2], [2, 3]], 0.1, 0.5)
        assert_array_equal(np.sort(pairs), np.empty((0, 2)))

        pairs, _ = _find_pairs([[2, 2], [2, 2.05]], 0.1, 0.5)
        assert_array_equal(np.sort(pairs), [[0, 1]])

    def test_three_particles(self):
        pairs, _ = _find_pairs([[2, 2], [2, 3], [5, 5]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[0, 1]])

        pairs, _ = _find_pairs([[2, 2], [2, 3], [3, 3]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[0, 1], [1, 2]])

        pairs, _ = _find_pairs([[2, 2], [2, 3], [2.5, 3]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[1, 2]])

        pairs, _ = _find_pairs([[2, 2], [2, 3], [2.2, 3]], 1, 0.5)
        assert_array_equal(np.sort(pairs), [[1, 2]])

        pairs, _ = _find_pairs([[2, 2], [2, 2.2], [2.2, 2.2]], 1, 0.5)
        assert_array_equal(np.sort(pairs), np.empty((0, 2)))


class TestRandomWalker(unittest.TestCase):
    def setUp(self):
        self.stepsize = 1
        self.N = 10
        self.length = 5
        self.shape = (64, 64)
        self.rw = RandomWalker(self.N, self.length, self.stepsize, self.shape)
        self.max_pairs = ((self.N * (self.N - 1)) // 2) * (self.length - 1)

    def test_extract_pairs_all(self):
        pairs, _ = extract_pairs(self.rw.f(), 100, 0)
        assert_equal(len(pairs), self.max_pairs)

    def test_extract_pairs_dist2(self):
        prev_count = self.max_pairs
        for dist2 in [20, 10, 5, 1]:
            pairs, _ = extract_pairs(self.rw.f(), dist2, 0.5)
            assert len(pairs) <= prev_count
            assert_array_less(pairs['s0'], dist2)
            prev_count = len(pairs)

    def test_extract_pairs_dist3(self):
        prev_count = self.max_pairs
        for dist3 in [0, 0.5, 2, 10]:
            count = len(extract_pairs(self.rw.f(), 10, dist3)[0])
            assert count <= prev_count
            prev_count = count

    def test_extract_pairs_inperfect(self):
        f = self.rw.f()
        halflen = len(f) // 2
        mask = [True] * halflen + [False] * (len(f) - halflen)
        np.random.shuffle(mask)
        f = f[mask].copy()
        extract_pairs(f, 10, 0.5)

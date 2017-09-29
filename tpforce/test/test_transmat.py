from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np
import pandas as pd

from tpforce.sampling import extract_pairs
from tpforce.transmat import transition_matrix, stationary

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


class TestTransitionMatrix(unittest.TestCase):
    def setUp(self):
        self.stepsize = 1
        self.N = 100
        self.length = 100
        self.shape = (64, 64)
        self.rw = RandomWalker(self.N, self.length, self.stepsize, self.shape)

    def test_trans_mat_many(self):
        pairs, _ = extract_pairs(self.rw.f(), 10, 0.5)
        bins, mat, counts = transition_matrix(pairs, 0, 10, 0.1)
        assert_almost_equal(mat.sum(1), 1.)
        assert mat.shape[0] == mat.shape[1]

    def test_trans_mat_few(self):
        pairs, _ = extract_pairs(self.rw.f(), 10, 1)
        bins, mat, counts = transition_matrix(pairs, 0, 10, 0.1)
        assert mat.shape[0] == mat.shape[1]


class TestStationaryDistribution(unittest.TestCase):
    def test_flat_1d(self):
        rw = RandomWalker(100, 1000, 1, (1024,))

        def out_of_bounds(df):
            return (df['x'] < 0) | (df['x'] >= 1024)

        pairs, _ = extract_pairs(rw.f(), 10, 0, out_of_bounds=out_of_bounds)
        bins, mat, counts = transition_matrix(pairs['s0'].values,
                                              pairs['s1'].values,
                                              bins=np.arange(0, 10.1, 0.1),
                                              norm='2d')

        distr = stationary(mat)

        # mask = (distr_std / distr) < REL_STD_CUTOFF
        # distr = distr[mask]
        # bins = bins[mask]

        import matplotlib.pyplot as plt
        plt.plot(distr)
        plt.show()
        assert np.std(distr)/np.mean(distr) < 0.1
        slope, intercept = np.polyfit(bins, distr, 1)
        assert slope * max(bins) / intercept < 0.05


    def test_flat_2d(self):
        rw = RandomWalker(100, 1000, 1, (64, 64))

        def out_of_bounds(df):
            return (df['x'] < 0) | (df['x'] >= 64) | (df['y'] < 0) | (df['y'] > 64)

        pairs, _ = extract_pairs(rw.f(), 10, 0, out_of_bounds=out_of_bounds)
        bins, mat, counts = transition_matrix(pairs['s0'].values,
                                              pairs['s1'].values,
                                              bins=np.arange(1, 5.1, 0.1),
                                              transform='radial_2d')

        distr = stationary(mat)

        # mask = (distr_std / distr) < REL_STD_CUTOFF
        # distr = distr[mask]
        # bins = bins[mask]

        import matplotlib.pyplot as plt
        plt.plot(distr, marker='o')
        plt.ylim([0, 1.2*np.max(distr)])
        plt.show()
        assert np.std(distr)/np.mean(distr) < 0.1
        slope, intercept = np.polyfit(bins, distr, 1)
        assert slope * max(bins) / intercept < 0.05


    def test_distr_lagt2(self):
        rw = RandomWalker(100, 1000, 1, (64, 64))

        def out_of_bounds(df):
            return (df['x'] < 0) | (df['x'] >= 64) | (df['y'] < 0) | (df['y'] > 64)

        pairs, _ = extract_pairs(rw.f(), 10, 0, out_of_bounds=out_of_bounds,
                                 lagt=2)
        bins, mat, counts = transition_matrix(pairs, 0, 10, 0.1, norm='2d')
        distr = stationary(mat)

        distr = distr[5:]
        bins = bins[5:]

        assert np.std(distr)/np.mean(distr) < 0.1
        slope, intercept = np.polyfit(bins, distr, 1)
        assert slope * max(bins) / intercept < 0.05

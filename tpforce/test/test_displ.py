from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np
import pandas as pd

from rdf.transition import (_find_pairs, extract_pairs_sphere,
                            extract_pairs, transition_matrix, stationary,
                            transition_error, stationary_error)

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


class RandomWalkerSphere(RandomWalker):
    def __init__(self, N, length, ds, R):
        self.R = R
        super(RandomWalkerSphere, self).__init__(N, length, ds, (None, None),
                                                 ['th', 'phi'])

    def generate(self):
        phi = np.random.rand(self.N) * 2*np.pi
        th = np.concatenate((np.arcsin(np.random.rand(self.N // 2)),
                             -np.arcsin(np.random.rand(self.N - (self.N // 2))))) + np.pi/2
        result = np.empty((len(self), self.N, 2), dtype=np.float)
        result[0] = np.asarray([th, phi]).T
        steps = (np.random.random((self._len - 1, self.N, 3)) - 0.5) * self.ds

        for i, step in enumerate(steps):
            th, phi = result[i].T
            z = np.cos(th) * self.R + step[:, 0]
            y = np.sin(th)*np.sin(phi) * self.R + step[:, 1]
            x = np.sin(th)*np.cos(phi) * self.R + step[:, 2]
            R = np.sqrt(x**2 + y**2 + z**2)
            result[i + 1, :, 0] = np.pi/2 - np.arcsin(z / R)
            result[i + 1, :, 1] = np.arctan2(y, x)

        self.pos = result


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


class TestRandomWalkerSphere(unittest.TestCase):
    def setUp(self):
        self.stepsize = 1
        self.N = 10
        self.length = 5
        self.R = 10
        self.rw = RandomWalkerSphere(self.N, self.length, self.stepsize, self.R)
        self.max_pairs = ((self.N * (self.N - 1)) // 2) * (self.length - 1)

    def test_extract_pairs_all(self):
        pairs, _ = extract_pairs_sphere(self.rw.f(), self.R, np.pi*self.R, 0)
        assert_equal(len(pairs), self.max_pairs)

    def test_extract_pairs_dist2(self):
        prev_count = self.max_pairs
        for dist2 in [20, 10, 5, 1]:
            pairs, _ = extract_pairs_sphere(self.rw.f(), self.R, dist2, 0.5)
            assert len(pairs) <= prev_count
            assert_array_less(pairs['s0'], dist2)
            prev_count = len(pairs)

    def test_extract_pairs_dist3(self):
        prev_count = self.max_pairs
        for dist3 in [0, 0.5, 2, 10]:
            count = len(extract_pairs_sphere(self.rw.f(), self.R, 10, dist3)[0])
            assert count <= prev_count
            prev_count = count

    def test_extract_pairs_inperfect(self):
        f = self.rw.f()
        halflen = len(f) // 2
        mask = [True] * halflen + [False] * (len(f) - halflen)
        np.random.shuffle(mask)
        f = f[mask].copy()
        extract_pairs_sphere(f, self.R, 10, 0.5)


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

    def test_distr_sphere(self):
        radius = 10
        rw = RandomWalkerSphere(100, 1000, 1, radius)

        pairs, _ = extract_pairs_sphere(rw.f(), radius, radius*np.pi/2, 0)
        bins, mat, counts = transition_matrix(pairs['s0'].values, pairs['s1'].values,
                                              np.arange(0, radius*np.pi/2, 0.1),
                                              norm='sphere', radius=radius)
        distr = stationary(mat)

        distr = distr[5:]
        bins = bins[5:]


        import matplotlib.pyplot as plt
        plt.plot(distr, marker='o')
        plt.ylim([0, 0.1])
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

    def test_distr_error(self):
        rw = RandomWalker(100, 1000, 1, (64, 64))

        def out_of_bounds(df):
            return (df['x'] < 0) | (df['x'] >= 64) | (df['y'] < 0) | (df['y'] > 64)

        pairs, _ = extract_pairs(rw.f(), 10, 0, out_of_bounds=out_of_bounds)
        bins, mat, counts, mat_std = transition_error(pairs, 0, 10, 0.1,
                                                      norm='2d')
        distr = stationary(mat)
        distr_std = stationary_error(distr, mat_std)
        print(distr_std)
        print(distr_std / distr)

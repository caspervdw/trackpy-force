from __future__ import division, unicode_literals
from rdf import rdf_hist, rdf_kde, rdf_series
import numpy as np
import unittest
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_less)


class _rdf(unittest.TestCase):
    def test_flat_hist(self):
        _, hist = rdf_hist(self.pos, self.mode, self.min_dist,
                           self.max_dist, self.bw, **self.kwargs)
        assert_array_almost_equal(hist, 1, 1)  # 1.0 up to 1 decimal

    def test_flat_kde(self):
        _, hist = rdf_hist(self.pos, self.mode, self.min_dist,
                           self.max_dist, self.bw, **self.kwargs)
        assert_array_almost_equal(hist[2:-2], 1, 1)  # 1.0 up to 1 decimal

    def test_fluct_hist(self):
        std = []
        for bw in [0.1, 0.2, 1, 2, 5]:
            _, hist = rdf_hist(self.pos, self.mode, self.min_dist,
                               self.max_dist, bw, **self.kwargs)
            std.append(hist.std())
        assert np.all(std[1:] < std[:-1])

    def test_fluct_kde(self):
        std = []
        for bw in [0.1, 0.2, 1, 2, 5]:
            _, hist = rdf_kde(self.pos, self.mode, self.min_dist,
                              self.max_dist, bw, **self.kwargs)
            std.append(hist.std())
        assert np.all(std[1:] < std[:-1])

    def test_flat_hist_few(self):
        posstack = self.pos.reshape(self.pos.shape[0] // 4, 4,
                                    self.pos.shape[1])
        _, hist, _ = rdf_series(posstack, 'hist', self.mode, self.min_dist,
                                self.max_dist, self.bw, **self.kwargs)
        assert hist.std() < .5  # 1.0 up to 1 decimal

    def test_flat_kde_few(self):
        posstack = self.pos.reshape(self.pos.shape[0] // 4, 4,
                                    self.pos.shape[1])
        _, hist, _ = rdf_series(posstack, 'kde', self.mode, self.min_dist,
                                self.max_dist, self.bw, **self.kwargs)
        assert hist[2:-2].std() < .5  # 1.0 up to 1 decimal


class TestRDF_2D(_rdf):
    def setUp(self):
        x = np.random.rand(500) * 100
        y = np.random.rand(500) * 100
        np.random.shuffle(x)
        np.random.shuffle(y)
        self.pos = np.asarray([y, x]).T
        self.mode = '2d_bounded'
        self.bw = 2.0
        self.min_dist = 5
        self.max_dist = 50
        self.kwargs = {'box': np.array([[0, 100], [0, 100]])}


class TestRDF_3D(_rdf):
    def setUp(self):
        x = np.random.rand(500) * 50
        y = np.random.rand(500) * 50
        z = np.random.rand(500) * 50
        np.random.shuffle(x)
        np.random.shuffle(y)
        np.random.shuffle(z)
        self.pos = np.asarray([z, y, x]).T
        self.mode = '3d_bounded'
        self.bw = 2.0
        self.min_dist = 3
        self.max_dist = 25
        self.kwargs = {'box': np.array([[0, 50], [0, 50], [0, 50]])}


class TestRDF_sphere(_rdf):
    def setUp(self):
        phi = np.random.rand(500) * 2*np.pi
        th = np.concatenate((np.arcsin(np.random.rand(250)),
                             -np.arcsin(np.random.rand(250)))) + np.pi/2
        np.random.shuffle(phi)
        np.random.shuffle(th)
        self.pos = np.asarray([th, phi]).T
        self.mode = 'sphere'
        self.bw = 2.0
        self.min_dist = 5
        self.max_dist = 50 * np.pi / 2
        self.kwargs = {'R': 50}


class TestDist(unittest.TestCase):
    def test_eucl_2D(self):
        x1 = np.random.random(100)
        y1 = np.random.random(100)
        x2 = np.random.random(100)
        y2 = np.random.random(100)
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        assert_array_almost_equal(dist, dist_eucl(np.array([x1, y1]).T,
                                                  np.array([x2, y2]).T))

    def test_eucl_3D(self):
        x1 = np.random.random(100)
        y1 = np.random.random(100)
        z1 = np.random.random(100)
        x2 = np.random.random(100)
        y2 = np.random.random(100)
        z2 = np.random.random(100)
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        assert_array_almost_equal(dist, dist_eucl(np.array([x1, y1, z1]).T,
                                                  np.array([x2, y2, z2]).T))

    def test_dist_sphere(self):
        phi = np.random.rand(1000) * 2*np.pi
        th = np.concatenate((np.arcsin(np.random.rand(500)),
                             -np.arcsin(np.random.rand(500)))) + np.pi/2
        R = np.random.random() * 100
        x = R*np.sin(th)*np.cos(phi)
        y = R*np.sin(th)*np.sin(phi)
        z = R*np.cos(th)
        d = dist_eucl(np.array([x, y, z]).T, np.array([0,0,R]))
        dist = 2*R*np.arcsin(d/(2*R))
        assert_array_almost_equal(dist, dist_sphere(np.array([th, phi]).T,
                                                    np.array([0, 0]).T, R))


class TestArcLenAndArea(unittest.TestCase):
    def setUp(self):
        self.N = 10  # some tests go with N**2!

    def test_limits(self):
        assert_almost_equal(circle_cap_arclen(0, 1), np.pi)
        assert_almost_equal(circle_corner_arclen(0, 0, 1), np.pi/2)
        assert_almost_equal(sphere_cap_area(0, 1), 2*np.pi)
        assert_almost_equal(sphere_edge_area(0, 0, 1), np.pi)
        assert_almost_equal(sphere_corner_area(0, 0, 0, 1), np.pi/2)

        e = 1 - 10**-10
        e2 = np.sqrt(1/2) - 10**-10
        e3 = np.sqrt(1/3) - 10**-10
        assert_almost_equal(circle_cap_arclen(e, 1), 0, 3)
        assert_almost_equal(circle_corner_arclen(e2, e2, 1), 0, 3)
        assert_almost_equal(sphere_cap_area(e, 1), 0, 3)
        assert_almost_equal(sphere_edge_area(e2, e2, 1), 0, 3)
        assert_almost_equal(sphere_corner_area(e3, e3, e3, 1), 0, 3)

    def test_scaling_circle_cap(self):
        for i in range(self.N):
            h0 = np.random.random()
            R = np.random.random(self.N) * 100
            result = circle_cap_arclen(h0*R, R) / R
            assert_array_almost_equal(result[1:], result[:-1])

    def test_scaling_circle_corner(self):
        for i in range(self.N):
            h0 = np.random.random()
            h1 = np.random.random() * np.sqrt(1-h0**2)
            R = np.random.random(self.N) * 100
            result = circle_corner_arclen(h0*R, h1*R, R) / R
            assert_array_almost_equal(result[1:], result[:-1])

    def test_scaling_sphere_cap(self):
        for i in range(self.N):
            h0 = np.random.random()
            R = np.random.random(self.N) * 100
            result = sphere_cap_area(h0*R, R) / R**2
            assert_array_almost_equal(result[1:], result[:-1])

    def test_scaling_sphere_edge(self):
        for i in range(self.N):
            h0 = np.random.random()
            h1 = np.random.random() * np.sqrt(1-h0**2)
            R = np.random.random(self.N) * 100
            result = sphere_edge_area(h0*R, h1*R, R) / R**2
            assert_array_almost_equal(result[1:], result[:-1])

    def test_scaling_sphere_corner(self):
        for i in range(self.N):
            h0 = np.random.random()
            h1 = np.random.random() * np.sqrt(1-h0**2)
            h2 = np.random.random() * np.sqrt(1-h0**2-h1**2)
            R = np.random.random(self.N) * 100
            result = sphere_corner_area(h0*R, h1*R, h2*R, R) / R**2
            assert_array_almost_equal(result[1:], result[:-1])

    def test_symmetry_circle_corner(self):
        h0 = np.random.random(self.N)
        h1 = np.random.random(self.N) * np.sqrt(1-h0**2)
        R = np.random.random(self.N) * 100
        assert_array_almost_equal(circle_corner_arclen(h0, h1, R),
                                  circle_corner_arclen(h1, h0, R))

    def test_symmetry_sphere_edge(self):
        h0 = np.random.random(self.N)
        h1 = np.random.random(self.N) * np.sqrt(1-h0**2)
        R = np.random.random(self.N) * 100
        assert_array_almost_equal(sphere_edge_area(h0, h1, R),
                                  sphere_edge_area(h1, h0, R))

    def test_symmetry_sphere_corner(self):
        h0 = np.random.random(self.N)
        h1 = np.random.random(self.N) * np.sqrt(1-h0**2)
        h2 = np.random.random(self.N) * np.sqrt(1-h0**2-h1**2)
        R = np.random.random(self.N) * 100
        result = np.array([sphere_corner_area(h0, h1, h2, R),
                           sphere_corner_area(h1, h2, h0, R),
                           sphere_corner_area(h2, h0, h1, R),
                           sphere_corner_area(h2, h1, h0, R),
                           sphere_corner_area(h0, h2, h1, R),
                           sphere_corner_area(h1, h0, h2, R)])
        assert_array_almost_equal(result[1:], result[:-1])


class TestNormCircle(unittest.TestCase):
    def setUp(self):
        self.R = np.random.random() * 100
        R = self.R
        self.point = np.repeat([[R, R]], 4, axis=0)
        self.box = np.array([[0, 2*R], [0, 2*R]])

    def test_norm_circle_inside(self):
        dist = np.array([0.001, 0.5, 0.9, 1.0])*self.R
        result = arclen_2d_bounded(dist, self.point, self.box)

        assert_array_almost_equal(result, 2*np.pi*dist)

    def test_norm_circle_trunc(self):
        dist = np.array([1.0001, 1.1, 1.2, np.sqrt(2)-0.01])*self.R
        result = arclen_2d_bounded(dist, self.point, self.box)

        assert_array_less(result, 2*np.pi*dist)

    def test_norm_circle_outside(self):
        dist = np.array([np.sqrt(2)+0.01, 2, 10, 100])*self.R
        result = arclen_2d_bounded(dist, self.point, self.box)

        assert_array_almost_equal(result, np.nan, 5)

    def test_norm_circle_limits(self):
        R = self.R
        box = self.box
        # center
        point = np.array([[1, 1]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = arclen_2d_bounded(dist, point, box)
        assert_array_almost_equal(result, 2*np.pi*R)
        # edges
        point = np.array([[0, 1], [1, 0], [1, 2], [2, 1]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = arclen_2d_bounded(dist, point, box)
        assert_array_almost_equal(result, np.pi*R)
        # corners
        point = np.array([[0, 0], [2, 0], [0, 2], [2, 2]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = arclen_2d_bounded(dist, point, box)
        assert_array_almost_equal(result, R*np.pi/2)


class TestNormSphere(unittest.TestCase):
    def setUp(self):
        self.R = np.random.random() * 100
        R = self.R
        self.point = np.repeat([[R, R, R]], 4, axis=0)
        self.box = np.array([[0, 2*R], [0, 2*R], [0, 2*R]])

    def test_norm_sphere_inside(self):
        dist = np.array([0.001, 0.5, 0.9, 1.0])*self.R
        result = area_3d_bounded(dist, self.point, self.box)

        assert_array_almost_equal(result, 4*np.pi*dist**2)

    def test_norm_sphere_trunc(self):
        dist = np.array([1.0001, 1.1, 1.2, np.sqrt(3)-0.01])*self.R
        result = area_3d_bounded(dist, self.point, self.box)

        assert_array_less(result, 4*np.pi*dist**2)

    def test_norm_sphere_outside(self):
        dist = np.array([np.sqrt(3)+0.01, 2, 10, 100])*self.R
        result = area_3d_bounded(dist, self.point, self.box)

        assert_array_almost_equal(result, np.nan, 5)

    def test_norm_sphere_limits(self):
        R = self.R
        box = self.box
        # center
        point = np.array([[1, 1, 1]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = area_3d_bounded(dist, point, box)
        assert_array_almost_equal(result, 4*np.pi*R**2)
        # planes
        point = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [2, 1, 1],
                          [1, 1, 2], [1, 2, 1]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = area_3d_bounded(dist, point, box)
        assert_array_almost_equal(result, 2*np.pi*R**2)
        # edges
        point = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [2, 0, 1],
                          [0, 2, 1], [1, 0, 2], [1, 2, 0], [0, 1, 2],
                          [2, 1, 0], [2, 2, 1], [1, 2, 2], [2, 1, 2]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = area_3d_bounded(dist, point, box)
        assert_array_almost_equal(result, np.pi*R**2)
        # corners
        point = np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0], [2, 0, 0],
                          [0, 2, 2], [2, 2, 0], [2, 0, 2], [2, 2, 2]]) * R
        dist = np.repeat([1], point.shape[0], axis=0) * R
        result = area_3d_bounded(dist, point, box)
        assert_array_almost_equal(result, 0.5*np.pi*R**2)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs'], exit=False)

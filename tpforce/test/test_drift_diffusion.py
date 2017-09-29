import numpy as np
from scipy.stats import norm as normal_distribution
import unittest
from displcode.drift_diffusion import Diffusion1D, ConstantDrift1D, \
    _Piecewise1D, PiecewiseForce1D, PiecewiseForce2DGrid, PiecewiseEnergy2DGrid, \
    _Piecewise2D
from numpy.testing import assert_allclose

def simulate_1D(f_func, n_particles, n_steps, D, dt, dt_sim, force=0., kT=1.):
    # this simulation is valid when F and D are constant locally
    # that is  (2*D*dt_sim)**0.5  <<  length at which force varies
    x = np.empty((n_steps, n_particles))
    x[0] = np.random.random(n_particles) * 10 - 5

    dilution = int(dt // dt_sim)
    n_steps_sim = n_steps * dilution

    gamma = D / kT
    rv = normal_distribution(loc=force*dt_sim*gamma, scale=(2*D * dt_sim)**0.5)

    x_curr = x[0].copy()
    for i in range(1, n_steps_sim):
        force = f_func(x_curr)
        dx = rv.rvs(n_particles)
        # the mean of the distribution displaces by gamma * F * dt
        # F is in units of kT/um, so gamma = D / kT = D
        x_curr += dx + force * D * dt_sim
        if i % dilution == 0:
            x[i // dilution] = x_curr

    return x


def simulate_2D_radial(f_func, n_particles, n_steps, D, dt, dt_sim,
                       initial_radius=6):
    x, y = np.empty((2, n_steps, n_particles))

    # random numbers in circle of radius 6
    x0 = np.random.random(n_particles * 2) * (
    initial_radius * 2) - initial_radius
    y0 = np.random.random(n_particles * 2) * (
    initial_radius * 2) - initial_radius

    mask = x0 ** 2 + y0 ** 2 < initial_radius ** 2
    x[0] = x0[mask][:n_particles]
    y[0] = y0[mask][:n_particles]

    # this simulation is valid when F and D are constant locally
    dilution = int(dt // dt_sim)
    n_steps_sim = n_steps * dilution

    rv = normal_distribution(loc=0., scale=(2 * D * dt_sim) ** 0.5)

    x_curr = x[0].copy()
    y_curr = y[0].copy()
    for i in range(1, n_steps_sim):
        r_curr = np.sqrt(x_curr ** 2 + y_curr ** 2)
        force = f_func(r_curr)
        force_x = force * (x_curr / r_curr)
        force_x[~np.isfinite(force_x)] = 0.
        force_y = force * (y_curr / r_curr)
        force_y[~np.isfinite(force_y)] = 0.

        dx, dy = rv.rvs((2, n_particles))

        # the mean of the distribution displaces by gamma * F * dt
        # F is in units of kT/um, so gamma = D / kT = D
        x_curr += dx + force_x * D * dt_sim
        y_curr += dy + force_y * D * dt_sim
        if i % dilution == 0:
            x[i // dilution] = x_curr
            y[i // dilution] = y_curr

    return x, y


class TestInterp(unittest.TestCase):
    def test_parabola1d(self):
        N = 200
        x0 = np.linspace(0, 10, 100)
        func = lambda x: 3*x**2 + 2*x - 5
        y0 = func(x0)
        x1 = np.random.random(N) * 10
        y1 = _Piecewise1D(x0, x1)(y0)

        assert_allclose(y1, func(x1), rtol=0.1)

    def test_parabola2d(self):
        N = 200
        extent = -5, 5
        bw = 0.5

        x_vect = y_vect = np.arange(extent[0], extent[1] + bw, bw)
        y_grid, x_grid = np.meshgrid(x_vect, y_vect, indexing='ij')

        func = lambda x, y: 0.3 * x ** 2 + 0.3 * y ** 2 - 2 * x * y - 10 + 2 * x
        z_grid = func(x_grid, y_grid)

        y1, x1 = np.random.random((2, N)) * 10 - 5
        z1 = _Piecewise2D(y_vect, x_vect, y1, x1)(z_grid)

        assert_allclose(z1, func(x1, y1), atol=0.1)


class TestDiffusion1D(unittest.TestCase):
    def test_fit(self):
        N = 10000
        D = 0.5

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=1, dt_sim=1)
        model = Diffusion1D(x[:-1].ravel(), x[1:].ravel())
        actual = model.fit()[0]

        assert_allclose(actual, D, rtol=0.05)

    def test_dt(self):
        N = 10000
        D = 0.5

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=0.1, dt_sim=0.1)
        model = Diffusion1D(x[:-1].ravel(), x[1:].ravel(), dt=0.1)
        actual = model.fit()[0]

        assert_allclose(actual, D, rtol=0.05)

    def test_err(self):
        M = 10
        N = 10000
        D = 0.5
        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=1, dt_sim=1)
        x0 = x[:-1].ravel()
        x1 = x[1:].ravel()
        for sqrtN in np.linspace(np.sqrt(40), np.sqrt(N), M):
            N = int(sqrtN**2)
            model = Diffusion1D(x0[:N], x1[:N])
            mu, sigma = model.sample_bayesian(p0=[D], nburn=0, nwalkers=10,
                                              nsteps=500, est_alpha=1e-7)

            expected_sigma = mu/np.sqrt(0.5*N)  # why the half?
            assert_allclose(sigma, expected_sigma, rtol=0.2, atol=0.001)


class TestDriftDiffusion1D(unittest.TestCase):
    def test_nodrift(self):
        N = 10000
        D = 0.5

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=1, dt_sim=1)
        model = ConstantDrift1D(x[:-1].ravel(), x[1:].ravel())

        Fs, Ds = model.fit()
        assert_allclose(Fs, 0, atol=0.1)
        assert_allclose(Ds, D, rtol=0.05)

    def test_drift(self):
        N = 10000
        D = 0.5
        force = 1.4

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        tau=1, dt_sim=1, force=force)
        model = ConstantDrift1D(x[:-1].ravel(), x[1:].ravel())

        Fs, Ds = model.fit()
        assert_allclose(Fs, force, rtol=0.1)
        assert_allclose(Ds, D, rtol=0.05)

    def test_dt(self):
        N = 10000
        D = 0.5
        force = 1.4

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=0.2, dt_sim=0.2, force=force)
        model = ConstantDrift1D(x[:-1].ravel(), x[1:].ravel(), tau=0.2)

        Fs, Ds = model.fit()
        assert_allclose(Fs, force, rtol=0.1)
        assert_allclose(Ds, D, rtol=0.05)

    def test_kT(self):
        N = 10000
        D = 0.5
        force = 16
        kT = 10.

        x = simulate_1D(lambda x: np.zeros_like(x), N, 2, D=D,
                        dt=1, dt_sim=1, force=force, kT=kT)
        model = ConstantDrift1D(x[:-1].ravel(), x[1:].ravel(), kT=kT)

        Fs, Ds = model.fit()
        assert_allclose(Fs, force, rtol=0.1)
        assert_allclose(Ds, D, rtol=0.05)


class TestPiecewise1D(unittest.TestCase):
    def test_force(self):
        N = 100000
        D = 0.5
        f_func = lambda r: (4 * r ** 3 - 40 * r) / -50  # kT / um
        force_x = np.linspace(-5, 5, 20)

        x = simulate_1D(f_func, N, 2, D=D, dt=1, dt_sim=1)
        model = PiecewiseForce1D(x[0], x[1],
                                 force_x=force_x, D=D)

        Fs = model.fit()
        assert_allclose(Fs, f_func(force_x), atol=0.1)


class TestPiecewise2D(unittest.TestCase):
    def test_force(self):
        N = 100000
        D = 0.5
        f_func = lambda r: (4 * r ** 3 - 40 * r) / -50  # kT / um
        force_x = np.linspace(-5, 5, 10)

        x, y = simulate_2D_radial(f_func, N, 2, D=D, dt=1, dt_sim=1)
        xy = np.array([x, y]).transpose(0, 2, 1)
        model = PiecewiseForce2DGrid(xy[0], xy[1], grid_x0=force_x,
                                     grid_x1=force_x, D=D)

        Fs = model.fit()

        x_result, y_result = np.meshgrid(force_x, force_x, indexing='ij')
        r_result = np.sqrt(x_result**2 + y_result**2)
        f_r = f_func(r_result)
        f_x = x_result / r_result * f_r
        f_y = y_result / r_result * f_r
        assert_allclose(Fs[:, :, 0], f_x, atol=0.1)
        assert_allclose(Fs[:, :, 1].reshape(10, 10), f_y, atol=0.1)

    def test_energy(self):
        N = 100000
        D = 0.5
        f_func = lambda r: (4 * r ** 3 - 40 * r) / -50  # kT / um
        u_func = lambda r: (r ** 4 - 20 * r ** 2) / 50  # kT
        u_x = np.linspace(-3, 3, 20)

        y, x = simulate_2D_radial(f_func, 2*N, 2, D=D, dt=1, dt_sim=1,
                                  initial_radius=3 * np.sqrt(2))

        yx = np.array([y, x]).transpose((1, 2, 0))
        mask = np.all(np.abs(yx[0, :]) < 3, axis=1)
        yx = yx[:, mask][:, :N]

        model = PiecewiseEnergy2DGrid(yx[0], yx[1], grid_x0=u_x,
                                      grid_x1=u_x, D=D)
        Us = model.fit()

        y_result, x_result = np.meshgrid(u_x, u_x, indexing='ij')
        r_result = np.sqrt(x_result**2 + y_result**2)
        u = u_func(r_result)

        Us += np.mean(u - Us)
        assert_allclose(Us, u, atol=0.2)

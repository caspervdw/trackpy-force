from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from six import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline


def _check_displ_data(x_t0, x_t1):
    x_t0 = np.array(x_t0)
    x_t1 = np.array(x_t1)
    if x_t0.ndim == 1:
        x_t0 = x_t0[:, np.newaxis]
    if x_t1.ndim == 1:
        x_t1 = x_t1[:, np.newaxis]
    assert x_t0.shape[0] == x_t1.shape[0]
    assert x_t0.shape[1] == x_t1.shape[1]
    return x_t0.copy(), x_t1.copy()


class MLEBase(with_metaclass(ABCMeta, object)):
    """Basemodel to fit Brownian motion in a potential field.

    The Smoluchowski equation is a stochastic partial differential
    that describes random motion in a potential field. Given data of
    these movements, the underlying potential field is estimated via
    a log likelihood maximization.

    This BaseClass defines an architecture how to transform a certain Model
    of Force vectors (F) and Diffusion tensors (D) into data coordinates, and
    provides fitting methods to find the Maximum Likelihood Estimate (mle) of
    F and D, as well as a simple Bayesian method to sample the credibility
    intervals.

    We use `x` to denote the (M-dimensional) coordinate of an observation. The
    corresponding coordinate system can be anything: Euclidean (x, y) position,
    an angle, a distance between two particles, or the simultaneous positions
    of 6 particles.

    The bulk of the computation is done by the function `log_likelihood`. This
    computes the probability of displacement `dX` given F and D, assuming
    that F and D are constant during the displacement `dX`.

    Parameters
    ----------
    x_t0: ndarray
        The measured positions at t.
        Either 1D with length N, or 2D with shape (N, M). N stands
        for the number of measured positions and M stands for the
        number of dimensions.
    x_t1: ndarray
        The measured positions at t + tau, where tau denotes the
        sampling time interval.
        Either 1D with length N, or 2D with shape (N, M). N stands
        for the number of measured positions and M stands for the
        number of dimensions
    tau : float
        The time interval over which is measured. Default 1, so that
        the modelled diffusion constant actually is `D * tau`.
    kT : float
        The value of the thermal energy `kT`. Default 1, so that the
        modelled force is given in units of `kT`.
    **kwargs: other arguments are passed to `self.initialize`
    """

    def __init__(self, x_t0, x_t1, tau=1., kT=1., **kwargs):
        self.measurements = _check_displ_data(x_t0, x_t1)
        self.tau = tau
        self.kT = kT

        # possibly x_t0 and x_t1 are adapted in initialize
        self.initialize(**kwargs)

        self.measurements = _check_displ_data(*self.measurements)
        self.N, self.M = self.measurements[0].shape
        self.dx = self.measurements[1] - self.measurements[0]

    def initialize(self, *args, **kwargs):
        return

    @abstractmethod
    def model_compute(self, p):
        """Define the Model here that, given an array of parameters
        `p`, provides a model for the force vectors `F` and diffusion
        tensors `D` for each displacement `dX`.

        The expected return values are:
            (F * D * tau / kT, D * tau)
        Vectors and tensors should be given in the coordinate system of `dX`.

        Allowed shapes for the first return value (rescaled F):
            scalar; (N,); (N, M)
        Allowed shapes for the second return value (rescaled D):
            scalar; (N,); (N, M); (N, M, M)
        """
        return  # (F * D * self.tau / self.kT, D * self.tau)


    @abstractproperty
    def n_params(self):
        """ The number of parameters in the model """
        return

    @abstractproperty
    def default_p0(self):
        """ Set some appropriate p0 for the model. This also determines n_params. """
        return

    @property
    def model_bounds(self):
        """Optimally, set bounds for model parameters of shape (N_params, 2)"""
        return

    def postprocess(self, p):
        """For convenience, postprocess the parameters"""
        return  # return None by default

    def log_prior(self, p):
        """Prior on model params, for Bayesian statistics"""
        return 0.

    def log_likelihood(self, p):
        F, D = self.model_compute(p)
        return _log_likelihood(self.dx, F, D)

    def log_posterior(self, p):
        result = self.log_prior(p)
        if not np.isfinite(result):
            return result
        return result + self.log_likelihood(p)

    def fit(self, p0=None, detail=False, **kwargs):
        if p0 is None:
            p0 = self.default_p0
        res = minimize(lambda x: -self.log_likelihood(x), p0,
                       bounds=self.model_bounds, **kwargs)

        processed = self.postprocess(res['x'])
        if processed is None:
            processed = res['x']

        if detail:
            res['x_processed'] = processed
            return res
        else:
            return processed

    def sample_bayesian(self, p0=None, nwalkers=None, est_alpha=0.1, nburn=100,
                        nsteps=200):
        try:
            import emcee
        except ImportError:
            raise ImportError('The package "emcee" is required for Bayesian sampling')
        if p0 is None:
            p0 = self.fit(p0=None, detail=False)
        if nwalkers is None:
            nwalkers = self.n_params * 2

        # uniform distributed starting guesses between 1-alpha and 1+alpha
        noise = np.random.random((nwalkers, self.n_params)) * (2 * est_alpha) + (
        1 - est_alpha)
        starting_guesses = noise * p0

        sampler = emcee.EnsembleSampler(nwalkers, self.n_params, self.log_posterior)
        sampler.run_mcmc(starting_guesses, nsteps)
        self._traces = sampler.chain[:, nburn:, :].reshape(-1, self.n_params).T

        # try postprocess
        if self.postprocess(self._traces[:, 0]) is not None:
            self.traces = []
            for i_trace in range(self._traces.shape[1]):
                self.traces.append(self.postprocess(self._traces[:, i_trace]))
        else:
            self.traces = self._traces

        try:
            mu = np.mean(self.traces, axis=1)
            sigma = np.std(self.traces, axis=1)
            return mu, sigma
        except:
            return self.traces


def _log_likelihood(dX, F, D):
    """Log likelihood of N observations in M dimensions.

        scalar; (N,); (N, M)
        Allowed shapes for the second return value (rescaled D):
            scalar; (N,); (N, M); (N, M, M)

    Parameters
    ----------
    dX : ndarray
        N observations in M dimensions. Shape (N, M)
    F : float or ndarray
        force.
        Scalar, (N,) or (N, M) shaped.
    D : float or ndarray
        diffusion coefficient.
        Scalar, (N,), (N, M), or (N, M, M) shaped.
        (N, M) is useful for diagonal diffusion tensors.
    """
    N, M = dX.shape
    F = np.atleast_1d(F)
    D = np.atleast_1d(D)

    if F.size == 1:
        F = F.ravel()[0]
    elif F.ndim == 1:
        assert F.shape[0] == N
        F = F[:, np.newaxis]
    else:
        assert F.shape[0] == N
        assert F.shape[1] == M

    mu = dX - F
    if D.size == 1:  # constant diffusion coefficient
        D = D.ravel()[0]
        P = -0.5 * M * np.log(4 * np.pi * D) - np.sum(mu ** 2, axis=1) / (4 * D)
    elif D.ndim == 1:  # diffusion coefficient per measurement
        assert D.shape[0] == N
        P = -0.5 * M * np.log(4 * np.pi * D) - 0.25*np.sum(mu**2/D[:, np.newaxis], axis=1)
    elif D.ndim == 2:  # diagonal diffusion matrix per measurement
        assert D.shape[0] == N
        assert D.shape[1] == M
        det = np.prod(D, axis=1)
        P = -0.5 * M * np.log(4 * np.pi) - 0.5 * np.log(det) - 0.25*np.sum(mu**2/D, axis=1)
    elif D.ndim == 3:  # diffusion matrix per measurement
        assert D.shape[0] == N
        assert D.shape[1] == M
        assert D.shape[2] == M
        inv = np.linalg.inv(D)
        det = np.linalg.det(D)
        P = -0.5 * M * np.log(4 * np.pi) - 0.5 * np.log(det) - \
            0.25 * np.einsum('...ij,...i,...j->...', inv, mu, mu)
    return P.sum()


def integrate_traces(x, y, reference=None):
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if reference is None:
        # take the smallest error in y
        reference = np.argmin(np.std(y, axis=1))
    else:
        reference = np.searchsorted(x, reference)

    y_int = cumtrapz(y, x, axis=0, initial=0)
    y_int = y_int - y_int[reference]  # inplace skips the last element ??

    mu = np.mean(y_int, axis=1)
    sigma = np.std(y_int, axis=1)
    return mu, sigma


class Diffusion1D(MLEBase):
    @property
    def model_bounds(self):
        return [[1e-7, np.inf]]

    @property
    def default_p0(self):
        return [1]

    @property
    def n_params(self):
        return 1

    def model_compute(self, p):
        return 0., p[0] * self.tau  # F * D * tau / kT, D * tau

    def log_prior(self, p):
        if p[0] <= 0.:
            return -np.inf
        else:
            return 0.


class ConstantDrift1D(MLEBase):
    @property
    def model_bounds(self):
        return [[-np.inf, np.inf], [1e-7, np.inf]]

    @property
    def default_p0(self):
        return [0, 1]

    @property
    def n_params(self):
        return 2

    def log_prior(self, p):
        """Prior on model params"""
        if p[1] <= 0.:
            return -np.inf
        else:
            return 0.

    def model_compute(self, p):
        Dt = p[1] * self.tau
        return p[0] * Dt / self.kT, Dt


class Piecewise1D(object):
    """Fast piecewise linear interpolation in one dimension.

    Optimized for the special case that the interpolation is done many
    times on the same positions, on datapoints with a constant x grid,
    but varying y values.

    Parameters
    ----------
    x_grid : ndarray
        Horizontal (x) grid on which the datapoints will be given
    x_interp : ndarray
        Values on which interpolation will be performed
    """
    def __init__(self, x_grid, x_interp):
        # check if x0 is sorted
        if not np.all(x_grid[1:] > x_grid[:-1]):
            raise ValueError("x_grid must be sorted in ascending order")

        self.x_grid = x_grid
        self.x_interp = x_interp
        self.index = np.searchsorted(x_grid, x_interp) - 1

    def __call__(self, y_grid):
        """Perform interpolation using given `y_grid` values.
        """
        x_grid = self.x_grid
        x_interp = self.x_interp

        a = (y_grid[1:] - y_grid[:-1]) / (x_grid[1:] - x_grid[:-1])
        b = y_grid[1:] - a * x_grid[1:]
        return a.take(self.index) * x_interp + b.take(self.index)


class Piecewise2D(object):
    def __init__(self, y_grid, x_grid, y_interp, x_interp):
        if not np.all(x_grid[1:] > x_grid[:-1]):
            raise ValueError("x_grid must be sorted in ascending order")
        if not np.all(y_grid[1:] > y_grid[:-1]):
            raise ValueError("y_grid must be sorted in ascending order")

        self.x_grid = x_grid
        self.y_grid = y_grid
        self.x_interp = x_interp
        self.y_interp = y_interp
        self.ix = np.searchsorted(x_grid, x_interp) - 1
        self.iy = np.searchsorted(y_grid, y_interp) - 1
        self.i = ((self.ix + len(x_grid) * self.iy,
                  self.ix + 1 + len(x_grid) * self.iy),
                  (self.ix + len(x_grid) * (self.iy + 1),
                  self.ix + 1 + len(x_grid) * (self.iy + 1)))

        x1 = x_grid.take(self.ix)
        x2 = x_grid.take(self.ix + 1)
        y1 = y_grid.take(self.iy)
        y2 = y_grid.take(self.iy + 1)

        norm = (x2 - x1) * (y2 - y1)
        a11 = (x2 - x_interp) * (y2 - y_interp) / norm
        a21 = - (x1 - x_interp) * (y2 - y_interp) / norm
        a12 = - (x2 - x_interp) * (y1 - y_interp) / norm
        a22 = (x1 - x_interp) * (y1 - y_interp) / norm
        self.a = ((a11, a21), (a12, a22))

    def __call__(self, z_grid):
        """Perform interpolation using given `z_grid` values.
        """
        a = self.a
        i = self.i

        result = z_grid.take(i[0][0]) * a[0][0] + \
                 z_grid.take(i[0][1]) * a[0][1] + \
                 z_grid.take(i[1][0]) * a[1][0] + \
                 z_grid.take(i[1][1]) * a[1][1]
        return result


class PiecewiseForce1D(MLEBase):
    def initialize(self, force_x, D=None):
        self.D = D
        self.force_x = np.sort(force_x)
        self.force = Piecewise1D(self.force_x, self.measurements[0])

    @property
    def n_params(self):
        if self.D is None:
            return len(self.force_x) + 1
        else:
            return len(self.force_x)

    @property
    def default_p0(self):
        result = [0] * self.n_params
        if self.D is None:
            result += [1]
        return result

    @property
    def model_bounds(self):
        result = [[None, None]] * self.n_params
        if self.D is None:
            result += [[1e-7, np.inf]]
        return result

    def log_prior(self, p):
        """Prior on model params"""
        if self.D is None and p[-1] <= 0.:
            return -np.inf
        else:
            return 0.

    def model_compute(self, p):
        if self.D is None:
            D = p[-1]
            F = p[:-1]
        else:
            D = self.D
            F = p
        Dt = D * self.tau
        force = self.force(F) * Dt / self.kT
        return force, Dt  # force, diffusion coefficient


class PiecewiseForce2DGrid(MLEBase):
    def initialize(self, grid_x0, grid_x1, D):
        self.D = D
        self.grid_x0 = grid_x0
        self.grid_x1 = grid_x1

    @property
    def n_params(self):
        return len(self.grid_x0) * len(self.grid_x1) * 2

    @property
    def model_bounds(self):
        return [[None, None]] * self.n_params

    @property
    def default_p0(self):
        return [0] * self.n_params

    def postprocess(self, p):
        p_force0 = p[::2].reshape(len(self.grid_x0), len(self.grid_x1))
        p_force1 = p[1::2].reshape(len(self.grid_x0), len(self.grid_x1))
        return np.array([p_force0, p_force1]).transpose(1, 2, 0)

    def model_compute(self, p):
        p_force0 = p[::2].reshape(len(self.grid_x0), len(self.grid_x1))
        p_force1 = p[1::2].reshape(len(self.grid_x0), len(self.grid_x1))
        x0_t0 = self.measurements[0][:, 0]
        x1_t0 = self.measurements[0][:, 1]
        interpolator = RectBivariateSpline(self.grid_x0, self.grid_x1,
                                           p_force0, kx=1, ky=1)
        force0 = interpolator(x0_t0, x1_t0, grid=False)
        interpolator = RectBivariateSpline(self.grid_x0, self.grid_x1,
                                           p_force1, kx=1, ky=1)
        force1 = interpolator(x0_t0, x1_t0, grid=False)
        force = np.array([force0, force1]).T
        return force * (self.D * self.tau / self.kT), self.D * self.tau


class PiecewiseEnergy2DGrid(MLEBase):
    def initialize(self, grid_x0, grid_x1, D):
        self.D = D
        self.grid_x0 = grid_x0
        self.grid_x1 = grid_x1

        dx0 = np.diff(grid_x0)
        if np.allclose(dx0[1:], dx0[:-1]):
            self.dx0 = dx0[0]
        else:
            self.dx0 = dx0
        dx1 = np.diff(grid_x1)
        if np.allclose(dx1[1:], dx1[:-1]):
            self.dx1 = dx1[0]
        else:
            self.dx1 = dx1

        self.interpolator = Piecewise2D(grid_x0, grid_x1,
                                        self.measurements[0][:, 0],
                                        self.measurements[0][:, 1])

    @property
    def n_params(self):
        return len(self.grid_x0) * len(self.grid_x1)

    @property
    def model_bounds(self):
        return [[None, None]] * self.n_params

    @property
    def default_p0(self):
        return [0] * self.n_params

    def postprocess(self, p):
        u = p.reshape(len(self.grid_x0), len(self.grid_x1))
        return u - u[u.shape[0] // 2, u.shape[1] // 2]

    def model_compute(self, p):
        u = p.reshape(len(self.grid_x0), len(self.grid_x1))
        dudx0, dudx1 = np.gradient(u, self.dx0, self.dx1)
        f_x0_interp = self.interpolator(-dudx0)
        f_x1_interp = self.interpolator(-dudx1)
        force = np.array([f_x0_interp, f_x1_interp]).T
        return force * (self.D * self.tau / self.kT), self.D * self.tau

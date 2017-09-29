from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np
from itertools import combinations
from scipy.special import erf
from scipy.stats import t as student_t
from rdf.algebraic import *


def ensemble_dist(pos, mode='auto', min_dist=1, max_dist=None, **kwargs):
    """Calculates distances between all combinations of particles.

    Parameters
    ----------
    pos : numpy array
        numpy array of all particle positions
    mode : {auto, 2d, 2d_bounded, 3d, 3d_bounded, sphere}, default auto
        method of distance and weight calculation
    min_dist : number, default 1
        lower distance cutoff. should be > 0
    max_dist : number, optional
        higher distance cutoff
    **kwargs :
        for 2d or 3d: box
        for sphere: R

    Returns
    -------
        distances : numpy array of all distances
        weights : normalized weights of all distances
    """
    if mode == 'auto':
        if pos.shape[1] == 2 and 'box' in kwargs:
            mode = '2d_bounded'
        elif pos.shape[1] == 2:
            mode = '2d'
        elif pos.shape[1] == 3 and 'box' in kwargs:
            mode = '3d_bounded'
        elif pos.shape[1] == 3:
            mode = '3d'
    if mode == '2d':
        dist_func = dist_eucl
        weight_func = lambda x, y, z: 1 / (2 * np.pi * x)
        volume = 1
    elif mode == '2d_bounded':
        dist_func = dist_eucl
        box = kwargs['box']
        _max_dist = np.sqrt(((np.diff(box, axis=1)/2)**2).sum())
        if max_dist is None:
            max_dist = _max_dist
        elif max_dist > _max_dist:
            raise ValueError('For every particle, the maximum distance should '
                             'fit inside the box.')
        weight_func = lambda x, y, z: (1 / arclen_2d_bounded(x, y, box) +
                                       1 / arclen_2d_bounded(x, z, box))
        volume = np.prod(box[:, 1] - box[:, 0])
    elif mode == '3d':
        dist_func = dist_eucl
        weight_func = lambda x, y, z: 1 / (4 * np.pi * x**2)
        volume = 1
    elif mode == '3d_bounded':
        dist_func = dist_eucl
        box = kwargs['box']
        _max_dist = np.sqrt(((np.diff(box, axis=1)/2)**2).sum())
        if max_dist is None:
            max_dist = _max_dist
        elif max_dist > _max_dist:
            raise ValueError('For every particle, the maximum distance should '
                             'fit inside the box.')
        weight_func = lambda x, y, z: (1 / area_3d_bounded(x, y, box) +
                                       1 / area_3d_bounded(x, z, box))
        volume = np.prod(box[:, 1] - box[:, 0])
    elif mode == 'sphere':
        R = kwargs['R']
        if max_dist is None:
            max_dist = np.pi*R
        elif max_dist > np.pi*R:
            raise ValueError('The maximum distance should fit on the sphere.')
        dist_func = lambda x, y: dist_sphere(x, y, R)
        weight_func = lambda x, y, z: 2 / arclen_sphere(x, R)
        volume = 4*np.pi*R**2
    else:
        raise ValueError('Unknown mode')
    if pos.shape[0] < 2:
        return (np.array([], dtype=np.float),
                np.array([], dtype=np.float), volume)
    combs = np.asarray([a for a in combinations(pos, 2)])
    dist = np.asarray(dist_func(combs[:, 0], combs[:, 1]))
    mask = dist >= min_dist
    if max_dist is not None:
        mask = mask & (dist <= max_dist)
    dist = dist[mask]
    combs = combs[mask]
    weights = weight_func(dist, combs[:, 0], combs[:, 1])
    mask = np.isfinite(weights)
    return dist[mask], weights[mask], volume


def kde(data, points, bw, weights=None):
    """Performs a kernel-density estimate using a Gaussian kernel.
    Possible to provide a `weights` array, that signifies for instance
    the number of measurements of each datapoint.

    Parameters
    ----------
    data : list or 1D array
        Datapoints to estimate from.
    points : list or 1D array
        Grid on which the pdf is calculated
    bw : float
        Width of Gaussian kernel (sigma)
    weights : list or 1D array
        Weights of datapoints.

    Returns
    -------
    Estimated pdf on given grid.
    """
    data = np.asarray(data)
    if data.ndim > 1:
        raise ValueError("`data` should be one-dimensional.")
    n = data.size
    data = data[np.newaxis, :]

    points = np.asarray(points)
    m = points.size
    points = points[np.newaxis, :]

    result = np.zeros((m,), dtype=np.float)
    if n == 0:
        return result

    sigmasq = 1 / (2 * bw**2)

    if weights is not None:
        if weights.size != data.size:
            raise ValueError("size of `data` and `weights` should be equal.")
    else:
        weights = np.ones((n,))
    weights = weights / (np.sqrt(2*np.pi) * bw)

    if m >= n:
        # there are more points than data, so loop over data
        for i in range(n):
            diff = data[:, i, np.newaxis] - points
            energy = np.sum(diff**2 * sigmasq, axis=0)
            result = result + weights[i]*np.exp(-energy)
    else:
        # loop over points
        for i in range(m):
            diff = data - points[:, i, np.newaxis]
            energy = np.sum(diff**2 * sigmasq, axis=0)
            result[i] = np.sum(weights*np.exp(-energy), axis=0)

    return result


def rdf_kde(pos, mode='auto', min_dist=1, max_dist=None, bw=1, n=100,
            **kwargs):
    """Calculates the radial distribution function over an ensemble of
    particles. This is the average of all individual rdfs.
    It uses Kernel Density Estimation to estimate the rdf."""
    if pos.shape[0] == 0:
        raise ValueError('Empty position array')
    if max_dist is None:
        cutoff = None
    else:
        cutoff = max_dist + 3*bw

    dist, weights, volume = ensemble_dist(pos, mode, min_dist, cutoff, **kwargs)

    if max_dist is None:
        right_edge = max(dist)
    else:
        right_edge = max_dist

    x_grid = np.arange(min_dist, right_edge, (right_edge - min_dist) / n)
    result = kde(dist, x_grid, bw, weights=weights)
    concentration = (pos.shape[0] - 1) / volume
    return x_grid, result / (concentration * pos.shape[0])


def rdf_hist(pos, mode='auto', min_dist=1, max_dist=None, bw=1, **kwargs):
    """Calculates the radial distribution function over an ensemble of
    particles. This is the average of all individual rdfs.
    It uses a histogram to estimate the rdf."""
    if pos.shape[0] == 0:
        raise ValueError('Empty position array')
    dist, weights, volume = ensemble_dist(pos, mode, min_dist, max_dist,
                                          **kwargs)
    if max_dist is None:
        right_edge = max(dist)
    else:
        right_edge = max_dist
    bins = np.arange(min_dist, right_edge, bw)
    if len(dist) > 0:
        hist, bins = np.histogram(dist, bins=bins, weights=weights)
    else:
        hist = np.zeros(len(bins) - 1, dtype=np.float)
    bins = (bins[1:] + bins[:-1]) / 2
    concentration = (pos.shape[0] - 1) / volume
    return bins, hist / (concentration * pos.shape[0] * bw)


def rdf_series(f, hist_mode='hist', dist_mode='auto', min_dist=1,
               max_dist=None, bw=1, **kwargs):
    """Analyzes the ensemble averaged rdf, averaged additionally over all
    frames.
    
    Parameters
    ----------
    f : iterable of ndarray
        contains the coordinates in a N x 2 or N x 3 array. Order should be
        y, x; z, y, x or theta, phi (theta being the angle with z axis)
    hist_mode : {hist, kde}
        histogram mode (histogram/kernel density estimation)
    dist_mode : {auto, 2d, 3d, sphere}, default auto
        method of distance and weight calculation
    min_dist : number, default 1
        lower distance cutoff. should be > 0
    max_dist : number, optional
        higher distance cutoff
    bw : float
        binwidth for hist, bandwidth for kde
    box : ndarray
        for dist_mode 2d_bounded or 3d_bounded: box as 2 x 2 or 3 x 2 array
    R : number
        for dist_mode sphere: radius of sphere

    Returns
    -------
    r, g(r), variance of g(r)
    """
    if hist_mode == 'hist':
        hist_func = rdf_hist
    elif hist_mode == 'kde':
        hist_func = rdf_kde
    else:
        raise ValueError('Unknown hist mode')

   # if max_dist is None:
   #     raise ValueError('Specify max distance')

    result = []
    for pos in f:
        if len(pos) > 1:
            x, hist = hist_func(pos, dist_mode, min_dist, max_dist,
                                bw, **kwargs)
            result.append(hist)

    if len(hist) == 0:
        raise ValueError('No positions found')

    result = np.atleast_2d(result)
    return x, result.mean(0), result.var(0)


def rdf_series_df(f, hist_mode='hist', dist_mode='auto', min_dist=1,
                  max_dist=None, bw=1, pos_columns=None, t_column=None,
                  real_t_column=None, particle_column=None, D=None,
                  confidence=0.95, **kwargs):
    """Analyzes the ensemble averaged rdf, averaged additionally over all
    frames. Experimental: confidence interval calculation

    Parameters
    ----------
    f : pandas DataFrame
        contains the coordinates and frame indices
    hist_mode : {hist, kde}
        histogram mode (histogram/kernel density estimation)
    dist_mode : {auto, 2d, 3d, sphere}, default auto
        method of distance and weight calculation
    min_dist : number, default 1
        lower distance cutoff. should be > 0
    max_dist : number, optional
        higher distance cutoff
    bw : float
        binwidth for hist, bandwidth for kde
    pos_columns : list of str
        DataFrame columns to read as positions.
    t_column : str
        DataFrame column for frame index
    real_t_column : str:
        DataFrame column for real time (units should comply with D)
    particle_column : str
        DataFrame column for particle index
    D : float
        diffusion coefficient
    confidence : float
        confidence interval to use (default 0.95)
    **kwargs :
        for 2d or 3d: box
        for sphere: R

    Returns
    -------
    r, g(r), confidence (symmetric, deviation from g(r))
    """
    if pos_columns is None:
        if dist_mode == 'sphere':
            pos_columns = ['th', 'phi']
        elif 'z' in f:
            pos_columns = ['z', 'y', 'x']
        else:
            pos_columns = ['y', 'x']
    if t_column is None:
        t_column = 'frame'
    if real_t_column is None and 't' in f:
        real_t_column = 't'
    if particle_column is None and 'particle' in f:
        particle_column = 'particle'

    x, gr, var = rdf_series((g[1][pos_columns].values for g in f.groupby(t_column)),
                            hist_mode, dist_mode, min_dist, max_dist, bw,
                            **kwargs)

    if np.all([a is not None for a in [real_t_column, particle_column, D]]):
        N = 0
        for _, f_p in f.groupby(particle_column):
            N += effective_N_df(f_p[real_t_column].values, D, bw)
        err = np.sqrt(var / N)  # sample stdev
        conf = err * student_t.ppf((1 + confidence)/2., N-1)  # two sided
    else:
        conf = None

    return x, gr, conf


def pair_potential(x, gr, x_bulk=None, conf=None):
    """Calculates the pair potential from g(r).

    Parameters
    ----------
    x : ndarray
        Array of r positions
    gr : ndarray
        Array of g(r)
    x_bulk : float
        If this value is given, u(r) will be shifted so that u=0 at x > x_bulk
    conf : ndarray
        Confidence interval. Single array with deviations from g(r)

    Returns
    -------
    u(r), [u_lower, u_higher]
    """
    if np.any(gr < 0):
        raise ValueError('Radial distribution function cannot be below zero.')

    u = -np.log(gr)  # units of kT

    if x_bulk is not None:
        u_bulk = np.nanmean(u[(x >= x_bulk) & np.isfinite(u)])
    else:
        u_bulk = 0

    if conf is not None:
        lowerbound = gr - conf
        lowerbound[lowerbound < 0] = 0
        higherbound = gr + conf

        u_lower = -np.log(higherbound) - u_bulk
        u_higher = -np.log(lowerbound) - u_bulk
    else:
        u_lower = None
        u_higher = None

    return u - u_bulk, [u_lower, u_higher]


def autocorrelation(dt, D, dr):
    """The autocorrelation of the concentration in a bin of size dr."""
    a = dr / np.sqrt(8 * D * dt)
    return erf(a) - (1 - np.exp(-a**2)) / (a * np.sqrt(np.pi))


def effective_N(N, dt, D, dr):
    """The effective number of independent measurements of concentration in a
    bin of width dr."""
    if np.isscalar(dt):
        dt = np.array([dt], dtype=np.float)
    t = np.arange(1, N)[:, np.newaxis]
    A = ((N - t)*autocorrelation(t * dt[np.newaxis, :], D, dr)).sum(0)
    return (N**2 / (N + 2 * A)).squeeze()


def effective_N_df(t_values, D, dr):
    """The effective number of independent measurements of concentration in a
    bin of width dr. The t values are taken from the DataFrame, gaps and
    other unequal time spacings (dt) are allowed."""
    N = len(t_values)
    all_dts = (t_values[:, np.newaxis] - t_values[np.newaxis, :]).ravel()
    # only need to calculate one half of the matrix, excluding dt = 0
    A = autocorrelation(all_dts[all_dts > 0], D, dr).sum()
    return N**2 / (N + 2 * A)


def guess_pos_columns(f):
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    return pos_columns


def auto_box(f, offset=None, pos_columns=None):
    """ Determines the rectangular box in which all particles are. If offset is
    nonzero, then the box is taken smaller than the actual region the particles
    are in. The particles that fall out of the box are dropped off.
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    if offset is None:
        offset = 0
    if not hasattr(offset, '__iter__'):
        offset = [offset] * len(pos_columns)

    e = 10**-7  # for numerical reasons, increase offset by 10**-7
    box = []
    mask = []
    for col, _offset in zip(pos_columns, offset):
        box.append([f[col].min() + (_offset + e), f[col].max() - (_offset + e)])
        mask.append((f[col] > box[-1][0]) & (f[col] < box[-1][1]))

    return f[np.all(mask, axis=0)].copy(), np.array(box)

from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def _take_coords(pos):
    shape = pos.shape
    if len(shape) == 1:
        pos = [np.array([pos[i]]) for i in range(shape[0])]
    elif len(shape) == 2:
        pos = [np.take(pos, i, 1) for i in range(shape[1])]
    elif len(shape) == 3:
        coord_dim = [d for d in range(3) if shape[d] > 1][-1]
        pos = [np.take(pos, i, coord_dim) for i in range(shape[coord_dim])]
    else:
        raise ValueError

    return pos


def dist_eucl(pos1, pos2):
    pos1 = np.array(_take_coords(pos1))
    pos2 = np.array(_take_coords(pos2))
    return np.sqrt(np.sum((pos1 - pos2)**2, axis=0))


def _norm_trans(arclen, trans):
    return trans * arclen[:, np.newaxis] / arclen[np.newaxis, :]


def norm_2d(s, trans):
    return _norm_trans(s, trans)


def norm_3d(s, trans):
    return _norm_trans(s**2, trans)


def norm_sphere(s, trans, R):
    return _norm_trans(np.sin(s / R), trans)


def geodesic2euclidean(dist, R):
    return 2 * R * np.sin(dist / (2 * R))


def euclidean2geodesic(dist, R):
    return 2 * R * np.arcsin(dist / (2 * R))


def _find_pairs(pos, max_dist, max_dist_3):
    """Determine isolated pairs of particles: closer than ``max_dist`` together with a
    possible third particle farther than ``max_dist_3`` from any of them.

    Optimal for few features. Should use cKDTree or something else for many features

    Returns array of pair indices (shape N x 2)"""
    # matrix of all combinations
    pos = np.atleast_2d(pos)
    kdt = cKDTree(pos)
    sparse_mat = kdt.sparse_distance_matrix(kdt, max_distance=max_dist)
    dist = dist_eucl(pos[..., np.newaxis], pos[np.newaxis, ...])
    compare_3 = (dist <= max_dist_3).sum(0)

    # particles having 0 neighbors closer than the three-body limit
    # they are free to pair with any particle from this list
    neighbors_0 = np.where(compare_3 == 1)[0]
    if len(neighbors_0) > 1:
        dist_masked = dist[neighbors_0][:, neighbors_0]
        pairs_0 = np.nonzero(np.tril(dist_masked <= max_dist, k=-1))
    else:
        pairs_0 = ([], [])
    pairs_0 = [neighbors_0[ind] for ind in pairs_0]

    # particles having 1 neighbor closer than the three-body limit
    # this neighbor is directly the pair
    neighbors_1 = np.where(compare_3 == 2)[0]
    if len(neighbors_1) > 1:
        dist_masked = dist[neighbors_1][:, neighbors_1]
        pairs_1 = np.nonzero(np.tril(dist_masked <= max_dist_3, k=-1))
    else:
        pairs_1 = ([], [])
    pairs_1 = [neighbors_1[ind] for ind in pairs_1]
    pairs = np.concatenate([pairs_0, pairs_1], axis=1)
    pairs_dist = dist[pairs[0], pairs[1]]
    return pairs.T, pairs_dist


def extract_pairs(f, max_dist, max_dist_3=None, out_of_bounds=None,
                  lagt=1, pos_columns=None, metric='euclidean'):
    """Extracts pairs from a feature positions for transition matrix estimation.

    Parameters
    ----------
    f : DataFrame
        having the pos_columns, a particle column and a frame column
    max_dist :
        distance cutoff
    max_dist_3 :
        pairs having a third particle closter than this will be ignored
    out_of_bounds : function
        use this to make sure that at both particles are still present at t + dt
    lagt : integer
    pos_columns : list of strings

    Notes
    -----
    Given any pair(t), pair(t + dt) should be unbiased. This is reached by:
    - both particles of pair(t) should be further than R3 away from other
      particles, to avoid three-body effects pair(t + dt) can have particles
      closer than R3.
    - both particles in pair(t) are such that they are always present at t + dt
     (far enough from the image margins)

    Other reasons for disappearing particles should be uncorrelated with
    distance between pairs. For instance close-by tracking artefacts are
    correlated.
    """
    if max_dist_3 is None:
        max_dist_3 = max_dist

    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['z', 'y', 'x']
        else:
            pos_columns = ['y', 'x']

    # sort the features df, making a copy
    try:
        f = f.sort_values(['particle', 'frame'])
    except AttributeError:
        f = f.sort(['particle', 'frame'])
    f.reset_index(drop=False, inplace=True)

    if out_of_bounds is not None:
        # drop particles in forbidden zones
        f_select = f[~out_of_bounds(f)].copy()
    else:
        f_select = f

    last_frame = f['frame'].max()
    pairs_0 = []
    pairs_1 = []
    #dists = []
    for frame_no, f_frame in f_select.groupby('frame'):
        if frame_no == last_frame:
            break
        if f_frame['particle'].nunique() != len(f_frame):
            raise RuntimeError('There are non-unique track ids!')

        pairs, dist = _find_pairs(f_frame[pos_columns].values, max_dist,
                                  max_dist_3)
        if len(pairs) == 0:
            continue

        pairs_0.extend(f_frame.index[pairs.T[1]].tolist())
        pairs_1.extend(f_frame.index[pairs.T[0]].tolist())
        #dists.extend(dist)

    pairs_0 = np.array(pairs_0, dtype=np.int)
    pairs_1 = np.array(pairs_1, dtype=np.int)
    #dists = np.array(dists, dtype=np.float)

    pairs_p0t0 = f.loc[pairs_0].reset_index(drop=True)
    pairs_p1t0 = f.loc[pairs_1].reset_index(drop=True)
    pairs_p0t1 = f.loc[pairs_0 + lagt].reset_index(drop=True)
    pairs_p1t1 = f.loc[pairs_1 + lagt].reset_index(drop=True)

    # check that each pair has a subsequent pair that is `lagt` frame(s) later
    mask = ((pairs_p0t0['frame'] + lagt == pairs_p0t1['frame']) &
            (pairs_p1t0['frame'] + lagt == pairs_p1t1['frame']) &
            (pairs_p0t0['particle'] == pairs_p0t1['particle']) &
            (pairs_p1t0['particle'] == pairs_p1t1['particle'])).values

    # acquire distances
    pos_t0p0 = pairs_p0t0.loc[mask, pos_columns]
    pos_t0p1 = pairs_p1t0.loc[mask, pos_columns]
    pos_t1p0 = pairs_p0t1.loc[mask, pos_columns]
    pos_t1p1 = pairs_p1t1.loc[mask, pos_columns]

    s0 = dist_eucl(pos_t0p0.values, pos_t0p1.values)
    s1 = dist_eucl(pos_t1p0.values, pos_t1p1.values)

    result = pd.DataFrame({'s0': s0, 's1': s1}, index=pos_t0p0.index)
    result['p0t0'] = pairs_p0t0.loc[mask, 'index']
    result['p1t0'] = pairs_p1t0.loc[mask, 'index']
    result['p0t1'] = pairs_p0t1.loc[mask, 'index']
    result['p1t1'] = pairs_p1t1.loc[mask, 'index']
    result['p0'] = pairs_p0t0.loc[mask, 'particle']
    result['p1'] = pairs_p1t0.loc[mask, 'particle']
    result['t0'] = pairs_p0t0.loc[mask, 'frame']
    result['t1'] = pairs_p0t1.loc[mask, 'frame']

    if mask.sum() != len(mask):
        s_lost = dist_eucl(pairs_p0t0.loc[~mask, pos_columns].values,
                           pairs_p1t0.loc[~mask, pos_columns].values)
    else:
        s_lost = None

    return result, s_lost


def extract_pairs_sphere(f, radius, max_dist, max_dist_3=None,
                         out_of_bounds=None, lagt=1):
    if max_dist_3 is None:
        max_dist_3 = max_dist

    if not all(col in f for col in ['z', 'y', 'x']):
        assert 'th' in f and 'phi' in f
        f['z'] = radius * np.cos(f['th'])
        f['y'] = radius * np.sin(f['th']) * np.sin(f['phi'])
        f['x'] = radius * np.sin(f['th']) * np.cos(f['phi'])

    max_dist = geodesic2euclidean(max_dist, radius)
    max_dist_3 = geodesic2euclidean(max_dist_3, radius)
    result, s_lost = extract_pairs(f, max_dist, max_dist_3,
                                   out_of_bounds, lagt, ['z', 'y', 'x'])

    result['s0'] = euclidean2geodesic(result['s0'], radius)
    result['s1'] = euclidean2geodesic(result['s1'], radius)
    if s_lost is not None:
        s_lost = euclidean2geodesic(s_lost, radius)
    return result, s_lost


def _norm_transition(bins, trans, norm, **kwargs):
    count = trans.sum(1)
    trans = np.nan_to_num(trans / count[:, np.newaxis])

    if norm == '2d':
        trans = norm_2d(bins, trans)
    elif norm == '3d':
        trans = norm_3d(bins, trans)
    elif norm == 'sphere':
        trans = norm_sphere(bins, trans, kwargs['radius'])

    return bins, trans, count


def transition_matrix(s0, s1, bins, transform=None, **kwargs):
    """Calculates the transition probability matrix given pairs of distances.

    Parameters
    ----------
    s0 : ndarray
        1d array containing initial distances
    s1 : 1d ndarray of initial distances
        1d array containing final distances
    bins : ndarray
        the binedges that discretizing the distances
    transform : {None | 'radial_2d' | 'radial_3d' | 'sphere'}
        defines the geometry
    radius : float
        required only if the geometry is 'sphere'

    Notes
    -----
    On axis 0, t = 0; axis 1, t = lagt.

    rho(x_i, dt) = sum_j [ trans_ij * rho(x_j, 0) ]
    new_distr = np.sum(trans*distr[:, np.newaxis], 0)
    """
    tr_dict = {'radial_2d': lambda x: x,
               'radial_3d': lambda x: x ** 2,
               'sphere': lambda x: np.sin(x / kwargs['radius'])}

    if transform is not None and transform not in tr_dict:
        raise KeyError('Unknown norm')

    # copy and filter the coordinates
    mask = (s0 >= bins[0]) & (s0 < bins[-1])
    s0 = s0[mask]
    s1 = s1[mask]

    # apply mirror edge conditions
    mirror_before = s1 < bins[0]
    mirror_after = s1 >= bins[-1]
    s1_mirror = s1.copy()
    s1_mirror[mirror_before] = 2 * bins[0] - s1[mirror_before]
    s1_mirror[mirror_after] = 2 * bins[-1] - s1[mirror_after]

    trans_count = np.histogram2d(s0, s1_mirror, bins=[bins, bins])[0]
    count_s0 = trans_count.sum(axis=1)

    if transform is None:
        trans = trans_count
    else:
        # calculate the weights in the transition matrix
        # this equals arclen(s0) / arclen(s1)
        # with arclen the length of equidistant circle with length s
        # TODO: what if geometry varies spatially so that arclen depends on pos?

        # more technically, for a coordinate transformation Cartesian -> Polar:
        # weight = J_0 / J_1
        # with J_n abs(det(Jacobian(dxdy/drdth)) evaluated at s_n
        # for 2D polar: J = s.   for 3D polar, J = s^2.
        # for curved, spherical: J = np.sin(s / radius)

        jacobian = tr_dict[transform]
        weights = jacobian(s0) / jacobian(s1_mirror)
        trans = np.histogram2d(s0, s1_mirror, bins=[bins, bins],
                               weights=weights)[0]

    # fix the transition matrix wherever count_s0 == 0
    # to avoid division by 0 later
    zero_counts = count_s0 == 0
    count_s0[zero_counts] = 1
    eye = np.eye(len(count_s0), dtype=trans.dtype)
    trans[zero_counts] = eye[zero_counts]

    # a particle always has a probability of 1 of going anywhere
    trans /= count_s0[:, np.newaxis]

    return bins, trans, count_s0


def stationary(trans, max_dev=0.0001, max_iter=10000):
    """Generate a stationary concentration profile from a transition matrix."""
    # taking the eigenvector is sensitive to experimental errors
    # just multiply the matrix repeatedly with a unit vector
    if np.any(trans.sum(1) == 0):
        raise ValueError('Every bin should have at least 1 occurence')
    distr = np.ones_like(trans[0])
    for i in range(max_iter):
        new_distr = np.sum(trans*distr[:, np.newaxis], 0)
        new_distr /= new_distr.sum()
        rms_dev = np.sqrt(np.sum((new_distr - distr)**2))
        distr = new_distr
        if rms_dev < max_dev:
            break

    return distr


from scipy.stats import norm
from scipy.integrate import quad

def hist_expectance(edges, hist, sigma):
    """Returns the expectance value of the histogram, given an uncertainty in
    the position. Assuming normal distributed, uncorrelated uncertainty."""
    distr = norm(scale=sigma)
    result = np.zeros(len(hist), dtype=np.float)
    for j in range(len(hist)):
        func = lambda x: distr.cdf(edges[j + 1] - x) - \
                         distr.cdf(edges[j] - x)
        for k, N in enumerate(hist):
            ds_k = edges[k + 1] - edges[k]
            result[j] += N / ds_k * quad(func, edges[k], edges[k + 1])[0]
    return result


def hist_std(edges, hist, sigma):
    """Returns the stdev value of the histogram, given an uncertainty in
    the position. Assuming normal distributed, uncorrelated uncertainty."""
    distr = norm(scale=sigma)
    result = np.zeros(len(hist), dtype=np.float)
    for j in range(len(hist)):
        def func(x):
            diff = distr.cdf(edges[j + 1] - x) - distr.cdf(edges[j] - x)
            return diff * (1 - diff)
        for k, N in enumerate(hist):
            ds_k = edges[k + 1] - edges[k]
            result[j] += N / ds_k * quad(func, edges[k], edges[k + 1])[0]
    return np.sqrt(result)


def stationary_error(distr, trans_err):
    """Compute the standard error in the measured concentration profile."""
    return np.sqrt(np.sum((trans_err*distr[:, np.newaxis])**2, 0))


def transition_error(pairs, min_s, max_s, ds, norm='None', s_err=0, **kwargs):
    """Calculates the transition probability matrix and the error in all the
    elements due to finite counts, given pairs of distances.

    Parameters
    ----------
    pairs : DataFrame
        having columns s0 and s1 being consequent distances between particles
    min_s : number
        minimum separation between particles.
        should be below the lowest possible.
    max_s : number
        maximum separation, mirror conditions will apply at this
    ds : number
        binwidth
    norm : {None | 2d | 3d | sphere}

    Notes
    -----
    On axis 0, t = 0; axis 1, t = lagt.

    rho(x_i, dt) = sum_j [ trans_ij * rho(x_j, 0) ]
    new_distr = np.sum(trans*distr[:, np.newaxis], 0)
    """
    norm = str(norm)
    if norm not in ('2d', '3d', 'sphere', 'None', ''):
        raise ValueError('Unknown norm "{}"'.format(norm))
    bins, trans = _transition_count(pairs, min_s, max_s, ds)

    # calculate uncertainty due to limited number of counts
    # This approach is actually incorrect, as it assumes a Gaussian distribution
    # of the bin count, which is actually a Poisson distribution that actually
    # differs a lot for low N. We do not have enough information to compute the
    # uncertainty in measured N as we do not know the actual N. Better is to
    # compute uncertainty around the model curve.
    # But anyway it does the job of showing N so here we go
    std = np.sqrt(trans)

    # calculate uncertainty due to position uncertainty
    if s_err > 0:
        edges = np.empty(len(bins) + 1, dtype=np.float)
        edges[:-1] = bins - ds / 2
        edges[1:] = bins + ds / 2
        trans_smooth = np.empty(trans.shape, dtype=np.float)
        std_pos = np.empty(trans.shape, dtype=np.float)
        for i in range(len(trans)):
            trans_smooth[i] = hist_expectance(edges, trans[i], s_err)
            std_pos[i] = hist_std(edges, trans[i], s_err)
            print('{0:.2f}'.format(i/len(trans)))

        std = np.sqrt(std**2 + std_pos**2)

    std_rel = np.nan_to_num(std / trans)
    bins, trans, count = _norm_transition(bins, trans, norm, **kwargs)
    return bins, trans, count, std_rel * trans

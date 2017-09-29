from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import six
import numpy as np


def _norm_trans(arclen, trans):
    return trans * arclen[:, np.newaxis] / arclen[np.newaxis, :]


def norm_2d(s, trans):
    return _norm_trans(s, trans)


def norm_3d(s, trans):
    return _norm_trans(s**2, trans)


def norm_sphere(s, trans, R):
    return _norm_trans(np.sin(s / R), trans)


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

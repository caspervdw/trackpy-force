from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import six

import numpy as np
import pandas as pd
from trackpy.utils import cKDTree  # protects against a scipy version issue


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


def _find_pairs(pos, max_dist, max_dist_3):
    """Determine isolated pairs of particles: closer than ``max_dist`` together
    with a possible third particle farther than ``max_dist_3`` from any of them.

    Optimal for few features. Should use cKDTree or something else for many
    features

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
                  lagt=1, pos_columns=None):
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

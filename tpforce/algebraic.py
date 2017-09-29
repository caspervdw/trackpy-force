from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import numpy as np


def dist_eucl(pos0, pos1):
    """Length of a straight line between pos0 and pos1, in N dimensions."""
    return np.sqrt(np.sum((pos0 - pos1)**2, axis=1))


def circle_cap_arclen(h, R):
    """ Length of a circle arc of circle with radius R that is bounded by
    a straight line `h` from the origin. h >= 0, h < R"""
    return 2*R*np.arccos(h / R)


def circle_corner_arclen(h1, h2, R):
    """ Length of a circle arc of circle with radius R that is bounded by
    two perpendicular straight lines `h1` and `h2` from the origin.
    h1**2 + h2**2 < R**2
    h1 >= R
    h2 >= R
    """
    return R*(np.arccos(h2 / R) - np.arcsin(h1 / R))


def sphere_cap_area(h, R):
    """ Area of a sphere cap of sphere with radius R that is bounded by
    a flat plane `h` from the origin. h >= 0, h < R"""
    return 2*np.pi*R*(R-h)


def sphere_edge_area(x, y, R):
    """ Area of a sphere 'edge' of sphere with radius R that is bounded by
    two perpendicular flat planes `h0`, `h1` from the origin. h >= 0, h < R"""
    p = np.sqrt(R**2 - x**2 - y**2)
    A = (R - x - y)*np.pi - 2*R*np.arctan(x*y/(p*R)) + \
        2*x*np.arctan(y/p) + 2*y*np.arctan(x/p)
    return A*R


def sphere_corner_area(x, y, z, R):
    """ Area of a sphere 'corner' of sphere with radius R that is bounded by
    three perpendicular flat planes `h0`, `h1`, `h2` from the origin. """
    pxy = np.sqrt(R**2 - x**2 - y**2)
    pyz = np.sqrt(R**2 - y**2 - z**2)
    pxz = np.sqrt(R**2 - x**2 - z**2)
    A = np.pi*(R - x - y - z)/2 + \
        x*(np.arctan(y/pxy) + np.arctan(z/pxz)) - R*np.arctan(y*z/(R*pyz)) + \
        y*(np.arctan(x/pxy) + np.arctan(z/pyz)) - R*np.arctan(x*z/(R*pxz)) + \
        z*(np.arctan(x/pxz) + np.arctan(y/pyz)) - R*np.arctan(x*y/(R*pxy))
    return A*R


def arclen_2d_bounded(R, pos, box):
    arclen = 2*np.pi*R

    h = np.array([pos[:, 0] - box[0, 0], box[0, 1] - pos[:, 0],
                  pos[:, 1] - box[1, 0], box[1, 1] - pos[:, 1]])

    for h0 in h:
        mask = h0 < R
        if mask.size == 1 and not mask.ravel()[0]:
            continue
        elif mask.size == 1:
            mask = 0
        arclen[mask] -= circle_cap_arclen(h0[mask], R[mask])

    for h1, h2 in [[0, 2], [0, 3], [1, 2], [1, 3]]:  # adjacent sides
        mask = h[h1]**2 + h[h2]**2 < R**2
        if mask.size == 1 and not mask.ravel()[0]:
            continue
        elif mask.size == 1:
            mask = 0
        arclen[mask] += circle_corner_arclen(h[h1, mask], h[h2, mask],
                                             R[mask])

    arclen[arclen < 10**-5 * R] = np.nan
    return arclen


def area_3d_bounded(dist, pos, box, min_z=None, min_x=None):
    """ Calculated using the surface area of a sphere equidistant
    to a certain point.

    When the sphere is truncated by the box boundaries, this distance
    is subtracted using the formula for the sphere cap surface. We
    calculate this by defining h = the distance from point to box edge.

    When for instance sphere is bounded by the top and right boundaries,
    the area in the edge may be counted double. This is the case when
    h1**2 + h2**2 < R**2. This double counted area is calculated
    and added if necessary.

    When the sphere is bounded by three adjacant boundaries,
    the area in the corner may be subtracted double. This is the case when
    h1**2 + h2**2 + h3**2 < R**2. This double counted area is calculated
    and added if necessary.

    The result is the sum of the weights of pos0 and pos1."""

    area = 4*np.pi*dist**2

    h = np.array([pos[:, 0] - box[0, 0], box[0, 1] - pos[:, 0],
                  pos[:, 1] - box[1, 0], box[1, 1] - pos[:, 1],
                  pos[:, 2] - box[2, 0], box[2, 1] - pos[:, 2]])

    if min_x is not None and min_z is not None:
        close_z = dist < min_z
        lower_cutoff = np.sqrt(dist[close_z]**2 - min_x**2)
        h_masked = h[:, close_z]
        h[:2, close_z] = np.vstack((np.minimum(lower_cutoff, h_masked[0]),
                                    np.minimum(lower_cutoff, h_masked[1])))

    for h0 in h:
        mask = h0 < dist
        if mask.size == 1 and not mask.ravel()[0]:
            continue
        elif mask.size == 1:
            mask = 0
        area[mask] -= sphere_cap_area(h0[mask], dist[mask])

    for h1, h2 in [[0, 2], [0, 3], [0, 4], [0, 5],
                   [1, 2], [1, 3], [1, 4], [1, 5],
                   [2, 4], [2, 5], [3, 4], [3, 5]]:  #2 adjacent sides
        mask = h[h1]**2 + h[h2]**2 < dist**2
        if mask.size == 1 and not mask.ravel()[0]:
            continue
        elif mask.size == 1:
            mask = 0
        area[mask] += sphere_edge_area(h[h1, mask], h[h2, mask],
                                       dist[mask])

    for h1, h2, h3 in [[0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
                       [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]]:  #3 adjacent sides
        mask = h[h1]**2 + h[h2]**2 + h[h3]**2 < dist**2
        if mask.size == 1 and not mask.ravel()[0]:
            continue
        elif mask.size == 1:
            mask = 0
        area[mask] -= sphere_corner_area(h[h1, mask], h[h2, mask],
                                         h[h3, mask], dist[mask])

    area[area < 10**-7 * dist**2] = np.nan

    return area


def dist_sphere(pos0, pos1, R):
    """Length of shortest path on a sphere between pos0 and pos1. This is
    the Great Circle Distance. Used here is the Vincenty formula.
    Coordinates are theta, phi.

    Theta is the angle with the z axis (0 <= th <= pi).
    Phi is the angle around the z axis (0 <= phi < 2 pi)
    """
    pos0, pos1 = np.atleast_2d(pos0), np.atleast_2d(pos1)
    p1, l1 = pos0.T
    p2, l2 = pos1.T
    Dl = np.abs(l1 - l2)
    cos = np.cos
    sin = np.sin
    num = np.sqrt((sin(p2) * sin(Dl))**2 +
                  (cos(p1) * sin(p2) * cos(Dl) - sin(p1) * cos(p2))**2)
    denom = cos(p1) * cos(p2) + sin(p1) * sin(p2) * cos(Dl)
    return R*np.arctan2(num, denom)


def arclen_sphere(dist, R):
    """ Calculated using the circumference of a circle equidistant
    to a certain point on a sphere. This is independent of the point
    and only depends on the distance and the sphere radius.

    The angle between point 1 and point 2 with distance `dist`
    from each other is `dist / R`, where R is the sphere radius.

    The circle radius is then the sine of this angle."""
    return 2*np.pi*R* np.sin(dist / R)

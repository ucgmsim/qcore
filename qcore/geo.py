"""
Various tools which may be needed in various processes.
"""

import functools
import itertools
from math import acos, asin, atan, atan2, cos, degrees, pi, radians, sin, sqrt
from subprocess import PIPE, Popen
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
import scipy as sp

from qcore.binary_version import get_unversioned_bin

R_EARTH = 6378.139


def get_distances(
    locations: np.ndarray, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]
):
    """
    Calculates the distance between the array of locations and
    the specified reference location / locations

    Parameters
    ----------
    locations : np.ndarray
        List of locations
        Shape [n_locations, 2], column format (lon, lat)
    lon : Union[float, np.ndarray]
        Array or singular float of Longitude reference locations to compare
    lat : Union[float, np.ndarray]
        Array or singular float of Latitude reference locations to compare

    Returns
    -------
    np.ndarray
        The distances, shape [n_locations] or shape [n_references, n_locations]
        based on the input lon, lat values being a single float or array
    """
    d = (
        np.sin(np.radians(np.expand_dims(locations[:, 1], axis=1) - lat) / 2.0) ** 2
        + np.cos(np.radians(lat))
        * np.cos(np.radians(np.expand_dims(locations[:, 1], axis=1)))
        * np.sin(np.radians(np.expand_dims(locations[:, 0], axis=1) - lon) / 2.0) ** 2
    )
    d = (R_EARTH * 2.0 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))).T
    return d[0] if d.shape[0] == 1 else d


def closest_location(locations, lon, lat):
    """
    Find position and distance of closest location in 2D np.array of (lon, lat).
    """
    d = get_distances(locations, lon, lat)
    i = np.argmin(d)

    return i, d[i]


def ll2gp_multi(
    coords,
    mlon,
    mlat,
    rot,
    nx,
    ny,
    hh,
    dx=1,
    dy=1,
    decimated=False,
    verbose=False,
    keep_outside=False,
):
    """
    Converts longitude/latitude positions to gridpoints.
    Three main modes of operation:
    1: No dx, dy (= 1): gridpoint
    2: dx or dy != 1: closest gridpoint considering dx/dy
    3: decimated: gridpoint number if only decimated points existed
    coords: 2d list in format [[lon0, lat0], [lon1, lat1], ...]
    keep_outside: False will remove values outside the sim domain,
            True will replace those entries with None
    """
    # derived parameters
    xlen = nx * hh
    ylen = ny * hh
    # where the x plane points
    xazim = (rot + 90) % 360

    # run binary, get output
    # output is displacement (x, y) from center, in kilometres
    cmd = [
        get_unversioned_bin("ll2xy"),
        "mlat=%s" % (mlat),
        "mlon=%s" % (mlon),
        "geoproj=1",
        "center_origin=1",
        "h=%s" % (hh),
        "xazim=%s" % (xazim),
        "xlen=%s" % (xlen),
        "ylen=%s" % (ylen),
    ]
    if verbose:
        print(" ".join(cmd))

    # Has to be a byte string
    p_conv = Popen(cmd, stdin=PIPE, stdout=PIPE)
    stdout = p_conv.communicate(
        "\n".join(["%s %s" % tuple(c) for c in coords]).encode()
    )[0].decode()
    xy = [list(map(float, line.split())) for line in stdout.rstrip().split("\n")]

    # convert displacement to grid points
    # has to be 'nx - 1', because the first gridpoint is offset 0km
    # nx = 1400 means the greatest offset is 1399 * hh km
    mid_x = (nx - 1) * hh * 0.5
    mid_y = (ny - 1) * hh * 0.5
    for i, c in enumerate(xy[::-1]):
        # make the distance relative to top corner
        # convert back to grid spacing
        # gridpoints are discrete
        c[0] = int(round((c[0] + mid_x) / hh))
        c[1] = int(round((c[1] + mid_y) / hh))

        # x values range from 0 -> nx - 1
        if not (0 <= c[0] < nx and 0 <= c[1] < ny):
            if keep_outside:
                xy[-(i + 1)] = None
            else:
                # this is why xy is looped in reverse
                xy.remove(c)
            continue

        if decimated:
            c[0] //= dx
            c[1] //= dy
        else:
            # closest gridpoint considering decimation
            c[0] -= c[0] % dx
            c[1] -= c[1] % dy

    return xy


def oriented_bearing_wrt_normal(
    from_direction: np.ndarray, to_direction: np.ndarray, normal: np.ndarray
) -> float:
    """Compute the oriented bearing between two directions with respect to a normal.

    This function is useful to calculate, for example, strike and dip
    directions. The orientation is established via the right hand rule
    (or refer to the diagram below).

        to_direction
           ^
           │
           │
           │
           │
           │
           │<┐  bearing
           │ └─┐
           ╳─────────────> from_direction
          ╱
         ╱
        ╱
       ╱
    normal

    Parameters
    ----------
    from_direction : np.ndarray
        The direction to measure the bearing from.
    to_direction : np.ndarray
        The direction to measure the bearing to.
    normal : np.ndarray
        The normal direction to orient the bearing with via the right hand rule.

    Returns
    -------
    float
        The bearing from from_direction to to_direction oriented with respect to
        the normal direction.

    Examples
    --------
    >>> oriented_bearing_wrt_normal(np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]))
    270
    >>> oriented_bearing_wrt_normal(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    90
    """

    from_dir_hat = from_direction / np.linalg.norm(from_direction)
    to_dir_hat = to_direction / np.linalg.norm(to_direction)
    angle_signed = np.arccos(np.dot(from_dir_hat, to_dir_hat))
    orientation = np.sign(np.dot(np.cross(from_direction, to_direction), normal))
    return np.degrees(angle_signed * orientation) % 360


def ll2gp(
    lat, lon, mlat, mlon, rot, nx, ny, hh, dx=1, dy=1, decimated=False, verbose=False
):
    """
    Converts latitude/longitude to a gridpoint position.
    """
    try:
        return ll2gp_multi(
            [[lon, lat]],
            mlon,
            mlat,
            rot,
            nx,
            ny,
            hh,
            dx=dx,
            dy=dy,
            decimated=decimated,
            verbose=verbose,
            keep_outside=False,
        )[0]
    except IndexError:
        raise ValueError("Input outside simulation domain.")


def gp2ll_multi(
    coords: list[list[float]],
    mlat: float,
    mlon: float,
    rot: float,
    nx: int,
    ny: int,
    hh: float,
) -> list[list[float]]:
    """
    Converts gridpoint positions to longitude, latitude.
    coords: 2d list in format [[x0, y0], [x1, y1], ...]
    """
    # derived parameters
    xlen = nx * hh
    ylen = ny * hh
    # where the x plane points
    xazim = (rot + 90) % 360

    # convert gridpoint to offset km
    max_x = (nx - 1) * hh
    max_y = (ny - 1) * hh
    for c in coords:
        # length from corner
        c[0] *= hh
        c[1] *= hh
        # length from centre origin
        c[0] -= max_x * 0.5
        c[1] -= max_y * 0.5

    # run binary, get output
    p_conv = Popen(
        [
            get_unversioned_bin("xy2ll"),
            "mlat=%s" % (mlat),
            "mlon=%s" % (mlon),
            "geoproj=1",
            "center_origin=1",
            "h=%s" % (hh),
            "xazim=%s" % (xazim),
            "xlen=%s" % (xlen),
            "ylen=%s" % (ylen),
        ],
        stdin=PIPE,
        stdout=PIPE,
    )
    stdout = p_conv.communicate(
        "\n".join(["%s %s" % tuple(c) for c in coords]).encode()
    )[0].decode()

    # lon, lat
    return [list(map(float, line.split())) for line in stdout.rstrip().split("\n")]


def gp2ll(x, y, mlat, mlon, rot, nx, ny, hh):
    """
    Converts a gridpoint position to latitude/longitude.
    """
    return gp2ll_multi([[x, y]], mlat, mlon, rot, nx, ny, hh)[0]


def gen_mat(mrot, mlon, mlat):
    """
    Precursor for xy2ll and ll2xy functions.
    mrot: model rotation
    mlon: model centre longitude
    mlat: model centre latitude
    """
    arg = radians(mrot)
    cosA = cos(arg)
    sinA = sin(arg)

    arg = radians(90.0 - mlat)
    cosT = cos(arg)
    sinT = sin(arg)

    arg = radians(mlon)
    cosP = cos(arg)
    sinP = sin(arg)

    amat = np.array(
        [
            [
                cosA * cosT * cosP + sinA * sinP,
                sinA * cosT * cosP - cosA * sinP,
                sinT * cosP,
            ],
            [
                cosA * cosT * sinP - sinA * cosP,
                sinA * cosT * sinP + cosA * cosP,
                sinT * sinP,
            ],
            [-cosA * sinT, -sinA * sinT, cosT],
        ],
        dtype="f",
    )
    ainv = amat.T * 1.0 / np.linalg.det(amat)

    return amat.flatten(), ainv.flatten()


def xy2ll(xy_km, amat):
    """
    Converts km offsets to longitude and latitude.
    xy_km: 2D np array of [X, Y] offsets from origin (km)
    amat: from gen_mat function
    """
    x = xy_km[:, 0] / R_EARTH
    sinB = np.sin(x)
    y = xy_km[:, 1] / R_EARTH
    sinG = np.sin(y)
    z = np.sqrt(1.0 + sinB * sinB * sinG * sinG)
    xp = sinG * np.cos(x) * z
    yp = sinB * np.cos(y) * z
    zp = np.sqrt(1.0 - xp * xp - yp * yp)

    xg = xp * amat[0] + yp * amat[1] + zp * amat[2]
    yg = xp * amat[3] + yp * amat[4] + zp * amat[5]
    zg = xp * amat[6] + yp * amat[7] + zp * amat[8]

    lat = np.where(
        zg == 0.0, 0.0, 90.0 - np.degrees(np.arctan(np.sqrt(xg * xg + yg * yg) / zg))
    )
    lat[np.where(zg < 0.0)] -= 180.0

    lon = np.where(xg == 0.0, 0.0, np.degrees(np.arctan(yg / xg)))
    lon[np.where(xg < 0.0)] -= 180.0
    lon[np.where(lon < -180.0)] += 360.0

    return np.column_stack((lon, lat))


def ll2xy(ll, ainv):
    """
    Converts longitude and latitude to km offsets.
    ll: 2D np array of [lon, lat]
    ainv: from gen_mat function
    """
    lon = np.radians(ll[:, 0])
    lat = np.radians(90.0 - ll[:, 1])

    xg = np.sin(lat) * np.cos(lon)
    yg = np.sin(lat) * np.sin(lon)
    zg = np.cos(lat)

    xp = xg * ainv[0] + yg * ainv[1] + zg * ainv[2]
    yp = xg * ainv[3] + yg * ainv[4] + zg * ainv[5]
    zp = xg * ainv[6] + yg * ainv[7] + zg * ainv[8]

    # X km offsets from centre origin, Y km offsets from centre origin
    return np.column_stack(
        (
            R_EARTH * np.arcsin(yp / np.sqrt(1.0 - xp * xp)),
            R_EARTH * np.arcsin(xp / np.sqrt(1.0 - yp * yp)),
        )
    )


def xy2gp(xy, nx, ny, hh):
    """
    Converts km offsets to grid points.
    xy: 2D np array of [X, Y] offsets from origin (km)
    nx: number of X grid positions
    ny: number of Y grid positions
    hh: grid spacing
    """
    gp = np.copy(xy)

    # distance from corner
    gp[:, 0] += (nx - 1) * hh * 0.5
    gp[:, 1] += (ny - 1) * hh * 0.5

    # gridpoint from top corner
    gp /= hh

    return np.round(gp).astype(np.int32, copy=False)


def gp2xy(gp, nx, ny, hh):
    """
    Converts grid points to km offsets.
    xy: 2D np array of [X, Y] gridpoints
    nx: number of X grid positions
    ny: number of Y grid positions
    hh: grid spacing
    """
    xy = gp.astype(np.float32) * hh

    # shift for centre origin
    xy[:, 0] -= (nx - 1) * hh * 0.5
    xy[:, 1] -= (ny - 1) * hh * 0.5

    return xy


def ll_shift(lat, lon, distance, bearing):
    """
    Shift lat/long by distance at bearing.
    """
    # formula is for radian values
    lat, lon, bearing = list(map(radians, [lat, lon, bearing]))

    shift = distance / R_EARTH
    lat2 = asin(sin(lat) * cos(shift) + cos(lat) * sin(shift) * cos(bearing))
    lon2 = lon + atan2(
        sin(bearing) * sin(shift) * cos(lat), cos(shift) - sin(lat) * sin(lat2)
    )

    return degrees(lat2), degrees(lon2)


def ll_mid(lon1, lat1, lon2, lat2):
    """
    Return midpoint between a pair of lat, long points.
    """
    # functions based on radians
    lon1, lat1, lat2, dlon = list(map(radians, [lon1, lat1, lat2, (lon2 - lon1)]))

    Bx = cos(lat2) * cos(dlon)
    By = cos(lat2) * sin(dlon)

    lat3 = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + Bx) ** 2 + By**2))
    lon3 = lon1 + atan2(By, cos(lat1) + Bx)

    return degrees(lon3), degrees(lat3)


def ll_dist(lon1, lat1, lon2, lat2):
    """
    Return distance between a pair of lat, long points.
    """
    # functions based on radians
    lat1, lat2, dlon, dlat = list(
        map(radians, [lat1, lat2, (lon2 - lon1), (lat2 - lat1)])
    )

    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return R_EARTH * 2.0 * atan2(sqrt(a), sqrt(1 - a))


def ll_bearing(lon1, lat1, lon2, lat2, midpoint=False):
    """
    Initial bearing when traveling from 1 -> 2.
    Direction facing from point 1 when looking at point 2.
    """
    if midpoint:
        lon1, lat1 = ll_mid(lon1, lat1, lon2, lat2)
    lat1, lat2, lon_diff = map(radians, [lat1, lat2, (lon2 - lon1)])
    return (
        degrees(
            atan2(
                cos(lat2) * sin(lon_diff),
                cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon_diff),
            )
        )
        % 360
    )


def ll_cross_along_track_dist(
    lon1, lat1, lon2, lat2, lon3, lat3, a12=None, a13=None, d13=None
):
    """
    Returns both the distance of point 3 to the nearest point on the great circle line that passes through point 1 and
    point 2 and how far away that nearest point is from point 1, along the great line circle
    If any of a12, a13, d13 are given the calculations for them are skipped
    If all of a12, a13, d13 are given, none of the lon, lat values are used and may be junk data
    Taken from https://www.movable-type.co.uk/scripts/latlong.html
    :param lon1, lat1: The lon, lat coordinates for point 1
    :param lon2, lat2: The lon, lat coordinates for point 2
    :param lon3, lat3: The lon, lat coordinates for point 3
    :param a12: The angle between point 1 (lon1, lat1) and point 2 (lon2, lat2) in radians
    :param a13: The angle between point 1 (lon1, lat1) and point 3 (lon3, lat3) in radians
    :param d13: The distance between point 1 (lon1, lat1) and point 3 (lon3, lat3) in km
    """
    if a12 is None:
        a12 = radians(ll_bearing(lon1, lat1, lon2, lat2))
    if a13 is None:
        a13 = radians(ll_bearing(lon1, lat1, lon3, lat3))
    if d13 is None:
        d13 = ll_dist(lon1, lat1, lon3, lat3)
    d13 /= R_EARTH
    # Only keeping the cross track angle saves a division on the following line
    xta = asin(sin(d13) * sin(a13 - a12))
    # The sign factor of this line is a modification of the original formula to distinguish between up/down strike
    # locations
    ata = acos(cos(d13) / cos(xta)) * np.sign(np.cos(a13 - a12))

    # Multiply the angles by radius to get the distance across the earth surface
    return xta * R_EARTH, ata * R_EARTH


def angle_diff(b1: float, b2: float) -> float:
    """
    Return smallest difference (clockwise, -180 -> 180) from b1 to b2.
    """
    r = (b2 - b1) % 360
    if r > 180:
        return r - 360
    return r


def avg_wbearing(angles):
    """
    Return average angle given angles and weightings.
    NB: angles are clockwise from North, not anti-clockwise from East.
    angles: 2d list of (angle, weight)
    """
    x = 0
    y = 0
    for a in angles:
        x += a[1] * sin(radians(a[0]))
        y += a[1] * cos(radians(a[0]))
    q_diff = 0
    if y < 0:
        q_diff = pi
    elif x < 0:
        q_diff = 2 * pi
    return degrees(atan(x / y) + q_diff)


def build_corners(origin, rot, xlen, ylen):
    """
    Return 4 coordinates at corners centered at origin in [longitude, latitude] format.
    Parameters
    ----------
    origin :  a tuple of (lon,lat)
    rot : bearing (in degrees 0~360)
    xlen : width between corner points
    ylen : height between corner points

    Returns
    -------
    (c1, c2, c3, c4) where each c1, c2, c3 and c4 are in tuple (lon,lat)  format
    """
    # amount to shift from middle
    x_shift = xlen / 2.0
    y_shift = ylen / 2.0

    y_len_mid_shift = R_EARTH * asin(sin(y_shift / R_EARTH) / cos(x_shift / R_EARTH))

    top_mid = ll_shift(
        lat=origin[1], lon=origin[0], distance=y_len_mid_shift, bearing=rot
    )[::-1]
    bottom_mid = ll_shift(
        lat=origin[1],
        lon=origin[0],
        distance=y_len_mid_shift,
        bearing=(rot + 180) % 360,
    )[::-1]

    top_mid_bearing = ll_bearing(*top_mid, *origin) + 180 % 360
    bottom_mid_bearing = ll_bearing(*bottom_mid, *origin)

    c2 = ll_shift(
        lat=top_mid[1],
        lon=top_mid[0],
        distance=x_shift,
        bearing=(top_mid_bearing - 90) % 360,
    )[::-1]
    c1 = ll_shift(
        lat=top_mid[1],
        lon=top_mid[0],
        distance=x_shift,
        bearing=(top_mid_bearing + 90) % 360,
    )[::-1]

    c3 = ll_shift(
        lat=bottom_mid[1],
        lon=bottom_mid[0],
        distance=x_shift,
        bearing=(bottom_mid_bearing - 90) % 360,
    )[::-1]
    c4 = ll_shift(
        lat=bottom_mid[1],
        lon=bottom_mid[0],
        distance=x_shift,
        bearing=(bottom_mid_bearing + 90) % 360,
    )[::-1]

    # at this point we have a perfect square (by corner distance)
    # c1 -> c4 == c2 -> c3 (right == left), c1 -> c2 == c3 -> c4 (top == bottom)
    return c1, c2, c3, c4


def path_from_corners(
    corners=None, output="sim.modelpath_hr", min_edge_points=100, close=True
):
    """
    corners: python list (4 by 2) containing (lon, lat) in order
        otherwise take from velocity model
    output: where to store path of (lon, lat) values
    min_edge_points: at least this many points wanted along edges
    """

    # close the box by going back to origin
    if close:
        corners.append(corners[0])

    # until each side has at least wanted number of points
    while len(corners) < 4 * min_edge_points:
        # work backwards, insertions don't change later indexes
        for i in range(len(corners) - 1, 0, -1):
            val = ll_mid(
                corners[i][0], corners[i][1], corners[i - 1][0], corners[i - 1][1]
            )
            corners.insert(i, val)

    # write points the make the path
    if output != None:
        with open(output, "w") as mp:
            for point in corners:
                mp.write("%s %s\n" % (point[0], point[1]))
    else:
        return corners


def compute_intermediate_latitudes(lon_lat1, lon_lat2, lon_in):
    """
    Calculates the latitudes of points along the shortest path between the points lon_lat1 and lon_lat2, taking the
    shortest path on the sphere, using great circle calculations.
    Note that this is different from the path obtained by interpolating the latitude and longitude between the two points.
    Equation taken from http://www.edwilliams.org/avform.htm#Int
    :param lon_lat1: A tuple containing the longitude and latitude of the start point
    :param lon_lat2: A tuple containing the longitude and latitude of the end point
    :param lon_in: A float or iterable of floats compliant with numpy of the longitude(s) to have latitudes calculated for
    :return: A float or iterable of floats which represent the latitude(s) of the value(s) given in lon_in
    """
    conversion_factor = np.pi / 180
    lon1, lat1 = lon_lat1
    lon2, lat2 = lon_lat2
    lat1 *= conversion_factor
    lon1 *= conversion_factor
    lat2 *= conversion_factor
    lon2 *= conversion_factor
    lon = lon_in * conversion_factor
    if lon1 == lon2:
        return np.linspace(lat1, lat2, len(lon_in)) / conversion_factor
    return (
        np.arctan(
            (
                np.sin(lat1) * np.cos(lat2) * np.sin(lon - lon2)
                - np.sin(lat2) * np.cos(lat1) * np.sin(lon - lon1)
            )
            / (np.cos(lat1) * np.cos(lat2) * np.sin(lon1 - lon2))
        )
    ) / conversion_factor


def homogenise_point(p: np.ndarray) -> np.ndarray:
    """Express a point in homogenous coordinates.

    Given a point p in R^3 return the affine point in PG(3, R) associated with p. Informally, it maps
    (x, y, z) -> (x, y, z, 1).

    Parameters
    ----------
    p : np.ndarray
        A point in R^3.

    Returns
    -------
    np.ndarray
        The associated point in PG(3, R).

    Examples
    --------
    >>> homogenise_point(np.array([1, 0, 0]))
    array([1,0,0,1])
    """

    return np.append(p, 1)


def projective_span(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Find the projective plane spanned by p, q, and r.

    Parameters
    ----------
    p : np.ndarray
        The homogenous coordinates of p.
    q : np.ndarray
        The homogenous coordinates of q.
    r : np.ndarray
        The homogenous coordinates of r.

    Returns
    -------
    np.ndarray
        The dual coordinates of the plane spanning p, q, and r

    Raises
    ------
    ValueError
        If the supplied points don't span a plane in PG(3, R)

    Examples
    --------
    >>> projective_span(np.array([1, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([0, 0, 0, 1]))
    np.array([0, 0, 1, 0])
    >>> projective_span(np.array([0, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([0.5, 0.5, 0, 1]))
    ValueError: Points supplied do not span a plane. # NOTE: these points lie on the line x = y.
    """
    M = np.vstack([p, q, r])
    null_space = sp.linalg.null_space(M)
    # The null space of M describes the space of all dual coordinates x such
    # that x * p == 0, x * q == 0, and x * r == 0. Such coordinates, if p, q,
    # and r are not collinear, should be unique up to scalar multiples. In that
    # case, the null_space has rank 1 to represent these scalar multiples.
    # Hence, if null_space does not have rank 1 (checked via shape[1]), the
    # points supplied cannot be contained within a unique plane.
    if null_space.shape[1] != 1:
        raise ValueError("Points supplied do not span a plane.")
    # As previously discussed, the dual coordinates for any plane are not
    # unique. The plane x = 0 could be given dual coordinates (1, 0, 0, 0), or
    # equivalently (2, 0, 0, 0). We need to choose one of these coordinates
    # deterministically. This is usually done by rescaling the coordinates such
    # that the last non-zero coordinate is one (a process called normalisation).
    #
    # NOTE: This choice to normalise by the last coordinate is arbitrary. Any
    # method of normalisation would work.
    null_space = null_space.ravel()
    c = next(x for x in reversed(null_space) if not np.isclose(x, 0))
    return null_space / c


def plane_from_three_points(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Find the plane spanning three points.

    Returns the coefficient vector of the affine plane spanning three points.
    Note this is distinct from projective_span in that p, q, and r are three
    points in R^3.

    Parameters
    ----------
    p : np.ndarray
        a point on the plane.
    q : np.ndarray
        a point on the plane.
    r : np.ndarray
        a point on the plane.

    Returns
    -------
    np.ndarray
        The dual coordinates of the plane spanning p, q, and r.

    Examples
    --------
    >>> plane_from_three_points(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    array([ 1., -0., -0., -0.]) # Plane x = 0
    >>> plane_from_three_points(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    array([-1., -1., -1.,  1.]) # Plane x + y + z = 1
    """
    p = homogenise_point(p)
    q = homogenise_point(q)
    r = homogenise_point(r)
    return projective_span(p, q, r)


def orthogonal_plane(pi: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Find the orthogonal plane to pi through p and q.

    Given two points p and q, find the unique orthogonal plane that meets pi at
    p and q.

    Parameters
    ----------
    pi : np.ndarray
        Dual coordinates of the plane.
    p : np.ndarray
        Coordinates of the point p.
    q : np.ndarray
        Coordinates of the point q

    Returns
    -------
    np.ndarray
        The dual coordinates of the homogenous plane.

    Examples
    --------
    >>> pi = plane_from_three_points(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    >>> orthogonal_plane(pi, np.array([0, 0, 0]), np.array([0, 1, 0]))
    array([0., 0., 1., 0.]) # The orthogonal plane z = 0
    >>> pi = np.array([0, 0, 1, 0])
    >>> orthogonal_plane(pi, np.array([1, 0, 0]), np.array([0, 1, 0]))
    array([-1., -1.,  0.,  1.]) # The plane x + y = 1. You should check that it is orthogonal, and it contains the points [1, 0, 0] and [0, 1, 0]
    """
    # We can do this efficiently with some projective geometry.
    # We are looking for a plane gamma with dual coordinates x such that:
    # - The plane pi is orthogonal to x: pi * x = 0
    # - The plane x contains p: p * x = 0
    # - The plane x contains q: q * x = 0
    # (assuming pi is a row vector, and x a column vector)
    # So we want the null space of [pi; p; q]
    # This will be a one-dimensional subspace of R^4 (i.e. a projective point).
    # So we again have to normalise like in plane_from_three_points.
    # ...actually, this is exactly identical to asking for projective_span(pi, p', q'),
    # where p' and q' are the homogenised equivalent of p and q!
    p = homogenise_point(p)
    q = homogenise_point(q)
    return projective_span(pi, p, q)


def oriented_bounding_planes(
    plane_dual_coordinates, plane_corners: np.ndarray
) -> List[np.ndarray]:

    plane_centroid = np.average(plane_corners, axis=0)
    # For each side of the plane p1, we construct a plane orthogonal to p1
    # passing through the side.
    bounding_planes = [
        orthogonal_plane(
            plane_dual_coordinates,
            plane_corners[i],
            plane_corners[(i + 1) % len(plane_corners)],
        )
        for i in range(len(plane_corners))
    ]

    # The dual coordinates of a plane are defined up to scalar multiples. Here we
    # scale the coordinates to ensure that the normal vectors point towards the
    # centre.
    #
    # The picture should look like
    #
    #      x
    #     / \
    # p1'/  |
    #   /    x--------x
    #  X  ---> norm  /
    #   \  /   .    / p1
    #   | /   c    /
    #    x--------x
    #
    # Where p1 is the plane, p1' the plane orthogonal to p1, and norm the normal
    # vector pointing towards c, the centroid. This ensures that the inequality
    # norm * x >= 0 describes all points on the same side of p1' as c. Adding a
    # condition like this for each edge of p1 bounds our search to just points
    # on p1.
    for i, norm in enumerate(bounding_planes):
        if norm.dot(homogenise_point(plane_centroid)) < 0:
            bounding_planes[i] *= -1

    return bounding_planes


def closest_points_between_planes(
    p1_corners: np.ndarray, p2_corners: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the closest points between two finite planes.

    This function solves the closest point problem by phrasing it as a
    constrained quadratic optimisation problem, and then solving using scipy's
    optimisation package.

    Parameters
    ----------
    p1_corners : np.ndarray
        The corners of the first plane.
        p1_corners[i] and p1_corners[( i + 1 ) % 4] must be adjacent corners.
    p2_corners : np.ndarray
        The corners of the second plane.
        p1_corners[i] and p1_corners[( i + 1 ) % 4] must be adjacent corners.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The point pair (p, q) with p in p1 and q in p2 such that ||p - q|| is minimised.
    """
    # These dual coordinates are used to enforce the condition that the found
    # points lie within p1 and p2 respectively.
    p1_dual_coordinates = plane_from_three_points(*p1_corners[:3])
    p2_dual_coordinates = plane_from_three_points(*p2_corners[:3])

    p1_in_plane_constraint = {
        "type": "eq",
        "fun": lambda x: p1_dual_coordinates.dot(homogenise_point(x[:3])),
    }

    p2_in_plane_constraint = {
        "type": "eq",
        "fun": lambda x: p2_dual_coordinates.dot(homogenise_point(x[3:])),
    }

    p1_ortho_planes = oriented_bounding_planes(p1_dual_coordinates, p1_corners)
    p2_ortho_planes = oriented_bounding_planes(p2_dual_coordinates, p2_corners)

    def p1_constraint(ortho_coords, x):
        return ortho_coords.dot(homogenise_point(x[:3]))

    def p2_constraint(ortho_coords, x):
        return ortho_coords.dot(homogenise_point(x[3:]))

    p1_bounds_constraints = [
        {
            "type": "ineq",
            "fun": functools.partial(p1_constraint, ortho_coords),
        }  # we use functools.partial because we want to copy ortho_coords by value rather than by reference.
        for ortho_coords in p1_ortho_planes
    ]
    p2_bounds_constraints = [
        {"type": "ineq", "fun": functools.partial(p2_constraint, ortho_coords)}
        for ortho_coords in p2_ortho_planes
    ]

    def distance(plane_point_positions):
        # x is a numpy array of length 6 containing the position of the points p in x[:3] and q in x[3:].
        # returns ||p - q||^2
        return sp.spatial.distance.cdist(
            plane_point_positions[:3].reshape((1, 3)),
            plane_point_positions[3:].reshape((1, 3)),
            metric="sqeuclidean",
        )

    # Our initial guess is just the centroid of each plane.
    p1_centroid = np.average(p1_corners, axis=0)
    p2_centroid = np.average(p2_corners, axis=0)

    plane_centroids = [*p1_centroid, *p2_centroid]

    result = sp.optimize.minimize(
        distance,
        plane_centroids,
        constraints=[
            p1_in_plane_constraint,
            p2_in_plane_constraint,
        ]
        + p1_bounds_constraints
        + p2_bounds_constraints,
        options={"ftol": 1e-10},
    )

    solution = result.x
    return (solution[:3], solution[3:])


def closest_points_between_plane_sequences(
    sequence1_planes: np.ndarray, sequence2_planes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the closest points between two sequences of planes.

    Parameters
    ----------
    sequence1_planes : np.ndarray
        A sequence of planes (tensor of shape (n x 4 x 3)).
    sequence2_planes : np.ndarray
        A sequence of planes (tensor of shape (n x 4 x 3)).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Points (p, q) minimising ||p - q||, such that p lies on sequence1_planes and
        q lies on sequence2_planes.

    """
    return min(
        (
            closest_points_between_planes(p1, p2)
            for (p1, p2) in itertools.product(sequence1_planes, sequence2_planes)
        ),
        key=lambda pq: sp.spatial.distance.cdist(
            pq[0].reshape((1, 3)), pq[1].reshape((1, 3)), metric="sqeuclidean"
        ),
    )

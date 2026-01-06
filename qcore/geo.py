"""
qcore geometry utilities.
"""

from math import asin, atan, atan2, cos, degrees, pi, radians, sin, sqrt
from typing import Union

import numpy as np
import numpy.typing as npt

R_EARTH = 6378.139


def get_distances(
    locations: np.ndarray, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]
) -> np.ndarray:
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


def closest_location(
    locations: np.ndarray, lon: float, lat: float
) -> tuple[int, float]:
    """
    Find position and distance of closest location in 2D np.array of (lon, lat).
    """
    d = get_distances(locations, lon, lat)
    i = np.argmin(d)

    return i, d[i]


def oriented_bearing_wrt_normal(
    from_direction: np.ndarray, to_direction: np.ndarray, normal: np.ndarray
) -> np.float64:
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
    # If the from_direction ~ +/- to_direction, orientation (and so
    # bearing) will be zero. The expression (orientation or 1) ensures
    # that bearings of 180 degrees are handled correctly.
    return np.degrees(angle_signed * (orientation or 1)) % 360


def gen_mat(mrot: float, mlon: float, mlat: float) -> tuple[np.ndarray, np.ndarray]:
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


def xy2ll(xy_km: np.ndarray, amat: np.ndarray) -> np.ndarray:
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


def ll2xy(ll: np.ndarray, ainv: np.ndarray) -> np.ndarray:
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


def xy2gp(xy: np.ndarray, nx: int, ny: int, hh: float) -> np.ndarray:
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


def gp2xy(gp: np.ndarray, nx: int, ny: int, hh: float) -> np.ndarray:
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


def ll_shift(
    lat: float, lon: float, distance: float, bearing: float
) -> tuple[float, float]:
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


def ll_mid(lon1: float, lat1: float, lon2: float, lat2: float) -> tuple[float, float]:
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


def ll_dist(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Return distance between a pair of lat, long points.
    """
    # functions based on radians
    lat1, lat2, dlon, dlat = list(
        map(radians, [lat1, lat2, (lon2 - lon1), (lat2 - lat1)])
    )

    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return R_EARTH * 2.0 * atan2(sqrt(a), sqrt(1 - a))


def ll_bearing(
    lon1: float, lat1: float, lon2: float, lat2: float, midpoint: bool = False
):
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


def angle_diff(b1: float, b2: float) -> float:
    """
    Return smallest difference (clockwise, -180 -> 180) from b1 to b2.
    """
    r = (b2 - b1) % 360
    if r > 180:
        return r - 360
    return r


def avg_wbearing(angles: list[list[float]]) -> float:
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


def path_from_corners(
    corners: list[tuple[float, float]],
    output: str = "sim.modelpath_hr",
    min_edge_points: int = 100,
    close: bool = True,
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
    if output is not None:
        with open(output, "w", encoding="utf-8") as mp:
            for point in corners:
                mp.write(f"{point[0]} {point[1]}\n")
    else:
        return corners


def rotation_matrix(angle: float) -> np.ndarray:
    """Returns the 2D rotation matrix for a given angle.

    Parameters
    ----------
    angle : float
        The angle to rotate by in radians.

    Returns
    -------
    np.ndarray
        The 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def point_to_segment_distance(
    p: npt.ArrayLike, q: npt.ArrayLike, r: npt.ArrayLike
) -> float:
    """Compute the shortest distance between a point and a line segment.

    See [1] for a concise explanation of the calculations involved.

    Parameters
    ----------
    p : npt.ArrayLike
        A point to measure distance to.
    q : npt.ArrayLike
        The first point of the line segment.
    r : npt.ArrayLike
        The second point of the line segment.

    Returns
    -------
    float
        The distance between r and the closest point on the line
        segment pq to r.

    References
    ----------
    [1]: https://math.stackexchange.com/questions/2193720/find-a-point-on-a-line-segment-which-is-the-closest-to-other-point-not-on-the-li/2193733#2193733
    """
    p = np.asarray(p)
    q = np.asarray(q)
    r = np.asarray(r)

    qr = r - q
    qp = p - q

    if np.allclose(qr, 0):
        raise ValueError("Degenerate line segment q -> r!")

    t = np.clip(np.dot(qp, qr) / np.dot(qr, qr), 0, 1)

    closest_point = q + t * qr

    return float(np.linalg.norm(p - closest_point))

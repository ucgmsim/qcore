"""
qcore geometry utilities.
"""

import numpy as np
import numpy.typing as npt

R_EARTH = 6378.139


def get_distances(
    locations: np.ndarray, lon: float | np.ndarray, lat: float | np.ndarray
) -> np.ndarray:
    """
    Calculates the distance between the array of locations and
    the specified reference location / locations

    Parameters
    ----------
    locations : np.ndarray
        list of locations
        Shape [n_locations, 2], column format (lon, lat)
    lon : float | np.ndarray
        Array or singular float of Longitude reference locations to compare
    lat : float | np.ndarray
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
    Find the index and distance of the closest location to a reference point.

    Parameters
    ----------
    locations : np.ndarray
        Array of shape (n_locations, 2) containing longitude and latitude pairs (degrees).
    lon : float
        Reference longitude in degrees.
    lat : float
        Reference latitude in degrees.

    Returns
    -------
    tuple of (int, float)
        Index of the closest location and its distance in kilometers.
    """
    d = get_distances(locations, lon, lat)
    i = np.argmin(d)

    return int(i), float(d[i])


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
    Generate model rotation and inverse matrices for coordinate transformations.

    Parameters
    ----------
    mrot : float
        Model rotation angle in degrees.
    mlon : float
        Model center longitude in degrees.
    mlat : float
        Model center latitude in degrees.

    Returns
    -------
    tuple of np.ndarray
        Flattened (3×3) transformation matrix and its inverse.
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
    Convert XY offsets (in km) to geographic coordinates (longitude, latitude).

    Parameters
    ----------
    xy_km : np.ndarray
        Array of shape (n, 2) representing offsets in kilometers from the model origin.
    amat : np.ndarray
        Transformation matrix generated by `gen_mat`.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) containing longitude and latitude in degrees.
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
    Convert geographic coordinates to XY offsets (in km).

    Parameters
    ----------
    ll : np.ndarray
        Array of shape (n, 2) containing longitude and latitude (degrees).
    ainv : np.ndarray
        Inverse transformation matrix from `gen_mat`.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) containing X, Y offsets in kilometers.
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


def gp2xy(gp: np.ndarray, nx: int, ny: int, hh: float) -> np.ndarray:
    """
    Convert grid indices to XY offsets in kilometers.

    Parameters
    ----------
    gp : np.ndarray
        Grid points array of shape (n, 2).
    nx : int
        Number of grid points along X.
    ny : int
        Number of grid points along Y.
    hh : float
        Grid spacing in kilometers.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) containing X, Y offsets (km) relative to the grid center.
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
    Compute a new latitude and longitude by shifting from a point by a given distance and bearing.

    Parameters
    ----------
    lat : float
        Starting latitude in degrees.
    lon : float
        Starting longitude in degrees.
    distance : float
        Distance to move in kilometers.
    bearing : float
        Bearing angle in degrees clockwise from north.

    Returns
    -------
    tuple of (float, float)
        New latitude and longitude in degrees.
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
    Compute the geographic midpoint between two points.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the first point (degrees).
    lon2, lat2 : float
        Longitude and latitude of the second point (degrees).

    Returns
    -------
    tuple of (float, float)
        Midpoint longitude and latitude in degrees.
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
    Compute great-circle distance between two points.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the first point (degrees).
    lon2, lat2 : float
        Longitude and latitude of the second point (degrees).

    Returns
    -------
    float
        Distance between the points in kilometers.
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
    Compute the initial bearing from one geographic point to another.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the first point (degrees).
    lon2, lat2 : float
        Longitude and latitude of the second point (degrees).
    midpoint : bool, optional
        If True, compute bearing at the midpoint. Default is False.

    Returns
    -------
    float
        Bearing in degrees clockwise from north.
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
    Compute the signed smallest difference between two bearings.

    Parameters
    ----------
    b1 : float
        First bearing in degrees.
    b2 : float
        Second bearing in degrees.

    Returns
    -------
    float
        Difference in degrees, within (-180, 180].
    """
    r = (b2 - b1) % 360
    if r > 180:
        return r - 360
    return r


def avg_wbearing(angles: list[list[float]]) -> float:
    """
    Compute the weighted average of a set of bearings.

    Parameters
    ----------
    angles : list of [float, float]
        Each element is [angle, weight], where angle is in degrees.

    Returns
    -------
    float
        Weighted average bearing in degrees clockwise from north.
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


@overload
def path_from_corners(
    corners: list[tuple[float, float]],
    output: str | None = None,
    min_edge_points: int = ...,
    close: bool = ...,
) -> list[tuple[float, float]]: ...


@overload
def path_from_corners(
    corners: list[tuple[float, float]],
    output: str = ...,
    min_edge_points: int = ...,
    close: bool = ...,
) -> None: ...


def path_from_corners(
    corners: list[tuple[float, float]],
    output: str | None = "sim.modelpath_hr",
    min_edge_points: int = 100,
    close: bool = True,
) -> list[tuple[float | int, float | int]] | None:
    """
    Generate a path connecting the corners of a region with optional subdivision and output.

    Parameters
    ----------
    corners : list of tuple of float
        List of (lon, lat) coordinates defining the polygon corners in order.
    output : str or None, optional
        Path to save the generated path points, or None to return them.
    min_edge_points : int, optional
        Minimum number of points per edge. Default is 100.
    close : bool, optional
        Whether to close the polygon by connecting back to the first corner. Default is True.

    Returns
    -------
    list of tuple of float or None
        List of (lon, lat) coordinates if `output` is None, otherwise None.
    """

    # close the box by going back to origin
    if close:
        corners.append(corners[0])

    # until each side has at least the wanted number of points
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


def ll_cross_along_track_dist(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    lon3: float,
    lat3: float,
    a12: float | None = None,
    a13: float | None = None,
    d13: float | None = None,
) -> tuple[float, float]:
    """
    Compute the cross-track and along-track distances from a third point to a great-circle path.

    This function calculates both:
    - The shortest (cross-track) distance from point 3 to the great-circle line
      defined by points 1 and 2.
    - The along-track distance from point 1 to the nearest point on that line.

    If any of `a12`, `a13`, or `d13` are provided, the corresponding calculations
    are skipped. If all three are given, longitude and latitude values are ignored.

    Based on:
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lon1 : float
        Longitude of point 1 in degrees.
    lat1 : float
        Latitude of point 1 in degrees.
    lon2 : float
        Longitude of point 2 in degrees.
    lat2 : float
        Latitude of point 2 in degrees.
    lon3 : float
        Longitude of point 3 in degrees.
    lat3 : float
        Latitude of point 3 in degrees.
    a12 : float, optional
        Initial bearing from point 1 to point 2 in radians. If None, computed automatically.
    a13 : float, optional
        Initial bearing from point 1 to point 3 in radians. If None, computed automatically.
    d13 : float, optional
        Distance between point 1 and point 3 in kilometers. If None, computed automatically.

    Returns
    -------
    tuple of float
        A tuple containing:
        - Cross-track distance (float): Distance from point 3 to the great-circle path, in kilometers.
        - Along-track distance (float): Distance from point 1 to the nearest point on the great-circle path, in kilometers.
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

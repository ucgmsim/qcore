"""
Module for coordinate conversions between WGS84 (latitude and longitude) and
NZTM (New Zealand Transverse Mercator) coordinate systems.

Functions
----------
- wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    Converts WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.
- nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    Converts NZTM coordinates (y, x, depth) to WGS84 coordinates.

References
----------
This module provides functions for converting coordinates between WGS84 and NZTM coordinate systems.
See LINZ[0] for a description of the NZTM coordinate system.

[0]: https://www.linz.govt.nz/guidance/geodetic-system/coordinate-systems-used-new-zealand/projections/new-zealand-transverse-mercator-2000-nztm2000
"""

from typing import Union

import numpy as np
import pyproj

from qcore import geo

# Module level conversion constants for internal use only.
_WGS_CODE = 4326
_NZTM_CODE = 2193

_WGS2NZTM = pyproj.Transformer.from_crs(_WGS_CODE, _NZTM_CODE)
_NZTM2WGS = pyproj.Transformer.from_crs(_NZTM_CODE, _WGS_CODE)

_REF_ELLIPSOID = pyproj.Geod(ellps="GRS80")


def wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.

    Parameters
    ----------
    wgs_depth_coordinates : np.ndarray
        An array of shape (N, 3), (N, 2), (2,) or (3,) containing WGS84
        coordinates latitude, longitude and, optionally, depth.

    Returns
    -------
    np.ndarray
        An array with the same shape as wgs_depth_coordinates containing NZTM
        coordinates y, x and, optionally, depth.

    Raises
    ------
    ValueError
        If the given coordinates are not in the valid range to be converted
        to NZTM coordinates.

    Examples
    --------
    >>> wgs_depth_to_nztm(np.array([-43.5320, 172.6366]))
    array([5180040.61473068, 1570636.6812821])
    >>> wgs_depth_to_nztm(np.array([-43.5320, 172.6366, 1]))
    array([5180040.61473068, 1570636.6812821 ,       1.        ])
    >>> wgs_depth_to_nztm(np.array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]]))
    array([[5.92021456e+06, 1.75731133e+06, 1.00000000e+02],
           [5.42725716e+06, 1.74893148e+06, 0.00000000e+00]])
    """
    nztm_coordinates = np.array(_WGS2NZTM.transform(*wgs_depth_coordinates.T)).T
    if not np.all(np.isfinite(nztm_coordinates)):
        raise ValueError(
            "Latitude and longitude coordinates given are invalid (did you input lon, lat instead of lat, lon?)"
        )

    return nztm_coordinates


def nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert NZTM coordinates (y, x, depth) to WGS84 coordinates.

    Parameters
    ----------
    nztm_coordinates : np.ndarray
        An array of shape (N, 3), (N, 2), (2,) or (3,) containing NZTM
        coordinates y, x and, optionally, depth.

    Returns
    -------
    np.ndarray
        An array with the same shape as nztm_coordinates containing WGS84
        coordinates latitude, longitude and, optionally, depth.

    Raises
    ------
    ValueError
        If the given NZTM coordinates are not valid.

    Examples
    --------
    >>> nztm_to_wgs_depth(np.array([5180040.61473068, 1570636.6812821]))
    array([-43.5320, 172.6366])
    >>> nztm_to_wgs_depth(np.array([5180040.61473068, 1570636.6812821, 0]))
    array([-43.5320, 172.6366, 0])
    >>> wgs_depth_to_nztm(array([[5.92021456e+06, 1.75731133e+06, 100],
                                 [5.42725716e+06, 1.74893148e+06, 0]]))
    array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]])
    """
    wgs_coordinates = np.array(_NZTM2WGS.transform(*nztm_coordinates.T)).T
    if not np.all(np.isfinite(wgs_coordinates)):
        raise ValueError(
            "NZTM coordinates given are invalid (did you input x, y instead of y, x?)"
        )
    return wgs_coordinates


def bearing_between(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Return the bearing between two points in lat, lon format.

    Parameters
    ----------
    point_a : np.ndarray
        The first point (lat, lon).
    point_b : np.ndarray
        The second point (lat, lon).

    Returns
    -------
    float
        The bearing (in degrees) between point_a and point_b.

    Examples
    --------
    >>> nztm_bearing_between(
        np.array([-44.88388755, 166.8699418])
        np.array([-42.73774027, 171.03176429])
    )
    55.999999966075194
    """
    fwd_azimuth, _, _ = _REF_ELLIPSOID.inv(
        point_a[1], point_a[0], point_b[1], point_b[0]
    )
    return fwd_azimuth


def nztm_bearing_between(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Return the bearing between two points in NZTM format.

    Parameters
    ----------
    point_a : np.ndarray
            The first point (y, x).
    point_b : np.ndarray
            The second point (y, x).

    Returns
    -------
    float
        The bearing (in degrees) between point_a and point_b.

    Examples
    --------
    >>> nztm_bearing_between(
        np.array([5011637.39446645, 1115879.1174585 ]),
        np.array([5266429.66487645, 1438889.1789583 ])
    )
    55.999999966075194
    """
    return bearing_between(nztm_to_wgs_depth(point_a), nztm_to_wgs_depth(point_b))


def forward_bearing(point_a: np.ndarray, bearing: float, distance: float) -> np.ndarray:
    """Return the point at a given bearing and distance from point_a.

    Parameters
    ----------
    point_a : np.ndarray
        The origin point (lat, lon).
    bearing : float
        The bearing (in degrees) from point_a.
    distance : float
        The distance (in metres) to shift.

    Returns
    -------
    np.ndarray
        The new point (lat, lon) at the given bearing and distance from
        point_a.

    Examples
    --------
    >>> forward_bearing(
        np.array([-43.0, -172.0]),
        45.0,
        1000.0
    )
    array([-42.99363465, 172.0086709 ])
    """
    lon, lat, _ = _REF_ELLIPSOID.fwd(
        point_a[1], point_a[0], bearing, distance, radians=False
    )
    return np.array([lat, lon])


def nztm_forward_bearing(
    point_a: np.ndarray, bearing: float, distance: float
) -> np.ndarray:
    """Return the point at a given bearing and distance from point_a.

    Parameters
    ----------
    point_a : np.ndarray
            The origin point (y, x).
    bearing : float
            The bearing (in degrees) from point_a.
    distance : float
            The distance (in metres) to shift.

    Returns
    -------
    np.ndarray
            The new point (y, x) at the given bearing and distance from
            point_a.

    Examples
    --------
    >>> forward_bearing(
        array([5128644.8819868 , 2823486.41350085]),
        45.0,
        1000.0
    )
    array([5239415.32038242, 1519189.76925286])
    """
    return wgs_depth_to_nztm(
        forward_bearing(nztm_to_wgs_depth(point_a), bearing, distance)
    )


def distance_between_wgs_depth_coordinates(
    point_a: np.ndarray, point_b: np.ndarray
) -> Union[float, np.ndarray]:
    """Return the distance between two points in lat, lon, depth format.

    Valid only for points that can be converted into NZTM format.

    Parameters
    ----------
    point_a : np.ndarray
        The first point (lat, lon, optionally depth). May have shape
        (2,), (3,), (n, 2), (n, 3).

    point_b : np.ndarray
        The second point (lat, lon, optionally depth). May have shape
        (2, ), (3,), (n, 2), (n, 3)

    Returns
    -------
    float or np.ndarray
        The distance (in metres) between point_a and point_b. Will
        return an array of floats if input contains multiple points
    """
    if len(point_a.shape) > 1:
        return np.linalg.norm(
            wgs_depth_to_nztm(point_a) - wgs_depth_to_nztm(point_b), axis=1
        )
    return np.linalg.norm(wgs_depth_to_nztm(point_a) - wgs_depth_to_nztm(point_b))

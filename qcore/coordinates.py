"""
Module for coordinate conversions between WGS84 (latitude and longitude) and
NZTM (New Zealand Transverse Mercator) coordinate systems.

Functions
----------
- wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    Converts WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.
- nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    Converts NZTM coordinates (x, y, depth) to WGS84 coordinates.

References
----------
This module provides functions for converting coordinates between WGS84 and NZTM coordinate systems.
See LINZ[0] for a description of the NZTM coordinate system.

[0]: https://www.linz.govt.nz/guidance/geodetic-system/coordinate-systems-used-new-zealand/projections/new-zealand-transverse-mercator-2000-nztm2000
"""

from typing import Union

import numpy as np
import pyproj

# Module level conversion constants for internal use only.
_WGS_CODE = 4326
_NZTM_CODE = 2193

_WGS2NZTM = pyproj.Transformer.from_crs(_WGS_CODE, _NZTM_CODE)
_NZTM2WGS = pyproj.Transformer.from_crs(_NZTM_CODE, _WGS_CODE)


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
        coordinates x, y and, optionally, depth.

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
    return np.array(_WGS2NZTM.transform(*wgs_depth_coordinates.T)).T


def nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert NZTM coordinates (x, y, depth) to WGS84 coordinates.

    Parameters
    ----------
    nztm_coordinates : np.ndarray
        An array of shape (N, 3), (N, 2), (2,) or (3,) containing NZTM
        coordinates x, y and, optionally, depth.

    Returns
    -------
    np.ndarray
        An array with the same shape as nztm_coordinates containing WGS84
        coordinates latitude, longitude and, optionally, depth.

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
    return np.array(_NZTM2WGS.transform(*nztm_coordinates.T)).T


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

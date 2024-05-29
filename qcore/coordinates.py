"""
Module for coordinate conversions between WGS84 (latitude and longitude) and NZTM (New Zealand Transverse Mercator) coordinate systems.

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
        An array of shape (N, 3) or (3,) containing WGS84 coordinates
        (latitude, longitude, depth).

    Returns
    -------
    np.ndarray
        An array of shape (N, 3) containing NZTM coordinates (x, y, depth).

    Examples
    --------
    >>> wgs_depth_to_nztm(np.array([-43.5320, 172.6366, 0]))
    array([5180040.61473068, 1570636.6812821 ,       0.        ])
    >>> wgs_depth_to_nztm(np.array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]]))
    array([[5.92021456e+06, 1.75731133e+06, 1.00000000e+02],
           [5.42725716e+06, 1.74893148e+06, 0.00000000e+00]])
    """
    if wgs_depth_coordinates.shape == (3,):
        return np.array(_WGS2NZTM.transform(*wgs_depth_coordinates))
    return np.array(
        _WGS2NZTM.transform(
            wgs_depth_coordinates[:, 0],
            wgs_depth_coordinates[:, 1],
            wgs_depth_coordinates[:, 2],
        )
    ).T


def nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert NZTM coordinates (x, y, depth) to WGS84 coordinates.

    Parameters
    ----------
    nztm_coordinates : np.ndarray
        An array of shape (N, 3) or (3,) containing NZTM coordinates (x, y, depth).

    Returns
    -------
    np.ndarray
        An array of shape (N, 3) containing WGS84 coordinates (latitude, longitude, depth).

    Examples
    --------
    >>> nztm_to_wgs_depth(np.array([5180040.61473068, 1570636.6812821, 0]))
    array([-43.5320, 172.6366, 0])
    >>> wgs_depth_to_nztm(array([[5.92021456e+06, 1.75731133e+06, 100],
                                 [5.42725716e+06, 1.74893148e+06, 0]]))
    array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]])
    """
    if nztm_coordinates.shape == (3,):
        return np.array(_NZTM2WGS.transform(*nztm_coordinates))
    return np.array(
        _NZTM2WGS.transform(
            nztm_coordinates[:, 0],
            nztm_coordinates[:, 1],
            nztm_coordinates[:, 2],
        )
    ).T

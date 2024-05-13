"""
Module for coordinate conversions between WGS84 (latitude and longitude) and NZTM (New Zealand Transverse Mercator) coordinate systems.

This module provides functions for converting coordinates between WGS84 and NZTM coordinate systems.
See linz.govt.nz for a description of the NZTM coordinate system.

Functions:
- wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    Converts WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.
- nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    Converts NZTM coordinates (x, y, depth) to WGS84 coordinates.
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
        An array of shape (N, 3) containing WGS84 coordinates (latitude, longitude, depth).

    Returns
    -------
    np.ndarray
        An array of shape (N, 3) containing NZTM coordinates (x, y, depth).
    """
    nztm_coords = np.array(
        _WGS2NZTM.transform(wgs_depth_coordinates[:, 0], wgs_depth_coordinates[:, 1]),
    ).T
    return np.append(nztm_coords, wgs_depth_coordinates[:, 2].reshape((-1, 1)), axis=-1)


def nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert NZTM coordinates (x, y, depth) to WGS84 coordinates.

    Parameters
    ----------
    nztm_coordinates : np.ndarray
        An array of shape (N, 3) containing NZTM coordinates (x, y, depth).

    Returns
    -------
    np.ndarray
        An array of shape (N, 3) containing WGS84 coordinates (latitude, longitude, depth).
    """
    wgs_coords = np.array(
        _NZTM2WGS.transform(nztm_coordinates[:, 0], nztm_coordinates[:, 1]),
    ).T
    return np.append(wgs_coords, nztm_coordinates[:, 2].reshape((-1, 1)), axis=-1)

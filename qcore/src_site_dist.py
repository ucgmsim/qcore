"""
Originally written form IM_calculation, but is relocated and modified for qcore.
History of this file:
https://github.com/ucgmsim/IM_calculation/commits/afa9bf02d5e197300e3a91f87a9136b4ebcabd62/IM_calculation/source_site_dist/src_site_dist.py
"""

from typing import Literal, overload

import numpy as np

from qcore import geo


@overload
def calc_rrup_rjb(
    srf_points: np.ndarray,
    locations: np.ndarray,
    n_stations_per_iter: int = 1000,
    return_rrup_points: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # numpydoc ignore=GL08


@overload
def calc_rrup_rjb(
    srf_points: np.ndarray,
    locations: np.ndarray,
    n_stations_per_iter: int = 1000,
    return_rrup_points: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...  # numpydoc ignore=GL08


def calc_rrup_rjb(
    srf_points: np.ndarray,
    locations: np.ndarray,
    n_stations_per_iter: int = 1000,
    return_rrup_points: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates rrup and rjb distance

    Parameters
    ----------
    srf_points : np.ndarray
        The fault points from the srf file (qcore, srf.py, read_srf_points),
        format (lon, lat, depth).
    locations : np.ndarray
        The locations for which to calculate the distances,
        format (lon, lat, depth).
    n_stations_per_iter : int
        Number of stations to iterate over, default to 1000.
        Change based on memory requirements
    return_rrup_points : bool (optional) default False
        If True, returns the lon, lat, depth of the rrup points on the srf.

    Returns
    -------
    np.ndarray
        The rrup distance for the locations, shape/order same as locations.
    np.ndarray
        The rjb distance for the locations, shape/order same as locations.
    np.ndarray (optional)
        The lon, lat, depth of the rrup points, shape/order same as locations.
    """
    rrups = np.empty(locations.shape[0], dtype=np.float32)
    rjb = np.empty(locations.shape[0], dtype=np.float32)
    rrup_points = np.empty((locations.shape[0], 3), dtype=np.float32)

    srf_points = srf_points.astype(np.float32)
    locations = locations.astype(np.float32)

    # Split locations by a given limit for memory capacity
    split_locations = np.split(
        locations,
        np.arange(n_stations_per_iter, locations.shape[0], n_stations_per_iter),
    )

    for ix, cur_locations in enumerate(split_locations):
        h_dist = np.atleast_2d(
            geo.get_distances(srf_points, cur_locations[:, 0], cur_locations[:, 1])
        )

        v_dist = srf_points[:, 2, np.newaxis] - cur_locations[:, 2]

        d = np.sqrt(h_dist**2 + v_dist.T**2)

        start_ix = n_stations_per_iter * ix
        end_ix = start_ix + cur_locations.shape[0]
        rrups[start_ix:end_ix] = np.min(d, axis=1)
        rjb[start_ix:end_ix] = np.min(h_dist, axis=1)

        if return_rrup_points:
            rrup_points[start_ix:end_ix] = srf_points[np.argmin(d, axis=1)]

    if return_rrup_points:
        return rrups, rjb, rrup_points
    else:
        return rrups, rjb

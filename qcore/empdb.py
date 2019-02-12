#!/usr/bin/env python3
"""
empirical db access api

create db using empdb_create.py
"""

import h5py


def hazard_curve(empdb_file, station, im):
    """
    Load hazard curve data for a given station / IM.
    station: load hazard curve for this station name
    im: load hazard curve for this IM
    """

    with h5py.File(empdb_file, "r") as empdb:
        return empdb["hazard/{}/{}".format(station, im)][...]


def deagg_grid(empdb_file, station, im):
    """
    Load deagg data grid for a given station / IM.
    station: load deagg grid for this station name
    im: load deagg grid for this IM
    """

    with h5py.File(empdb_file, "r") as empdb:
        # TODO: labels for x, y, z axis
        return empdb["deagg/{}/{}".format(station, im)][...]

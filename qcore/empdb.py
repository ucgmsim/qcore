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

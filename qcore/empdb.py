#!/usr/bin/env python3
"""
empirical db access api

create db using empdb_create.py
"""

import h5py
import numpy as np


def hazard_curve(empdb_file, station, im):
    """
    Load hazard curve data for a given station / IM.
    station: load hazard curve for this station name
    im: load hazard curve for this IM
    """

    with h5py.File(empdb_file, "r") as empdb:
        return empdb["hazard/{}/{}".format(station, im)][...]


def deagg_grid(empdb_file, station, im, exceedance):
    """
    Load deagg data grid for a given station / IM.
    station: load deagg grid for this station name
    im: load deagg grid for this IM
    """

    with h5py.File(empdb_file, "r") as empdb:
        return (
            empdb["deagg/{}/{}".format(station, im)][exceedance],
            empdb.attrs["values_x"],
            empdb.attrs["values_y"],
            empdb.attrs["values_z"],
        )

def deagg_top(empdb_file, station, im, exceedance):
    """
    Return top contributing faults to hazard.
    """

    with h5py.File(empdb_file, "r") as empdb:
        index_value = empdb["deagg/{}/SUMM_{}".format(station, im)][exceedance]
        n = np.argmax(index_value["f1"] < 0)
        if n == 0:
            n = index_value.size
        faults = empdb["faults"][...][index_value['f0'][:n]].astype(np.unicode_)
        result = np.empty(n, dtype=[("fault", faults.dtype.descr[0][1]), ("contribution", np.float16)])
        result["fault"] = faults
        result["contribution"] =  index_value['f1'][:n]
        return result










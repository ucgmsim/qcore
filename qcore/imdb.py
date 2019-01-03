#!/usr/bin/env python3
"""
imdb access api

create db using imdb_create.py
"""

import h5py
import numpy as np
import pandas as pd


def ims(imdb_file, fmt="imdb"):
    """
    Returns list of IMs available in IMDB
    """

    with h5py.File(imdb_file, "r") as imdb:
        ims = imdb.attrs["ims"].tolist()

    if fmt == "file":
        fmt_file = (
            lambda im: im if not im.startswith("pSA") else im[1:].replace(".", "p")
        )
        return list(map(fmt_file, ims))

    return ims


def station_ims(imdb_file, station, im=None, fmt="imdb"):
    """
    Load IMs for a given station.
    station: load IMs for this station
    im: only give this IM
    """

    with h5py.File(imdb_file, "r") as imdb:
        df = pd.DataFrame(
            imdb["station_data/%s" % (station)][...],
            index=imdb["simulations"][...][imdb["station_index/%s" % (station)][...]],
            columns=ims(fmt=fmt),
        )

    if im is not None:
        if fmt == "file":
            return df["p%s" % (im.replace("p", "."))]
        return df[im]
    return df


def closest_station(imdb_file, lon, lat):
    """
    Find closest station.
    imdb_file: H5 database file
    lon: target longitude
    lat: target latitude
    returns: numpy.record with fields: id, name, lon, lat, dist
    """
    with h5py.File(imdb_file, "r") as imdb:
        stations = imdb["stations"][...]
    dtype = stations.dtype.descr
    dtype.append(("dist", "f4"))
    r = np.rec.array(np.zeros(stations.size, dtype=dtype))
    for k in dtype[:-1]:
        r[k[0]] = stations[k[0]]
    del stations

    d = (
        np.sin(np.radians(r.lat - lat) / 2.0) ** 2
        + np.cos(np.radians(lat))
        * np.cos(np.radians(r.lat))
        * np.sin(np.radians(r.lon - lon) / 2.0) ** 2
    )
    r.dist = 6378.139 * 2.0 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return r[np.argmin(r.dist)]


def station_details(imdb_file, station_name=None):
    """
    Give station details given name or id. Return all stations if no selection.
    """
    with h5py.File(imdb_file, "r") as imdb:
        # TODO: don't read all station data, just where "name" == station_name
        stations = np.rec.array(imdb["stations"][...])

    if station_name is not None:
        return stations[np.where(stations.name == station_name)[0][0]]
    return stations

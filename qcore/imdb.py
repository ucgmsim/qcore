"""
imdb access api

create db using imdb_create.py
"""

import h5py
import numpy as np
import pandas as pd


# workaround for python 3 use, TODO: clean up
stat_dtype = np.dtype([("name", "|U7"), ("lon", np.float32), ("lat", np.float32)])


def ims(imdb_file, fmt="imdb"):
    """
    Returns list of IMs available in IMDB
    """

    with h5py.File(imdb_file, "r") as imdb:
        ims = imdb.attrs["ims"].astype(np.unicode_).tolist()
    if fmt == "imdb":
        return ims

    if fmt == "file":
        fmt = lambda im: im if not im.startswith("pSA") else im[1:].replace(".", "p")
    elif fmt == "human":
        fmt = (
            lambda im: im
            if not im.startswith("pSA")
            else "pSA ({}s)".format(im.split("_")[1])
        )
    else:
        return

    return list(map(fmt, ims))


def simulations(imdb_file):
    """
    Return Simulation/Realisation names.
    """

    with h5py.File(imdb_file, "r") as imdb:
        return sorted(imdb["simulations"][...].astype(np.unicode_))


def simulation_station_ims(imdb_file, simulation, station, im=None, fmt="imdb"):
    """
    Load IMs for a given simulation, station combination.
    simulation: load IMs for this simulation name
    station: load IMs for this station name
    im: only give this IM
    """

    with h5py.File(imdb_file, "r") as imdb:
        try:
            sim_index = np.where(imdb["simulations"][...] == simulation.encode())[0][0]
            sim_stat_index = np.where(
                imdb["station_index/%s" % (station)][...] == sim_index
            )[0][0]
        except IndexError:
            # invalid simulation/station name combination
            return pd.Series()
        df = pd.Series(
            imdb["station_data/%s" % (station)][sim_stat_index].astype(np.unicode_),
            index=ims(imdb_file, fmt=fmt),
        )

    if im is not None:
        return df.loc[im]
    return df


def station_ims(imdb_file, station, im=None, fmt="imdb", rates_as_index=False):
    """
    Load IMs for a given station.
    station: load IMs for this station
    im: only give this IM
    rates_as_index: index will be annualised rupture rate rather than sim name
    """

    with h5py.File(imdb_file, "r") as imdb:
        station_index = imdb["station_index/%s" % (station)][...]
        if rates_as_index:
            index = imdb["simulations_arr"][...][station_index]
        else:
            index = imdb["simulations"][...][station_index].astype(np.unicode_)

        if im is None:
            return pd.DataFrame(
                imdb["station_data/%s" % (station)][...],
                index=index,
                columns=ims(imdb_file, fmt=fmt),
            )
        else:
            return pd.Series(
                imdb["station_data/%s" % (station)][
                    :, ims(imdb_file, fmt=fmt).index(im)
                ],
                index=index,
            )


def station_simulations(imdb_file, station, sims=True, rates=False):
    """
    List of simulation names and/or rupture rates for given station.
    station: retrieve simulations/rupture rates for this station
    sims: retrieve simulation names
    rates: retrieve rupture rates
    """
    try:
        assert sims or rates
    except AssertionError:
        return
    with h5py.File(imdb_file, "r") as imdb:
        station_index = imdb["station_index/%s" % (station)][...]
        if sims:
            sims = imdb["simulations"][...][station_index].astype(np.unicode_)
            if not rates:
                return sims
        if rates:
            rates = imdb["simulations_arr"][...][station_index]
            # testing for bool would fail for sims=numpy.bool_(False)
            if not type(sims) == np.ndarray:
                return rates
        return sims, rates


def closest_station(imdb_file, lon, lat, event=None):
    """
    Find closest station.
    imdb_file: H5 database file
    lon: target longitude
    lat: target latitude
    returns: numpy.record with fields: id, name, lon, lat, dist
    """
    with h5py.File(imdb_file, "r") as imdb:
        stations = imdb["stations"][...].astype(stat_dtype)
        if event is not None:
            stations = stations[imdb["sim_stats/{}".format(event)][...]]
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
        stations = np.rec.array(imdb["stations"][...].astype(stat_dtype))

    if station_name is not None:
        return stations[np.where(stations.name == station_name)[0][0]]
    return stations


def faults(imdb_file):
    """
    Returns list of fault names in imdb.
    requires imdb V6
    """
    with h5py.File(imdb_file, "r") as imdb:
        return list(imdb["fault_traces"].attrs.keys())


def fault_traces(imdb_file, named=False, fault_name=None):
    """
    Give a list of fault traces (2D array of lon, lat).
    Or give a single fault trace given fault name.
    requires imdb V6
    """
    with h5py.File(imdb_file, "r") as imdb:
        traces = imdb["fault_traces"].attrs
        if fault_name is None:
            if named:
                return list(traces.items())
            return list(traces.values())
        return traces[fault_name]

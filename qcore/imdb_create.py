#!/usr/bin/env python3
"""
Creates IMDB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os

import h5py
from mpi4py import MPI
import numpy as np
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
master = 0
is_master = not rank


def __csv_values(csv, dtype):
    """
    Loads individual simulation CSV file.
    """
    c = pd.read_csv(csv, index_col=0, engine="c", dtype=dtype)
    return c.loc[c["component"] == "geom"].drop("component", axis="columns")


def __csv_ims(csv):
    """
    Minimal load function that returns list of IMs.
    """
    with open(csv, "r") as c:
        return list(map(str.strip, c.readline().split(",")[2:]))


def __csv_stations(csv):
    """
    Minimal load function that returns stations available in CSV file.
    """
    c = pd.read_csv(
        csv,
        index_col=0,
        usecols=[0, 1],
        dtype={"station": np.string_, "component": np.string_},
        engine="c",
    )
    return c.loc[c["component"] == "geom"].index.values


def __add_dict_counters(dict_a, dict_b, datatype):
    """
    MPI function to sum a dictionary of counters.
    """
    for station in dict_b:
        try:
            dict_a[station] += dict_b[station]
        except KeyError:
            dict_a[station] = dict_b[station]
    return dict_a


NSIMSUM = MPI.Op.Create(__add_dict_counters, commute=True)


# collect required arguements
args = None
csvs = None
if is_master:
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("runs_dir", help="Location of Runs folder")
    arg("station_file", help="Location of station (ll) file")
    arg("db_file", help="Where to store IMDB")
    try:
        args = parser.parse_args()
    except SystemExit:
        # invalid arguments or -h
        comm.Abort()
    # add some additional info
    # TODO: sort by file size, with many realisations it will be ok
    csvs = glob(os.path.join(args.runs_dir, "*", "IM_calc", "*", "*.csv"))
args = comm.bcast(args, root=master)
csvs = comm.bcast(csvs, root=master)
n_csvs = len(csvs)
rank_csvs = csvs[rank::size]
del csvs
sims = np.array(list(map(lambda path: os.path.splitext(os.path.basename(path))[0], rank_csvs)), dtype=np.string_)
# TODO: also store fault names, needs to work with Python 2 and 3 (str vs unicode)
#faults = np.array(list(map(lambda sim: sim.split("_HYP")[0], sims)), dtype=np.string_)
sims_dtype = '|S%d' % (comm.allreduce(sims.dtype.itemsize, op=MPI.MAX))
if n_csvs <= 65536:
    simsi_dtype = np.uint16
else:
    simsi_dtype = np.uint32

# create imdb for potentially a subset of all stations in IM CSVs
# station list needed in all cases as location is extracted from here
station_ll = {}
with open(args.station_file, "r") as sf:
    for line in sf:
        lon, lat, name = line.split()
        station_ll[name] = (float(lon), float(lat))
# TODO: use numpy to load the station file with this dtype, change below accordingly
station_dtype = np.dtype([('name', '|S7'), ('lon', 'f4'), ('lat', 'f4')])

###
### PASS 1 : determine work size
###

if is_master:
    t0 = MPI.Wtime()

# determine number of simulations for each station
stat_nsim = {}
for csv in rank_csvs:
    for stat in __csv_stations(csv):
        if stat not in station_ll:
            continue
        try:
            stat_nsim[stat][rank] += 1
        except KeyError:
            stat_nsim[stat] = np.zeros(size)
            stat_nsim[stat][rank] = 1
# get a complete set of simulations for each station by rank
stat_nsim = comm.allreduce(stat_nsim, op=NSIMSUM)

# determine IMs available
ims = None
if is_master:
    # make sure IMs are the same, in the same order
    ims = __csv_ims(rank_csvs[0])
ims = comm.bcast(ims, root=master)
n_im = len(ims)
csv_dtype = {"station": np.string_, "component": np.string_}
for im in ims:
    csv_dtype[im] = np.float32

if is_master:
    print("gathering metadata complete (%.2fs)" % (MPI.Wtime() - t0))


###
### PASS 2 : store data
###

if is_master:
    t0 = MPI.Wtime()

# create structure together
h5 = h5py.File(args.db_file, "w", driver="mpio", comm=comm)

# ims as columns
h5.attrs["ims"] = np.array(ims, dtype=np.string_)

# stations reference
h5_ll = h5.create_dataset("stations", (len(stat_nsim),), dtype=station_dtype)
for i, stat in enumerate(list(stat_nsim.keys())[rank::size]):
    h5_ll[rank + i * size] = (stat, station_ll[stat][0], station_ll[stat][1])
del h5_ll

# simulations reference
h5_sims = h5.create_dataset("simulations", (n_csvs,), dtype=sims_dtype)
for i in range(len(sims)):
    h5_sims[rank + i * size] = sims[i]
del h5_sims

# per station IM and simulation datasets
h5_stats = {}
h5_sims = {}
for stat in stat_nsim:
    h5_stats[stat] = h5.create_dataset(
        "station_data/%s" % (stat), (sum(stat_nsim[stat]), n_im), dtype='f4',
    )
    h5_sims[stat] = h5.create_dataset(
        "station_index/%s" % (stat), (sum(stat_nsim[stat]),), dtype=simsi_dtype,
    )

if is_master:
    print("hdf datastructures created (%.2fs)" % (MPI.Wtime() - t0))

# TODO: reduce stat_nsim to contain relevant number for current rank from this point
# store IM and simulation name values
t0 = MPI.Wtime()
for i, csv in enumerate(rank_csvs):
    if (i + 1) % 5 == 0:
        print("[%03d] %03d / %03d" % (rank, i + 1, len(rank_csvs)))
    df = __csv_values(csv, csv_dtype)
    # all CSVs have to have same IM order, expected, cheaper than explicit order
    assert np.array_equal(df.columns.values, ims)
    for stat in df.index.values:
        if stat not in station_ll:
            continue
        stat_nsim[stat][rank] -= 1
        # slowest part for optimisation (too much indexing)
        # TODO: prepare rank sets and save them when complete or mem limit reached?
        h5_stats[stat][sum(stat_nsim[stat][: rank + 1])] = df.loc[stat].values
        h5_sims[stat][sum(stat_nsim[stat][: rank + 1])] = rank + i * size
        if stat_nsim[stat][rank] == 0:
            del h5_stats[stat], h5_sims[stat]
print("[%03d] %03d in %.2fs" % (rank, len(rank_csvs), MPI.Wtime() - t0))


# will hang for a long time if datasets are not deleted
del h5_stats, h5_sims
h5.close()

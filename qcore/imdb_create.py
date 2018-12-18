#!/usr/bin/env python3
"""
Creates IMDB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os
import sqlite3

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
rank_csvs = csvs[rank::size]
del csvs
sims = list(map(lambda path: os.path.splitext(os.path.basename(path))[0], rank_csvs))
faults = list(map(lambda sim: sim.split("_HYP")[0], sims))


###
### PASS 1 : determine work size
###

if is_master:
    t0 = MPI.Wtime()

# determine number of simulations for each station
stat_nsim = {}
for csv in rank_csvs:
    for stat in __csv_stations(csv):
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
h5_stats = {}
for stat in stat_nsim:
    h5_stats[stat] = h5.create_dataset(
        "station_data/%s" % (stat), (sum(stat_nsim[stat]), n_im),
    )
if is_master:
    print("hdf datastructures created (%.2fs)" % (MPI.Wtime() - t0))

# store im values
t0 = MPI.Wtime()
for i, csv in enumerate(rank_csvs):
    if (i + 1) % 5 == 0:
        print("[%03d] %03d / %03d" % (rank, i + 1, len(rank_csvs)))
    df = __csv_values(csv, csv_dtype)
    # all CSVs have to have same IM order, expected, cheaper than explicit order
    assert np.array_equal(df.columns.values, ims)
    for stat in df.index.values:
        stat_nsim[stat][rank] -= 1
        #h5_stats[stat][sum(stat_nsim[stat][: rank + 1])] = df.loc[stat].values
print("[%03d] %03d in %.2fs" % (rank, len(rank_csvs), MPI.Wtime() - t0))


# will hang for a long time if datasets are not deleted
del h5_stats
h5.close()

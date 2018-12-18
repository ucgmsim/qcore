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


def __csv_ims(csv):
    """
    Loads individual simulation CSV files.
    """
    c = pd.read_csv(csv, index_col=0)
    return c.loc[c["component"] == "geom"].drop("component", axis="columns")


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


statSumOp = MPI.Op.Create(__add_dict_counters, commute=True)


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
    csvs = glob(os.path.join(args.runs_dir, "*", "IM_calc", "*", "*.csv"))
args = comm.bcast(args, root=master)
csvs = comm.bcast(csvs, root=master)
rank_csvs = csvs[rank::size]
sims = list(map(lambda path: os.path.splitext(os.path.basename(path))[0], csvs))
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
stat_nsim = comm.allreduce(stat_nsim, op=statSumOp)

# other metadata
ims = __csv_ims(csvs[0]).columns.values
n_im = len(ims)

if is_master:
    print("gathering metadata complete (%.2fs)" % (MPI.Wtime() - t0))


###
### PASS 2 : store data
###

if is_master:
    t0 = MPI.Wtime()

# create structure together
h5 = h5py.File(args.db_file, "w", driver="mpio", comm=comm)
ims = {}
for stat in stat_nsim:
    ims[stat] = h5.create_dataset(
        "station_data/%s" % (stat), (sum(stat_nsim[stat]), n_im), dtype="f4"
    )
if is_master:
    print("hdf datastructures created (%.2fs)" % (MPI.Wtime() - t0))

# store im values
for i, csv in enumerate(rank_csvs):
    print("[%03d] %03d / %03d..." % (rank, i, len(rank_csvs)))
    df = __csv_ims(csv)
    for stat in df.index.values:
        stat_nsim[stat][rank] -= 1
        ims[stat][sum(stat_nsim[stat][:rank + 1])] = df.loc[stat].values


# will hang for a long time if datasets are not deleted
del ims
h5.close()

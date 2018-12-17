#!/usr/bin/env python3
"""
Creates IMDB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os
import sqlite3

from mpi4py import MPI
import numpy as np
import pandas as pd


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
master = 0
is_master = not rank

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
sims = list(map(lambda path: os.path.splitext(os.path.basename(path))[0], csvs))
faults = list(map(lambda sim: sim.split("_HYP")[0], sims))

###
### PASS 1 : determine work size
###
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


# determine number of simulations for each station
stat_nsim = {}
for csv in csvs[rank::size]:
    for stat in __csv_stations(csv):
        try:
            stat_nsim[stat] += 1
        except KeyError:
            stat_nsim[stat] = 1
statSumOp = MPI.Op.Create(__add_dict_counters, commute=True)
stat_nsim = comm.allreduce(stat_nsim, op=statSumOp)

###
### PASS 2 : store data
###
h5 = h5py.File(args.db_file, "w", driver="mpio", comm=comm)

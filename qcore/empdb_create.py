#!/usr/bin/env python3
"""
Creates Empirical DB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os

import h5py
from mpi4py import MPI
import numpy as np

from qcore.geo import closest_location
from qcore import imdb

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
master = 0
is_master = not rank

# collect required arguements
args = None
if is_master:
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("imdb", help="Location of imdb.h5")
    arg("emp_src", help="Location of empiricals dir containing IM subfolders")
    arg("empdb", help="Where to store Empirical DB empdb.h5")
    arg("--n-hazard", help="Number of points making up hazard curve.", type=int, default=100)
    arg("--mag-d", help="magnitude spacing", type=float, default=0.5)
    arg("--mag-min", help="magnitude minimum", type=float, default=5.0)
    arg("--mag-n", help="magnitude blocks from minimum at spacing", type=int, default=8)
    arg("--rrup-d", help="rrup spacing", type=float, default=10.0)
    arg("--rrup-n", help="rrup blocks at spacing", type=int, default=20)
    try:
        args = parser.parse_args()
    except SystemExit:
        # invalid arguments or -h
        comm.Abort()
args = comm.bcast(args, root=master)

emp_ims = [i for i in os.listdir(args.emp_src) if os.path.isdir(os.path.join(args.emp_src, i))]
#emp_ims = ["PGA"]

for im in emp_ims:
    files = glob(
        os.path.join(args.emp_src, im, "EmpiricalPsha_Lat*")
    )

    extract_ll = lambda path: [
        float(ll[3:].replace("p", ".")) for ll in os.path.basename(path).split("_")[1:3]
    ]

    locations = np.array(list(map(extract_ll, files)))

    for station in imdb.station_details(args.imdb):
        i, d = closest_location(locations, station.lat, station.lon)
        if d > 0.1:
            print("WARNING: missing station:", station.name, im)
        #mag_rrup_mean_dev_prob = np.loadtxt(files[i], skiprows=1, usecols=(1, 2, 5, 6, 7), dtype=np.float32)
        #print(mag_rrup_mean_dev_prob)
        #break

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

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
MASTER = 0
IS_MASTER = not RANK

N_TYPES = 3
N_SERIES = 1 + N_TYPES

def extract_ll(path):
    return [float(ll[3:].replace("p", ".")) for ll in os.path.basename(path).split("_")[1:3]]

def emp_data(emp_file):
    # read through file only once
    emp = np.rec.array(
        np.loadtxt(
            emp_file,
            skiprows=1,
            usecols=(0, 1, 2, 5, 6, 7),
            dtype=[
                ("fault", "|U18"),
                ("mag", np.float32),
                ("rrup", np.float32),
                ("med", np.float32),
                ("dev", np.float32),
                ("prob", np.float32),
            ],
        )
    )

    # all faults except for distributed seismicity has incorrect probabilities
    c = np.sort(np.unique(emp.fault, return_index=True)[1])
    for i in range(c.size):
        # startswith will break if code ran with python2 (wanted behaviour)
        if emp.fault[c[i]].startswith("PointEqkSource"):
            # distributed seismicity
            continue
        try:
            block = emp.prob[c[i] : c[i + 1]]
        except IndexError:
            block = emp.prob[c[i] :]
        # prevent new input rules being undetected
        assert sum(block == 0) == block.size - 1
        block[...] = max(block) / block.size
    return emp

from time import time
def process_emp_file(emp_file, station, im):
    t0 = time()
    emp = emp_data(emp_file)
    types = np.zeros(emp.size, dtype=np.uint8) + 2
    types_a = np.vectorize(imdb_faults.__contains__)(emp.fault)
    types -= types_a + 2 * (np.inv(types_a))
    print(types)
    print("%.2fs" % (time() - t0))
    return

###
### STEP 0: prepare/validate inputs
###

args = None
if IS_MASTER:
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("imdb", help="Location of imdb.h5")
    arg("emp_src", help="Location of empiricals dir containing IM subfolders")
    arg("empdb", help="Where to store Empirical DB empdb.h5")
    arg(
        "--hazard-n",
        help="Number of points making up hazard curve.",
        type=int,
        default=100,
    )
    arg("--mag-d", help="magnitude spacing", type=float, default=0.5)
    arg("--mag-min", help="magnitude minimum", type=float, default=5.0)
    arg("--mag-n", help="magnitude blocks from minimum at spacing", type=int, default=8)
    arg("--rrup-d", help="rrup spacing", type=float, default=10.0)
    arg("--rrup-n", help="rrup blocks at spacing", type=int, default=20)
    try:
        args = parser.parse_args()
    except SystemExit:
        # invalid arguments or -h
        COMM.Abort()
args = COMM.bcast(args, root=MASTER)

emp_ims = [
    i for i in os.listdir(args.emp_src) if os.path.isdir(os.path.join(args.emp_src, i))
]
emp_ims = ["PGA"]
# imdb_ims =
imdb_stations = imdb.station_details(args.imdb)
imdb_faults = set(map(lambda sim: sim.split("_HYP")[0], imdb.simulations(args.imdb)))


###
### STEP 1: create hdf structures (groups, datasets, attributes)
###

h5 = h5py.File(args.empdb, "w", driver="mpio", comm=COMM)

# per station IM and simulation datasets
h5_hazard = []
h5_deagg = []
for stat in imdb_stations.name:
    stat_hazards = []
    stat_deaggs = []
    for im in emp_ims:
        stat_hazards.append(
            h5.create_dataset(
                "hazard/{}/{}".format(stat, im), (N_SERIES, args.hazard_n), dtype="f4"
            )
        )
        stat_deaggs.append(
            h5.create_dataset(
                "deagg/{}/{}".format(stat, im),
                (args.rrup_n, args.mag_n, N_TYPES),
                dtype="f4",
            )
        )
    h5_hazard.append(stat_hazards)
    h5_deagg.append(stat_deaggs)
del stat_hazards, stat_deaggs

if IS_MASTER:
    print("HDF datastructures created.", h5_deagg.__sizeof__())


###
### STEP 2: distribute work to fill datasets
###

if IS_MASTER:
    status = MPI.Status()
    for i in range(len(emp_ims)):
        files = glob(os.path.join(args.emp_src, emp_ims[i], "EmpiricalPsha_Lat*"))
        locations = np.array(list(map(extract_ll, files)))
        for j in range(imdb_stations.size):
            # match station to file
            f, d = closest_location(locations, imdb_stations[j].lat, imdb_stations[j].lon)
            if d > 0.1:
                print("WARNING: missing station:", imdb_stations[j].name, emp_ims[i])
                continue
            # previous job
            value = COMM.recv(source=MPI.ANY_SOURCE, status=status)
            # next job
            COMM.send(obj=(files[f], j, i), dest=status.Get_source())

    # end slave loops
    for _ in range(SIZE - 1):
        value = COMM.recv(source=MPI.ANY_SOURCE, status=status)
        COMM.send(obj=StopIteration, dest=status.Get_source())
else:
    # ask for work loop
    value = None
    for task in iter(lambda: COMM.sendrecv(value, dest=MASTER), StopIteration):
        value = process_emp_file(task[0], task[1], task[2])

del h5_hazard, h5_deagg
h5.close()

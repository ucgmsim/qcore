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
from scipy.stats import norm

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
    return [
        float(ll[3:].replace("p", ".")) for ll in os.path.basename(path).split("_")[1:3]
    ]


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
    types = np.zeros(emp.size, dtype=np.uint8)
    types += np.invert(np.vectorize(imdb_faults.__contains__)(emp.fault)).astype(
        np.uint8
    )

    # all faults except for distributed seismicity has incorrect probabilities
    c = np.sort(np.unique(emp.fault, return_index=True)[1])
    for i in range(c.size):
        # startswith will break if code ran with python2 (wanted behaviour)
        if emp.fault[c[i]].startswith("PointEqkSource"):
            # distributed seismicity
            try:
                types[c[i] : c[i + 1]] = 2
            except IndexError:
                types[c[i] :] = 2
            continue
        try:
            block = emp.prob[c[i] : c[i + 1]]
        except IndexError:
            block = emp.prob[c[i] :]
        # prevent new input rules being undetected
        assert sum(block == 0) == block.size - 1
        block[...] = max(block) / block.size

    return emp, types


def process_emp_file(emp_file, station_i, im_i):
    try:
        emp, types = emp_data(emp_file)
    except ValueError:
        print("corrupt file", emp_file)
        return
    mn = np.argmin(emp.med)
    mx = np.argmax(emp.med)
    mn = np.e ** (emp[mn].med - 3 * emp[mn].dev)
    mx = np.e ** (emp[mx].med + 3 * emp[mx].dev)
    hazard = h5_hazard[station_i][im_i]
    hazard[0] = np.logspace(np.log10(mn), np.log10(mx), num=args.hazard_n)
    for t in range(3):
        # would probably be faster to have split arrays per type
        hazard[t + 1] = [
            np.dot(
                norm.sf(np.log(i), emp.med[types == t], emp.dev[types == t]),
                emp.prob[types == t],
            )
            for i in hazard[0]
        ]


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

h5.attrs["ims"] = np.array(emp_ims, dtype=np.string_)

# stations reference
station_dtype = np.dtype([("name", "|S7"), ("lon", "f4"), ("lat", "f4")])
h5_ll = h5.create_dataset("stations", (imdb_stations.size,), dtype=station_dtype)
for i, stat in enumerate(imdb_stations[RANK::SIZE]):
    # TODO: use location from actual empirical file
    h5_ll[RANK + i * SIZE] = (stat.name.astype(np.string_), stat.lon, stat.lat)
del h5_ll

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
    print("HDF datastructures created.")


###
### STEP 2: distribute work to fill datasets
###
for i in range(len(emp_ims)):
    files = glob(os.path.join(args.emp_src, emp_ims[i], "EmpiricalPsha_Lat*"))
    locations = np.array(list(map(extract_ll, files)))
    for j in range(RANK, imdb_stations.size, SIZE):
        # match station to file
        f, d = closest_location(locations, imdb_stations[j].lat, imdb_stations[j].lon)
        if d > 0.1:
            print("WARNING: missing station:", imdb_stations[j].name, emp_ims[i])
            continue
        # next job
        process_emp_file(files[f], j, i)
del h5_hazard, h5_deagg
h5.close()

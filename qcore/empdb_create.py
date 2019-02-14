#!/usr/bin/env python3
"""
Creates Empirical DB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os
from time import time

import h5py
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.stats import norm

from qcore.geo import closest_location

# from qcore import imdb
import sys

sys.path.append("/home/vap30/ucgmsim/qcore/qcore")
import imdb


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
MASTER = 0
IS_MASTER = not RANK

# A cs, A emp, B emp, DS emp
N_TYPES = 4
# X, A emp, B emp, DS emp
N_SERIES = 1 + 3


def extract_ll(path):
    return [
        float(ll[3:].replace("p", ".")) for ll in os.path.basename(path).split("_")[1:3]
    ]


def emp_data(emp_file):
    # read through file only once
    emp = pd.read_csv(
        emp_file,
        sep="\t",
        names=("fault", "mag", "rrup", "med", "dev", "prob"),
        usecols=(0, 1, 2, 5, 6, 7),
        dtype={
            "fault": object,
            "mag": np.float32,
            "rrup": np.float32,
            "med": np.float32,
            "dev": np.float32,
            "prob": np.float32,
        },
        engine="c",
        skiprows=1,
    )
    emp["type"] = pd.Series(0, index=emp.index, dtype=np.uint8)
    # type B faults
    emp.type += np.invert(np.vectorize(imdb_faults.__contains__)(emp.fault))
    # distributed seismicity
    emp.loc[emp.fault=="PointEqkSource", "type"] = 2

    # assume magnitude correct where prob is given
    mag = {}
    rrup = {}
    # all faults except for distributed seismicity has incorrect probabilities
    # pandas painfully slow at indexing, use numpy here
    f, fl = np.unique(emp.fault, return_index=True)
    x = np.argsort(fl)
    f = f[x]
    fl = fl[x]
    for i in range(fl.size - 1):
        rows = emp.iloc[fl[i]:fl[i+1]]
        prob = rows.prob.values
        # prevent new input rules being undetected
        assert np.sum(prob != 0) == 1
        # fault properties
        mag[f[i]] = rows.mag[np.argmax(prob) + fl[i]]
        rrup[f[i]] = rows.rrup[fl[i]]
        # because pandas is incapabale of storing a view
        emp.iloc[fl[i]:fl[i+1], 5] = np.average(prob)

    return emp, mag, rrup


def process_emp_file(args, emp_file, station, im):
    t0 = time()
    try:
        emp, mags_d, rrups_d = emp_data(emp_file)
    except ValueError:
        print("corrupt file", emp_file)
        return
    mn = emp.loc[emp.med.idxmin()]
    mx = emp.loc[emp.med.idxmax()]
    mn = np.e ** (mn.med - 3 * mn.dev)
    mx = np.e ** (mx.med + 3 * mx.dev)
    hazard = h5["hazard/{}/{}".format(station, im)]
    hazard[0] = np.logspace(np.log10(mn), np.log10(mx), num=args.hazard_n)
    for t in range(3):
        rows = emp.loc[emp.type==t]
        hazard[t + 1] = [
            np.dot(norm.sf(np.log(i), rows.med, rows.dev), rows.prob)
            for i in hazard[0]
        ]
    # close pointer, use local RO copy
    hazard = hazard[...]
    t1 = time()

    # deaggregation
    bins_rrup = (np.arange(args.rrup_n, dtype=np.float32) + 1) * args.rrup_d
    bins_mag = (np.arange(args.mag_n + 1, dtype=np.float32) * args.mag_d) + args.mag_min
    e = 1 / 500.0
    block = h5["deagg/{}/{}".format(station, im)]
    # TODO: exceedance_rate: type A (hazard[1]) replaced with cybershake, exceedance_empirical: type A from empirical
    im_level = np.exp(
        np.interp(
            np.log(e) * -1, np.log(np.sum(hazard[1:], axis=0)) * -1, np.log(hazard[0])
        )
    )
    # pandas is slow
    rrups = emp.rrup.values
    mags = emp.mag.values
    types = emp.type.values
    # survival function (1 - cdf)
    sf = norm.sf(np.log(im_level), emp.med.values, emp.dev.values) * emp.prob.values
    r = np.digitize(rrups, bins_rrup)
    m = np.digitize(mags, bins_mag) - 1
    i = (r < bins_rrup.size) & (m < bins_mag.size) & (m != -1)
    u = np.unique(np.dstack((r[i], m[i], types[i]))[0], axis=0)
    for x, y, z in u:
        block[x, y, z + 1] = sum(sf[(r == x) & (m == y) & (types == z)])
    t2 = time()

    # deagg - cybershake
    ims = imdb.station_ims(args.imdb, station, im=im, fmt="file")
    rates = imdb.station_simulations(args.imdb, station, sims=False, rates=True)
    faults = list(map(lambda sim: sim.split("_HYP")[0], ims.index.values))
    for i, sim in enumerate(ims.index.values):
        if ims[i] < e:
            # below exceedance, not contributing
            continue
        # check if out of range
        try:
            r = np.digitize([rrups_d[i]], bins_rrup)[0]
            m = np.digitize([mags_d[faults[i]]], bins_mag)[0]
        except KeyError:
            continue
        if m == 0:
            continue
        try:
            block[r, m-1, 0] += rates[i]
        except ValueError:
            continue

    print("%.2fs %.2fs %.2fs" % (t1 - t0, t2 - t1, time() - t2))


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
    arg("--mag-d", help="magnitude spacing", type=float, default=0.25)
    arg("--mag-min", help="magnitude minimum", type=float, default=5.0)
    arg(
        "--mag-n", help="magnitude blocks from minimum at spacing", type=int, default=16
    )
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
h5.attrs["values_x"] = (
    np.arange(args.rrup_n, dtype=np.float32) * args.rrup_d + args.rrup_d / 2.0
)
h5.attrs["values_y"] = (
    np.arange(args.mag_n, dtype=np.float32) * args.mag_d
    + args.mag_min
    + args.mag_d / 2.0
)
h5.attrs["values_z"] = np.array(["A", "B", "DS"], dtype=np.string_)

# stations reference
station_dtype = np.dtype([("name", "|S7"), ("lon", "f4"), ("lat", "f4")])
h5_ll = h5.create_dataset("stations", (imdb_stations.size,), dtype=station_dtype)
for i, stat in enumerate(imdb_stations[RANK::SIZE]):
    # TODO: use location from actual empirical file
    h5_ll[RANK + i * SIZE] = (stat.name.astype(np.string_), stat.lon, stat.lat)
del h5_ll

# per station IM and simulation datasets
for stat in imdb_stations.name:
    for im in emp_ims:
        h5.create_dataset(
            "hazard/{}/{}".format(stat, im), (N_SERIES, args.hazard_n), dtype="f4"
        )
        h5.create_dataset(
            "deagg/{}/{}".format(stat, im),
            (args.rrup_n, args.mag_n, N_TYPES),
            dtype="f4",
        )

if IS_MASTER:
    print("HDF datastructures created.")


###
### STEP 2: distribute work to fill datasets
###
for i in range(len(emp_ims)):
    emp_files = glob(os.path.join(args.emp_src, emp_ims[i], "EmpiricalPsha_Lat*"))
    locations = np.array(list(map(extract_ll, emp_files)))
    for j in range(RANK, imdb_stations.size, SIZE):
        print(j, "/", imdb_stations.size)
        # match station to file
        f, d = closest_location(locations, imdb_stations[j].lat, imdb_stations[j].lon)
        if d > 0.1:
            print("WARNING: missing station:", imdb_stations[j].name, emp_ims[i])
            continue
        # next job
        process_emp_file(args, emp_files[f], imdb_stations[j].name, emp_ims[i])
h5.close()

#!/usr/bin/env python3
"""
Creates Empirical DB (using MPI).
Usage: Run with mpirun/similar. See help (run with -h).
"""

from argparse import ArgumentParser
from glob import glob
import os
from subprocess import call
from time import time

import h5py
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.stats import norm

from qcore.geo import closest_location

from qcore import imdb


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
MASTER = 0
IS_MASTER = not RANK

# A (cs), B (emp), DS (emp), e 0->7 (cs)
N_TYPES = 11
# X, A emp, B emp, DS emp
N_SERIES = 1 + 3


def fault_names(nhm_file):
    faults = ["PointEqkSource"]
    with open(nhm_file, "r") as nf:
        db = nf.readlines()
        dbi = 15
        dbl = len(db)
    while dbi < dbl:
        faults.append(db[dbi].strip())
        dbi += 13 + int(db[dbi + 11])

    return np.sort(faults)


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
    emp.loc[emp.fault == "PointEqkSource", "type"] = 2

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
        rows = emp.iloc[fl[i] : fl[i + 1]]
        prob = rows.prob.values
        # prevent new input rules being undetected
        assert np.sum(prob != 0) == 1
        # fault properties
        mag[f[i]] = rows.mag[np.argmax(prob) + fl[i]]
        rrup[f[i]] = rows.rrup[fl[i]]
        # because pandas is incapabale of storing a view
        emp.iloc[fl[i] : fl[i + 1], 5] = np.average(prob)

    return emp, mag, rrup


def process_emp_file(args, all_faults, emp_file, station, im):
    try:
        emp, mags_d, rrups_d = emp_data(emp_file)
    except ValueError:
        print("corrupt file", emp_file)
        return
    mx = emp.loc[emp.med.idxmax()]
    if im == "PGV":
        mn = 1.0
    else:
        mn = 0.01
    mx = np.e ** (mx.med + 3 * mx.dev)
    hazard = h5["hazard/{}/{}".format(station, im)]
    hazard[0] = np.logspace(
        np.log10(mn), np.log10(mx), num=args.hazard_n, dtype=np.float16
    )
    for t in range(3):
        rows = emp.loc[emp.type == t]
        hazard[t + 1] = [
            np.dot(norm.sf(np.log(i), rows.med, rows.dev), rows.prob) for i in hazard[0]
        ]
    # close pointer, use local RO copy
    hazard = hazard[...]

    # deaggregation
    bins_rrup = (np.arange(args.rrup_n, dtype=np.float32) + 1) * args.rrup_d
    bins_mag = (np.arange(args.mag_n + 1, dtype=np.float32) * args.mag_d) + args.mag_min
    bins_epsilon = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    # only interested in data within deagg blocks and not empirical type A
    r = np.digitize(emp.rrup.values, bins_rrup)
    m = np.digitize(emp.mag.values, bins_mag) - 1
    vi = (r < bins_rrup.size) & (m < bins_mag.size) & (m != -1) & (emp.type.values != 0)
    emp = emp[vi]
    # pandas is slow (fast for reading csv)
    rrups = emp.rrup.values
    mags = emp.mag.values
    types = emp.type.values
    meds = emp.med.values
    devs = emp.dev.values
    # for summing empirical accross faults
    emp_fault_u_bins, emp_fault_u = pd.factorize(emp.fault.values)
    # deagg blocks
    r = np.digitize(rrups, bins_rrup)
    m = np.digitize(mags, bins_mag) - 1
    u = np.unique(np.dstack((r, m, types))[0], axis=0)
    # cybershake details
    ims = imdb.station_ims(args.imdb, station, im=im, fmt="file")
    rates = imdb.station_simulations(args.imdb, station, sims=False, rates=True)
    faults = np.array(list(map(lambda sim: sim.split("_HYP")[0], ims.index.values)))
    # results storage
    block = h5["deagg/{}/{}".format(station, im)]
    for b, e in enumerate(
        map(lambda tp: -1.0 / tp[0] * np.log(1 - tp[1]), args.deagg_e)
    ):
        # store max contributors to epsilon and type charts
        summ_contrib = {}
        # exceedance -> im
        im_level = np.exp(
            np.interp(
                np.log(e) * -1,
                np.log(np.sum(hazard[1:], axis=0)) * -1,
                np.log(hazard[0]),
            )
        )
        # survival function (1 - cdf)
        sf = norm.sf(np.log(im_level), meds, devs) * emp.prob.values
        # fault based contribution
        for i, contrib in enumerate(np.bincount(emp_fault_u_bins, weights=sf)):
            summ_contrib[emp_fault_u[i]] = contrib
        # type based contribution
        for x, y, z in u:
            block[b, x, y, z] = sum(sf[(r == x) & (m == y) & (types == z)])
        # epsilon based contribution
        epsilon = np.digitize((im_level - meds) / devs, bins_epsilon)
        ue = np.unique(np.dstack((r, m, epsilon))[0], axis=0)
        for x, y, z in ue:
            block[b, x, y, z + 3] = sum(sf[(r == x) & (m == y) & (epsilon == z)])

        # sums and totals for each fault
        s_im = {}
        s_rate = {}
        # deagg - cybershake
        for i, sim in enumerate(ims.index.values):
            if ims[i] >= e:
                # below exceedance, not contributing
                continue
            try:
                s_im[faults[i]] += 1
            except TypeError:
                # out of range from previous message (None)
                continue
            except KeyError:
                s_im[faults[i]] = 1
            # check if out of range
            try:
                cs_r = np.digitize([rrups_d[faults[i]]], bins_rrup)[0]
                cs_m = np.digitize([mags_d[faults[i]]], bins_mag)[0]
            except KeyError:
                # fault not found in empirical file
                s_im[faults[i]] = None
                continue
            if cs_m == 0 or cs_r == bins_rrup.size or cs_m == bins_mag.size:
                # outside range being displayed
                s_im[faults[i]] = None
                continue
            try:
                block[b, cs_r, cs_m - 1, 0] += rates[i]
                s_rate[faults[i]] += rates[i]
            except ValueError:
                continue
            except KeyError:
                s_rate[faults[i]] = rates[i]
        for fault, count in np.column_stack(np.unique(faults, return_counts=True)):
            try:
                epsilon = np.digitize(
                    [norm.ppf(s_im[fault] / float(count))], bins_epsilon
                )[0]
            except KeyError:
                # s_im[fault] = 0, no valid contributing realisations
                continue
            except TypeError:
                # out of range
                continue
            try:
                cs_r = np.digitize([rrups_d[fault]], bins_rrup)[0]
                cs_m = np.digitize([mags_d[fault]], bins_mag)[0]
            except KeyError:
                # not found in empirical file
                continue
            if cs_m == 0 or cs_r == bins_rrup.size or cs_m == bins_mag.size:
                continue
            # fault_contrib = np.sum(rates[(faults == fault) and ()])
            try:
                fault_contrib = s_rate[fault]
            except KeyError:
                continue
            block[b, cs_r, cs_m - 1, epsilon + 3] += fault_contrib
            try:
                summ_contrib[fault] += fault_contrib
            except KeyError:
                summ_contrib[fault] = fault_contrib
        # find top 50 contributors
        summ_block = h5["deagg/{}/SUMM_{}".format(station, im)]
        percent_factor = sum(summ_contrib.values()) / 100.0
        top50 = np.argsort(list(summ_contrib.values()))[::-1][:50]
        names = np.array(list(summ_contrib.keys()))[top50]
        summ_block[b, : top50.size] = list(
            zip(
                np.searchsorted(all_faults, names),
                np.array(list(summ_contrib.values()))[top50] / percent_factor,
            )
        )
        if top50.size < 50:
            summ_block[b, top50.size :] = -1, 0


###
### STEP 0: prepare/validate inputs
###

args = None
if IS_MASTER:
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("imdb", help="Location of imdb.h5")
    arg("emp_src", help="Location of empiricals dir containing IM subfolders")
    arg("nhm_file", help="Location of NHM file for fault names.")
    arg("empdb", help="Where to store Empirical DB empdb.h5")
    arg(
        "--hazard-n",
        help="Number of points making up hazard curve.",
        type=int,
        default=50,
    )
    arg("--mag-d", help="magnitude spacing", type=float, default=0.25)
    arg("--mag-min", help="magnitude minimum", type=float, default=5.0)
    arg(
        "--mag-n", help="magnitude blocks from minimum at spacing", type=int, default=16
    )
    arg("--rrup-d", help="rrup spacing", type=float, default=10.0)
    arg("--rrup-n", help="rrup blocks at spacing", type=int, default=20)
    arg(
        "--deagg-e",
        help="exceedence for deagg",
        nargs=2,
        metavar=("years", "probability"),
        type=float,
        action="append",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        # invalid arguments or -h
        COMM.Abort()
    if args.deagg_e is None:
        args.deagg_e = [[50.0, 0.5], [50.0, 0.1], [50.0, 0.02]]
args = COMM.bcast(args, root=MASTER)

all_faults = fault_names(args.nhm_file)
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

h5 = h5py.File(args.empdb + ".P", "w", driver="mpio", comm=COMM)

h5.attrs["ims"] = np.array(emp_ims, dtype=np.string_)
h5.attrs["values_x"] = (
    np.arange(args.rrup_n, dtype=np.float32) * args.rrup_d + args.rrup_d / 2.0
)
h5.attrs["values_y"] = (
    np.arange(args.mag_n, dtype=np.float32) * args.mag_d
    + args.mag_min
    + args.mag_d / 2.0
)
h5.attrs["values_z"] = np.array(
    [
        "A (CS)",
        "B",
        "DS",
        "E < -2",
        "-2 < E < -1",
        "-1 < E < -0.5",
        "-0.5 < E < 0",
        "0 < E < 0.5",
        "0.5 < E < 1",
        "1 < E < 2",
        "2 < E",
    ],
    dtype=np.string_,
)
h5.attrs["deagg_e"] = args.deagg_e

# stations reference
station_dtype = np.dtype([("name", "|S7"), ("lon", "f4"), ("lat", "f4")])
h5_ll = h5.create_dataset("stations", (imdb_stations.size,), dtype=station_dtype)
for i, stat in enumerate(imdb_stations[RANK::SIZE]):
    # TODO: use location from actual empirical file
    h5_ll[RANK + i * SIZE] = (stat.name.astype(np.string_), stat.lon, stat.lat)
del h5_ll

# fault list reference
h5_fault = h5.create_dataset(
    "faults", (len(all_faults),), dtype="|S{}".format(len(max(all_faults, key=len)))
)
for i, fault in enumerate(all_faults[RANK::SIZE]):
    h5_fault[RANK + i * SIZE] = fault.encode()
del h5_fault

# per station IM and simulation datasets
top_dtype = np.dtype([("fault", np.int16), ("contribution", np.float16)])
for stat in imdb_stations.name:
    for im in emp_ims:
        h5.create_dataset(
            "hazard/{}/{}".format(stat, im), (N_SERIES, args.hazard_n), dtype="f4"
        )
        h5.create_dataset(
            "deagg/{}/{}".format(stat, im),
            (len(args.deagg_e), args.rrup_n, args.mag_n, N_TYPES),
            dtype="f2",
        )
        h5.create_dataset(
            "deagg/{}/SUMM_{}".format(stat, im),
            (len(args.deagg_e), 50),
            dtype=top_dtype,
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
        process_emp_file(
            args, all_faults, emp_files[f], imdb_stations[j].name, emp_ims[i]
        )
h5.close()

###
### STEP 3: master to compress deagg data
###
COMM.barrier()
if IS_MASTER:
    call(["h5repack", "-f", "deagg:GZIP=4", args.empdb + ".P", args.empdb])
    os.remove(args.empdb + ".P")

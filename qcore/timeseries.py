"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

import base64
from glob import glob
from io import BytesIO
import math
import os
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Union, Set

from qcore.constants import MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION
from qcore.formats import load_e3d_par
from qcore.utils import compare_versions


try:
    from scipy.signal import butter, resample
except ImportError:
    print("SciPy not installed. Certain functions will fail.")
# sosfilt new in scipy 0.16
# sosfiltfilt new in scipy 0.18
try:
    butter
    from scipy.signal import sosfiltfilt
except NameError:
    pass
except ImportError:
    from qcore.sosfiltfilt import sosfiltfilt
import numpy as np

rfft = np.fft.rfft
irfft = np.fft.irfft


# butterworth filter
# bandpass not necessary as sampling frequency too low
def bwfilter(data, dt, freq, band, match_powersb=True):
    """
    data: np.array to filter
    dt: timestep of data
    freq: cutoff frequency
    band: One of {'highpass', 'lowpass', 'bandpass', 'bandstop'}
    match_powersb: shift the target frequency so that the given frequency has no attenuation
    """
    # power spectrum based LF/HF filter (shift cutoff)
    # readable code commented, fast code uncommented
    # order = 4
    # x = 1.0 / (2.0 * order)
    # if band == 'lowpass':
    #    x *= -1
    # freq *= exp(x * math.log(sqrt(2.0) - 1.0))
    nyq = 1.0 / (2.0 * dt)
    highpass_shift = 0.8956803352330285
    lowpass_shift = 1.1164697500474103
    if match_powersb:
        if band == "highpass":
            freq *= highpass_shift
        elif band == "bandpass" or band == "bandstop":
            freq = np.asarray((freq * highpass_shift, freq * lowpass_shift))
        elif band == "lowpass":
            freq *= lowpass_shift
    return sosfiltfilt(
        butter(4, freq / nyq, btype=band, output="sos"), data, padtype=None
    )


def ampdeamp(timeseries, ampf, amp=True):
    """
    Amplify or Deamplify timeseries.
    """
    nt = len(timeseries)

    # length the fourier transform should be
    ft_len = ampf.size + ampf.size

    # taper 5% on the right using the hanning method
    ntap = int(nt * 0.05)
    timeseries[nt - ntap :] *= np.hanning(ntap * 2 + 1)[ntap + 1 :]

    # extend array, fft
    timeseries = np.resize(timeseries, ft_len)
    timeseries[nt:] = 0
    fourier = rfft(timeseries)

    # ampf modified for de-amplification
    if not amp:
        ampf = 1.0 / ampf
    # last value of fft is some identity value
    fourier[:-1] *= ampf

    return irfft(fourier)[:nt]


def transf(
    vs_soil,
    rho_soil,
    damp_soil,
    height_soil,
    vs_rock,
    rho_rock,
    damp_rock,
    nt,
    dt,
    ft_freq=None,
):
    """
    Used in deconvolution. Made by Chris de la Torre.
    vs = shear wave velocity (upper soil or rock)
    rho = density
    damp = damping ratio
    height_soil = height of soil above rock
    nt = number of timesteps
    dt = delta time in timestep (seconds)
    """
    if ft_freq is None:
        ft_len = int(2.0 ** math.ceil(math.log(nt) / math.log(2)))
        ft_freq = np.arange(0, ft_len / 2 + 1) / (ft_len * dt)
    omega = 2.0 * math.pi * ft_freq
    Gs = rho_soil * vs_soil**2.0
    Gr = rho_rock * vs_rock**2.0

    kS = omega / (vs_soil * (1.0 + 1j * damp_soil))
    kR = omega / (vs_rock * (1.0 + 1j * damp_rock))

    alpha = Gs * kS / (Gr * kR)

    H = 2.0 / (
        (1.0 + alpha) * np.exp(1j * kS * height_soil)
        + (1.0 - alpha) * np.exp(-1j * kS * height_soil)
    )
    H[0] = 1
    return H


def read_ascii(filepath, meta=False, t0=False):
    """
    Read timeseries data from standard ascii file to numpy array.
    meta: also return first 2 lines (metadata) as string lists
    t0: adjust data to start from t = 0 (uses secs and not hr:min)
    """
    with open(filepath, "r") as ts:
        info1 = ts.readline().split()
        info2 = ts.readline().split()

        vals = np.array(
            list(map(float, " ".join(list(map(str.rstrip, ts.readlines()))).split()))
        )

    # check if header length correct for integrity
    try:
        assert len(vals) == int(info2[0])
    except AssertionError:
        print("File entries don't match NT: %s" % (filepath))
        raise

    # start from t = 0 by (truncating) or (inserting 0s) at beginning
    if t0:
        # t start / dt = number of missing points
        diff = int(round(float(info2[4]) / float(info2[1])))
        if diff < 0:
            # remove points before t = 0
            vals = vals[abs(diff) :]
        elif diff > 0:
            # insert zeros between t = 0 and start
            vals = np.append(np.zeros(diff), vals)

    if meta:
        note = ""
        if len(info1) >= 3:
            note = " ".join(info1[2:])
        # int(float()) to not crash when there is a float (invalid) in hr or min
        return (
            vals,
            {
                "name": info1[0],
                "comp": info1[1],
                "note": note,
                "nt": int(info2[0]),
                "dt": float(info2[1]),
                "hr": int(float(info2[2])),
                "min": int(float(info2[3])),
                "sec": float(info2[4]),
                "e_dist": float(info2[5]),
                "az": float(info2[6]),
                "baz": float(info2[7]),
            },
        )
    return vals


def vel2acc(timeseries, dt):
    """
    Differentiate following Rob Graves' code logic.
    """
    return np.diff(np.hstack(([0], timeseries)) * (1.0 / dt))


def vel2acc3d(timeseries, dt):
    """
    vel2acc for x,y,z arrays
    """
    return np.diff(np.vstack(([0, 0, 0], timeseries)), axis=0) * (1.0 / dt)


def acc2vel(timeseries, dt):
    """
    Integrates following Rob Graves' code logic (simple).
    also works for x,y,z arrays
    """
    return np.cumsum(timeseries, axis=0) * dt


def pgv2MMI(pgv):
    """
    Calculates MMI from pgv based on Worden et al (2012)
    A maximum function is applied to floor the value to 1
    """
    return np.maximum(
        np.where(
            np.log10(pgv) < 0.53,
            3.78 + 1.47 * np.log10(pgv),
            2.89 + 3.16 * np.log10(pgv),
        ),
        1,
    )


def timeseries_to_text(
    timeseries: np.ndarray,
    filename: Path,
    dt: float,
    stat: str,
    comp: str,
    values_per_line: int = 6,
    start_hr: int = 0,
    start_min: int = 0,
    start_sec: float = 0.0,
    edist: float = 0.0,
    az: float = 0.0,
    baz: float = 0.0,
    title: str = "",
):
    """
    Store timeseries data into a text file.

    Parameters
    ----------
    timeseries : np.ndarray
        The timeseries data to store
    filename : Path
        The full file path to store the file
    dt : float
        The time step of the data
    stat : str
        The station name
    comp : str
        The component name
    values_per_line : int, optional
        The number of values per line, by default 6
    start_hr : int, optional
        The start hour of the data, by default 0
    start_min : int, optional
        The start minute of the data, by default 0
    start_sec : float, optional
        The start second of the data, by default 0.0
    edist : float, optional
        The epicentral distance, by default 0.0
    az : float, optional
        The azimuth forward A->B in degrees, by default 0.0
    baz : float, optional
        The azimuth backwards B->A in degrees, by default 0.0
    title : str, optional
        The optional title added to header
    """
    nt = timeseries.shape[0]
    with open(filename, "wb") as txt:
        # same format strings as fdbin2wcc
        txt.write(("%-10s %3s %s\n" % (stat, comp, title)).encode())
        txt.write(
            (
                "%d %12.5e %d %d %12.5e %12.5e %12.5e %12.5e\n"
                % (nt, dt, start_hr, start_min, start_sec, edist, az, baz)
            ).encode()
        )
        # values below header lines, split into lines
        divisible = nt - nt % values_per_line
        np.savetxt(
            txt, timeseries[:divisible].reshape(-1, values_per_line), fmt="%13.5e"
        )
        np.savetxt(txt, np.atleast_2d(timeseries[divisible:]), fmt="%13.5e")


###
### PROCESSING OF LF BINARY CONTAINER
###
class LFSeis:
    # format constants
    HEAD_STAT = 0x30
    N_COMP = 9
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, outbin):
        """
        Load LF binary store.
        outbin: path to OutBin folder containing seis files
        """
        self.seis = sorted(glob(os.path.join(outbin, "*seis-*.e3d")))
        # try load e3d.par at the same directory first
        # otherwise try look for one folder above
        self.e3dpar = os.path.join(outbin, "e3d.par")
        if not os.path.isfile(self.e3dpar):
            self.e3dpar = os.path.join(outbin, "../e3d.par")
            if not os.path.isfile(self.e3dpar):
                raise ValueError(
                    "Cannot find e3d.par in the given directory or a folder above. "
                    "Either move or create a symlink to the correct file please."
                )
            else:
                print(
                    "e3d.par was not found under the same folder but found in one level above"
                )
                print("e3d.par path: {}".format(self.e3dpar))

        # determine endianness by checking file size
        lfs = os.stat(self.seis[0]).st_size
        with open(self.seis[0], "rb") as lf0:
            nstat, nt = np.fromfile(lf0, dtype="<i4", count=6)[0::5]
            if (
                lfs
                == 4
                + np.int64(nstat) * self.HEAD_STAT
                + np.int64(nstat) * nt * self.N_COMP * 4
            ):
                endian = "<"
                self.nt = nt
            elif (
                lfs
                == 4
                + np.int64(nstat.byteswap()) * self.HEAD_STAT
                + np.int64(nstat.byteswap()) * nt.byteswap() * self.N_COMP * 4
            ):
                endian = ">"
                self.nt = nt.byteswap()
            else:
                raise ValueError("File is not an LF seis file: %s" % (self.seis[0]))
            self.i4 = "%si4" % (endian)
            self.f4 = "%sf4" % (endian)
            # load rest of common metadata from first station in first file
            self.dt, self.hh, self.rot = np.fromfile(lf0, dtype=self.f4, count=3)
            self.duration = self.nt * self.dt

        pars = load_e3d_par(self.e3dpar)
        self.flo = float(pars["flo"])
        self.emod3d_version = pars["version"]

        if self.flo is None or self.emod3d_version is None:
            raise ValueError(
                "The e3d.par file in the OutBin directory did not contain at least one of flo and version, "
                "please add the correct values and run again."
            )
        self.start_sec = -1 / self.flo
        if (
            compare_versions(self.emod3d_version, MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION)
            > 0
        ):
            self.start_sec *= 3

        # rotation matrix for converting to 090, 000, ver is inverted (* -1)
        theta = math.radians(self.rot)
        self.rot_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [-math.sin(theta), -math.cos(theta), 0],
                [0, 0, -1],
            ]
        )

        # load nstats to determine total size
        nstats = np.zeros(len(self.seis), dtype="i")
        for i, s in enumerate(self.seis):
            nstats[i] = np.fromfile(s, dtype=self.i4, count=1)
        # container for station data
        stations = np.rec.array(
            np.zeros(
                np.sum(nstats),
                dtype=[
                    ("x", "i4"),
                    ("y", "i4"),
                    ("z", "i4"),
                    ("seis_idx", "i4", 2),
                    ("lat", "f4"),
                    ("lon", "f4"),
                    ("name", "|S8"),
                ],
            )
        )
        # populate station data from headers
        for i, s in enumerate(self.seis):
            with open(s) as f:
                f.seek(4)
                stations_n = np.fromfile(
                    f,
                    count=nstats[i],
                    dtype=np.dtype(
                        {
                            "names": [
                                "stat_pos",
                                "x",
                                "y",
                                "z",
                                "seis_idx",
                                "lat",
                                "lon",
                                "name",
                            ],
                            "formats": [
                                self.i4,
                                self.i4,
                                self.i4,
                                self.i4,
                                (self.i4, 2),
                                self.f4,
                                self.f4,
                                "|S8",
                            ],
                            "offsets": [0, 4, 8, 12, 16, 32, 36, 40],
                        }
                    ),
                )
            stations_n["seis_idx"][:, 0] = i
            stations_n["seis_idx"][:, 1] = np.arange(nstats[i])
            stations[stations_n["stat_pos"]] = stations_n[
                list(stations_n.dtype.names[1:])
            ]
        # protect against duplicated stations between processes
        # results in too many stations entries created, last ones are empty
        # important to keep indexes correct, only remove empty items from end
        if stations.name[-1] == "":
            stations = stations[: -np.argmin((stations.name == "")[::-1])]
        # store station names as unicode (python 3 strings)
        stat_type = stations.dtype.descr
        stat_type[6] = stat_type[6][0], "U7"
        self.stations = np.rec.fromrecords(stations, dtype=stat_type)

        self.nstat = self.stations.size
        # allow indexing by station names
        self.stat_idx = dict(list(zip(self.stations.name, np.arange(self.nstat))))

        # information for timeseries retrieval
        self.ts_pos = 4 + nstats * self.HEAD_STAT
        self.ts0_type = "3%sf4" % (endian)
        self.ts_type = [
            np.dtype(
                {
                    "names": ["xyz"],
                    "formats": [self.ts0_type],
                    "offsets": [nstats[i] * self.N_COMP * 4 - 3 * 4],
                }
            )
            for i in range(nstats.size)
        ]

    def vel(self, station, dt=None):
        """
        Returns timeseries (velocity, cm/s) for station.
        station: station name, must exist
        """
        file_no, file_idx = self.stations[self.stat_idx[station]]["seis_idx"]
        ts = np.empty((self.nt, 3))
        with open(self.seis[file_no], "r") as data:
            data.seek(self.ts_pos[file_no] + file_idx * self.N_COMP * 4)
            ts[0] = np.fromfile(data, dtype=self.ts0_type, count=1)
            ts[1:] = np.fromfile(data, dtype=self.ts_type[file_no])["xyz"]
            ts = np.dot(ts, self.rot_matrix)
        if dt is None or dt == self.dt:
            return ts
        return resample(ts, int(round(self.duration / dt)))

    def acc(self, station, dt=None):
        """
        Like vel but also converts to acceleration (cm/s/s).
        """
        if dt is None:
            dt = self.dt
        return vel2acc3d(self.vel(station, dt=dt), dt)

    def vel2txt(self, station, prefix="./", title="", dt=None, acc=False):
        """
        Creates standard EMOD3D text files for the station.
        """
        if dt is None:
            dt = self.dt
        if acc:
            f = self.acc
        else:
            f = self.vel
        for i, c in enumerate(f(station, dt=dt).T):
            timeseries_to_text(
                c,
                f"{prefix}{station}.{self.COMP_NAME[i]}",
                dt,
                station,
                self.COMP_NAME[i],
                start_sec=self.start_sec,
                title=title,
            )

    def all2txt(self, prefix="./", dt=None, f="vel"):
        """
        Creates waveforms in text files for all stations.
        Note: This function is not designed to be used other than for single/debug use.
        Make a parallel wrapper for any "real" use cases.
        Produces text files previously done by script called `winbin-aio`.
        For compatibility. Consecutive file indexes in parallel for performance.
        Slowest part is numpy formating numbers into text and number of lines.

        Parameters
        ----------
        prefix : str, optional, default="./"
            The prefix is an output path combined with an optional filename prefix.
            If prefix ends with /, its prefix_filename is empty, otherwise, prefix_filename is the last part of the prefix
            eg. prefix = "dir1/dir2/XXX", prefix_filename = "XXX" and prefix_dirname = "dir1/dir2"
            eg. prefix = "dir1/dir2/", prefix_filename = "" and prefix_dirname = "dir1/dir2"
        dt : float, optional
            The time step of the data, by default None
        f : str, optional
            The type of data to save, by default "acc". Options are "acc" and "vel"
        """

        if dt is None:
            dt = self.dt
        acc = f == "acc"

        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations.name:
            self.vel2txt(s, prefix=prefix, title=prefix_filename, dt=dt, acc=acc)


###
### PROCESSING OF HF BINARY CONTAINER
###
class HFSeis:
    # format constants
    HEAD_SIZE = 0x200
    HEAD_STAT = 0x18
    N_COMP = 3
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, hf_path):
        """
        Load HF binary store.
        hf_path: path to the HF binary file
        """

        hfs = os.stat(hf_path).st_size
        hff = open(hf_path, "rb")
        # determine endianness by checking file size
        nstat, nt = np.fromfile(hff, dtype="<i4", count=2)
        if (
            hfs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            endian = "<"
        elif (
            hfs
            == self.HEAD_SIZE
            + np.int64(nstat.byteswap()) * self.HEAD_STAT
            + np.int64(nstat.byteswap()) * nt.byteswap() * self.N_COMP * 4
        ):
            endian = ">"
        else:
            hff.close()
            raise ValueError("File is not an HF seis file: %s" % (hf_path))
        hff.seek(0)

        # read header - integers
        (
            self.nstat,
            self.nt,
            self.seed,
            siteamp,
            self.pdur_model,
            nrayset,
            rayset1,
            rayset2,
            rayset3,
            rayset4,
            self.nbu,
            self.ift,
            self.nlskip,
            icflag,
            same_seed,
            site_specific_vm,
        ) = np.fromfile(hff, dtype="%si4" % (endian), count=16)
        self.siteamp = bool(siteamp)
        self.rayset = [rayset1, rayset2, rayset3, rayset4][:nrayset]
        self.icflag = bool(icflag)
        self.seed_inc = not bool(same_seed)
        self.site_specific_vm = bool(site_specific_vm)
        # read header - floats
        (
            self.duration,
            self.dt,
            self.start_sec,
            self.sdrop,
            self.kappa,
            self.qfexp,
            self.fmax,
            self.flo,
            self.fhi,
            self.rvfac,
            self.rvfac_shal,
            self.rvfac_deep,
            self.czero,
            self.calpha,
            self.mom,
            self.rupv,
            self.vs_moho,
            self.vp_sig,
            self.vsh_sig,
            self.rho_sig,
            self.qs_sig,
            self.fa_sig1,
            self.fa_sig2,
            self.rv_sig1,
        ) = np.fromfile(hff, dtype="%sf4" % (endian), count=24)
        # read header - strings
        self.stoch_file, self.velocity_model = np.fromfile(hff, dtype="|S64", count=2)

        # load station info
        hff.seek(self.HEAD_SIZE)
        stations = np.fromfile(
            hff,
            count=self.nstat,
            dtype=[
                ("lon", "%sf4" % (endian)),
                ("lat", "%sf4" % (endian)),
                ("name", "|S8"),
                ("e_dist", "%sf4" % (endian)),
                ("vs", "%sf4" % (endian)),
            ],
        )
        hff.close()
        # store station names as unicode (python 3 strings)
        stat_type = stations.dtype.descr
        stat_type[2] = stat_type[2][0], "U7"
        self.stations = np.rec.fromrecords(stations, dtype=stat_type)
        if np.min(self.stations.vs) == 0:
            print("WARNING: looks like an incomplete file: %s" % (hf_path))

        # allow indexing by station names
        self.stat_idx = dict(zip(self.stations.name, np.arange(self.nstat)))
        # keep location for data retrieval
        self.path = hf_path
        # location to start of 3rd (data) block
        self.ts_pos = self.HEAD_SIZE + nstat * self.HEAD_STAT
        # data format
        self.dtype = "3%sf4" % (endian)

    def acc(self, station, comp=Ellipsis, dt=None):
        """
        Returns timeseries (acceleration, cm/s/s) for station.
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        with open(self.path, "r") as data:
            data.seek(self.ts_pos + self.stat_idx[station] * self.nt * 3 * 4)
            ts = np.fromfile(data, dtype=self.dtype, count=self.nt)
        if dt is None or dt == self.dt:
            return ts
        return resample(ts, int(round(self.duration / dt)))

    def vel(self, station, dt=None):
        """
        Like acc but also converts to velocity (cm/s).
        """
        if dt is None:
            dt = self.dt
        return acc2vel(self.acc(station, dt=dt), dt)

    def acc2txt(self, station, prefix="./", title="", dt=None):
        """
        Creates standard EMOD3D text files for the station.
        """
        if dt is None:
            dt = self.dt
        stat_idx = self.stat_idx[station]
        for i, c in enumerate(self.acc(station, dt=dt).T):
            timeseries_to_text(
                c,
                f"{prefix}{station}.{self.COMP_NAME[i]}",
                dt,
                station,
                self.COMP_NAME[i],
                start_sec=self.start_sec,
                edist=self.stations.e_dist[stat_idx],
                title=title,
            )

    def all2txt(self, prefix="./", dt=None):
        """
        Creates waveforms in text files for all stations.

        Note: This function is not designed to be used other than for single/debug use.
        Make a parallel wrapper for any "real" use cases.
        Produces text files previously done by script called `winbin-aio`.
        For compatibility. Consecutive file indexes in parallel for performance.
        Slowest part is numpy formating numbers into text and number of lines.

        Parameters
        ----------
        prefix : str, optional, default="./"
            The prefix is an output path combined with an optional filename prefix.
            If prefix ends with /, its prefix_filename is empty, otherwise, prefix_filename is the last part of the prefix
            eg. prefix = "dir1/dir2/XXX", prefix_filename = "XXX" and prefix_dirname = "dir1/dir2"
            eg. prefix = "dir1/dir2/", prefix_filename = "" and prefix_dirname = "dir1/dir2"
        dt : float, optional
            The time step of the data, by default None
        """

        if dt is None:
            dt = self.dt

        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations.name:
            self.acc2txt(s, prefix=prefix, title=prefix_filename, dt=dt)


###
### PROCESSING OF BB BINARY CONTAINER
###
class BBSeis:
    # format constants
    HEAD_SIZE = 0x500
    HEAD_STAT = 0x2C
    N_COMP = 3
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, bb_path):
        """
        Load BB binary store.
        bb_path: path to the BB binary file
        """

        bbs = os.stat(bb_path).st_size
        bbf = open(bb_path, "rb")
        # determine endianness by checking file size
        nstat, nt = np.fromfile(bbf, dtype="<i4", count=2)
        if (
            bbs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            endian = "<"
        elif (
            bbs
            == self.HEAD_SIZE
            + np.int64(nstat.byteswap()) * self.HEAD_STAT
            + np.int64(nstat.byteswap()) * nt.byteswap() * self.N_COMP * 4
        ):
            endian = ">"
        else:
            bbf.close()
            raise ValueError("File is not an BB seis file: %s" % (bb_path))
        bbf.seek(0)

        # read header - integers
        self.nstat, self.nt = np.fromfile(bbf, dtype="%si4" % (endian), count=2)
        # read header - floats
        self.duration, self.dt, self.start_sec = np.fromfile(
            bbf, dtype="%sf4" % (endian), count=3
        )
        # read header - strings
        self.lf_dir, self.lf_vm, self.hf_file = np.fromfile(
            bbf, count=3, dtype="|S256"
        ).astype(np.unicode_)

        # load station info
        bbf.seek(self.HEAD_SIZE)
        stations = np.rec.array(
            np.fromfile(
                bbf,
                count=self.nstat,
                dtype=[
                    ("lon", "f4"),
                    ("lat", "f4"),
                    ("name", "|S8"),
                    ("x", "i4"),
                    ("y", "i4"),
                    ("z", "i4"),
                    ("e_dist", "f4"),
                    ("hf_vs_ref", "f4"),
                    ("lf_vs_ref", "f4"),
                    ("vsite", "f4"),
                ],
            )
        )
        bbf.close()
        # store station names as unicode (python 3 strings)
        stat_type = stations.dtype.descr
        stat_type[2] = stat_type[2][0], "U7"
        self.stations = np.rec.fromrecords(stations, dtype=stat_type)
        if np.min(self.stations.vsite) == 0:
            print("WARNING: looks like an incomplete file: %s" % (bb_path))

        # allow indexing by station names
        self.stat_idx = dict(list(zip(self.stations.name, np.arange(self.nstat))))
        # keep location for data retrieval
        self.path = bb_path
        # location to start of 3rd (data) block
        self.ts_pos = self.HEAD_SIZE + nstat * self.HEAD_STAT
        # data format
        self.dtype = "3%sf4" % (endian)

    def acc(self, station, comp=Ellipsis):
        """
        Returns timeseries (acceleration, g) for station.
        TODO: select component by changing dtype
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        with open(self.path, "r") as data:
            data.seek(self.ts_pos + self.stat_idx[station] * self.nt * 3 * 4)
            return np.fromfile(data, dtype=self.dtype, count=self.nt)[:, comp]

    def vel(self, station, comp=Ellipsis):
        """
        Returns timeseries (velocity, cm/s) for station.
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        return acc2vel(self.acc(station, comp=comp) * 981.0, self.dt)

    def save_txt(self, station, prefix="./", title="", f="acc"):
        """
        Creates standard EMOD3D text files for the station.
        Prefix is the name of file before station.component,
            use None to retun the 3 component files as byte arrays.
        """
        stat_idx = self.stat_idx[station]
        if f == "vel":
            f = self.vel
        else:
            f = self.acc
        xyz = []
        for i, c in enumerate(f(station).T):
            xyz.append(
                timeseries_to_text(
                    c,
                    f"{prefix}{station}.{self.COMP_NAME[i]}",
                    self.dt,
                    station,
                    self.COMP_NAME[i],
                    start_sec=self.start_sec,
                    edist=self.stations.e_dist[stat_idx],
                    title=title,
                )
            )
        if prefix is None:
            return xyz

    def all2txt(self, prefix="./", f="acc"):
        """
        Extracts waveform data from the binary file and produces output in text format.

        Parameters
        ----------
        prefix : str, optional, default="./"
            The prefix is an output path combined with an optional filename prefix.
            If prefix ends with /, its prefix_filename is empty, otherwise, prefix_filename is the last part of the prefix
            eg. prefix = "dir1/dir2/XXX", prefix_filename = "XXX" and prefix_dirname = "dir1/dir2"
            eg. prefix = "dir1/dir2/", prefix_filename = "" and prefix_dirname = "dir1/dir2"
        f : str, optional
            The type of data to save, by default "acc". Options are "acc" and "vel"
        """
        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations.name:
            self.save_txt(s, prefix=prefix, title=prefix_filename, f=f)

    def save_ll(self, path):
        """
        Saves station list to text file containing: lon lat station_name.
        """
        np.savetxt(path, self.stations[["lon", "lat", "name"]], fmt="%f %f %s")


def get_observed_stations(observed_data_folder: Union[str, Path]) -> Set[str]:
    """
    returns a list of station names that can be found in the observed data folder

    :param observed_data_folder: path to the record folder, e.g. observed/events/vol1/data/accBB/
    :type observed_data_folder: str/os.path
    :return: list of unique station names
    :rtype: Set[str]
    """
    search_path = os.path.abspath(os.path.join(observed_data_folder, "*"))
    files = glob(search_path)
    station_names = {
        os.path.splitext(os.path.basename(filename))[0] for filename in files
    }

    return station_names

"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

import math
import multiprocessing
import os
import warnings
from enum import StrEnum, auto
from glob import glob
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyfftw
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import scipy as sp
import xarray as xr

from qcore.constants import MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION
from qcore.utils import compare_versions

# The `sosfiltfilt` function So we instead shift the cutoff frequency
# up for a highpass filter (resp. down for a lowpass filter) so that
# power at the cutoff frequencies becomes 1/sqrt(2). The factor by
# which we have to shift the cutoff factors for an order 4 highpass
# butterworth filter is (sqrt(2) - 1) ^ (1/8). Symmetrically, we have
# to shift by (sqrt(2) - 1) ^ (-1/8) = 1 / highpass shift for a
# lowpass filter. See https://dsp.stackexchange.com/a/19491 for a more
# detailed explanation. Note that `sosfiltfilt` applies the filter
# twice, so the attenuation at the target frequency is usually
# 1/sqrt(2) * 1/sqrt(2) = 1/2!
_BW_HIGHPASS_SHIFT = (np.sqrt(2) - 1) ** (1 / 8)
_BW_LOWPASS_SHIFT = 1 / _BW_HIGHPASS_SHIFT


class Band(StrEnum):
    """Filter types for `bwfilter`."""

    HIGHPASS = auto()
    """High-pass filter, filters all frequencies lower than taper frequency."""
    BANDPASS = auto()
    """Band-pass filter, filters all frequencies outside bounding frequencies."""
    BANDSTOP = auto()
    """Band-pass filter, filters all frequencies between bounding frequencies."""
    LOWPASS = auto()
    """Low-pass filter, filters all frequencies greater than taper frequencies."""


def bwfilter(
    waveform: np.ndarray,
    dt: float,
    taper_frequency: float | np.ndarray,
    band: Band,
) -> np.ndarray:
    """Construct and apply a Butterworth filter to a waveform.

    This function constructs an order-4 Butterworth filter with cutoff
    frequencies. It is applied forward and backward to eliminate phase
    lag. The `taper_frequency` is used to set cutoff frequencies for
    the filter such that the power at `taper_frequency` is 1/sqrt(2)
    of the original power.

    Parameters
    ----------
    waveform : np.ndarray
        The input waveform.
    dt : float
        The timestep of the input waveform.
    taper_frequency : float
        The tapering frequency. If `band` is highpass or lowpass then
        `taper_frequency` is the upper or lower tapering frequency, respectively.
        If `band` is either bandpass or bandstop, this should be an
        array of tapering frequencies `[low_cutoff, high_cutoff]`.
    band : Band
        Changes the kind of filter. See `Band` for details.

    Returns
    -------
    np.ndarray
        The filtered waveform.

    See Also
    --------
    https://en.wikipedia.org/wiki/Butterworth_filter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
    (specifically, the examples comparing sosfilt and sosfiltfilt)
    """

    cutoff_frequencies: np.ndarray | float = taper_frequency

    match band:
        case Band.HIGHPASS:
            cutoff_frequencies = taper_frequency * _BW_HIGHPASS_SHIFT
        case Band.BANDPASS | Band.BANDSTOP:
            cutoff_frequencies = taper_frequency * np.array(
                [_BW_HIGHPASS_SHIFT, _BW_LOWPASS_SHIFT]
            )
        case Band.LOWPASS:
            cutoff_frequencies = taper_frequency * _BW_LOWPASS_SHIFT

    return sp.signal.sosfiltfilt(
        sp.signal.butter(4, cutoff_frequencies, btype=band, output="sos", fs=1.0 / dt),
        waveform,
        padtype=None,
    )


def ampdeamp(
    waveform: np.ndarray,
    amplification_factor: np.ndarray,
    amplify: bool = True,
    cores: int = multiprocessing.cpu_count(),
) -> np.ndarray:
    """Apply amplification factor to waveforms.

    Parameters
    ----------
    waveform : np.ndarray
        The input waveform.
    amplification_factor : np.ndarray
        The frequency amplification factors. If `waveform` has
        length `2^i`, then `amplification_factor` should have length `2^(ceil(i) -
        1)`.
    amplify : bool
        Setting `amplify = False` is equivalent to setting
        `amplification_factor = np.reciprocal(amplification_factor)`.
    cores : int
        The number of cores to use for FFT. Defaults to all cores
        available on the system as reported by
        `muliprocessing.cpu_count()`.

    Returns
    -------
    np.ndarray
        The input waveform (de)ampilfied at frequencies according to
        the values of `amplification_factor`.
    """

    pyfftw.config.NUM_THREADS = cores

    nt = waveform.shape[-1]
    waveform_dtype = waveform.dtype

    # Taper 5% on the right using the Hanning method
    ntap = int(nt * 0.05)

    if ntap > 0:
        # Create a Hanning window for the taper, ensuring it's float32
        hanning_window = np.hanning(ntap * 2)[ntap:].astype(waveform_dtype)
        waveform[..., nt - ntap :] *= hanning_window

    n_fft = 2 * amplification_factor.shape[-1]

    fourier = pyfftw_fft.rfft(waveform, n=n_fft, axis=-1)

    # Amplification factor modified for de-amplification
    if not amplify:
        # Ensure ampf_modified is float32 for consistent operations.
        # Handle potential division by zero if ampf contains zeros.
        ampf_modified = np.where(
            amplification_factor != 0, 1.0 / amplification_factor, np.inf
        ).astype(waveform_dtype)
    else:
        ampf_modified = amplification_factor.astype(waveform_dtype)

    # Apply amplification/de-amplification. fourier[..., :-1]
    # corresponds to the first `n_fft // 2` frequency bins.
    fourier[..., :-1] *= ampf_modified

    result_full = pyfftw_fft.irfft(fourier, n=n_fft, axis=-1)

    # Trim to original length
    return result_full[..., :nt]


def transf(
    vs_soil: np.ndarray,
    rho_soil: np.ndarray,
    damp_soil: np.ndarray,
    height_soil: np.ndarray,
    vs_rock: np.ndarray,
    rho_rock: np.ndarray,
    damp_rock: np.ndarray,
    nt: int,
    dt: float,
    ft_freq: np.ndarray | None = None,
) -> np.ndarray:
    """Used in de-convolution of high-frequency site-response modelling.

    Can be used instead of traditional Vs30-based site-response when
    the relevant input parameters are known. Made by Chris de la
    Torre. It is part of the workflow described in [0]_.

    Parameters
    ----------
    vs_soil : array of floats
        The shear wave velocity in upper soil.
    rho_soil : array of floats
        The upper soil density.
    damp_soil : array of floats
        The upper soil damping ratio.
    height_soil : array of floats
        The height of the upper soil.
    vs_rock : array of floats
        The shear wave velocity in rock.
    rho_rock : array of floats
        The rock density.
    damp_rock : array of floats
        The rock damping ratio.
    nt : float
        The number of timesteps in input waveform.
    dt : float
        Waveform timestep.
    ft_freq : array of floats, optional
        Frequency space of transformed waveform.

    Returns
    -------
    np.ndarray
        A transfer function `H` used for waveform de-convolution.

    References
    ----------
    ..[0] de la Torre, C. A., Bradley, B. A., & Lee, R. L. (2020). Modeling
    nonlinear site effects in physics-based ground motion simulations
    of the 2010–2011 Canterbury earthquake sequence. Earthquake
    Spectra, 36(2), 856-879.
    """
    if ft_freq is None:
        ft_len = int(2.0 ** np.ceil(np.log(nt) / np.log(2)))
        ft_freq = np.arange(0, ft_len / 2 + 1) / (ft_len * dt)
    omega = 2.0 * np.pi * ft_freq
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


def _velocity_to_acceleration(
    velocities: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:
    """
    Convert velocity to acceleration for an array of shape (ns, nt, 3).

    Uses backward difference method for compatibility with original implementation.

    Parameters
    ----------
    velocities : numpy.ndarray
        Array of velocity data with shape (ns, nt, 3)
    dt : float
        Time step in seconds

    Returns
    -------
    numpy.ndarray
        Array of acceleration data with the same shape as input
    """
    # Create a copy of the input array shifted one time step
    # For each station, prepend a zero vector [0,0,0] and remove the last time step
    ns, nt, nc = velocities.shape
    zeros = np.zeros((ns, 1, nc))
    shifted = np.concatenate([zeros, velocities[:, :-1, :]], axis=1)

    # Calculate the backward difference and scale by 1/dt
    return (velocities - shifted) / dt


_HEAD_STAT = 48  # Header size per station
_N_COMP = 9  # Number of components in LF seis files


def _lfseis_dtypes(seis_file: Path) -> tuple[str, str]:
    """Determine the data types for reading LF seis files.

    Parameters
    ----------
    seis_file : Path
        Path to the LF seis file.

    Returns
    -------
    tuple[str, str]
        Tuple containing the int and float types accounting for file endianness.
    """
    with open(seis_file, "rb") as f:
        nstat: np.int32
        nt: np.int32
        nstat, nt = np.fromfile(f, dtype="<i4", count=6)[0::5]
        file_size = seis_file.stat().st_size

        if file_size == 4 + nstat * _HEAD_STAT + nstat * nt * _N_COMP * 4:
            endian = "<"
        elif (
            file_size
            == 4
            + nstat.byteswap() * _HEAD_STAT
            + nstat.byteswap() * nt.byteswap() * _N_COMP * 4
        ):
            endian = ">"
        else:
            raise ValueError(f"File is not an LF seis file: {seis_file}")

        i4 = f"{endian}i4"
        f4 = f"{endian}f4"

        return i4, f4


def _read_lfseis_file(seis_file: Path) -> xr.Dataset:
    """Read a single LF seis file.

    Parameters
    ----------
    seis_file : Path
            Path to the LF seis file.

    Returns
    -------
    xr.Dataset
            xarray Dataset containing LF seis data.
    """
    i4, f4 = _lfseis_dtypes(seis_file)
    with open(seis_file, "rb") as f:
        # Read number of stations in this file
        try:
            nstat_file = int(np.fromfile(f, dtype=i4, count=1)[0])
        except IndexError:
            raise ValueError(f"File {seis_file} is empty")

        # Read station headers
        dtype_header = np.dtype(
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
                "formats": [i4, i4, i4, i4, (i4, 2), f4, f4, "|S8"],
                "offsets": [0, 4, 8, 12, 16, 32, 36, 40],
            }
        )

        station_headers = np.fromfile(f, dtype=dtype_header, count=nstat_file)
        x_coords: npt.NDArray[np.int32] = station_headers["x"]
        y_coords: npt.NDArray[np.int32] = station_headers["y"]
        lat_coords: npt.NDArray[np.int32] = station_headers["lat"]
        lon_coords: npt.NDArray[np.int32] = station_headers["lon"]
        # Decode station names from UTF-8 encoded bytes.

        station_names: list[str] = [
            station_name.decode("utf-8", errors="replace").strip("\x00")
            for station_name in station_headers["name"]
        ]
    nt = (seis_file.stat().st_size - 4 - nstat_file * _HEAD_STAT) // (
        nstat_file * _N_COMP * 4
    )
    offset = 4 + nstat_file * _HEAD_STAT
    waveform_data = np.memmap(
        seis_file,
        dtype=f4,
        mode="r",
        offset=offset,
        shape=(nt, nstat_file, _N_COMP),
    )

    valid_indices: npt.NDArray[np.int_] = np.array(
        [i for i, name in enumerate(station_names) if name]
    )
    if len(valid_indices) < len(station_names):
        station_names = [station_names[i] for i in valid_indices]
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        lat_coords = lat_coords[valid_indices]
        lon_coords = lon_coords[valid_indices]
        waveform_data = waveform_data[valid_indices]

    return xr.Dataset(
        data_vars={
            "waveforms": (
                ["station", "time", "component"],
                # Swap from (nt, nstat_file, _N_COMP) to (nstat_file, nt, 3)
                np.swapaxes(waveform_data[:, :, :3], 0, 1),
            ),
        },
        coords={
            "station": station_names,
            "component": ["x", "y", "z"],
            "x": ("station", x_coords),
            "y": ("station", y_coords),
            "lat": ("station", lat_coords),
            "lon": ("station", lon_coords),
        },
    )


def _lfseis_header(header_file: Path) -> tuple[int, float, float, float]:
    """Read the header of an LF seis file.

    Parameters
    ----------
    header_file : Path
        Path to the header file.

    Returns
    -------
    tuple[int, float, float, float]
        Tuple containing the number of timesteps, timestep, resolution, and rotation.
    """
    i4, f4 = _lfseis_dtypes(header_file)
    with open(header_file, "rb") as f:
        _ = f.seek(
            20
        )  # 4 * i4 for nstat, station index, x, y, z grid point of the first station
        nt = int(np.fromfile(f, dtype=i4, count=1)[0])
        dt, resolution, rotation = map(float, np.fromfile(f, dtype=f4, count=3))
    return nt, dt, resolution, rotation


def read_lfseis_directory(outbin: Path | str, start_sec: float = 0) -> xr.Dataset:
    """Read LF station seismograms in a single pass.

    Parameters
    ----------
    outbin : Pathlike
        Path to the OutBin directory containing seis files. Should contain `*seis-*.e3d` files.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing LF seis data.

    Raises
    ------
    ValueError
        If the directory does not contain LF seis files.
    """
    outbin = Path(outbin)
    seis_files = list(sorted(outbin.glob("*seis-*.e3d")))

    if not seis_files:
        raise ValueError(f"No LF seis files found in {outbin}")

    header_file = next(
        seis_file for seis_file in seis_files if seis_file.stat().st_size
    )
    nt, dt, resolution, rotation = _lfseis_header(header_file)
    # Calculate rotation matrix
    theta: float = np.radians(rotation)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [-np.sin(theta), -np.cos(theta), 0],
            [0, 0, -1],
        ]
    )

    # Read all station data and waveforms in a single pass per file
    ds = xr.concat(
        [_read_lfseis_file(seis_file) for seis_file in seis_files],
        dim="station",
    ).assign_coords(time=np.arange(start_sec, start_sec + nt * dt, dt))

    # Rotate waveforms and differentiate to get acceleration
    ds["waveforms"] = (
        ("station", "time", "component"),
        np.dot(ds["waveforms"], rotation_matrix),
    )

    # Set global attributes
    ds.attrs["dt"] = dt
    ds.attrs["resolution"] = resolution
    ds.attrs["rotation"] = rotation
    ds.attrs["units"] = "m/s"

    return ds


def load_e3d_par(fp: str, comment_chars=("#",)):
    """
    Loads an emod3d parameter file as a dictionary
    As the original file does not have type data all values will be strings. Typing must be done manually.
    Crashes if duplicate keys are found
    :param fp: The path to the parameter file
    :param comment_chars: Any single characters that denote the line as a comment if they are the first non whitespace character
    :return: The dictionary of key:value pairs, as found in the parameter file
    """
    vals = {}
    with open(fp) as e3d:
        for line in e3d:
            if line.lstrip()[0] in comment_chars:
                pass
            key, value = line.split("=")
            if key in vals:
                raise KeyError(
                    f"Key {key} is in the emod3d parameter file at least twice. Resolve this before re running."
                )
            vals[key] = value
    return vals


def acc2vel(timeseries, dt):
    """
    Integrates following Rob Graves' code logic (simple).
    also works for x,y,z arrays
    """
    return np.cumsum(timeseries, axis=0) * dt


def vel2acc3d(timeseries, dt):
    """
    vel2acc for x,y,z arrays
    """
    return np.diff(np.vstack(([0, 0, 0], timeseries)), axis=0) * (1.0 / dt)


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
        warnings.warn(
            "LFSeis is deprecated, use read_lfseis_directory instead",
            DeprecationWarning,
        )
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
                print(f"e3d.par path: {self.e3dpar}")

        # determine endianness by checking file size
        lfs = os.stat(self.seis[0]).st_size
        with open(self.seis[0], "rb") as lf0:
            nstat: np.int32
            nt: np.int32
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
        if stations.name[-1] in ["", b""]:
            stations = stations[
                : -np.argmin((stations.name == stations.name[-1])[::-1])
            ]

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
        return sp.signal.resample(ts, int(round(self.duration / dt)))

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

    def all2txt(self, prefix: str = "./", dt: float = None, f: str = "vel"):
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
            eg. prefix = "dir1/dir2/XXX", prefix_filename = "XXX" and prefix_dirname = "dir1/dir2"
            eg. prefix = "dir1/dir2/", prefix_filename = "" and prefix_dirname = "dir1/dir2"
        dt : float, optional, default=None
            The time step of the data
        f : str, optional, default="vel"
            The type of data to save. Options are "acc" and "vel"
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
        warnings.warn(
            "HFSeis is deprecated, use the xarray interface for new workflows",
            DeprecationWarning,
        )
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

    def all2txt(self, prefix: str = "./", dt: float = None):
        """
        Creates waveforms in text files for all stations.

        Note: For compatibility. Consecutive file indexes in parallel for performance.
        Slowest part is numpy formating numbers into text and number of lines.

        Parameters
        ----------
        prefix : str, optional, default="./"
            The prefix is an output path combined with an optional filename prefix.
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
        warnings.warn(
            "BBSeis is deprecated, use the xarray interface for new workflows",
            DeprecationWarning,
        )
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

    def all2txt(self, prefix: str = "./", f: str = "acc"):
        """
        Extracts waveform data from the binary file and produces output in text format.
        Note: For compatibility. Should run slices in parallel for performance.
        Slowest part is numpy formating numbers into text and number of lines.

        Parameters
        ----------
        prefix : str, optional, default="./"
            The prefix is an output path combined with an optional filename prefix.
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


def get_observed_stations(observed_data_folder: str | Path) -> set[str]:
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

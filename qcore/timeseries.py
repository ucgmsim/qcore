"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

import io
import math
import multiprocessing
import os
import warnings
from enum import StrEnum, auto
from glob import glob
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
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
    taper: bool = True,
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
    taper : bool, optional
        If true, taper the waveform to avoid spectral leakage. Default
        True.

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

    if ntap > 0 and taper:
        # Create a Hanning window for the taper, ensuring it's float32
        hanning_window = np.hanning(ntap * 2 + 1)[ntap + 1 :].astype(waveform_dtype)
        # Create a copy of the original waveform so-as not to modify it in-place.
        waveform = waveform.copy()
        waveform[..., nt - ntap :] *= hanning_window

    n_fft = 2 * amplification_factor.shape[-1]

    # NOTE: The old code had the following resizing behaviour
    # timeseries = np.resize(timeseries, ft_len)
    # timeseries[nt:] = 0
    # this is actually unnecessary as setting `n=n_fft` will automatically do the same thing
    # See: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
    # and the PYFFTW equivalent:
    # https://pyfftw.readthedocs.io/en/latest/source/pyfftw/interfaces/numpy_fft.html#pyfftw.interfaces.numpy_fft.rfft

    fourier = pyfftw_fft.rfft(waveform, n=n_fft, axis=-1)

    # Amplification factor modified for de-amplification
    if not amplify:
        # Ensure ampf_modified is float32 for consistent operations.
        # Handle potential division by zero if ampf contains zeros.
        if np.any(np.isclose(amplification_factor, 0.0)):
            raise ZeroDivisionError("Would divide by zero in amplification factor.")
        ampf_modified = np.reciprocal(amplification_factor).astype(waveform_dtype)
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
    Gs = rho_soil * vs_soil**2.0  # noqa: N806
    Gr = rho_rock * vs_rock**2.0  # noqa: N806

    kS = omega / (vs_soil * (1.0 + 1j * damp_soil))  # noqa: N806
    kR = omega / (vs_rock * (1.0 + 1j * damp_rock))  # noqa: N806

    alpha = Gs * kS / (Gr * kR)

    H = 2.0 / (  # noqa: N806
        (1.0 + alpha) * np.exp(1j * kS * height_soil)
        + (1.0 - alpha) * np.exp(-1j * kS * height_soil)
    )
    H[0] = 1
    return H


_HEAD_STAT = 48  # Header size per station
_N_COMP = 9  # Number of components in LF seis files


class LFSeisHeader(NamedTuple):
    """LFSeis File Header."""

    nstat: int
    """int: The number of stations in the file."""
    nt: int
    """int: The number of timesteps the file."""
    dt: float
    """float: The temporal resolution of the file."""
    resolution: float
    """float: The spatial resolution of the file."""
    rotation: float
    """float: The model rotation."""


class LFSeisParser:
    """A parser for LFSeis files.

    Parameters
    ----------
    handle : io.BufferedReader
        A buffered reader object representing a file object. Cannot be a
        file-like object due to the use of `np.fromfile` which requires a
        `fileno`.
    """

    def __init__(self, handle: io.BufferedReader):  # noqa: D107
        self.handle = handle
        self.length = self._extract_length()
        self.i4, self.f4 = self._lfseis_dtypes()

    def _extract_length(self) -> int:
        """Extract the length of the file, in bytes.

        Returns
        -------
        int
            The file length.
        """
        current_position = self.handle.tell()
        self.handle.seek(0, os.SEEK_END)
        length = self.handle.tell()
        self.handle.seek(current_position)
        return length

    def _lfseis_dtypes(self) -> tuple[str, str]:
        """Heuristically evaluate the byte order of the file.



        Returns
        -------
        tuple[str, str]
            The numpy dtype strings representing the `self.i4` and
            `self.f4` types accounting for machine byteorder.

        Raises
        ------
        ValueError
            If the given file does not have an `nt`, `nstat` value
            consistent with the file size. Will try byteorder swapping
            the values to find a match.
        """
        current_position = self.handle.tell()
        nstat_raw = np.fromfile(self.handle, dtype="<i4", count=1)[0]
        _ = self.handle.seek(
            16, os.SEEK_CUR
        )  # 4 * i4 for station index, x, y, z grid point of the first station
        nt_raw = np.fromfile(self.handle, dtype="<i4", count=1)[0]
        nstat = int(nstat_raw)
        nt = int(nt_raw)
        nstat_bw = int(nstat_raw.byteswap())
        nt_bw = int(nt_raw.byteswap())
        if self.length == 4 + nstat * _HEAD_STAT + nstat * nt * _N_COMP * 4:
            endian = "<"
        elif self.length == 4 + nstat_bw * _HEAD_STAT + nstat_bw * nt_bw * _N_COMP * 4:
            endian = ">"
        else:
            raise ValueError("Handle does not read from an LFSeis file.")

        i4 = f"{endian}i4"
        f4 = f"{endian}f4"
        self.handle.seek(current_position)
        return i4, f4

    def read_header(self) -> LFSeisHeader:
        """Read the header values from an LFSeis file.

        Returns
        -------
        LFSeisHeader
            The header object representing this file.
        """
        nstat = int(np.fromfile(self.handle, dtype=self.i4, count=1)[0])
        current_position = self.handle.tell()
        _ = self.handle.seek(
            16, os.SEEK_CUR
        )  # 4 * i4 for station index, x, y, z grid point of the first station
        nt = int(np.fromfile(self.handle, dtype=self.i4, count=1)[0])
        dt, resolution, rotation = map(
            float, np.fromfile(self.handle, dtype=self.f4, count=3)
        )
        self.handle.seek(current_position)
        return LFSeisHeader(
            nstat=nstat, nt=nt, dt=dt, resolution=resolution, rotation=rotation
        )

    def read_stations(self, nstat: int) -> pd.DataFrame:
        """Read the stations table from the LFSeis file.

        Parameters
        ----------
        nstat : int
            The number of stations to read.

        Returns
        -------
        pd.DataFrame
            A dataframe of all the stations contained in this file.
        """
        dtype_header = np.dtype(
            {
                "names": [
                    "stat_pos",
                    "x",
                    "y",
                    "lat",
                    "lon",
                    "station",
                ],
                "formats": [
                    self.i4,
                    self.i4,
                    self.i4,
                    self.f4,
                    self.f4,
                    "|S8",
                ],
                # Stations are packed like:
                # x y z seis_idx nt timesteps resolution rotation lat lon name
                # with timesteps, resolution and rotation repeated for
                # each station. So we skip these values using the
                # offsets.
                "offsets": [0, 4, 8, 32, 36, 40],
            }
        )
        station_headers = np.fromfile(self.handle, dtype=dtype_header, count=nstat)
        x_coords: npt.NDArray[np.int32] = station_headers["x"]
        y_coords: npt.NDArray[np.int32] = station_headers["y"]
        lat_coords: npt.NDArray[np.int32] = station_headers["lat"]
        lon_coords: npt.NDArray[np.int32] = station_headers["lon"]
        # Decode station names from UTF-8 encoded bytes.
        station_names: list[str] = [
            station_name.decode("utf-8", errors="replace").strip("\x00")
            for station_name in station_headers["station"]
        ]
        return pd.DataFrame(
            {
                "x": x_coords,
                "y": y_coords,
                "lat": lat_coords,
                "lon": lon_coords,
                "station": station_names,
            }
        )

    def read_waveform(self, shape: tuple[int, ...]) -> np.memmap:
        """Memory map the waveform array from the seis file.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the resulting waveform array.

        Returns
        -------
        np.memmap
            A memory-mapped array containing floating point values
            read from the seis file in the provided shape.
        """
        return np.memmap(
            self.handle, dtype=self.f4, mode="r", offset=self.handle.tell(), shape=shape
        )


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

    with open(seis_file, "rb") as f:
        parser = LFSeisParser(f)
        header = parser.read_header()
        stations = parser.read_stations(header.nstat)
        waveform = parser.read_waveform(shape=(header.nt, header.nstat, _N_COMP))
        # Swap from (nt, nstat_file, _N_COMP) to (3 (x, y, z),
        # nstat_file, nt). The reason for the transposition is that we
        # currently process broadband data by component, That is
        # working with component x, y, and then z with arrays of shape
        # (nstat_file, nt). Rearranging the data in this way optimises
        # the memory layout for broadband processing. It is all
        # transparent the user interacting with xarray anyhow, so the
        # order of the dimensions hardly matters.
        waveform = np.transpose(waveform[:, :, :3], (2, 1, 0))
        # Have numpy re-arrange the waveform in-memory to reflect the
        # transposition change above.
        waveform = np.ascontiguousarray(waveform)
        # When reading station names from the file, encoding errors may occur.
        # These errors can result in invalid station names being replaced with empty strings or NaN values.
        # To ensure only valid stations are processed, we create a mask that selects stations with non-null names.
        valid_station_mask = pd.notnull(stations["station"])
        waveform = waveform[:, valid_station_mask, :]
        stations = stations.loc[valid_station_mask]
        return xr.Dataset(
            data_vars={
                "waveform": (
                    ["component", "station", "time"],
                    waveform,
                ),
            },
            coords={
                "station": stations["station"],
                "component": ["x", "y", "z"],
                "x": ("station", stations["x"]),
                "y": ("station", stations["y"]),
                "lat": ("station", stations["lat"]),
                "lon": ("station", stations["lon"]),
            },
        )


def _postprocess_waveform(
    waveform: np.ndarray, rotation: float, dt: float
) -> np.ndarray:
    """Post-process a waveform array by rotating, reflecting and differentiating.

    This function:

    1. Rotates and reflects the waveform components so that x points
    north, y points east, and z points down.
    2. Differentiates the waveform array so velocities become
    accelerations.

    Parameters
    ----------
    waveform : np.ndarray
        The waveform array to process.
    rotation : float
        The model rotation.
    dt : float
        The model timestep.


    Returns
    -------
    np.ndarray
        The waveform array rotated and differentiated.
    """
    theta = np.radians(np.float32(rotation))
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [-np.sin(theta), -np.cos(theta), 0],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )
    # NOTE: not *strictly* a rotation matrix. It also swaps the
    # y-axis (so north is up), and reflects the vertical axis (why?
    # not entirely sure).
    # Using np.einsum to specify the summation axis
    # 'ij' for the rotation_matrix (rows, columns)
    # 'jkl' for the waveform (component, station, time)
    # The 'j' axis is summed over, and the result has shape 'ikl'
    rotated = np.einsum("ij,jkl->ikl", rotation_matrix, waveform)
    # NOTE: Rotation matrix R was originally designed to be applied
    # like W * R (where W is the (nt x 3) waveform for a station). We
    # swap the order of the time and component axes from the Rob
    # Graves original file, so we have to swap the order of arguments
    # in the dot product. You should transpose the rotation matrix in
    # general. But, the rotation matrix is symmetric, so this is
    # unnecessary.

    # Differentiate waveform to get acceleration
    acceleration = np.gradient(rotated, dt, axis=-1)
    return acceleration


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
    with open(header_file, "rb") as f:
        parser = LFSeisParser(f)
        header = parser.read_header()

    # Read all station data and waveforms in a single pass per file
    ds = xr.concat(
        [_read_lfseis_file(seis_file) for seis_file in seis_files],
        dim="station",
    ).assign_coords(
        time=np.arange(start_sec, start_sec + header.nt * header.dt, header.dt)
    )

    ds["waveform"] = (
        ("component", "station", "time"),
        _postprocess_waveform(ds["waveform"].values, header.rotation, header.dt),
    )

    # Set global attributes
    ds.attrs["dt"] = header.dt
    ds.attrs["resolution"] = header.resolution
    ds.attrs["rotation"] = header.rotation
    ds.attrs["units"] = "cm/s^2"
    # start second = -3 / flo = -3 * (0.5 / (5 * dx))
    ds.attrs["start_sec"] = -3 * (0.5 / (5 * header.resolution))

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

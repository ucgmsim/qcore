"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

import io
import multiprocessing
import os
from enum import StrEnum, auto
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyfftw
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import scipy as sp
import xarray as xr

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

    btype: Literal["highpass"] | Literal["lowpass"] = (
        "highpass" if band == Band.HIGHPASS else "lowpass"
    )
    return sp.signal.sosfiltfilt(
        sp.signal.butter(4, cutoff_frequencies, btype=btype, output="sos", fs=1.0 / dt),
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

    # PyFFTW sets the globals in the config module dynamically.
    # So type-checking must be ignored here.
    pyfftw.config.NUM_THREADS = cores  # type: ignore

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

    handle: io.BufferedIOBase
    length: int
    i4: str
    f4: str

    def __init__(self, handle: io.BufferedIOBase):  # noqa: D107 # numpydoc ignore=GL08
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
        # transparent to the user interacting with xarray anyhow, so the
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
) -> None:
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

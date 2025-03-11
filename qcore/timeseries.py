"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.signal import butter, sosfiltfilt

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
    tuple[str, np.dtype, np.dtype]
        Tuple containing the int and float types accounting for file endianness.
    """
    with open(seis_file, "rb") as f:
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
            nt = nt.byteswap()
            nstat = nstat.byteswap()
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
            nstat_file = np.fromfile(f, dtype=i4, count=1)[0]
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
        x_coords = station_headers["x"]
        y_coords = station_headers["y"]
        lat_coords = station_headers["lat"]
        lon_coords = station_headers["lon"]
        # Decode station names from UTF-8 encoded bytes.
        station_names = [
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

    valid_indices = np.array([i for i, name in enumerate(station_names) if name])
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


def _lfseis_header(header_file: Path) -> tuple[int, float, float]:
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
        f.seek(
            20
        )  # 4 * i4 for nstat, station index, x, y, z grid point of the first station
        nt = np.fromfile(f, dtype=i4, count=1)[0]
        dt, resolution, rotation = np.fromfile(f, dtype=f4, count=3)
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
    theta = np.radians(rotation)
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

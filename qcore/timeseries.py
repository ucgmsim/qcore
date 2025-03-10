"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

import os
from glob import glob
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Set, Union

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.signal import butter, resample, sosfiltfilt

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


def read_lfseis(outbin: Path | str) -> xr.Dataset:
    """Read LF station seismograms.

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
    # Constants from the original code
    HEAD_STAT = 48  # Header size per station
    N_COMP = 3  # Number of components (x, y, z)

    outbin = Path(outbin)
    seis_files = list(sorted(outbin.glob("*seis-*.e3d")))
    if not seis_files:
        raise ValueError(f"No LF seis files found in {outbin}")

    # Determine endianness by checking file size
    file_size = seis_files[0].stat().st_size
    with open(seis_files[0], "rb") as f:
        nstat, nt = np.fromfile(f, dtype="<i4", count=6)[0::5]
        if (
            file_size
            == 4 + np.int64(nstat) * HEAD_STAT + np.int64(nstat) * nt * N_COMP * 4
        ):
            endian = "<"
        elif (
            file_size
            == 4
            + np.int64(nstat.byteswap()) * HEAD_STAT
            + np.int64(nstat.byteswap()) * nt.byteswap() * N_COMP * 4
        ):
            endian = ">"
            nt = nt.byteswap()
            nstat = nstat.byteswap()
        else:
            raise ValueError(f"File is not an LF seis file: {seis_files[0]}")

        _i4 = f"{endian}i4"
        _f4 = f"{endian}f4"

        # Read common metadata
        dt, resolution, rotation = np.fromfile(f, dtype=_f4, count=3)

    # Calculate rotation matrix
    theta = np.radians(rotation)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [-np.sin(theta), -np.cos(theta), 0],
            [0, 0, -1],
        ]
    )

    # Load nstats to determine total size
    nstats = np.zeros(len(seis_files), dtype="i")
    for i, s in enumerate(seis_files):
        with open(s, "rb") as f:
            nstats[i] = np.fromfile(f, dtype=_i4, count=1)[0]

    # Container for station data
    total_stations = np.sum(nstats)
    stations = np.recarray(
        total_stations,
        dtype=[
            ("x", "i4"),
            ("y", "i4"),
            ("z", "i4"),
            ("seis_idx", "i4", 2),
            ("lat", "f4"),
            ("lon", "f4"),
            ("name", "U8"),
        ],
    )

    # Prepare arrays for waveform data
    velocity_waveforms = np.zeros((total_stations, nt, 3), dtype=np.float32)

    # Read station data and waveforms
    station_offset = 0
    for i, seis_file in enumerate(seis_files):
        with open(seis_file, "rb") as f:
            # Read number of stations in this file
            nstat_file = np.fromfile(f, dtype=_i4, count=1)[0]

            # Read station headers
            stations_data = np.fromfile(
                f,
                count=nstat_file,
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
                        "formats": [_i4, _i4, _i4, _i4, (_i4, 2), _f4, _f4, "|S8"],
                        "offsets": [0, 4, 8, 12, 16, 32, 36, 40],
                    }
                ),
            )

            # Set seis_idx for tracking file and position within file
            stations_data["seis_idx"][:, 0] = i
            stations_data["seis_idx"][:, 1] = np.arange(nstat_file)

            # Store station data
            for field in ["x", "y", "z", "seis_idx", "lat", "lon"]:
                stations[field][station_offset : station_offset + nstat_file] = (
                    stations_data[field]
                )

            # Convert station names from bytes to unicode
            for j in range(nstat_file):
                name_bytes = stations_data["name"][j]
                stations.name[station_offset + j] = name_bytes.decode(
                    "utf-8", errors="replace"
                ).strip("\x00")

            # Calculate position of time series data
            ts_pos = 4 + nstat_file * HEAD_STAT
            f.seek(ts_pos)

            # Read the waveform data
            for j in range(nstat_file):
                # Read first time step
                first_step = np.fromfile(f, dtype=f"3{endian}f4", count=1)[0]
                velocity_waveforms[station_offset + j, 0, :] = first_step

                # Read remaining time steps
                remaining_data = np.fromfile(f, dtype=f"3{endian}f4", count=(nt - 1))
                velocity_waveforms[station_offset + j, 1:, :] = remaining_data.reshape(
                    nt - 1, 3
                )

        station_offset += nstat_file

    # Check for duplicated stations and remove empty entries at the end
    if stations.name[-1] == "":
        last_valid_index = len(stations) - np.argmin((stations.name == "")[::-1])
        stations = stations[:last_valid_index]
        velocity_waveforms = velocity_waveforms[:last_valid_index]

    # Apply rotation matrix to velocity data
    rotated_velocity = np.einsum("ijk,kl->ijl", velocity_waveforms, rotation_matrix)

    # Convert velocity to acceleration
    acceleration_waveforms = _velocity_to_acceleration(rotated_velocity, dt)

    # Create time array
    time = np.arange(0, nt) * dt

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "waveforms": (
                ["station", "time", "component"],
                acceleration_waveforms,
            ),
        },
        coords={
            "station": np.array([name.strip() for name in stations.name]),
            "time": time,
            "component": ["x", "y", "z"],
            "x": ("station", stations.x),
            "y": ("station", stations.y),
            "z": ("station", stations.z),
            "lat": ("station", stations.lat),
            "lon": ("station", stations.lon),
            "file_index": ("station", stations.seis_idx[:, 0]),
            "station_index": ("station", stations.seis_idx[:, 1]),
        },
        attrs={
            "resolution": resolution,
            "rotation": rotation,
            "dt": dt,
        },
    )

    return ds

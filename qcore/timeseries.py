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
    timeseries : numpy.ndarray
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
    ns, nt, nc = timeseries.shape
    zeros = np.zeros((ns, 1, nc))
    shifted = np.concatenate([zeros, timeseries[:, :-1, :]], axis=1)

    # Calculate the backward difference and scale by 1/dt
    return (timeseries - shifted) / dt


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
    outbin = Path(outbin)
    seis_files = sorted(outbin.glob("*seis-*.e3d"))
    if not seis_files:
        raise ValueError(f"No LF seis files found in {outbin}")

    endian = "<"
    with open(seis_files[0], "rb") as f:  # Fixed: use first file in list
        # read first 6 integers
        nstat, nt = np.fromfile(f, dtype="<i4", count=6)[0::5]  # Fixed: removed self.
        # determine endianness
        if f.read(4) == nstat * 48 * nt * 4:
            endian = "<"
        elif f.read(4) == nstat.byteswap() * 48 * nt.byteswap() * 4:
            endian = ">"
            nt = nt.byteswap()
        else:
            raise ValueError(f"File is not an LF seis file: {seis_files[0]}")

        _f4 = f"{endian}f4"  # Fixed: removed self.
        _i4 = f"{endian}i4"  # Fixed: removed self.
        dt, resolution, rotation = np.fromfile(  # Fixed: removed self.
            f, dtype=_f4, count=3
        )

    x = []
    y = []
    z = []
    lat = []
    lon = []
    station = []
    velocity_waveforms = []

    for file in seis_files:
        with open(file, "rb") as f:
            nstat_file = np.fromfile(f, dtype=_i4, count=1)[0]  # Fixed: extract value
            # read station headers
            station_data = np.fromfile(  # Fixed: store result
                f,
                dtype=[
                    ("x", _i4),  # Fixed: use _i4 variable
                    ("y", _i4),
                    ("z", _i4),
                    ("lat", _f4),  # Fixed: use _f4 variable
                    ("lon", _f4),
                    ("name", "S8"),
                ],
                count=nstat_file,
            )

            x.extend(station_data["x"].tolist())
            y.extend(station_data["y"].tolist())
            z.extend(station_data["z"].tolist())
            lat.extend(station_data["lat"].tolist())
            lon.extend(station_data["lon"].tolist())
            station.extend(
                [
                    name.decode("utf-8", errors="replace").strip()
                    for name in station_data["name"]
                ]
            )

            waveform_data = np.fromfile(
                f, dtype=_f4, count=nstat_file * nt * 3
            ).reshape(nstat_file, nt, 3)
            velocity_waveforms.append(waveform_data)

    # Combine waveform data from all files
    if velocity_waveforms:
        combined_velocity_waveforms = np.concatenate(velocity_waveforms, axis=0)
    else:
        combined_velocity_waveforms = np.array([])

    theta = np.radians(rotation)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [-np.sin(theta), -np.cos(theta), 0],
            [0, 0, -1],
        ]
    )

    combined_velocity_waveforms = np.dot(combined_velocity_waveforms, rotation_matrix)
    combined_acceleration_waveforms = _velocity_to_acceleration(
        combined_velocity_waveforms, dt
    )

    # Create time array
    time = np.arange(0, nt) * dt

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "waveforms": (
                ["station", "time", "component"],
                combined_acceleration_waveforms,
            ),
        },
        coords={
            "station": station,
            "time": time,
            "component": ["x", "y", "z"],
            "x": ("station", x),
            "y": ("station", y),
            "z": ("station", z),
            "lat": ("station", lat),
            "lon": ("station", lon),
        },
        attrs={
            "resolution": resolution,
            "rotation": rotation,
            "dt": dt,
        },
    )

    return ds

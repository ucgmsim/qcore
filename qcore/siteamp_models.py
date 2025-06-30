"""
site amplification functionality from wcc_siteamp.c
Acceleration amplification models.

@date 24 June 2016
@author Viktor Polak
@contact viktor.polak@canterbury.ac.nz

Implemented Models
==============================
cb_amp (version = "2008"):
    Based on Campbell and Bozorgnia 2008 - added 24 June 2016
cb_amp (version = "2014"):
    Based on Campbell and Bozorgnia 2014 - added 22 September 2016
ba18_amp (version 2018):
    Based on Bayless Fourier Amplitude Spectra Empirical Model - added 11 June 2020

Usage
==============================
from siteamp_models import cb_amp (or *)
cb_amp(variables, ...)
"""

# math functions faster than numpy for non-vector data
from math import ceil, exp, log
import os

from numba import njit
import numpy as np
import pandas as pd

from qcore.uncertainties import distributions

ba18_coefs_df = None


def amplification_uncertainty(
    amplification_factors, frequencies, seed=None, std_dev_limit=2
):
    """
    Applies an uncertainty factor to each value in an amplification spectrum
    :param amplification_factors: numpy array of amplification factors
    :param frequencies: numpy array of frequencies for the amplification factors
    :param seed: Seed to use for the prng. None for a random seed
    :return: amplification factors with uncertainty applied
    """
    sigma_x = (
        0.6167
        - 0.1495 / (1 + np.exp(-3.6985 * np.log(frequencies / 0.7248)))
        + 0.3640 / (1 + np.exp(-2.2497 * np.log(frequencies / 13.457)))
    )
    amp_function_output = np.ones_like(amplification_factors)
    amp_function_output[1:] = distributions.truncated_log_normal(
        amplification_factors[1:],
        sigma_x,
        std_dev_limit=std_dev_limit,
        seed=seed,
    )
    return amp_function_output


def init_ba18():
    global ba18_coefs_df
    __location__ = os.path.realpath(os.path.dirname(__file__))
    ba18_coefs_file = os.path.join(
        __location__, "siteamp_coefs_files", "Bayless_ModelCoefs.csv"
    )
    ba18_coefs_df = pd.read_csv(ba18_coefs_file, index_col=0)


def nt2n(nt):
    """
    Length the fourier transform should be
    given timeseries length nt.
    """
    return int(2 ** ceil(log(nt) / log(2)))


@njit(cache=True)
def _fs_low(
    T_idx: int,
    vs30: float,
    a1100: float,
    c10: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
) -> np.ndarray:
    scon_c = 1.88
    scon_n = 1.18
    return c10[T_idx] * log(vs30 / k1[T_idx]) + k2[T_idx] * log(
        (a1100 + scon_c * exp(scon_n * log(vs30 / k1[T_idx]))) / (a1100 + scon_c)
    )


@njit(cache=True)
def _fs_mid(
    T_idx: int, vs30: float, c10: np.ndarray, k1: np.ndarray, k2: float
) -> np.ndarray:
    scon_n = 1.18
    return (c10[T_idx] + k2[T_idx] * scon_n) * log(vs30 / k1[T_idx])


@njit(cache=True)
def _fs_high(T_idx: int, c10: np.ndarray, k1: np.ndarray, k2: np.ndarray):
    """Compute site factor based on vs30 value - high code path

    Parameters
    ----------
    T_idx : nt
        The index to compute for.
    vs30 : float
        The reference vs30
    a1100, c10, k1, k2 : np.ndarray
        Parameters for calculation
    """
    scon_n = 1.18
    return (c10[T_idx] + k2[T_idx] * scon_n) * log(1100.0 / k1[T_idx])


@njit(cache=True)
def _compute_fs_value(
    T_idx: int,
    vs30: float,
    a1100: np.ndarray,
    c10: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
):
    """Compute site factor based on vs30 value

    Parameters
    ----------
    T_idx : nt
        The index to compute for.
    vs30 : float
        The reference vs30
    a1100, c10, k1, k2 : np.ndarray
        Parameters for calculation
    """
    if vs30 < k1[T_idx]:
        return _fs_low(T_idx, vs30, a1100, c10, k1, k2)
    elif vs30 < 1100.0:
        return _fs_mid(T_idx, vs30, c10, k1, k2)
    else:
        return _fs_high(T_idx, c10, k1, k2)


@njit(cache=True)
def _cb_amp(
    dt: float,
    n: int,
    vref: float,
    vsite: float,
    vpga: float,
    pga: float,
    version: int = 2014,  # Changed to integer
    flowcap: float = 0.0,
    fmin: float = 0.2,
    fmidbot: float = 0.5,
    fhightop: float = 10.0,
    fmax: float = 15.0,
) -> np.ndarray:
    """
    Numba translation of cb_amp.

    Parameters:
    -----------
    dt : float
        Time step
    n : int
        Number of points
    vref : float
        Reference Vs30 value (m/s)
    vsite : float
        Site Vs30 value (m/s)
    vpga : float
        Vs30 value for PGA calculation (m/s)
    pga : float
        Peak ground acceleration value (g)
    version : int, optional
        CB version (2008 or 2014), default 2014
    flowcap : float, optional
        Flow capacity constraint, default 0.0
    fmin, fmidbot, fhightop, fmax : float, optional
        Bandpass filter parameters

    Returns:
    --------
    results : ndarray
        Amplification factors, shape (output_length,)
        where output_length depends on dt and n
    """
    # Pre-computed frequency array
    freqs = 1.0 / np.array(
        [
            0.001,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
            1.50,
            2.00,
            3.00,
            4.00,
            5.00,
            7.50,
            10.0,
        ]
    )

    # Version-specific constants (converted to integer logic)
    if version == 2008:
        c10 = np.array(
            [
                1.058,
                1.058,
                1.102,
                1.174,
                1.272,
                1.438,
                1.604,
                1.928,
                2.194,
                2.351,
                2.460,
                2.587,
                2.544,
                2.133,
                1.571,
                0.406,
                -0.456,
                -0.82,
                -0.82,
                -0.82,
                -0.82,
                -0.82,
            ]
        )
    elif version == 2014:
        # named c11 in cb2014
        c10 = np.array(
            [
                1.090,
                1.094,
                1.149,
                1.290,
                1.449,
                1.535,
                1.615,
                1.877,
                2.069,
                2.205,
                2.306,
                2.398,
                2.355,
                1.995,
                1.447,
                0.330,
                -0.514,
                -0.848,
                -0.793,
                -0.748,
                -0.664,
                -0.576,
            ]
        )
    else:
        # Default to 2014 version instead of raising exception
        c10 = np.array(
            [
                1.090,
                1.094,
                1.149,
                1.290,
                1.449,
                1.535,
                1.615,
                1.877,
                2.069,
                2.205,
                2.306,
                2.398,
                2.355,
                1.995,
                1.447,
                0.330,
                -0.514,
                -0.848,
                -0.793,
                -0.748,
                -0.664,
                -0.576,
            ]
        )

    k1 = np.array(
        [
            865.0,
            865.0,
            865.0,
            908.0,
            1054.0,
            1086.0,
            1032.0,
            878.0,
            748.0,
            654.0,
            587.0,
            503.0,
            457.0,
            410.0,
            400.0,
            400.0,
            400.0,
            400.0,
            400.0,
            400.0,
            400.0,
            400.0,
        ]
    )
    k2 = np.array(
        [
            -1.186,
            -1.186,
            -1.219,
            -1.273,
            -1.346,
            -1.471,
            -1.624,
            -1.931,
            -2.188,
            -2.381,
            -2.518,
            -2.657,
            -2.669,
            -2.401,
            -1.955,
            -1.025,
            -0.299,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # Calculate a1100
    # fs1100 - fs_vpga for T=0
    fs_high_0 = _compute_fs_value(0, 1100.0, pga, c10, k1, k2)  # fs_high for T=0
    fs_vpga_0 = _compute_fs_value(0, vpga, pga, c10, k1, k2)  # fs_auto for T=0
    a1100 = pga * exp(fs_high_0 - fs_vpga_0)

    # Calculate amplification factors for each period
    # Replace generator with explicit loop
    ampf0 = np.zeros(freqs.size, dtype=np.float64)
    for T_idx in range(freqs.size):
        fs_site = _compute_fs_value(T_idx, vsite, a1100, c10, k1, k2)
        fs_ref = _compute_fs_value(T_idx, vref, a1100, c10, k1, k2)
        ampf0[T_idx] = exp(fs_site - fs_ref)

    # Apply flow capacity constraint
    # Replace try/except with conditional check
    flow_indices = np.where(freqs <= flowcap)[0]
    if len(flow_indices) > 0:
        T_cap = flow_indices[0]
        ampf0[T_cap:] = ampf0[T_cap]

    # Interpolate and apply bandpass filter
    ampv, ftfreq = interpolate_frequency(freqs, ampf0, dt, n)
    ampf = amp_bandpass(ampv, fhightop, fmax, fmidbot, fmin, ftfreq)

    return ampf


@njit(cache=True)
def _cb_amp_multi(
    dt,
    n,
    vref,
    vsite,
    vpga,
    pga,
    version=2014,
    flowcap=0.0,
    fmin=0.2,
    fmidbot=0.5,
    fhightop=10.0,
    fmax=15.0,
):
    """
    Numba version of cb_amp that processes multiple parameter sets.

    Parameters:
    -----------
    dt : float
        Time step
    n : int
        Number of points
    vref : array_like
        Reference Vs30 values (m/s) - shape (N,)
    vsite : array_like
        Site Vs30 values (m/s) - shape (N,)
    vpga : array_like
        Vs30 values for PGA calculation (m/s) - shape (N,)
    pga : array_like
        Peak ground acceleration values (g) - shape (N,)
    version : int, optional
        CB version (2008 or 2014), default 2014
    flowcap : float, optional
        Flow capacity constraint, default 0.0
    fmin, fmidbot, fhightop, fmax : float, optional
        Bandpass filter parameters

    Returns:
    --------
    results : ndarray
        Amplification factors, shape (N, output_length)
        where N is the number of input parameter sets
        and output_length depends on dt and n
    """

    # Convert inputs to arrays and get dimensions
    vref_arr = np.asarray(vref, dtype=np.float64)
    vsite_arr = np.asarray(vsite, dtype=np.float64)
    vpga_arr = np.asarray(vpga, dtype=np.float64)
    pga_arr = np.asarray(pga, dtype=np.float64)

    # Check that all arrays have the same shape
    n_cases = vref_arr.size

    # Flatten arrays to handle both 1D and scalar inputs
    vref_flat = vref_arr.flatten()
    vsite_flat = vsite_arr.flatten()
    vpga_flat = vpga_arr.flatten()
    pga_flat = pga_arr.flatten()

    # Determine output size by running one case
    # This is needed since we don't know the output size a priori
    sample_result = _cb_amp(
        dt,
        n,
        vref_flat[0],
        vsite_flat[0],
        vpga_flat[0],
        pga_flat[0],
        version,
        flowcap,
        fmin,
        fmidbot,
        fhightop,
        fmax,
    )
    output_length = sample_result.size

    # Pre-allocate results array
    results = np.zeros((n_cases, output_length), dtype=np.float64)
    results[0, :] = sample_result  # Store the sample result

    # Process remaining cases
    for i in range(1, n_cases):
        results[i, :] = _cb_amp(
            dt,
            n,
            vref_flat[i],
            vsite_flat[i],
            vpga_flat[i],
            pga_flat[i],
            version,
            flowcap,
            fmin,
            fmidbot,
            fhightop,
            fmax,
        )

    return results


def cb_amp_multi(
    df: pd.DataFrame,
    dt: float,
    n: int,
    version: int = 2014,
    flowcap: float = 0.0,
    fmin: float = 0.2,
    fmidbot: float = 0.5,
    fhightop: float = 10.0,
    fmax: float = 15.0,
    vref_col: str = "vref",
    vsite_col: str = "vsite",
    vpga_col: str = "vpga",
    pga_col: str = "pga",
):
    """
    Compute CB amplification factors for multiple parameter sets from a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the input parameters
    dt : float
        Time step for frequency domain
    n : int
        Number of points for frequency domain
    version : int, optional
        CB version (2008 or 2014), default 2014
    flowcap : float, optional
        Flow capacity constraint, default 0.0
    fmin, fmidbot, fhightop, fmax : float, optional
        Bandpass filter parameters
    vref_col, vsite_col, vpga_col, pga_col : str, optional
        Column names for the respective parameters

    Returns:
    --------
    results : numpy.ndarray
        Amplification factors, shape (len(df), output_length)
        Each row corresponds to one row in the input DataFrame

    Raises:
    -------
    KeyError
        If required columns are missing from the DataFrame
    ValueError
        If DataFrame is empty or contains invalid values
    """

    # Input validation
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check required columns exist
    required_cols = [vref_col, vsite_col, vpga_col, pga_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Extract arrays from DataFrame
    vref = df[vref_col].values
    vsite = df[vsite_col].values
    vpga = df[vpga_col].values
    pga = df[pga_col].values

    # Check for missing values
    arrays = [vref, vsite, vpga, pga]
    array_names = ["vref", "vsite", "vpga", "pga"]
    for arr, name in zip(arrays, array_names):
        if np.any(pd.isna(arr)):
            raise ValueError(f"Column '{name}' contains NaN values")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Column '{name}' contains infinite values")
        if np.any(arr <= 0):
            raise ValueError(f"Column '{name}' contains non-positive values")

    # Call the numba-accelerated function
    results = _cb_amp_multi(
        dt=dt,
        n=n,
        vref=vref,
        vsite=vsite,
        vpga=vpga,
        pga=pga,
        version=version,
        flowcap=flowcap,
        fmin=fmin,
        fmidbot=fmidbot,
        fhightop=fhightop,
        fmax=fmax,
    )
    return results


@njit(cache=True)
def interpolate_frequency(freqs: np.ndarray, ampf0: np.ndarray, dt: float, n: int):
    # frequencies of fourier transform
    ftfreq = np.arange(1, n / 2) * (1.0 / (n * dt))
    # transition indexes
    digi = np.digitize(freqs, ftfreq)[::-1]
    # only go down to 2nd frequency
    ampf0[0] = ampf0[1]
    freqs[0] = freqs[1]
    # special case, first frequency in transition range
    ftfreq0 = int(digi[0] == 0)
    # all possible dadf factors
    dadf0 = np.zeros(freqs.size)
    for i in range(1, freqs.size - 1):
        # start with dadf = 0.0 if no freq change at pos 0
        dadf0[-i - 1 - ftfreq0] = (ampf0[i] - ampf0[i + 1]) / np.log(
            freqs[i] / freqs[i + 1]
        )
    # calculate amplification factors
    digi = np.append(digi, [np.float64(ftfreq.size)])
    a0 = np.zeros(ftfreq.size)
    f0 = np.zeros(ftfreq.size)
    dadf = np.zeros(ftfreq.size)
    start = 0
    start_dadf = 0
    for i in range(freqs.size):
        end = max(start + 1, digi[i + 1])
        end_dadf = max(start_dadf + 1, digi[i + ftfreq0])
        a0[start:end] = ampf0[-i - 1]
        f0[start:end] = freqs[-i - 1]
        dadf[start_dadf:end_dadf] = dadf0[i]
        start = max(end, digi[i + 1])
        start_dadf = end_dadf
    ampv = a0 + dadf * np.log(ftfreq / f0)
    return ampv, ftfreq


def get_ft_freq(dt, n):
    return np.arange(1, n / 2) * (1.0 / (n * dt))


@njit(cache=True)
def amp_bandpass(
    ampv: np.ndarray,
    fhightop: float,
    fmax: float,
    fmidbot: float,
    fmin: float,
    ftfreq: np.ndarray,
):
    # default amplification is 1.0 (keeping values the same)
    ampf = np.ones(ftfreq.size + 1, dtype=np.float64)
    # amplification factors applied differently at different bands
    ampf[1:] += (
        np.where(
            (ftfreq >= fhightop) & (ftfreq < fmax),
            -1 + ampv + np.log(ftfreq / fhightop) * (1.0 - ampv) / log(fmax / fhightop),
            0,
        )
        + np.where((ftfreq >= fmidbot) & (ftfreq < fhightop), -1 + ampv, 0)
        + np.where(
            (ftfreq >= fmin) & (ftfreq < fmidbot),
            np.log(ftfreq / fmin) * (ampv - 1.0) / log(fmidbot / fmin),
            0,
        )
    )
    return ampf


def ba18_amp(
    dt,
    n,
    vref,
    vs,
    vpga,
    pga,
    version=None,
    flowcap=0.0,
    fmin=0.00001,
    fmidbot=0.0001,
    fmid=1.0,
    fhigh=10 / 3.0,
    fhightop=999.0,
    fmax=1000,
):
    """

    :param dt:
    :param n:
    :param vref: Reference vs used for waveform
    :param vs: Actual vs30 value of the site
    :param vpga: Reference vs for HF
    :param pga: PGA value from HF
    :param version: unused
    :param flowcap: unused
    :param fmin:
    :param fmidbot:
    :param fmid:
    :param fhigh:
    :param fhightop:
    :param fmax:
    :return:
    """
    if vs > 1000:
        vs = 999  # maximum vs30 supported by the model is 999, so caps the vsite to that value

    # overwrite these two values to their default value, so changes by the caller function do not override this
    fmin = 0.00001
    fmidbot = 0.0001

    ref, __ = ba_18_site_response_factor(vref, pga, vpga)
    vsite, freqs = ba_18_site_response_factor(vs, pga, vpga)

    amp = np.exp(vsite - ref)
    ftfreq = get_ft_freq(dt, n)

    ampi = np.interp(ftfreq, freqs, amp)
    ampfi = amp_bandpass(
        ampi, fhightop, fmax, fmidbot, fmin, ftfreq
    )  # With these values it is effectively no filtering
    ampfi[0] = ampfi[1]  # Copies the first value, which isn't necessarily 1

    return ampfi


def ba_18_site_response_factor(vs, pga, vpga, f=None):
    vsref = 1000

    if ba18_coefs_df is None:
        print(
            "You need to call the init_ba18 function before using the site_amp functions"
        )
        exit()
    coefs = type("coefs", (object,), {})  # creates a custom object for coefs

    if f is None:
        freq_indices = ...
        coefs.freq = ba18_coefs_df.index.values
    else:
        freq_index = np.argmin(np.abs(ba18_coefs_df.index.values - f))
        if freq_index > f:
            freq_indices = [freq_index - 1, freq_index]
        else:
            freq_indices = [freq_index, freq_index + 1]
        coefs.freq = ba18_coefs_df.index.values[freq_indices]

    # Non-linear site parameters
    coefs.f3 = ba18_coefs_df.f3.values[freq_indices]
    coefs.f4 = ba18_coefs_df.f4.values[freq_indices]
    coefs.f5 = ba18_coefs_df.f5.values[freq_indices]
    coefs.b8 = ba18_coefs_df.c8.values[freq_indices]

    lnfas = coefs.b8 * np.log(min(vs, 1000) / vsref)

    fas_lin = np.exp(lnfas)

    if f is None:
        # Extrapolate to 100 Hz
        maxfreq = 23.988321
        imax = np.where(coefs.freq == maxfreq)[0][0]
        fas_maxfreq = fas_lin[imax]
        # Kappa
        kappa = np.exp(-0.4 * np.log(vs / 760) - 3.5)
        # Diminuition operator
        D = np.exp(-np.pi * kappa * (coefs.freq[imax:] - maxfreq))

        fas_lin = np.append(fas_lin[:imax], fas_maxfreq * D)
        lnfas = np.log(fas_lin)

    # Compute non-linear site response
    if pga is not None:
        v_model_ref = 760
        if vpga != v_model_ref:
            IR = pga * exp(
                ba_18_site_response_factor(
                    vs=v_model_ref, pga=None, vpga=v_model_ref, f=5
                )[0]
                - ba_18_site_response_factor(vs=vpga, pga=pga, vpga=v_model_ref, f=5)[0]
            )
        else:
            IR = pga

        coefs.f2 = coefs.f4 * (
            np.exp(coefs.f5 * (min(vs, v_model_ref) - 360))
            - np.exp(coefs.f5 * (v_model_ref - 360))
        )
        fnl0 = coefs.f2 * np.log((IR + coefs.f3) / coefs.f3)

        fnl0[np.where(fnl0 == min(fnl0))[0][0] :] = min(fnl0)
        result = fnl0 + lnfas
    else:
        result = lnfas

    if f is not None:
        return np.interp(f, coefs.freq, result), f
    else:
        return result, coefs.freq


def hashash_get_pgv(fnorm, mag, rrup, ztor):
    # get the EAS_rock at 5 Hz (no c8, c11 terms)
    b4a = -0.5
    mbreak = 6.0

    coefs = type("coefs", (object,), {})  # creates a custom object for coefs
    coefs.freq = ba18_coefs_df.index.values

    coefs.b1 = ba18_coefs_df.c1.values
    coefs.b2 = ba18_coefs_df.c2.values
    coefs.b3quantity = ba18_coefs_df["(c2-c3)/cn"].values
    coefs.bn = ba18_coefs_df.cn.values
    coefs.bm = ba18_coefs_df.cM.values
    coefs.b4 = ba18_coefs_df.c4.values
    coefs.b5 = ba18_coefs_df.c5.values
    coefs.b6 = ba18_coefs_df.c6.values
    coefs.bhm = ba18_coefs_df.chm.values
    coefs.b7 = ba18_coefs_df.c7.values
    coefs.b8 = ba18_coefs_df.c8.values
    coefs.b9 = ba18_coefs_df.c9.values
    coefs.b10 = ba18_coefs_df.c10.values
    # row = df.iloc[df.index == 5.011872]
    i5 = np.where(coefs.freq == 5.011872)
    lnfasrock5Hz = coefs.b1[i5]
    lnfasrock5Hz += coefs.b2[i5] * (mag - mbreak)
    lnfasrock5Hz += coefs.b3quantity[i5] * np.log(
        1 + np.exp(coefs.bn[i5] * (coefs.bm[i5] - mag))
    )
    lnfasrock5Hz += coefs.b4[i5] * np.log(
        rrup + coefs.b5[i5] * np.cosh(coefs.b6[i5] * max(mag - coefs.bhm[i5], 0))
    )
    lnfasrock5Hz += (b4a - coefs.b4[i5]) * np.log(np.sqrt(rrup**2 + 50**2))
    lnfasrock5Hz += coefs.b7[i5] * rrup
    lnfasrock5Hz += coefs.b9[i5] * min(ztor, 20)
    lnfasrock5Hz += coefs.b10[i5] * fnorm
    # Compute PGA_rock extimate from 5 Hz FAS
    IR = np.exp(1.238 + 0.846 * lnfasrock5Hz)
    return IR

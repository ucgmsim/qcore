"""Site amplification models."""


from enum import Enum

import numpy as np
import pandas as pd
from numba import njit

from qcore.uncertainties import distributions


def amplification_uncertainty(
    amplification_factors: np.ndarray,
    frequencies: np.ndarray,
    seed: int | None = None,
    std_dev_limit: int = 2,
) -> np.ndarray:
    """Compute uncertainties for site amplification models.

    Parameters
    ----------
    amplification_factors : np.ndarray
        An array of amplification factors to compute for.
    frequencies : np.ndarray
        An array of amplification frequencies.
    seed : int | None
        Seed for `distributions.truncated_trucated_log_normal`.
    std_dev_limit : int
        The +/- standard deviation limit for uncertainties.

    Returns
    -------
    np.ndarray
        An array of sampled amplification factors, with a mean on the
        values of `amplification_factors` and standard deviation
        determined by the frequencies. Uncertainties are distributed
        log normally about the mean.
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


@njit
def _fs_low(
    t_idx: int,
    vs30: float,
    a1100: np.ndarray,
    c10: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
) -> np.ndarray:
    """Compute site factor based on vs30 value - low code path

    Parameters
    ----------
    t_idx : nt
        The index to compute for.
    vs30 : float
        The reference vs30
    a1100, c10, k1, k2 : np.ndarray
        Parameters for calculation

    Returns
    -------
    np.ndarray
         Site amplification factor.
    """

    scon_c = 1.88
    scon_n = 1.18
    return c10[t_idx] * np.log(vs30 / k1[t_idx]) + k2[t_idx] * np.log(
        (a1100 + scon_c * np.exp(scon_n * np.log(vs30 / k1[t_idx]))) / (a1100 + scon_c)
    )


@njit
def _fs_mid(
    t_idx: int, vs30: float, c10: np.ndarray, k1: np.ndarray, k2: np.ndarray
) -> np.ndarray:
    """Compute site factor based on vs30 value - mid code path

    Parameters
    ----------
    t_idx : int
        The index to compute for.
    vs30 : float
        The reference vs30
    c10, k1, k2 : np.ndarray
        Parameters for calculation

    Returns
    -------
    np.ndarray
         Site amplification factor.

    """

    scon_n = 1.18
    return (c10[t_idx] + k2[t_idx] * scon_n) * np.log(vs30 / k1[t_idx])


@njit
def _fs_high(t_idx: int, c10: np.ndarray, k1: np.ndarray, k2: np.ndarray):
    """Compute site factor based on vs30 value - high code path

    Parameters
    ----------
    t_idx : int
        The index to compute for.
    c10, k1, k2 : np.ndarray
        Parameters for calculation

    Returns
    -------
    np.ndarray
         Site amplification factor.

    """
    scon_n = 1.18
    return (c10[t_idx] + k2[t_idx] * scon_n) * np.log(1100.0 / k1[t_idx])


@njit
def _compute_fs_value(
    t_idx: int,
    vs30: float,
    a1100: np.ndarray,
    c10: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
):
    """Compute site factor based on vs30 value

    Parameters
    ----------
    t_idx : int
        The index to compute for.
    vs30 : float
        The reference vs30
    a1100, c10, k1, k2 : np.ndarray
        Parameters for calculation

    Returns
    -------
    np.ndarray
         Site amplification factor.
    """
    if vs30 < k1[t_idx]:
        return _fs_low(t_idx, vs30, a1100, c10, k1, k2)
    elif vs30 < 1100.0:
        return _fs_mid(t_idx, vs30, c10, k1, k2)
    else:
        return _fs_high(t_idx, c10, k1, k2)


AMPLIFICATION_FREQUENCIES = 1.0 / np.array(
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


@njit
def _cb_amp(
    vref: float,
    vsite: float,
    vpga: float,
    pga: float,
    version: int = 2014,
    flowcap: float = 0.0,
    freqs: np.ndarray = AMPLIFICATION_FREQUENCIES,
) -> np.ndarray:
    """
    Numba translation of cb_amp.

    Parameters
    ----------
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
    freqs : np.ndarray, optional
        Frequencies to compute amplification values for using model
        explicitly.

    Returns
    -------
    np.ndarray
        Amplification factors, shaped like `freqs`

    """

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
    a1100 = pga * np.exp(fs_high_0 - fs_vpga_0)

    # Calculate amplification factors for each period
    ampf0 = np.zeros_like(freqs)
    t_idx = 0
    while t_idx < freqs.size and freqs[t_idx] > flowcap:
        fs_site = _compute_fs_value(t_idx, vsite, a1100, c10, k1, k2)
        fs_ref = _compute_fs_value(t_idx, vref, a1100, c10, k1, k2)
        ampf0[t_idx] = np.exp(fs_site - fs_ref)
        t_idx += 1
    ampf0[t_idx:] = ampf0[t_idx]

    return ampf0


class CBModelVersion(Enum):
    """Campbell and Bozorgnia model versions"""

    CB2008 = 2008
    CB2014 = 2014


@njit(parallel=True)
def _cb_amp_multi(
    vref: np.ndarray,
    vsite: np.ndarray,
    vpga: np.ndarray,
    pga: np.ndarray,
    version: int,
    flowcap: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Numba version of cb_amp that processes multiple parameter sets.

    Parameters
    ----------
    vref : array_like
        Reference Vs30 values (m/s) - shape (N,)
    vsite : array_like
        Site Vs30 values (m/s) - shape (N,)
    vpga : array_like
        Vs30 values for PGA calculation (m/s) - shape (N,)
    pga : array_like
        Peak ground acceleration values (g) - shape (N,)
    version : CBModelVersion
        CB version (2008 or 2014)
    flowcap : float
        Flow capacity constraint
    freqs : np.ndarray
        Frequencies to compute amplification values for using model
        explicitly.

    Returns
    -------
    np.ndarray
        Amplification factors, shape (N, output_length)
        where N is the number of input parameter sets
        and output_length depends on dt and n

    See Also
    --------
    cb_amp_multi : Public interface to this function. More details on the model are explained here.
    """

    # Convert inputs to arrays and get dimensions
    vref_arr = np.asarray(vref)
    vsite_arr = np.asarray(vsite)
    vpga_arr = np.asarray(vpga)
    pga_arr = np.asarray(pga)

    n_cases = vref_arr.size

    # Flatten arrays to handle both 1D and scalar inputs
    vref_flat = vref_arr.flatten()
    vsite_flat = vsite_arr.flatten()
    vpga_flat = vpga_arr.flatten()
    pga_flat = pga_arr.flatten()

    # Pre-allocate results array
    results = np.zeros((n_cases, freqs.size), dtype=vref_arr.dtype)

    for i in range(n_cases):
        results[i, :] = _cb_amp(
            vref_flat[i],
            vsite_flat[i],
            vpga_flat[i],
            pga_flat[i],
            version,
            flowcap,
            freqs,
        )

    return results


def cb_amp_multi(
    df: pd.DataFrame,
    version: CBModelVersion = CBModelVersion.CB2014,
    flowcap: float = 0.0,
    vref_col: str = "vref",
    vsite_col: str = "vsite",
    vpga_col: str = "vpga",
    pga_col: str = "pga",
    freqs: np.ndarray = AMPLIFICATION_FREQUENCIES,
):
    """Compute CB amplification factors for multiple parameter sets from a pandas DataFrame.

    This code compute site-amplification factors, which adjust
    response spectra computed for a generic site into response spectra
    for a site with known Vs30 values. The model used is the CB2014
    (or CB2008) models _[0], an empirical model that predicts pSA at
    sites, which we use to scale the FAS of the high-frequency
    waveforms.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the input parameters
    version : int, optional
        CB version (2008 or 2014), default 2014
    flowcap : float, optional
        Flow capacity constraint, default 0.0
    vref_col, vsite_col, vpga_col, pga_col : str, optional
        Column names for the respective parameters
    freqs : np.ndarray
        Frequencies to compute amplification values for using model
        explicitly.

    Returns
    -------
    np.ndarray
        Amplification factors, shape (len(df), output_length)
        Each row corresponds to one row in the input DataFrame

    Raises
    ------
    KeyError
        If required columns are missing from the DataFrame
    ValueError
        If DataFrame is empty or contains invalid values

    References
    ----------
    .. [0] Campbell KW, Bozorgnia Y. NGA-West2 Ground Motion Model for
    the Average Horizontal Components of PGA, PGV, and 5% Damped
    Linear Acceleration Response Spectra. Earthquake Spectra.
    2014;30(3):1087-1115. doi:10.1193/062913EQS175M
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
        if not np.isdtype(arr.dtype, 'real floating'):
            raise ValueError(f"Column '{name}' has incorrect kind, must be real floating")

    # Use pga for reference dtype because it is more reliably a float,
    # where vref can sometimes be an int.
    freqs = freqs.astype(pga.dtype)
    # Call the numba-accelerated function
    results = _cb_amp_multi(
        vref=vref,
        vsite=vsite,
        vpga=vpga,
        pga=pga,
        version=version.value,
        flowcap=flowcap,
        freqs=freqs,
    )
    return results


def cb2014_to_fas_amplification_factors(
    ampf0: np.ndarray,
    dt: float,
    n: int,
    fmin: float = 0.2,
    fmidbot: float = 0.5,
    fhightop: float = 10.0,
    fmax: float = 15.0,
    freqs: np.ndarray = AMPLIFICATION_FREQUENCIES,
) -> np.ndarray:
    """Converts the CB2014 site-amplification factors for suitable use with FAS.

    CB2014 predicts site-amplification for pSA, but we need it for FAS
    in simulations. This function interpolates frequencies, and
    applies a bandpass filter to convert between SA-based
    amplification factors and FAS amplification factors.

    Parameters
    ----------
    freqs : np.ndarray
        The SA frequencies corresponding to site-amplification factors.
    ampf0 : np.ndarray
        The amplification factors.
    dt : float
        The timestep delta for the waveforms to amplify.
    n : int
        The number of timesteps of the waveforms.
    fmin, fmidbot, fhightop, fmax : float, optional
        Bandpass filter parameters, see `amp_bandpass`.

    Returns
    -------
    np.ndarray
        The amplification factors `ampf0` interpolated to FFT frequencies
        matching `dt` and `n`, and amplified according to the bandpass
        filter `amp_bandpass`.
    """
    interpolated, ftfreq = interpolate_amplification_factors(freqs, ampf0, dt, n)
    return amp_bandpass(interpolated, fhightop, fmax, fmidbot, fmin, ftfreq)


@njit(
    parallel=True,
)
def interp_2d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Perform interpolation of a vector-valued function f at `x` with interpolation nodes `xp` and `fp`.

    This handles the case where `fp` is not 1-D. Interpolation is
    performed in parallel over the last axis.

    Parameters
    ----------
    x : np.ndarray, 1-D
        The points to interpolate.
    xp : np.ndarray, 1-D
        The interpolation nodes for `fp`.
    fp : np.ndarray, 2-D
        The function values at `xp`. The last axis must be the same as
        the length of `xp`.

    Returns
    -------
    np.ndarray
        The function `f` interpolated at `x`. Has the same `dtype` as
        `fp`.
    """
    out = np.zeros((fp.shape[0], len(x)), dtype=fp.dtype)
    for i in range(0, fp.shape[0]):
        out[i] = np.interp(x, xp, fp[i])
    return out


def interpolate_amplification_factors(
    freqs: np.ndarray,
    ampf0: np.ndarray,
    dt: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform logarithmic interpolation of amplification factors.

    Amplification factors are interpolated to frequencies typical for
    Fourier amplitude spectra of waveforms. Interpolation is performed in log-frequency space:

    A(f) ~ log(f)

    Parameters
    ----------
    freqs : np.ndarray
        The frequencies to interpolate between.
    ampf0 : np.ndarray
        The amplification factors to interpolate.
    dt : float
         Timestep.
    n : int
        The waveform length.

    Returns
    -------
    ampv : np.ndarray
        The interpolated amplification factors.
    ftfreq : np.ndarray
        The interpolated frequencies.
    """
    # Handle both 1-D and 2-D inputs uniformly
    ampf0 = np.atleast_2d(ampf0)

    # Copy inputs to avoid in-place modification
    freqs = freqs.copy()
    ampf0 = ampf0.copy()

    # Match original behaviour: discard first entry by overwriting with second
    freqs[0] = freqs[1]
    ampf0[:, 0] = ampf0[:, 1]

    # Ensure ascending order for np.interp
    freqs = freqs[::-1]
    ampf0 = ampf0[:, ::-1]

    # Target Fourier frequencies (skip 0 and Nyquist)
    ftfreq = np.fft.rfftfreq(n, dt)[1:-1].ravel().astype(freqs.dtype)

    # Interpolate in log-frequency space
    log_fftfreq = np.log(ftfreq)
    log_cb_freq = np.log(freqs)
    ampv = interp_2d(log_fftfreq, log_cb_freq, ampf0)

    return ampv, ftfreq.astype(freqs.dtype)


@njit(
    parallel=True,
)
def amp_bandpass(
    ampv: np.ndarray,
    fhightop: float,
    fmax: float,
    fmidbot: float,
    fmin: float,
    fftfreq: np.ndarray,
) -> np.ndarray:
    """Frequency-dependent amplification adjustment for site amplification factors.

    This function applies frequency-dependent amplification adjustments
    to site amplification factors in the frequency range [fmin, fmax].
    The adjustments are logarithmic in the ranges [fhightop, fmax) and
    (fmin, fmidbot]. The purpose of this adjustment is twofold:

    1. To address inconsistencies between the modelling of spectral
       acceleration (SA) in the CB2014 model and Fourier amplitude
       spectra (FAS) at high frequencies.
    2. To avoid double-counting low-frequency site amplification
       effects already captured by physics-based ground motion
       simulations and 3D velocity models.

    See _[0] for further details on why this filtering is applied.

    Parameters
    ----------
    ampv : np.ndarray
        A 1D array of raw amplification values from the CB2014 model.
        These values are adjusted based on the specified frequency
        ranges.
    fhightop : float
        The high-pass cutoff frequency. Amplification transitions
        logarithmically between [fhightop, fmax).
    fmax : float
        The maximum frequency. Amplification is attenuated above this
        frequency.
    fmidbot : float
        The low-pass cutoff frequency. Amplification transitions
        logarithmically between (fmin, fmidbot].
    fmin : float
        The minimum frequency. Amplification is set to 1 below this frequency.
    ftfreq : np.ndarray
        A 1D array of Fourier transform frequencies corresponding to
        the amplification values.

    Returns
    -------
    np.ndarray
        A 1D array of amplification factors with frequency-dependent
        adjustments applied.

    Notes
    -----
    The amplification adjustments are applied as follows:
    - For frequencies in [fhightop, fmax), amplification decreases
      logarithmically.
    - For frequencies in [fmidbot, fhightop), amplification is
      unchanged.
    - For frequencies in [fmin, fmidbot), amplification increases
      logarithmically.

    References
    ----------
    [0] Kuncar, Felipe, et al. Methods to account for shallow site
    effects in hybrid broadband ground-motion simulations. Earthquake
    Spectra 41.2 (2025): 1272-1313."""
    ampf = np.empty((ampv.shape[0], fftfreq.size + 1), dtype=ampv.dtype)
    ampf[:, 0] = 1.0

    log_fmax_diff = (np.log(fftfreq) - np.log(fhightop)) / (
        np.log(fmax) - np.log(fhightop)
    )
    log_fmin_diff = (np.log(fftfreq) - np.log(fmin)) / (np.log(fmidbot) - np.log(fmin))

    for i in range(ampf.shape[0]):
        for j in range(1, fftfreq.size + 1):
            if fhightop <= fftfreq[j - 1] < fmax:
                ampf[i, j] = ampv[i, j] + log_fmax_diff[j] * (1 - ampv[i, j])
            elif fmidbot <= fftfreq[j - 1] < fhightop:
                ampf[i, j] = ampv[i, j]
            elif fmin <= fftfreq[j - 1] < fmidbot:
                ampf[i, j] = 1.0 + log_fmin_diff[j] * (ampv[i, j] - 1.0)
            else:
                ampf[i, j] = 1.0

    return ampf

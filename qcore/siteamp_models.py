"""Site amplification models."""

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


@njit(cache=True)
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


@njit(cache=True)
def _fs_mid(
    t_idx: int, vs30: float, c10: np.ndarray, k1: np.ndarray, k2: np.ndarray
) -> np.ndarray:
    """Compute site factor based on vs30 value - mid code path

    Parameters
    ----------
    t_idx : nt
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


@njit(cache=True)
def _fs_high(t_idx: int, c10: np.ndarray, k1: np.ndarray, k2: np.ndarray):
    """Compute site factor based on vs30 value - high code path

    Parameters
    ----------
    t_idx : nt
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


@njit(cache=True)
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
    if vs30 < k1[t_idx]:
        return _fs_low(t_idx, vs30, a1100, c10, k1, k2)
    elif vs30 < 1100.0:
        return _fs_mid(t_idx, vs30, c10, k1, k2)
    else:
        return _fs_high(t_idx, c10, k1, k2)


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

    Parameters
    ----------
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

    Returns
    -------
    np.ndarray
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
    a1100 = pga * np.exp(fs_high_0 - fs_vpga_0)

    # Calculate amplification factors for each period
    ampf0 = np.zeros(freqs.size, dtype=np.float64)
    for t_idx in range(freqs.size):
        fs_site = _compute_fs_value(t_idx, vsite, a1100, c10, k1, k2)
        fs_ref = _compute_fs_value(t_idx, vref, a1100, c10, k1, k2)
        ampf0[t_idx] = np.exp(fs_site - fs_ref)

    # Apply flow capacity constraint
    flow_indices = np.where(freqs <= flowcap)[0]
    if len(flow_indices) > 0:
        t_cap = flow_indices[0]
        ampf0[t_cap:] = ampf0[t_cap]

    # Interpolate and apply bandpass filter
    ampv, ftfreq = interpolate_frequency(freqs, ampf0, dt, n)
    ampf = amp_bandpass(ampv, fhightop, fmax, fmidbot, fmin, ftfreq)

    return ampf


@njit(cache=True, parallel=True)
def _cb_amp_multi(
    dt: float,
    n: int,
    vref: np.ndarray,
    vsite: np.ndarray,
    vpga: np.ndarray,
    pga: np.ndarray,
    version: int = 2014,
    flowcap: float = 0.0,
    fmin: float = 0.2,
    fmidbot: float = 0.5,
    fhightop: float = 10.0,
    fmax: float = 15.0,
) -> np.ndarray:
    """
    Numba version of cb_amp that processes multiple parameter sets.

    Parameters
    ----------
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

    Returns
    -------
    np.ndarray
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

    Parameters
    ----------
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
    .. [0] Campbell KW, Bozorgnia Y. NGA-West2 Ground Motion Model for the Average Horizontal Components of PGA, PGV, and 5% Damped Linear Acceleration Response Spectra. Earthquake Spectra. 2014;30(3):1087-1115. doi:10.1193/062913EQS175M

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
def interpolate_frequency(
    freqs: np.ndarray, ampf0: np.ndarray, dt: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    # frequencies of fourier transform
    """Perform logarithmic interpolation of amplification factors.

    Amplification factors are interpolated to frequencies typical for
    Fourier amplitude spectra of waveforms.

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


@njit(cache=True)
def amp_bandpass(
    ampv: np.ndarray,
    fhightop: float,
    fmax: float,
    fmidbot: float,
    fmin: float,
    ftfreq: np.ndarray,
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
      constant.
    - For frequencies in [fmin, fmidbot), amplification increases
      logarithmically.

    References
    ----------
    [0] Kuncar, Felipe, et al. Methods to account for shallow site
    effects in hybrid broadband ground-motion simulations. Earthquake
    Spectra 41.2 (2025): 1272-1313."""
    # default amplification is 1.0 (keeping values the same)
    ampf = np.ones(ftfreq.size + 1, dtype=np.float64)
    # amplification factors applied differently at different bands
    ampf[1:] += (
        np.where(
            (ftfreq >= fhightop) & (ftfreq < fmax),
            -1
            + ampv
            + np.log(ftfreq / fhightop) * (1.0 - ampv) / np.log(fmax / fhightop),
            0,
        )
        + np.where((ftfreq >= fmidbot) & (ftfreq < fhightop), -1 + ampv, 0)
        + np.where(
            (ftfreq >= fmin) & (ftfreq < fmidbot),
            np.log(ftfreq / fmin) * (ampv - 1.0) / np.log(fmidbot / fmin),
            0,
        )
    )
    return ampf

"""Site amplification models."""

from enum import Enum

import numpy as np
import pandas as pd

from qcore.uncertainties import distributions

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


def _compute_fs_value(
    vs30: np.ndarray | float,
    a1100: np.ndarray | float,
    c10: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
) -> np.ndarray:
    """Compute site factor based on vs30 value.

    Vectorised over sites (leading axis) and periods (last axis): `vs30`
    and `a1100` broadcast against the coefficient arrays `c10`, `k1` and
    `k2`.

    Parameters
    ----------
    vs30 : np.ndarray | float
        The reference vs30, e.g. shape (N, 1).
    a1100 : np.ndarray | float
        Median PGA on rock (Vs30 = 1100 m/s), e.g. shape (N, 1).
    c10, k1, k2 : np.ndarray
        Model coefficients per period, shape (T,).

    Returns
    -------
    np.ndarray
         Site amplification factor, shape broadcast from the inputs, e.g. (N, T).
    """
    scon_c = 1.88
    scon_n = 1.18
    vs30 = np.asarray(vs30, dtype=np.float64)
    a1100 = np.asarray(a1100, dtype=np.float64)
    log_vs30_k1 = np.log(vs30 / k1)
    fs_low = c10 * log_vs30_k1 + k2 * np.log(
        (a1100 + scon_c * np.exp(scon_n * log_vs30_k1)) / (a1100 + scon_c)
    )
    fs_mid = (c10 + k2 * scon_n) * log_vs30_k1
    fs_high = (c10 + k2 * scon_n) * np.log(1100.0 / k1)
    return np.where(
        vs30 < k1,
        fs_low,
        np.where(vs30 < 1100.0, fs_mid, np.broadcast_to(fs_high, fs_mid.shape)),
    )


class CBModelVersion(Enum):
    """Campbell and Bozorgnia model versions"""

    CB2008 = 2008
    CB2014 = 2014


def _cb_amp_multi(
    vref: np.ndarray,
    vsite: np.ndarray,
    vpga: np.ndarray,
    pga: np.ndarray,
    version: int,
    flowcap: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Vectorised cb_amp that processes multiple parameter sets.

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
    version : int
        CB version (2008 or 2014)
    flowcap : float
        Flow capacity constraint
    freqs : np.ndarray
        Frequencies to compute amplification values for using model
        explicitly.

    Returns
    -------
    np.ndarray
        Amplification factors, shape (N, freqs.size)
        where N is the number of input parameter sets.

    Raises
    ------
    ValueError
        If `version` is not 2008 or 2014.

    See Also
    --------
    cb_amp_multi : Public interface to this function. More details on the model are explained here.
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
    else:
        raise ValueError(f"Unsupported CB model version: {version}")

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

    vref_arr = np.asarray(vref)
    # Column vectors of shape (N, 1) so they broadcast against the (T,)
    # per-period coefficients.
    vref_col = vref_arr.reshape(-1, 1)
    vsite_col = np.asarray(vsite).reshape(-1, 1)
    vpga_col = np.asarray(vpga).reshape(-1, 1)
    pga_col = np.asarray(pga).reshape(-1, 1).astype(np.float64)

    # Calculate a1100 from the T=0 (PGA) coefficients:
    # fs1100 - fs_vpga for T=0
    coeffs_0 = (c10[:1], k1[:1], k2[:1])
    fs_high_0 = _compute_fs_value(1100.0, pga_col, *coeffs_0)
    fs_vpga_0 = _compute_fs_value(vpga_col, pga_col, *coeffs_0)
    a1100 = pga_col * np.exp(fs_high_0 - fs_vpga_0)

    # Amplification factors are computed for leading frequencies above
    # flowcap. The remaining entries are filled with the value at the first
    # frequency below the cap, which has not been computed and is therefore 0.
    above_cap = np.asarray(freqs) > flowcap
    n_computed = freqs.size if above_cap.all() else int(np.argmin(above_cap))

    results = np.zeros((vref_arr.size, freqs.size), dtype=vref_arr.dtype)
    coeffs = (c10[:n_computed], k1[:n_computed], k2[:n_computed])
    fs_site = _compute_fs_value(vsite_col, a1100, *coeffs)
    fs_ref = _compute_fs_value(vref_col, a1100, *coeffs)
    results[:, :n_computed] = np.exp(fs_site - fs_ref)
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
    vref = df[vref_col].to_numpy()
    vsite = df[vsite_col].to_numpy()
    vpga = df[vpga_col].to_numpy()
    pga = df[pga_col].to_numpy()

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
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                f"Column '{name}' has incorrect kind, must be real floating"
            )

    # Use pga for reference dtype because it is more reliably a float,
    # where vref can sometimes be an int.
    freqs = freqs.astype(pga.dtype)  # type: ignore[no-matching-overload]
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
    ampf0 : np.ndarray
        The amplification factors.
    dt : float
        The timestep delta for the waveforms to amplify.
    n : int
        The number of timesteps of the waveforms.
    fmin, fmidbot, fhightop, fmax : float, optional
        Bandpass filter parameters, see `amp_bandpass`.
    freqs : np.ndarray, optional
        The SA frequencies corresponding to site-amplification factors.

    Returns
    -------
    np.ndarray
        The amplification factors `ampf0` interpolated to FFT frequencies
        matching `dt` and `n`, and amplified according to the bandpass
        filter `amp_bandpass`.
    """
    interpolated, ftfreq = interpolate_amplification_factors(freqs, ampf0, dt, n)
    return amp_bandpass(interpolated, fhightop, fmax, fmidbot, fmin, ftfreq)


def interp_2d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Perform interpolation of a vector-valued function f at `x` with interpolation nodes `xp` and `fp`.

    This handles the case where `fp` is not 1-D. Each row of `fp` is
    interpolated independently.

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
    for i in range(fp.shape[0]):
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
    ftfreq = np.fft.rfftfreq(n, dt).ravel().astype(freqs.dtype)

    if n % 2 == 1:
        # If n is odd, the highest frequency is *less* than the
        # Nyquist frequency so we can include it without aliasing.
        ftfreq = ftfreq[1:]
    else:
        # If n is even the highest frequency is the Nyquist frequency
        # so we should remove it to avoid aliasing.
        ftfreq = ftfreq[1:-1]

    # Interpolate in log-frequency space
    log_fftfreq = np.log(ftfreq)
    log_cb_freq = np.log(freqs)
    ampv = interp_2d(log_fftfreq, log_cb_freq, ampf0)

    return ampv, ftfreq.astype(freqs.dtype)


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
    fftfreq : np.ndarray
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
    n_freq = fftfreq.size
    ampf = np.ones((ampv.shape[0], n_freq + 1), dtype=ampv.dtype)

    # Log-frequency weights are computed in double precision regardless of
    # the input dtype.
    log_fftfreq = np.log(fftfreq.astype(np.float64))
    log_fmax_diff = (log_fftfreq - np.log(fhightop)) / (np.log(fmax) - np.log(fhightop))
    log_fmin_diff = (log_fftfreq - np.log(fmin)) / (np.log(fmidbot) - np.log(fmin))

    # Output column j (1 <= j <= n_freq) is selected by the band containing
    # fftfreq[j - 1], but takes its values from ampv[:, j] and the log
    # differences at index j. For j = n_freq that index is one past the end
    # of `fftfreq` (and of `ampv` when it has n_freq columns); in that case
    # the last in-range value is used.
    def shifted(values: np.ndarray, band: np.ndarray) -> np.ndarray:
        """Select `values` at index j for each band column j - 1.

        Parameters
        ----------
        values : np.ndarray
            Array to select from along its last axis.
        band : np.ndarray
            Boolean mask over `fftfreq` selecting the columns j - 1.

        Returns
        -------
        np.ndarray
            `values[..., j]`, clamped to the last in-range index.
        """
        idx = np.minimum(np.flatnonzero(band) + 1, values.shape[-1] - 1)
        return values[..., idx]

    high = (fhightop <= fftfreq) & (fftfreq < fmax)
    mid = (fmidbot <= fftfreq) & (fftfreq < fhightop)
    low = (fmin <= fftfreq) & (fftfreq < fmidbot)

    out = ampf[:, 1:]
    ampv_high = shifted(ampv, high)
    out[:, high] = ampv_high + shifted(log_fmax_diff, high) * (1 - ampv_high)
    out[:, mid] = shifted(ampv, mid)
    out[:, low] = 1.0 + shifted(log_fmin_diff, low) * (shifted(ampv, low) - 1.0)

    return ampf

"""Testing for qcore.siteamp module"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcore import siteamp_models


@pytest.fixture(scope="module")
def cb_2014_df() -> pd.DataFrame:
    """CB2014 Siteamp Dataframe.

    Values are site-amplification factors derived from CB2014 spreadsheet made by Felipe Kuncar.
    Spreadsheet link: https://www.dropbox.com/scl/fi/spbolh0iiy57pqlv9fx6h/CB14old.xlsx?rlkey=wtnqljwcwo1uhtczf1vz1771p&st=7wz5l89a&dl=0
    Spreadsheet dropbox path: /QuakeCoRE/Public/Test

    Returns
    -------
    pd.DataFrame
        CB2014 Siteamp reference database.
    """
    return pd.read_csv(Path(__file__).parent / "cb_2014_test.csv")


def test_cb_2014_siteamp_model(cb_2014_df: pd.DataFrame) -> None:
    """Test CB2014 values against reference implementation.

    Parameters
    ----------
    cb_2014_df : pd.DataFrame
        The dataframe containing the CB2014 reference dataset.
    """
    input_df = pd.DataFrame(
        {
            "vref": 500.0,
            "vpga": 500.0,
            "pga": cb_2014_df["PGA"],
            "vsite": cb_2014_df["Vs30"],
        }
    )
    # The first column value is 1000Hz but is not in the test dataset, so drop it with 1:
    output = siteamp_models.cb_amp_multi(input_df)[:, 1:]
    # Dataframe has PGA and Vs30 columns, drop those as they are inputs.
    expected = cb_2014_df.to_numpy()[:, 2:]
    # Test equality to within a 1% tolerance
    assert output == pytest.approx(expected, rel=0.01, abs=0.0)


def test_cb_2014_siteamp_bad_dtypes() -> None:
    """Check that the siteamp model throws errors for non-floating dtypes."""
    input_df = pd.DataFrame(
        {"vref": [np.float32(500.0)], "vpga": 500, "pga": np.float32(0.4), "vsite": 500}
    )
    with pytest.raises(
        ValueError, match="Column '.*' has incorrect kind, must be real floating"
    ):
        _ = siteamp_models.cb_amp_multi(input_df)


def test_cb_2014_siteamp_model_types() -> None:
    """Check that the siteamp model output respects numpy types."""
    input_df = pd.DataFrame(
        {
            "vref": [np.float32(500.0)],
            "vpga": np.float32(500.0),
            "pga": np.float32(0.4),
            "vsite": np.float32(500),
        }
    )
    output = siteamp_models.cb_amp_multi(input_df)
    assert output.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_different_dtypes(dtype: np.dtype):
    """Test with different numpy dtypes."""
    freqs = np.array([1.0, 10.0, 100.0], dtype=dtype)
    ampf0 = np.array([1.0, 2.0, 3.0], dtype=dtype)
    dt = 0.01
    n = 100

    ampv, ftfreq = siteamp_models.interpolate_amplification_factors(freqs, ampf0, dt, n)

    # Check that ftfreq maintains the input dtype
    assert ftfreq.dtype == dtype
    assert ampv.dtype == dtype


@pytest.fixture
def benchmark_interpolation_data() -> dict:
    with open(Path(__file__).parent / "interpolate_benchmark.pkl", "rb") as f:
        return pickle.load(f)


def test_interpolate_frequency(benchmark_interpolation_data: dict):
    freqs = benchmark_interpolation_data["freqs"].copy()
    ampf0 = benchmark_interpolation_data["ampf0"].copy()
    dt = benchmark_interpolation_data["dt"]
    n = benchmark_interpolation_data["n"]
    expected_ampv = benchmark_interpolation_data["expected_ampv"]
    expected_ftfreq = benchmark_interpolation_data["expected_ftfreq"]

    ampv, ftfreq = siteamp_models.interpolate_amplification_factors(freqs, ampf0, dt, n)

    # The old code behaves poorly on non-smooth data. The best we can hope
    # for is 5% equivalence.
    assert ampv.ravel() == pytest.approx(expected_ampv, rel=0.05)
    assert ftfreq == pytest.approx(expected_ftfreq)


@pytest.fixture
def benchmark_amp_bandpass_data() -> dict:
    with open(Path(__file__).parent / "amp_bandpass_benchmark.pkl", "rb") as f:
        return pickle.load(f)


def test_interpolate_amp_bandpass(benchmark_amp_bandpass_data: dict):
    ftfreq = benchmark_amp_bandpass_data["ftfreq"].copy()
    ampv = benchmark_amp_bandpass_data["ampv"].copy()
    fmin = benchmark_amp_bandpass_data["fmin"]
    fmidbot = benchmark_amp_bandpass_data["fmidbot"]
    fhightop = benchmark_amp_bandpass_data["fhightop"]
    fmax = benchmark_amp_bandpass_data["fmax"]
    expected_ampf = benchmark_amp_bandpass_data["expected_ampf"]
    ampf = siteamp_models.amp_bandpass(
        np.atleast_2d(ampv), fhightop, fmax, fmidbot, fmin, ftfreq
    )
    assert ampf.ravel() == pytest.approx(expected_ampf)


@pytest.mark.parametrize(
    "dt,n",
    [
        (0.01, 100),
        (0.1, 20),
        (0.001, 1000),
    ],
)
def test_frequency_range_calculation(dt: float, n: int):
    """Test that frequency range is calculated correctly for different dt and n."""
    freqs = np.array([1.0, 10.0, 100.0])
    ampf0 = np.array([1.0, 2.0, 3.0])

    _, ftfreq = siteamp_models.interpolate_amplification_factors(freqs, ampf0, dt, n)

    if len(ftfreq) > 0:
        freq_step = 1.0 / (n * dt)
        expected_min_freq = freq_step
        expected_max_freq_actual = (int(n / 2) - 1) * freq_step

        assert ftfreq[0] == pytest.approx(expected_min_freq)
        assert ftfreq[-1] == pytest.approx(expected_max_freq_actual)


@pytest.fixture
def basic_setup() -> dict:
    """Basic test setup with typical parameters."""
    return {
        "fmin": 0.1,
        "fmidbot": 1.0,
        "fhightop": 10.0,
        "fmax": 50.0,
        "fftfreq": np.logspace(-1, 2, 20),  # 0.1 to 100 Hz
        "ampv": np.random.uniform(0.5, 2.0, (3, 20)),  # 3 sites, 20 frequencies
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_types(basic_setup: dict, dtype: np.dtype) -> None:
    """Test that the output respects datatypes."""
    result = siteamp_models.amp_bandpass(
        basic_setup["ampv"].astype(dtype),
        basic_setup["fhightop"],
        basic_setup["fmax"],
        basic_setup["fmidbot"],
        basic_setup["fmin"],
        basic_setup["fftfreq"].astype(dtype),
    )
    assert result.dtype == dtype


def test_output_shape(basic_setup: dict):
    """Test that output has correct shape."""
    result = siteamp_models.amp_bandpass(
        basic_setup["ampv"],
        basic_setup["fhightop"],
        basic_setup["fmax"],
        basic_setup["fmidbot"],
        basic_setup["fmin"],
        basic_setup["fftfreq"],
    )
    expected_shape = (basic_setup["ampv"].shape[0], basic_setup["fftfreq"].size + 1)
    assert result.shape == expected_shape


def test_dc_component_is_unity(basic_setup: dict):
    """Test that DC component (first column) is always 1.0."""
    result = siteamp_models.amp_bandpass(
        basic_setup["ampv"],
        basic_setup["fhightop"],
        basic_setup["fmax"],
        basic_setup["fmidbot"],
        basic_setup["fmin"],
        basic_setup["fftfreq"],
    )
    np.all(result[:, 0] == 1.0)


def test_single_site():
    """Test function with single site (1D amplification array)."""
    fftfreq = np.logspace(-1, 2, 10)
    ampv = np.random.uniform(0.5, 2.0, (1, 10))

    result = siteamp_models.amp_bandpass(ampv, 10.0, 50.0, 1.0, 0.1, fftfreq)

    assert result.shape == (1, 11)
    assert result[0, 0] == 1.0


def test_multiple_sites():
    """Test function with multiple sites."""
    fftfreq = np.logspace(-1, 2, 10)
    ampv = np.random.uniform(0.5, 2.0, (5, 10))

    result = siteamp_models.amp_bandpass(ampv, 10.0, 50.0, 1.0, 0.1, fftfreq)

    assert result.shape == (5, 11)
    assert np.all(result[:, 0] == 1.0)


@pytest.fixture
def test_df() -> pd.DataFrame:
    """Test dataframe for regression test."""
    return pd.DataFrame(
        {
            "vref": [
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
            ],
            "vpga": [
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
                500.0,
            ],
            "vsite": [
                500.0,
                611.1111111111111,
                722.2222222222222,
                833.3333333333334,
                944.4444444444445,
                1055.5555555555557,
                1166.6666666666667,
                1277.7777777777778,
                1388.888888888889,
                1500.0,
            ],
            "pga": [
                1.3344444444444445,
                0.8377777777777778,
                1.5,
                1.1688888888888889,
                1.0033333333333334,
                0.5066666666666667,
                0.01,
                0.17555555555555558,
                0.6722222222222223,
                0.34111111111111114,
            ],
        }
    )


def amp_bandpass_old(
    ampv: np.ndarray,
    fhightop: float,
    fmax: float,
    fmidbot: float,
    fmin: float,
    ftfreq: np.ndarray,
) -> np.ndarray:
    """Old bandpass code, kept for testing"""
    # default amplification is 1.0 (keeping values the same)
    ampf = np.ones((ampv.shape[0], ftfreq.size + 1), dtype=np.float64)
    # amplification factors applied differently at different bands
    ampf[:, 1:] += (
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


def test_regression_amp_bandpass(test_df: pd.DataFrame) -> None:
    """Test amp_bandpass compatibility with old results."""
    ampv = siteamp_models.cb_amp_multi(test_df)
    dt = 0.005
    n = 64000
    ampv_new = siteamp_models.cb2014_to_fas_amplification_factors(ampv, dt, n)
    interpolated, ftfreq = siteamp_models.interpolate_amplification_factors(
        siteamp_models.AMPLIFICATION_FREQUENCIES, ampv, dt, n
    )

    ampv_old = amp_bandpass_old(
        interpolated, fmin=0.2, fmidbot=0.5, fhightop=10.0, fmax=15.0, ftfreq=ftfreq
    )
    assert ampv_new == pytest.approx(ampv_old, rel=0.01)


def test_cb_2014_siteamp_empty_dataframe() -> None:
    """Test that ValueError is raised for empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        siteamp_models.cb_amp_multi(empty_df)


def test_cb_2014_siteamp_missing_columns() -> None:
    """Test that KeyError is raised for missing required columns."""
    incomplete_df = pd.DataFrame({"vref": [500.0], "vpga": [500.0]})
    with pytest.raises(KeyError, match="Missing required columns"):
        siteamp_models.cb_amp_multi(incomplete_df)


def test_cb_2014_siteamp_nan_values() -> None:
    """Test that ValueError is raised for NaN values."""
    nan_df = pd.DataFrame(
        {"vref": [np.nan], "vpga": [500.0], "pga": [0.4], "vsite": [500.0]}
    )
    with pytest.raises(ValueError, match="Column 'vref' contains NaN values"):
        siteamp_models.cb_amp_multi(nan_df)


def test_cb_2014_siteamp_infinite_values() -> None:
    """Test that ValueError is raised for infinite values."""
    inf_df = pd.DataFrame(
        {"vref": [np.inf], "vpga": [500.0], "pga": [0.4], "vsite": [500.0]}
    )
    with pytest.raises(ValueError, match="Column 'vref' contains infinite values"):
        siteamp_models.cb_amp_multi(inf_df)


def test_cb_2014_siteamp_non_positive_values() -> None:
    """Test that ValueError is raised for non-positive values."""
    negative_df = pd.DataFrame(
        {"vref": [-500.0], "vpga": [500.0], "pga": [0.4], "vsite": [500.0]}
    )
    with pytest.raises(ValueError, match="Column 'vref' contains non-positive values"):
        siteamp_models.cb_amp_multi(negative_df)

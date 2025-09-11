"Tests for the qcore.timeseries module"

import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qcore import timeseries

ARRAY_SHAPE = [2**i for i in range(3, 9)]


@st.composite
def real_fourier_spectra(draw: st.DrawFn) -> np.ndarray:
    """Sample a real fourier spectra.

    Parameters
    ----------
    draw : st.DrawFn
        Draw function for hypothesis

    Returns
    -------
    np.ndarray
        The fourier spectrum of a real-valued function.
    """

    # We need to draw a fourier spectra corresponding to a real-valued function (otherwise tests will fail).
    # If f : R -> R is a periodic function on the reals then F(w) is a is function in the frequency space w with the following properties
    # 1. F(0) is real
    # 2. F(-w) = F(w)^* (NOTE: (2) implies (1), but this explicitly makes clear why `dc` is a float and not a complex number).
    #
    # In the discrete case (i.e. with vectors or numpy arrays), the same applies to the outputs of np.fft.fft (or fftw, ...). If v is a numpy array, and w = np.fft.fft(v) the DFT, then:
    # 1. v[0] is real
    # 2. v[i] == np.conj(v[-i])
    # The negative index is intentional, and implies the negative
    # frequencies are at the end of the DFT in the opposite order to
    # positive frequencies to have symmetry with the continuous case
    # above).
    #
    # The functions `np.fft.rfft` and `np.fft.irfft` assume
    # real-valued functions and spectra corresponding to real-valued
    # functions. Hence they exploit the symmetry and drop half the
    # spectra. So, you only need to generate `v[0]` (which should be
    # real) and `v[1:(n + 1) / 2]`. The (i)rfft function will assume
    # the other half is as implied by the conjugation theorem above.
    # This is why we generate only the float valued `dc` and the
    # complex valued half spectra.

    dc = draw(st.floats(min_value=0.1, max_value=100.0))
    half_spectra = draw(
        arrays(
            dtype=np.complex64,
            shape=st.sampled_from(ARRAY_SHAPE),
            elements=st.complex_numbers(min_magnitude=0.1, max_magnitude=100.0),
        )
    )
    return np.concatenate([np.atleast_1d(dc), half_spectra])


@st.composite
def fourier_spectra_with_amplification(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a real-fourier spectra and matching amplification factors.



    Parameters
    ----------
    draw : st.DrawFn
        Draw function for hypothesis


    Returns
    -------
    amplification_factors : np.ndarray
        Amplification factors to match the fourier spectra in shape.
    fas : np.ndarray
        The fourier spectrum of a real-valued function.
    """

    fas = draw(real_fourier_spectra())
    # Use the last axis size for amplification factor length
    amp_factor_length = fas.shape[-1] - 1
    amplification_factors = draw(
        arrays(
            dtype=np.float32,
            shape=st.just(amp_factor_length),
            elements=st.floats(min_value=0.1, max_value=2.0),
        )
    )
    return amplification_factors, fas


def test_bwfilter_lowpass() -> None:
    """Test lowpass filter with realistic expectations."""
    # Long signal because it seems to help the numerics more
    signal_length = 32768
    dt = 128 / signal_length
    # Total signal length = 128s
    # Now generate three sine waves, a low frequency wave (should pass
    # through), a high frequency wave (filtered), and a cutoff
    # frequency wave (which should also roughly pass through).

    t = np.arange(signal_length) * dt
    low_frequency = 1  # Very low, just 1Hz
    cutoff_frequency = 4  # cutoff frequency
    high_frequency = 16  # very high frequency
    signal = (
        np.sin(2 * np.pi * low_frequency * t)
        + np.sin(2 * np.pi * cutoff_frequency * t)
        + np.sin(2 * np.pi * high_frequency * t)
    )

    fas = pyfftw_fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=dt)
    low_freq_idx = np.argmin(np.abs(frequencies - low_frequency))
    cutoff_freq_idx = np.argmin(np.abs(frequencies - cutoff_frequency))
    high_freq_idx = np.argmin(np.abs(frequencies - high_frequency))

    # Sanity checks, at each input frequency, we get the appropriate signal strength.
    assert np.real(np.abs(fas[low_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )
    assert np.real(np.abs(fas[cutoff_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )
    assert np.real(np.abs(fas[high_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )

    # Now filter
    filtered = timeseries.bwfilter(
        signal, dt, cutoff_frequency, timeseries.Band.LOWPASS
    )
    filtered_fas = pyfftw_fft.rfft(filtered)
    # Low frequency should not be attenuated
    assert np.real(np.abs(filtered_fas[low_freq_idx])) == pytest.approx(
        signal_length / 2, rel=0.01
    )
    # Cutoff frequency should be attenuated to a power of 1/sqrt(2)
    assert np.real(np.abs(filtered_fas[cutoff_freq_idx])) == pytest.approx(
        signal_length / (2 * np.sqrt(2)), rel=0.01
    )
    # High frequencies should be almost eliminated
    assert (
        np.real(np.abs(filtered_fas[high_freq_idx])) < 16.0
    )  # NB: less than 5% of the original signal strength


def test_bwfilter_highpass() -> None:
    """Test lowpass filter with realistic expectations."""
    # Long signal because it seems to help the numerics more
    signal_length = 32768
    dt = 128 / signal_length
    # Total signal length = 128s
    # Now generate three sine waves, a low frequency wave (should pass
    # through), a high frequency wave (filtered), and a cutoff
    # frequency wave (which should also roughly pass through).
    t = np.arange(signal_length) * dt
    low_frequency = 1  # Very low, just 1Hz
    cutoff_frequency = 4  # cutoff frequency
    high_frequency = 16  # very high frequency
    signal = (
        np.sin(2 * np.pi * low_frequency * t)
        + np.sin(2 * np.pi * cutoff_frequency * t)
        + np.sin(2 * np.pi * high_frequency * t)
    )

    fas = pyfftw_fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=dt)
    low_freq_idx = np.argmin(np.abs(frequencies - low_frequency))
    cutoff_freq_idx = np.argmin(np.abs(frequencies - cutoff_frequency))
    high_freq_idx = np.argmin(np.abs(frequencies - high_frequency))

    # Sanity checks, at each input frequency, we get the appropriate signal strength.
    assert np.real(np.abs(fas[low_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )
    assert np.real(np.abs(fas[cutoff_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )
    assert np.real(np.abs(fas[high_freq_idx])) == pytest.approx(
        signal_length / 2, abs=0.001
    )

    # Now filter
    filtered = timeseries.bwfilter(
        signal, dt, cutoff_frequency, timeseries.Band.HIGHPASS
    )
    filtered_fas = pyfftw_fft.rfft(filtered)
    # Low frequency should be almost eliminated
    assert (
        np.real(np.abs(filtered_fas[low_freq_idx])) < 16.0
    )  # NB: less than 5% of the original signal strength
    # Cutoff frequency should be attenuated to a power of 1/sqrt(2)
    assert np.real(np.abs(filtered_fas[cutoff_freq_idx])) == pytest.approx(
        signal_length / (2 * np.sqrt(2)), rel=0.01
    )
    # High frequencies should not be attenuated
    assert np.real(np.abs(filtered_fas[high_freq_idx])) == pytest.approx(
        signal_length / 2, rel=0.01
    )


@given(
    inputs=fourier_spectra_with_amplification(),
)
def test_ampdeamp_with_fft_consistency(inputs: tuple[np.ndarray, np.ndarray]) -> None:
    """Test ampdeamp FFT consistency.

    Test that applying ampdeamp to a waveform derived from an inverse FFT
    modifies the Fourier spectra as expected.

    Parameters
    ----------
    inputs : pair of np.ndarray
        The input amplification factors and fas.
    """
    amplification_factor, fourier_spectra = inputs

    # Ensure amplification_factor is at least 2d
    amplification_factor = np.atleast_2d(amplification_factor)

    # Ensure fourier_spectra is at least 2D
    fourier_spectra = np.atleast_2d(fourier_spectra)

    # Compute the expected Fourier spectra after amplification
    expected_fourier = fourier_spectra[:, :-1] * amplification_factor

    # Compute the inverse FFT to generate the waveform
    n = 2 * amplification_factor.shape[-1]
    waveform = pyfftw_fft.irfft(fourier_spectra, n=n, axis=-1)

    # Apply ampdeamp to the waveform
    amplified_waveform = timeseries.ampdeamp(
        waveform, amplification_factor, amplify=True, taper=False
    )

    # Compute the FFT of the amplified waveform
    amplified_fourier = pyfftw_fft.rfft(amplified_waveform, n=n, axis=-1)

    # Verify that the amplified Fourier spectra matches the expected result
    assert amplified_fourier.shape[1] == expected_fourier.shape[1] + 1, "Shape mismatch"
    assert amplified_fourier[:, :-1] == pytest.approx(expected_fourier, abs=1e-6), (
        f"FFT of amplified waveform does not match expected Fourier spectra, {amplified_fourier[:, :-1]=}, {expected_fourier=}"
    )


def test_tapering_application() -> None:
    """Check that tapering works as expected.

    If taper = True, we should see a Hanning taper applied to the last 5% of the waveform
    """
    waveform = np.ones(100, dtype=np.float32)
    amplification_factor = np.ones(25, dtype=np.float32)  # n_fft = 50

    expected_ntap = int(100 * 0.05)
    # Manually calculate the expected tapered end
    expected_hanning_window = np.hanning(expected_ntap * 2 + 1)[
        expected_ntap + 1 :
    ].astype(np.float32)

    # Create the expected waveform after tapering
    expected_waveform_after_taper = waveform.copy()
    expected_waveform_after_taper[100 - expected_ntap :] = expected_hanning_window

    with (
        patch("pyfftw.interfaces.numpy_fft.rfft") as mock_rfft,
        patch("pyfftw.interfaces.numpy_fft.irfft") as mock_irfft,
    ):
        # Mock rfft and irfft to return predictable values, so we can focus on tapering
        mock_rfft.return_value = np.zeros(
            amplification_factor.shape[-1] + 1, dtype=np.complex64
        )
        mock_irfft.return_value = np.zeros(
            100, dtype=np.float32
        )  # Return original length

        _ = timeseries.ampdeamp(waveform, amplification_factor, taper=True)

        # Retrieve the waveform argument that was passed to rfft. This
        # is the waveform *after* tapering.
        actual_waveform_passed_to_rfft = mock_rfft.call_args[0][0]

        assert actual_waveform_passed_to_rfft == pytest.approx(
            expected_waveform_after_taper, abs=1e-6
        )
        # Check that the original waveform is not tapered.
        assert waveform == pytest.approx(np.ones(100, dtype=np.float32))


@given(
    inputs=fourier_spectra_with_amplification(),
)
def test_ampdeamp_reversibility(inputs: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that amplifying and then de-amplifying a waveform returns the original waveform.

    Parameters
    ----------
    inputs : pair of np.ndarray
        The input amplification factors and fas.
    """
    amplification_factor, fft = inputs
    n = 2 * amplification_factor.shape[-1]
    waveform = pyfftw_fft.irfft(fft, n, axis=-1)

    # Apply amplification
    amplified_waveform = timeseries.ampdeamp(
        waveform, amplification_factor, amplify=True, taper=False
    )

    # Apply de-amplification
    recovered_waveform = timeseries.ampdeamp(
        amplified_waveform, amplification_factor, amplify=False, taper=False
    )

    # Assert that the recovered waveform matches the original waveform
    # to within 0.1% or 10^-6.
    assert waveform == pytest.approx(recovered_waveform, rel=0.001, abs=1e-6)


def make_seis_file(
    endianness: Literal["<"] | Literal[">"],
    header: timeseries.LFSeisHeader,
    stations: pd.DataFrame,
    waveform: np.ndarray,
) -> bytearray:
    """Create an in-memory seis file from raw data.



    Parameters
    ----------
    header : timeseries.LFSeisHeader
        header parameters.
    stations : pd.DataFrame
        Station data
    waveform : np.ndarray
        Waveform array with shape (3, nt, nstat).

    Returns
    -------
    bytearray
        A bytearray contained the serialised seis file.
    """
    array = bytearray()
    byteorder = "little" if endianness == "<" else "big"
    array.extend(header.nstat.to_bytes(length=4, byteorder=byteorder))
    stations = stations.copy()
    stations["index"] = np.arange(len(stations))
    stations["nt"] = header.nt
    stations["dt"] = header.dt
    stations["rotation"] = header.rotation
    stations["resolution"] = header.resolution
    i4 = f"{endianness}i4"
    f4 = f"{endianness}f4"
    station_array = stations[
        [
            "index",
            "x",
            "y",
            "z",
            "nt",
            "dt",
            "resolution",
            "rotation",
            "lat",
            "lon",
            "station",
        ]
    ].to_records(
        index=False,
        column_dtypes=dict(
            [
                ("index", i4),
                ("x", i4),
                ("y", i4),
                ("z", i4),
                ("nt", i4),
                ("dt", f4),
                ("resolution", f4),
                ("rotation", f4),
                ("lat", f4),
                ("lon", f4),
                ("station", "|S8"),
            ]
        ),
    )

    array.extend(station_array.tobytes())
    array.extend(waveform.tobytes())
    return array


def header_strategy(
    nstat: int | None, nt: int | None
) -> st.SearchStrategy[timeseries.LFSeisHeader]:
    """Construct a strategy for generating a valid LFSeis header file.

    Parameters
    ----------
    nstat : int | None
        If provided, fix the number of stations in the file.
    nt : int | None
        If provided, fix the number of timesteps in the file.

    Returns
    -------
    SearchStrategy
        A search strategy returning valid LFSeis file headers.
    """
    return st.builds(
        timeseries.LFSeisHeader,
        nstat=st.integers(min_value=1, max_value=10)
        if nstat is None
        else st.just(nstat),
        nt=st.integers(min_value=1, max_value=100) if nt is None else st.just(nt),
        dt=st.floats(
            min_value=2**-8,
            max_value=2**-5,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        rotation=st.floats(
            min_value=-180.0,
            max_value=180.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        resolution=st.floats(
            min_value=2**-4,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


@st.composite
def seis_file_data(
    draw: st.DrawFn,
    nstat: int | None = None,
    nt: int | None = None,
) -> tuple[Literal["<"], timeseries.LFSeisHeader, pd.DataFrame, np.ndarray]:
    """A Hypothesis composite strategy to generate a tuple of valid arguments for the `make_seis_file` function.

    Parameters
    ----------
    draw : DrawFn
        The draw function.
    nstat : int | None
        If set, fix the number of stations in the file.
    nt : int | None
        If set fix the number of timesteps in the file.

    Returns
    -------
    byteorder : '<'
        The byteorder of the output file. Only machine byteorder for now.
    header : LFSeisHeader
        The header data.
    stations : DataFrame
        The stations dataframe.
    waveform : array of floats
        The synthetic waveform array.
    """

    # NOTE: Not easy to test with big endianess on a little endian machine due to limitations from numpy
    endianness = "<"

    header = draw(header_strategy(nstat, nt))

    nt = header.nt
    nstat = header.nstat

    stations_data = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "x": st.integers(min_value=-1000, max_value=1000),
                    "y": st.integers(min_value=-1000, max_value=1000),
                    "z": st.integers(min_value=-1000, max_value=1000),
                    "lat": st.floats(
                        min_value=-90,
                        max_value=90,
                        allow_nan=False,
                        allow_infinity=False,
                        width=32,
                    ),
                    "lon": st.floats(
                        min_value=-180,
                        max_value=180,
                        allow_nan=False,
                        allow_infinity=False,
                        width=32,
                    ),
                    "station": st.text(
                        min_size=0,
                        max_size=8,
                        alphabet=st.characters(
                            codec="ascii",
                            blacklist_categories=("Cs",),
                        ),
                    ),
                }
            ),
            min_size=nstat,
            max_size=nstat,
        )
    )

    stations = pd.DataFrame(stations_data)
    stations["x"] = stations["x"].astype(np.int32)
    stations["y"] = stations["y"].astype(np.int32)
    stations["z"] = stations["z"].astype(np.int32)
    stations["lat"] = stations["lat"].astype(np.float32)
    stations["lon"] = stations["lon"].astype(np.float32)

    dtype = f"{endianness}f4"
    waveform = draw(
        arrays(
            dtype=dtype,
            shape=st.just((nt, nstat, timeseries._N_COMP)),
            elements=st.floats(allow_nan=False, width=32, allow_infinity=False),
        )
    )

    return (endianness, header, stations, waveform)


@pytest.fixture(scope="module")
def basic_lfseis_fp() -> Path:
    """Return the path to the real-world seis file test.

    Returns
    -------
    Path
        A path to R01_seis-00041.e3d.
    """
    return Path(__file__).parent / "R01_seis-00041.e3d"


def test_lf_seis_real_dtypes(basic_lfseis_fp: Path) -> None:
    """Test the LFSeis dtype inference on a real file."""
    with open(basic_lfseis_fp, "rb") as f:
        parser = timeseries.LFSeisParser(f)
        assert parser.i4 == "<i4"
        assert parser.f4 == "<f4"


def test_lf_seis_real_header(basic_lfseis_fp: Path) -> None:
    """Test header extraction for a real LFSeis file."""
    with open(basic_lfseis_fp, "rb") as f:
        parser = timeseries.LFSeisParser(f)
        header = parser.read_header()
        assert header.nstat == 12
        assert header.nt == 64000
        assert header.dt == pytest.approx(0.005)
        assert header.resolution == pytest.approx(0.1)
        assert header.rotation == pytest.approx(41.9)


def test_lf_seis_real_stations(basic_lfseis_fp: Path) -> None:
    """Test station extraction for a real LFSeis file."""
    with open(basic_lfseis_fp, "rb") as f:
        parser = timeseries.LFSeisParser(f)
        header = parser.read_header()
        stations = parser.read_stations(header.nstat)
        expected_df = pd.DataFrame(
            {
                "x": [
                    3067,
                    3089,
                    3068,
                    2984,
                    3041,
                    3094,
                    3037,
                    3097,
                    3085,
                    3113,
                    3110,
                    3082,
                ],
                "y": [
                    1114,
                    1147,
                    1138,
                    1162,
                    1167,
                    1169,
                    1164,
                    1165,
                    1150,
                    1152,
                    1181,
                    1178,
                ],
                "lat": [
                    -37.561370849609375,
                    -37.59678649902344,
                    -37.5784912109375,
                    -37.546661376953125,
                    -37.58294677734375,
                    -37.61482238769531,
                    -37.57856750488281,
                    -37.61378479003906,
                    -37.59654235839844,
                    -37.61402893066406,
                    -37.63230895996094,
                    -37.61412048339844,
                ],
                "lon": [
                    178.30628967285156,
                    178.30142211914062,
                    178.28973388671875,
                    178.19940185546875,
                    178.24520874023438,
                    178.28976440429688,
                    178.24392700195312,
                    178.2952880859375,
                    178.29576110839844,
                    178.31863403320312,
                    178.29493713378906,
                    178.27279663085938,
                ],
                "station": [
                    "MXZ",
                    "HBZ",
                    "0201050",
                    "12018d3",
                    "12018d4",
                    "12018f5",
                    "12018f6",
                    "12018f7",
                    "2201324",
                    "2201325",
                    "2201326",
                    "2201327",
                ],
            }
        )
        expected_df["x"] = expected_df["x"].astype(np.int32)
        expected_df["y"] = expected_df["y"].astype(np.int32)
        expected_df["lat"] = expected_df["lat"].astype(np.float32)
        expected_df["lon"] = expected_df["lon"].astype(np.float32)
        pd.testing.assert_frame_equal(stations, expected_df)


def test_lf_seis_real_waveforms(basic_lfseis_fp: Path) -> None:
    """Test waveform extraction for a basic LFSeis file."""
    with open(basic_lfseis_fp, "rb") as f:
        parser = timeseries.LFSeisParser(f)
        header = parser.read_header()
        _ = parser.read_stations(header.nstat)
        waveform = parser.read_waveform((header.nt, header.nstat, timeseries._N_COMP))
        assert waveform.dtype == np.float32
        assert waveform.max() == pytest.approx(83.34158)
        assert waveform.min() == pytest.approx(-61.9567)
        assert waveform.mean() == pytest.approx(0.057460602)
        assert waveform.std() == pytest.approx(4.2015386)


@given(inputs=seis_file_data())
def test_lf_seis_random(inputs: tuple) -> None:
    """Test that LFSeis parsing correctly extracts header, stations and waveforms."""
    endianness, header, stations, waveform = inputs
    seis_file_bytes = make_seis_file(*inputs)
    with tempfile.TemporaryFile() as f:
        f.write(seis_file_bytes)
        f.seek(0)
        parser = timeseries.LFSeisParser(f)
        assert parser.i4 == f"{endianness}i4"
        assert parser.f4 == f"{endianness}f4"
        read_header = parser.read_header()
        read_stations = parser.read_stations(read_header.nstat)
        read_waveform = parser.read_waveform(
            (header.nt, header.nstat, timeseries._N_COMP)
        )

    assert read_header.nstat == header.nstat
    assert read_header.nt == header.nt
    assert read_header.dt == pytest.approx(header.dt)
    assert read_header.resolution == pytest.approx(header.resolution)
    assert read_header.rotation == pytest.approx(header.rotation)
    read_stations["x"] = read_stations["x"].astype(np.int32)
    read_stations["y"] = read_stations["y"].astype(np.int32)
    read_stations["lat"] = read_stations["lat"].astype(np.float32)
    read_stations["lon"] = read_stations["lon"].astype(np.float32)
    stations["station"] = stations["station"].str.strip("\x00")
    pd.testing.assert_frame_equal(
        stations[["x", "y", "lat", "lon", "station"]], read_stations
    )
    assert read_waveform == pytest.approx(waveform)


@given(inputs=seis_file_data())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_read_lf_seis_file(inputs: tuple, tmp_path: Path) -> None:
    """Test that _read_lfseis_file correctly reads an lfseis file from disk."""
    _, _, stations, waveform = inputs

    seis_file_bytes = make_seis_file(*inputs)
    seis_file_path = tmp_path / "seis_file"
    with open(seis_file_path, "wb") as f:
        f.write(seis_file_bytes)
    stations["station"] = stations["station"].str.strip("\x00")
    dset = timeseries._read_lfseis_file(seis_file_path)
    real_stations_mask = pd.notnull(stations["station"])
    waveform = np.transpose(waveform[:, real_stations_mask, :3], (2, 1, 0))
    stations = stations.loc[real_stations_mask]

    for column in ["x", "y", "lat", "lon", "station"]:
        np.testing.assert_array_equal(dset[column].values, stations[column].values)

    assert dset.waveform.dims == ("component", "station", "time")
    np.testing.assert_array_equal(dset["waveform"].values, waveform)


def test_waveform_postprocessing_basic() -> None:
    """Test waveform processing correctly handles unitary case."""
    nstat = 10
    nt = 100
    waveform = np.ones(shape=(3, nstat, nt), dtype=np.float32)
    rotated_diffed_waveform = timeseries._postprocess_waveform(waveform, 0.0, 1.0)
    assert rotated_diffed_waveform.dtype == np.float32
    assert rotated_diffed_waveform == pytest.approx(np.zeros_like(waveform))


def test_waveform_postprocessing_reflection() -> None:
    """Test waveform processing correctly reflects y and z axes."""
    nstat = 10
    nt = 100

    waveform = np.ones(shape=(3, nstat, nt), dtype=np.float32)
    waveform *= np.arange(nt)

    rotated_diffed_waveform = timeseries._postprocess_waveform(waveform, 0.0, 1.0)
    expected_waveform = np.ones_like(waveform)
    # Y and Z are both reflected
    expected_waveform[1] *= -1
    expected_waveform[2] *= -1
    assert rotated_diffed_waveform == pytest.approx(expected_waveform)


def test_waveform_postprocessing_rotation_45() -> None:
    """Test waveform processing handles rotation correctly."""
    nstat = 1
    nt = 100
    dt = 1.0

    waveform = np.zeros(shape=(3, nstat, nt), dtype=np.float32)
    x_signal = np.arange(nt)
    waveform[0, :, :] = x_signal

    # Expected rotation (before differentiation)
    # R_45 = [ [cos(45), -sin(45), 0],
    #          [-sin(45), -cos(45), 0],
    #          [0, 0, -1] ]
    #
    # rotated_x = cos(45)*x + (-sin(45))*y + 0*z
    # rotated_y = -sin(45)*x + (-cos(45))*y + 0*z
    # rotated_z = 0*x + 0*y + (-1)*z
    #
    # Since y and z are 0, this simplifies to:
    # rotated_x = cos(45) * x_signal = (1/sqrt(2)) * x_signal
    # rotated_y = -sin(45) * x_signal = -(1/sqrt(2)) * x_signal
    # rotated_z = 0
    #
    # After differentiation:
    # rotated_dx = (1/sqrt(2)) * d[x_signal]/dt = 1/sqrt(2)
    # # rotated_dx = -(1/sqrt(2)) * d[x_signal]/dt = -1/sqrt(2)

    rotated_diffed_waveform = timeseries._postprocess_waveform(waveform, 45.0, dt)

    expected_waveform = np.zeros_like(waveform)
    expected_waveform[0] = np.reciprocal(np.sqrt(2))
    expected_waveform[1] = -expected_waveform[0]
    assert rotated_diffed_waveform == pytest.approx(expected_waveform, rel=0.005)


def test_waveform_postprocessing_rotation_90() -> None:
    """Further test of waveform rotation processing."""
    nstat = 1
    nt = 100
    # Waveform with a signal only in the x-component
    waveform = np.zeros(shape=(3, nstat, nt), dtype=np.float32)
    waveform[0, :, :] = np.arange(nt)

    rotated_diffed_waveform = timeseries._postprocess_waveform(waveform, 90.0, 1.0)

    # Expected output after rotation and differentiation
    # The x-component is now the y-component (and reflected)
    # The y-component is now the x-component
    expected_waveform = np.zeros_like(waveform)
    expected_waveform[1, :, :] = -np.ones(
        shape=(nstat, nt)
    )  # becomes reflected y component
    # Due to rotation, we need to reduce the tolerance
    assert rotated_diffed_waveform == pytest.approx(expected_waveform, abs=1e-6)


def test_waveform_postprocessing_differentiation() -> None:
    """Test complex differentiation case."""
    nstat = 1
    nt = 100
    dt = 0.01

    # Create a sinusoidal waveform
    t = np.arange(nt) * dt
    waveform = np.zeros(shape=(3, nstat, nt), dtype=np.float32)
    waveform[0, :, :] = np.sin(2 * np.pi * t)

    rotated_diffed_waveform = timeseries._postprocess_waveform(waveform, 0.0, dt)

    # Expected output is the derivative of the sine wave
    expected_waveform = np.zeros_like(waveform)
    expected_waveform[0, :, :] = 2 * np.pi * np.cos(2 * np.pi * t)

    assert rotated_diffed_waveform == pytest.approx(expected_waveform, rel=0.005)


NT = 10
NSTAT = 10


@given(inputs=st.lists(seis_file_data(nt=NT, nstat=NSTAT), min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_read_lf_seis_file_plural(inputs: list[tuple], tmp_path: Path) -> None:
    """Test read_lfseis_directory extracts all seis files and assigns correct metadata."""
    # Clear up old files because `tmp_path` is not renewed between hypothesis tests.
    for fp in tmp_path.glob("*seis*"):
        fp.unlink()

    _, header, _, _ = inputs[0]

    for i, input in enumerate(inputs):
        seis_file_bytes = make_seis_file(*input)
        seis_file_path = tmp_path / f"file_seis-{i}.e3d"
        with open(seis_file_path, "wb") as f:
            f.write(seis_file_bytes)

    dset = timeseries.read_lfseis_directory(tmp_path)
    assert len(dset.station) == 10 * len(inputs)
    assert dset.waveform.shape == (3, 10 * len(inputs), header.nt)
    assert dset.attrs["dt"] == header.dt
    assert dset.attrs["resolution"] == header.resolution
    assert dset.attrs["rotation"] == header.rotation
    assert dset.attrs["units"] == "cm/s^2"

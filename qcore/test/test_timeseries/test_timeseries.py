"Tests for the qcore.timeseries module"

from unittest.mock import patch

import numpy as np
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import pytest
from hypothesis import given
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
    expected_hanning_window = np.hanning(expected_ntap * 2)[expected_ntap:].astype(
        np.float32
    )

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

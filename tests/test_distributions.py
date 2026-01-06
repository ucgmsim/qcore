import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from qcore.uncertainties.distributions import (
    rand_shyp,
    truncated_log_normal,
    truncated_normal,
    truncated_weibull,
    truncated_weibull_expected_value,
)


def test_truncated_normal_with_size_1() -> None:
    mean = 0
    std_dev = 1.0
    limit = 5.0
    samples = truncated_normal(mean, std_dev, limit, size=1, seed=0)
    assert isinstance(samples, float)


def test_truncated_weibull_with_size_1() -> None:
    upper = 1
    samples = truncated_weibull(upper, size=1, seed=0)
    assert isinstance(samples, float)


def test_truncated_log_normal_with_size_1() -> None:
    mean = 1
    std_dev = 0.1
    samples = truncated_log_normal(mean, std_dev, size=1, seed=0)
    assert isinstance(samples, float)


@given(mean=st.floats(-1e3, 1e3), std_dev=st.floats(0.1, 1e2), limit=st.floats(1, 10))
@settings(max_examples=20)
def test_truncated_normal_vectorized(mean: float, std_dev: float, limit: float) -> None:
    samples = truncated_normal(mean, std_dev, limit, size=200, seed=0)
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= mean - limit * std_dev)
    assert np.all(samples <= mean + limit * std_dev)


@given(upper=st.floats(0.1, 1e3), seed=st.integers(0, 1_000_000))
@settings(max_examples=20)
def test_truncated_weibull_vectorized(upper: float, seed: int) -> None:
    samples = truncated_weibull(upper, size=200, seed=seed)
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= 0)
    assert np.all(samples <= upper)
    # reproducibility check
    val1 = truncated_weibull(upper, seed=seed)
    val2 = truncated_weibull(upper, seed=seed)
    assert val1 == val2


@given(upper=st.floats(0.1, 1e3))
def test_truncated_weibull_expected_value_bounds(upper: float) -> None:
    val = truncated_weibull_expected_value(upper)
    assert 0 <= val <= upper


@given(
    mean=st.floats(1e-3, 1e3),
    std_dev=st.floats(1e-3, 1e2),
    seed=st.integers(0, 1_000_000),
)
@settings(max_examples=20)
def test_truncated_log_normal_vectorized(mean: float, std_dev: float, seed: int):
    samples = truncated_log_normal(mean, std_dev, size=200, seed=seed)
    assert np.all(np.isfinite(samples))
    log_mean = np.log(mean)
    assert np.all(np.log(samples) >= log_mean - 2 * std_dev)
    assert np.all(np.log(samples) <= log_mean + 2 * std_dev)
    # reproducibility
    val1 = truncated_log_normal(mean, std_dev, seed=seed)
    val2 = truncated_log_normal(mean, std_dev, seed=seed)
    assert val1 == val2


def test_rand_shyp_vectorized() -> None:
    samples = rand_shyp(size=200, seed=0)
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= -0.5)
    assert np.all(samples <= 0.5)

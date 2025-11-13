import numpy as np
import pytest

from qcore import src_site_dist


def test_calc_rrup_rjb_single_point():
    # Simple test with a single fault point and a single location
    srf_points = np.array([[172.0, -43.0, 5.0]])
    locations = np.array([[172.0, -43.0, 0.0]])

    out = src_site_dist.calc_rrup_rjb(srf_points, locations)
    assert len(out) == 2
    rrup, rjb = out
    assert rrup.shape == (1,)
    assert rjb.shape == (1,)
    assert rrup.dtype == np.float32
    assert rjb.dtype == np.float32
    # Since location is directly above the fault point, rjb should be ~0
    assert rjb[0] == pytest.approx(0.0, abs=1e-2)
    # rrup should be approximately the depth difference
    assert rrup[0] > 0


def test_calc_rrup_rjb_multiple_points():
    # Multiple fault points and multiple locations
    srf_points = np.array(
        [
            [172.0, -43.0, 5.0],
            [172.1, -43.1, 5.0],
            [172.2, -43.2, 5.0],
        ]
    )
    locations = np.array(
        [
            [172.0, -43.0, 0.0],
            [172.5, -43.5, 0.0],
        ]
    )

    out = src_site_dist.calc_rrup_rjb(srf_points, locations)
    assert len(out) == 2
    rrup, rjb = out

    assert rrup.shape == (2,)
    assert rjb.shape == (2,)
    assert np.all(rrup >= 0)
    assert np.all(rjb >= 0)


def test_calc_rrup_rjb_with_return_points():
    srf_points = np.array(
        [
            [172.0, -43.0, 5.0],
            [172.1, -43.1, 5.0],
        ]
    )
    locations = np.array([[172.0, -43.0, 0.0]])

    out = src_site_dist.calc_rrup_rjb(srf_points, locations, return_rrup_points=True)
    assert len(out) == 3

    rrup, rjb, rrup_points = out

    assert rrup.shape == (1,)
    assert rjb.shape == (1,)
    assert rrup_points.shape == (1, 3)
    assert rrup_points.dtype == np.float32


def test_calc_rrup_rjb_custom_iterations():
    # Test with custom n_stations_per_iter parameter
    srf_points = np.array(
        [
            [172.0, -43.0, 5.0],
            [172.1, -43.1, 6.0],
        ]
    )
    locations = np.array(
        [
            [172.0, -43.0, 0.0],
            [172.05, -43.05, 0.0],
            [172.1, -43.1, 0.0],
        ]
    )

    out = src_site_dist.calc_rrup_rjb(srf_points, locations, n_stations_per_iter=1)

    assert len(out) == 2
    rrup, rjb = out

    assert rrup.shape == (3,)
    assert rjb.shape == (3,)
    assert np.all(rrup >= 0)
    assert np.all(rjb >= 0)


def test_calc_rrup_rjb_rjb_less_than_rrup():
    # rjb should always be <= rrup since rjb is horizontal distance
    # and rrup includes vertical component
    srf_points = np.array([[172.0, -43.0, 10.0]])
    locations = np.array(
        [
            [172.0, -43.0, 0.0],
            [172.1, -43.1, 0.0],
        ]
    )

    out = src_site_dist.calc_rrup_rjb(srf_points, locations)
    assert len(out) == 2

    rrup, rjb = out
    # rjb should be less than or equal to rrup
    assert np.all(rjb <= rrup + 1e-5)  # Small tolerance for floating point

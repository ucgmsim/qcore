import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcore import geo


@pytest.mark.parametrize(
    "test_b1, test_b2, expected_angle",
    [(0, 360, 0), (80, 180, 100), (320, 0, 40), (180, -180, 0)],
)
def test_angle_diff(test_b1: float, test_b2: float, expected_angle: float) -> None:
    assert geo.angle_diff(test_b1, test_b2) == expected_angle


@pytest.mark.parametrize(
    "test_lon1, test_lat1, test_lon2, test_lat2, test_midpoint, expected_bearing",
    [
        (120, 90, 180, 90, True, 74.99999999999997),
        (0, 0, 180, 0, False, 90),
        (-45, 0, 90, 0, True, 90),
        (170, 90, 180, 90, False, 84.99999999999996),
    ],
)
def test_ll_bearing(
    test_lon1: float,
    test_lat1: float,
    test_lon2: float,
    test_lat2: float,
    test_midpoint: bool,
    expected_bearing: float,
) -> None:
    assert (
        geo.ll_bearing(test_lon1, test_lat1, test_lon2, test_lat2, test_midpoint)
        == expected_bearing
    )


@pytest.mark.parametrize(
    "test_lon1, test_lat1, test_lon2, test_lat2, expected_dist",
    [
        (0, 0, 0, 0, 0),
        (-45, 0, 90, 180, 5009.378656493638),
        (0, 0, 0, 180, 20037.51462597455),
        (45, 0, 90, 0, 5009.378656493638),
    ],
)
def test_ll_dist(
    test_lon1: float,
    test_lat1: float,
    test_lon2: float,
    test_lat2: float,
    expected_dist: float,
) -> None:
    assert geo.ll_dist(test_lon1, test_lat1, test_lon2, test_lat2) == expected_dist


@pytest.mark.parametrize(
    "test_lon1, test_lat1, test_lon2, test_lat2, expected_mid_lon, expected_mid_lat ",
    [
        (0, 0, 0, 0, 0, 0),
        (90, 0, 0, 0, 45, 0),
        (0, 90, 0, -90, 0, 0),
        (-10, 10, 15, 80, -6.323728615674871, 45.34616693081143),
    ],
)
def test_ll_mid(
    test_lon1: float,
    test_lat1: float,
    test_lon2: float,
    test_lat2: float,
    expected_mid_lon: float,
    expected_mid_lat: float,
) -> None:
    assert geo.ll_mid(test_lon1, test_lat1, test_lon2, test_lat2) == (
        expected_mid_lon,
        expected_mid_lat,
    )


@pytest.mark.parametrize(
    "test_lat1, test_lon1, test_distance, test_bearing, expected_lat, expected_lon ",
    [
        (0, 0, 0, 0, 0, 0),
        (90, 0, 10, 0, 89.91016849975625, 0),
        (-80, 50, 19, 180, -80.1706798504624, 50.0),
        (-90, 150, 0, 90, -90, 150),
    ],
)
def test_ll_shift(
    test_lat1: float,
    test_lon1: float,
    test_distance: float,
    test_bearing: float,
    expected_lat: float,
    expected_lon: float,
) -> None:
    assert geo.ll_shift(test_lat1, test_lon1, test_distance, test_bearing) == (
        expected_lat,
        expected_lon,
    )


@pytest.mark.parametrize(
    "test_angles, output_degrees",
    [
        ([[40, 1], [270, 1]], 335),
        ([[45, 10], [180, 1], [112.5, 2]], 59.252104114837415),
        ([[45, 1], [180, 1], [112.5, 2]], 112.5),
        ([[45, 1], [180, 1]], 112.49999999999999),
    ],
)
def test_avg_wbearing(test_angles: list[list[float]], output_degrees: float) -> None:
    assert geo.avg_wbearing(test_angles) == output_degrees


@given(target_bearing=st.floats(0, 360))
def test_oriented_bearing_wrt_normal(target_bearing: float) -> None:
    to_direction = np.array(
        [np.cos(np.radians(target_bearing)), np.sin(np.radians(target_bearing)), 0]
    )
    from_direction = np.array([1, 0, 0])
    up_direction = np.array([0, 0, 1])
    calculated_bearing = geo.oriented_bearing_wrt_normal(
        from_direction, to_direction, up_direction
    )
    assert (calculated_bearing == pytest.approx(target_bearing, abs=0.1)) or (
        target_bearing == pytest.approx(360, abs=0.1)
        and calculated_bearing == pytest.approx(0, abs=0.1)
    )


@pytest.mark.parametrize(
    "p, q, r, expected_distance",
    [
        # Point lies on the segment
        ([1, 0], [2, 0], [0, 0], 0.0),
        # Point is off the segment, perpendicular distance
        ([1, 1], [2, 0], [0, 0], 1.0),
        # Point coincides with one endpoint
        ([0, 0], [2, 0], [0, 0], 0.0),
        # Closest point is one of the end points
        ([3, 4], [2, 0], [0, 0], np.sqrt((3 - 2) ** 2 + 4**2)),
        # Vertical segment
        ([1, 1], [0, 2], [0, 0], 1.0),
        # Horizontal segment
        ([1, 1], [2, 0], [0, 0], 1.0),
        # Diagonal segment
        ([1, 0], [2, 2], [0, 0], np.sqrt(0.5)),
    ],
)
def test_point_to_segment_distance(
    p: list[float], q: list[float], r: list[float], expected_distance: float
) -> None:
    """Test the point_to_segment_distance function with various cases."""
    assert geo.point_to_segment_distance(p, q, r) == pytest.approx(expected_distance)


def test_point_to_segement_degenerate() -> None:
    """Test the failure case of a degenerate line."""
    with pytest.raises(ValueError):
        geo.point_to_segment_distance([1, 1], [0, 0], [0, 0])

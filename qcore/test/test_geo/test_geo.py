import numpy as np
import scipy as sp
import pytest

from qcore import geo
from qcore.test.tool import utils


@pytest.mark.parametrize(
    "test_b1, test_b2, expected_angle",
    [(0, 360, 0), (80, 180, 100), (320, 0, 40), (180, -180, 0)],
)
def test_angle_diff(test_b1, test_b2, expected_angle):
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
    test_lon1, test_lat1, test_lon2, test_lat2, test_midpoint, expected_bearing
):
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
def test_ll_dist(test_lon1, test_lat1, test_lon2, test_lat2, expected_dist):
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
    test_lon1, test_lat1, test_lon2, test_lat2, expected_mid_lon, expected_mid_lat
):
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
    test_lat1, test_lon1, test_distance, test_bearing, expected_lat, expected_lon
):
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
def test_avg_wbearing(test_angles, output_degrees):
    assert geo.avg_wbearing(test_angles) == output_degrees


@pytest.mark.parametrize(
    "test_lonlat, output_points",
    [
        ([[175, -45]], [[1757630.64073127, 5015103.82859388]]),
        (
            [[172, -41], [173, -43]],
            [[1515897.8655543, 5460761.41058073], [1600000.0, 5239185.20382672]],
        ),
    ],
)
def test_wgs_nztm2000x(test_lonlat, output_points):
    test_points = geo.wgs_nztm2000x(test_lonlat)
    sample_output_points = np.array(output_points)
    utils.compare_np_array(test_points, sample_output_points)


@pytest.mark.parametrize(
    "p, expected_p", [(np.array([1, 0, 0]), np.array([1, 0, 0, 1]))]
)
def test_homogenise_point(p, expected_p):
    assert np.allclose(geo.homogenise_point(p), expected_p)


@pytest.mark.parametrize(
    "p, q, r, dual",
    [
        (
            np.array([1, 0, 0, 1]),
            np.array([1, 1, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 1, 0]),
        )
    ],
)
def test_projective_span(p, q, r, dual):
    assert np.allclose(geo.projective_span(p, q, r), dual)


@pytest.mark.parametrize(
    "p, q, r, dual",
    [
        (
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0, 0]),
        ),
        (
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([-1, -1, -1, 1]),
        ),
    ],
)
def test_plane_from_three_points(p, q, r, dual):
    assert np.allclose(geo.plane_from_three_points(p, q, r), dual)


@pytest.mark.parametrize(
    "pi, p, q, dual",
    [
        (
            np.array([1, 0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1, 0]),
        ),
        (
            np.array([0, 0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([-1, -1, 0, 1]),
        ),
    ],
)
def test_orthogonal_plane(pi, p, q, dual):
    assert np.allclose(geo.orthogonal_plane(pi, p, q), dual)


@pytest.mark.parametrize(
    "p1_corners, p2_corners, p1_closest_point, p2_closest_point",
    [
        # edge to interior point
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            np.array([[1, 1.5, 1], [0, 1.5, 1], [0, 0, 1.5], [1, 0, 1.5]]),
            np.array([0.5, 1, 0]),
            np.array([0.5, 1.35, 1.05]),
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            np.array(
                [
                    [-1 / 2, 1 / 2, -3 / 4 - 1 / 2],
                    [-1 / 2, 1 / 4, -1 / 4 - 1 / 2],
                    [-1 / 2, 3 / 4, -1 / 4 - 1 / 2],
                    [-1 / 2, 1 / 2, 1 / 2],
                ]
            ),
            np.array([0, 1 / 2, 0]),
            np.array([-1 / 2, 1 / 2, 0]),
        ),
        # corner-to-corner
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            np.array([[1, 0, 1], [0, 0, 3 / 4], [0, 1, 1 / 2], [1, 1, 3 / 4]]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0.5]),
        ),
        # corner to interior
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            np.array(
                [
                    [1 / 2, 1 / 2, 1 / 4],
                    [1 / 2, 1 / 4, 3 / 4],
                    [1 / 2, 1 / 2, 1],
                    [1 / 2, 3 / 4, 3 / 4],
                ]
            ),
            np.array([1 / 2, 1 / 2, 0]),
            np.array([1 / 2, 1 / 2, 1 / 4]),
        ),
        # edge to edge
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            np.array(
                [
                    [-1 / 2, 1 / 2, -1],
                    [-1, 1 / 2, -1],
                    [-1, 1 / 2, 1],
                    [-1 / 2, 1 / 2, 1],
                ]
            ),
            np.array([0, 1 / 2, 0]),
            np.array([-1 / 2, 1 / 2, 0]),
        ),
    ],
)
def test_closest_points(p1_corners, p2_corners, p1_closest_point, p2_closest_point):
    (p1_computed, p2_computed) = geo.closest_points_between_planes(
        p1_corners, p2_corners
    )
    closest_distance = sp.spatial.distance.cdist(
        p1_closest_point.reshape((1, -1)), p2_closest_point.reshape((1, -1))
    )
    computed_distance = sp.spatial.distance.cdist(
        p1_computed.reshape((1, -1)), p2_computed.reshape((1, -1))
    )
    # distance check
    assert np.allclose(
        closest_distance,
        computed_distance,
        atol=1e-7,
    )
    # in plane check
    assert geo.in_finite_plane(p1_corners, p1_computed)
    assert geo.in_finite_plane(p2_corners, p2_computed)

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcore import geo


@pytest.mark.parametrize(
    "test_b1, test_b2, expected_angle",
    [
        (0, 360, 0),
        (80, 180, 100),
        (320, 0, 40),
        (180, -180, 0),
        (
            10,
            200,
            -170,
        ),  # Test case where result > 180: (200-10)%360=190, returns 190-360=-170
        (0, 270, -90),  # Another test case: (270-0)%360=270, returns 270-360=-90
    ],
)
def test_angle_diff(test_b1: float, test_b2: float, expected_angle: float) -> None:
    assert geo.angle_diff(test_b1, test_b2) == expected_angle


@pytest.mark.parametrize(
    "test_lon1, test_lat1, test_lon2, test_lat2, test_midpoint, expected_bearing",
    [
        (120, 90, 180, 90, True, 75),
        (0, 0, 180, 0, False, 90),
        (-45, 0, 90, 0, True, 90),
        (170, 90, 180, 90, False, 85),
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
    assert geo.ll_bearing(
        test_lon1, test_lat1, test_lon2, test_lat2, test_midpoint
    ) == pytest.approx(expected_bearing)


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
        ([[45, 1], [180, 1]], 112.5),
    ],
)
def test_avg_wbearing(test_angles: list[list[float]], output_degrees: float) -> None:
    assert geo.avg_wbearing(test_angles) == pytest.approx(output_degrees)


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


def test_get_distances_single_reference() -> None:
    """Test get_distances with a single reference point."""
    locations = np.array([[174.7645, -36.8509], [174.7787, -41.2924]])  # lon, lat
    ref_lon, ref_lat = 174.7645, -36.8509

    distances = geo.get_distances(locations, ref_lon, ref_lat)

    assert distances.shape == (2,)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)  # Same location
    assert distances[1] > 0  # Different location


def test_get_distances_multiple_references() -> None:
    """Test get_distances with multiple reference points."""
    locations = np.array([[174.7645, -36.8509], [174.7787, -41.2924]])
    ref_lons = np.array([174.7645, 174.7787])
    ref_lats = np.array([-36.8509, -41.2924])

    distances = geo.get_distances(locations, ref_lons, ref_lats)

    assert distances.shape == (2, 2)
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert distances[1, 1] == pytest.approx(0.0, abs=1e-6)


def test_closest_location() -> None:
    """Test closest_location function."""
    locations = np.array(
        [[174.7645, -36.8509], [174.7787, -41.2924], [172.6366, -43.5320]]
    )
    ref_lon, ref_lat = 174.7650, -36.8510

    idx, distance = geo.closest_location(locations, ref_lon, ref_lat)

    assert idx == 0  # First location should be closest
    assert isinstance(idx, int)
    assert isinstance(distance, float)
    assert distance < 1.0  # Should be very close (less than 1 km)


def test_gen_mat() -> None:
    """Test gen_mat transformation matrix generation."""
    mrot, mlon, mlat = 45.0, 174.0, -43.0

    amat, ainv = geo.gen_mat(mrot, mlon, mlat)

    assert amat.shape == (9,)  # Flattened 3x3 matrix
    assert ainv.shape == (9,)
    # Test that the matrices are valid (not all zeros)
    assert np.any(amat != 0)
    assert np.any(ainv != 0)
    # Test basic properties of transformation matrices
    amat_2d = amat.reshape(3, 3)
    ainv_2d = ainv.reshape(3, 3)
    # The determinant should be close to 1
    assert np.linalg.det(amat_2d) == pytest.approx(1)
    assert amat_2d @ ainv_2d == pytest.approx(np.eye(3, dtype=amat_2d.dtype), abs=1e-6)


def test_xy2ll_and_ll2xy_roundtrip() -> None:
    """Test that xy2ll and ll2xy are inverses of each other."""
    mrot, mlon, mlat = 0.0, 174.0, -43.0
    amat, ainv = geo.gen_mat(mrot, mlon, mlat)

    # Test with some XY offsets
    xy_km = np.array([[10.0, 20.0], [5.0, -15.0], [0.0, 0.0]])

    # Convert XY to lat/lon
    ll = geo.xy2ll(xy_km, amat)

    # Convert back to XY
    xy_recovered = geo.ll2xy(ll, ainv)

    assert xy_recovered == pytest.approx(xy_km, abs=1e-3)


def test_gp2xy() -> None:
    """Test gp2xy grid point to XY conversion."""
    gp = np.array([[0, 0], [1, 0], [0, 1], [2, 2]])
    nx, ny = 3, 3
    hh = 1.0  # 1 km spacing

    xy = geo.gp2xy(gp, nx, ny, hh)

    assert xy.shape == (4, 2)
    # Check that origin is at center
    # For nx=3, ny=3, center should be at index (1, 1)
    # gp[0,0] should be offset by -1.0 in both X and Y
    assert xy[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert xy[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_rotation_matrix() -> None:
    """Test rotation_matrix function."""
    angle = np.pi / 4  # 45 degrees

    rot = geo.rotation_matrix(angle)

    assert rot.shape == (2, 2)
    # Test that it's a proper rotation matrix (determinant = 1)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)
    # Test that it rotates a vector correctly
    vec = np.array([1.0, 0.0])
    rotated = rot @ vec
    expected = np.array([np.cos(angle), np.sin(angle)])
    assert rotated == pytest.approx(expected, abs=1e-6)


def test_path_from_corners_return() -> None:
    """Test path_from_corners when returning points."""
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    path = geo.path_from_corners(corners, output=None, min_edge_points=5, close=True)

    assert path is not None
    assert len(path) >= 20  # At least 5 points per edge * 4 edges
    assert path[0] == path[-1]  # Should be closed


def test_path_from_corners_output_file(tmp_path: Path) -> None:
    """Test path_from_corners when writing to a file."""
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    output_file = tmp_path / "test_path.txt"

    result = geo.path_from_corners(
        corners, output=str(output_file), min_edge_points=5, close=True
    )

    assert result is None
    assert output_file.exists()
    with open(output_file, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 20


def test_ll_cross_along_track_dist() -> None:
    """Test ll_cross_along_track_dist function."""
    # Define a great circle path from point 1 to point 2
    lon1, lat1 = 0.0, 0.0
    lon2, lat2 = 10.0, 0.0  # Path along equator
    lon3, lat3 = 5.0, 1.0  # Point slightly north of the path

    cross_track, along_track = geo.ll_cross_along_track_dist(
        lon1, lat1, lon2, lat2, lon3, lat3
    )

    # The function returns values, test that they are computed
    assert isinstance(cross_track, (float, np.floating))
    assert isinstance(along_track, (float, np.floating))
    # The cross track distance should have absolute value around 111 km (1 degree latitude)
    assert abs(cross_track) > 100
    assert abs(cross_track) < 120


def test_ll_cross_along_track_dist_with_precomputed() -> None:
    """Test ll_cross_along_track_dist with precomputed bearings and distance."""
    lon1, lat1 = 0.0, 0.0
    lon2, lat2 = 10.0, 0.0
    lon3, lat3 = 5.0, 1.0

    # Precompute values
    a12 = np.radians(geo.ll_bearing(lon1, lat1, lon2, lat2))
    a13 = np.radians(geo.ll_bearing(lon1, lat1, lon3, lat3))
    d13 = geo.ll_dist(lon1, lat1, lon3, lat3)

    cross_track_precomputed, along_track_precomputed = geo.ll_cross_along_track_dist(
        lon1, lat1, lon2, lat2, lon3, lat3, a12=a12, a13=a13, d13=d13
    )

    # Compute without precomputed values
    cross_track, along_track = geo.ll_cross_along_track_dist(
        lon1, lat1, lon2, lat2, lon3, lat3
    )

    # Results should be the same whether precomputed or not
    assert cross_track_precomputed == pytest.approx(cross_track, abs=1e-6)
    assert along_track_precomputed == pytest.approx(along_track, abs=1e-6)

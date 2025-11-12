import re

import numpy as np
import pyproj
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcore import coordinates
from qcore.coordinates import R_EARTH, SphericalProjection


@pytest.mark.parametrize(
    "coords,expected",
    [
        (np.array([-43.5320, 172.6366]), np.array([5180040.61473068, 1570636.6812821])),
        (
            np.array([-43.5320, 172.6366, 1]),
            np.array([5180040.61473068, 1570636.6812821, 1]),
        ),
        (
            np.array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 0]]),
            np.array(
                [
                    [5.92021456e06, 1.75731133e06, 1.00000000e02],
                    [5.42725716e06, 1.74893148e06, 0.00000000e00],
                ]
            ),
        ),
    ],
)
def test_wgs_depth_to_nztm_nominal(coords: np.ndarray, expected: np.ndarray) -> None:
    result = coordinates.wgs_depth_to_nztm(coords)
    assert np.allclose(result, expected, rtol=1e-6)
    assert result.shape == expected.shape
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "coords,expected",
    [
        (np.array([5180040.61473068, 1570636.6812821]), np.array([-43.5320, 172.6366])),
        (
            np.array([5180040.61473068, 1570636.6812821, 0]),
            np.array([-43.5320, 172.6366, 0]),
        ),
        (
            np.array(
                [
                    [5.92021456e06, 1.75731133e06, 100],
                    [5.42725716e06, 1.74893148e06, 100],
                ]
            ),
            np.array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]]),
        ),
    ],
)
def test_nztm_to_wgs_depth_nominal(coords: np.ndarray, expected: np.ndarray) -> None:
    result = coordinates.nztm_to_wgs_depth(coords)
    assert np.allclose(result, expected, rtol=1e-6)
    assert result.shape == expected.shape
    assert result.dtype == np.float64


def test_distance_between_wgs_depth_coordinates_single_point():
    # Two points in lat/lon
    point_a = np.array([-43.5320, 172.6366])
    point_b = np.array([-43.5310, 172.6376])

    dist = coordinates.distance_between_wgs_depth_coordinates(point_a, point_b)
    assert isinstance(dist, float)
    assert dist > 0  # distance should be positive


def test_distance_between_wgs_depth_coordinates_with_depth():
    point_a = np.array([-43.5320, 172.6366, 10])
    point_b = np.array([-43.5325, 172.6370, 20])

    dist = coordinates.distance_between_wgs_depth_coordinates(point_a, point_b)
    assert isinstance(dist, float)
    assert dist > 0


def test_distance_between_wgs_depth_coordinates_multiple_points():
    points_a = np.array([[-43.5320, 172.6366], [-41.2924, 174.7787]])
    points_b = np.array([[-43.5310, 172.6376], [-41.2920, 174.7790]])

    dist = coordinates.distance_between_wgs_depth_coordinates(points_a, points_b)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (2,)
    assert np.all(dist > 0)


def test_nztm_to_gc_bearing_inverse():
    origin = np.array([-43.5, 172.6])
    distance = 10.0  # km
    nztm_bearing = 45.0  # degrees

    gc_bearing = coordinates.nztm_bearing_to_great_circle_bearing(
        origin, distance, nztm_bearing
    )
    assert isinstance(gc_bearing, float)
    recovered_nztm = coordinates.great_circle_bearing_to_nztm_bearing(
        origin, distance, gc_bearing
    )
    assert isinstance(recovered_nztm, float)

    # The recovered NZTM bearing should be very close to the original
    assert np.isclose(recovered_nztm, nztm_bearing, atol=1e-6)


def latitude(
    min_value: float = -90.0, max_value: float = 90.0, **kwargs
) -> st.SearchStrategy:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        **kwargs,
    )


def longitude(
    min_value: float = -180.0, max_value: float = 180.0, **kwargs
) -> st.SearchStrategy:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        **kwargs,
    )


def test_wgs_depth_to_nztm_invalid_coordinates() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Latitude and longitude coordinates given are invalid (did you input lon, lat instead of lat, lon?)"
        ),
    ):
        coordinates.wgs_depth_to_nztm(np.array([-180.0, 0.0]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Latitude and longitude coordinates given are invalid (did you input lon, lat instead of lat, lon?)"
        ),
    ):
        coordinates.wgs_depth_to_nztm(np.array([np.nan, np.nan]))


def test_nztm_wgs_depth_invalid_coordinates() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "NZTM coordinates given are invalid (did you input x, y instead of y, x?)"
        ),
    ):
        coordinates.nztm_to_wgs_depth(np.array([1e10, 1e10]))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "NZTM coordinates given are invalid (did you input x, y instead of y, x?)"
        ),
    ):
        coordinates.nztm_to_wgs_depth(np.array([np.nan, np.nan]))


GEOD = pyproj.Geod(ellps="sphere", a=R_EARTH * 1000.0, b=R_EARTH * 1000.0)
HALF_SPHERE = R_EARTH * np.pi / 2.0  # Half-circumference of a hemisphere


@st.composite
def points_in_same_hemisphere(draw: st.DrawFn) -> tuple[float, float, float, float]:
    mlat = draw(latitude())
    mlon = draw(longitude())
    # Pick a second pair of points in the same hemisphere
    # but not the geographic hemisphere, the geometric hemisphere (+/- 90 degrees latitude)
    azimuth = draw(st.floats(min_value=-360, max_value=360))
    distance = draw(st.floats(min_value=0, max_value=HALF_SPHERE, exclude_max=True))
    lon, lat, _ = GEOD.fwd(mlon, mlat, azimuth, distance)
    return mlat, mlon, lat, lon


@given(points=points_in_same_hemisphere(), mrot=st.floats(-360, 360))
def test_projection_inverse_is_identity(
    points: tuple[float, float, float, float], mrot: float
) -> None:
    mlat, mlon, lat, lon = points
    proj = SphericalProjection(mlon, mlat, mrot)
    fwd = proj.project(lat, lon)
    back = proj.inverse(*fwd.T)
    # If lat is close to 90 degrees we should only test latitude == 90.0
    # and longitude is 0 degrees (because longitude at the poles is undefined)
    if np.isclose(np.abs(lat), 90):
        assert pytest.approx(lat, abs=1e-5) == back[0]
        assert pytest.approx(0.0, abs=1e-5) == back[1]
    else:
        assert pytest.approx(np.array([lat, lon]), rel=1e-4, abs=1e-4) == back


def test_projection_repr() -> None:
    proj = SphericalProjection(172.0, -43.0, 0.0)
    assert repr(proj) == "SphericalProjection(mlon=172.0, mlat=-43.0, mrot=0.0)"


# For this test we must exclude the poles because they don't behave
# like the other points wrt. the longitude. Namely the longitude is
# always 0 at the poles (because it is undefined), so shifting in
# longitude at the poles is equivalent to staying put (and hence the
# mlon +/- eps) tests will always fail.
@given(mlat=latitude(exclude_min=True, exclude_max=True), mlon=longitude())
def test_identity_rotation_preserves_axes(mlat: float, mlon: float) -> None:
    proj = SphericalProjection(mlon, mlat, 0)

    # The coordinate frame of reference is south is y-positive, west
    # is x-positive. This checks that south is
    eps = 1e-4
    assert proj(mlat + eps, mlon)[1] < 0.0
    assert proj(mlat - eps, mlon)[1] > 0.0
    assert pytest.approx(0.0, abs=1e-3) == proj(mlat, mlon + eps)[1]
    assert pytest.approx(0.0, abs=1e-3) == proj(mlat, mlon - eps)[1]
    assert proj(mlat, mlon + eps)[0] > 0.0
    assert proj(mlat, mlon - eps)[0] < 0.0


@given(mlat=latitude(), mlon=longitude(), mrot=st.floats(-360, 360))
def test_center_maps_to_origin(mlat: float, mlon: float, mrot: float) -> None:
    proj = SphericalProjection(mlon, mlat, mrot)
    out = proj.project(mlat, mlon)
    assert pytest.approx(np.zeros_like(out), abs=1e-3) == out


@given(points=points_in_same_hemisphere(), mrot=st.floats(-360, 360))
def test_projection_preserves_distance(
    points: tuple[float, float, float, float], mrot: float
) -> None:
    mlat, mlon, lat, lon = points
    proj = SphericalProjection(mlon, mlat, mrot)
    geod = proj.geod  # Pyproj spherical geodesic

    dist1 = geod.inv(mlon, mlat, lon, lat)[2] / 1000.0

    x, y = proj.project(lat, lon)
    dist2 = np.hypot(x, y)
    # The distances come from two different sources here, so we can
    # only test this very roughly.
    assert pytest.approx(dist1, abs=0.1) == dist2


def test_projection_preserves_depth() -> None:
    # Test that the projection does not change depth
    proj = SphericalProjection(0, 0, 0)  # Centered at the origin
    depth = 100.0
    lat, lon = 10.0, 20.0  # Arbitrary latitude and longitude

    projected = proj.project(lat, lon, depth)
    assert pytest.approx(projected[2]) == depth  # Depth should remain unchanged


def test_inverse_projection_preserves_depth() -> None:
    # Test that the projection does not change depth
    proj = SphericalProjection(0, 0, 0)  # Centered at the origin
    depth = 100.0
    x, y = 10.0, 20.0  # Arbitrary latitude and longitude

    projected = proj.inverse(x, y, depth)
    assert (
        pytest.approx(projected[2], abs=1e-6) == depth
    )  # Depth should remain unchanged


import numpy as np
import pyproj
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcore.coordinates import R_EARTH, SphericalProjection


def latitude(min_value: float = -90.0, max_value: float = 90.0, **kwargs):
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False, **kwargs)

def longitude(min_value: float = -180.0, max_value: float = 180.0, **kwargs):
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False, **kwargs)

GEOD = pyproj.Geod(ellps='sphere', a=R_EARTH * 1000.0, b=R_EARTH * 1000.0)
HALF_SPHERE = R_EARTH * np.pi / 2.0 # Half-circumference of a hemisphere

@st.composite
def points_in_same_hemisphere(draw):
    mlat = draw(latitude())
    mlon = draw(longitude())
    # Pick a second pair of points in the same hemisphere
    # but not the geographic hemisphere, the geometric hemisphere (+/- 90 degrees latitude)
    azimuth = draw(st.floats(min_value=-360, max_value=360))
    distance = draw(st.floats(min_value=0, max_value=HALF_SPHERE, exclude_max=True))
    lon, lat, _ = GEOD.fwd(mlon, mlat, azimuth, distance)
    return mlat, mlon, lat, lon


@given(points=points_in_same_hemisphere(), mrot=st.floats(-360, 360))
def test_projection_inverse_is_identity(points, mrot):
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



# For this test we must exclude the poles because they don't behave
# like the other points wrt. the longitude. Namely the longitude is
# always 0 at the poles (because it is undefined), so shifting in
# longitude at the poles is equivalent to staying put (and hence the
# mlon +/- eps) tests will always fail.
@given(mlat=latitude(exclude_min=True, exclude_max=True), mlon=longitude())
def test_identity_rotation_preserves_axes(mlat, mlon):
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
def test_center_maps_to_origin(mlat, mlon, mrot):
    proj = SphericalProjection(mlon, mlat, mrot)
    out = proj.project(mlat, mlon)
    assert pytest.approx(np.zeros_like(out), abs=1e-3) ==  out


@given(points=points_in_same_hemisphere(), mrot=st.floats(-360, 360))
def test_projection_preserves_distance(points, mrot):
    mlat, mlon, lat, lon = points
    proj = SphericalProjection(mlon, mlat, mrot)
    geod = proj.geod # Pyproj spherical geodesic

    dist1 = GEOD.inv(mlon, mlat, lon, lat)[2] / 1000.0

    x, y = proj.project(lat, lon)
    dist2 = np.hypot(x, y)
    # The distances come from two different sources here, so we can
    # only test this very roughly.
    assert pytest.approx(dist1, abs=0.1) == dist2


def test_projection_preserves_depth():
    # Test that the projection does not change depth
    proj = SphericalProjection(0, 0, 0)  # Centered at the origin
    depth = 100.0
    lat, lon = 10.0, 20.0  # Arbitrary latitude and longitude

    projected = proj.project(lat, lon, depth)
    assert pytest.approx(projected[2]) == depth  # Depth should remain unchanged

def test_inverse_projection_preserves_depth():
    # Test that the projection does not change depth
    proj = SphericalProjection(0, 0, 0)  # Centered at the origin
    depth = 100.0
    x, y = 10.0, 20.0  # Arbitrary latitude and longitude

    projected = proj.inverse(x, y, depth)
    assert pytest.approx(projected[2], abs=1e-6) == depth  # Depth should remain unchanged

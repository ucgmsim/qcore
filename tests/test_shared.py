import tempfile
from pathlib import Path

import pytest

from qcore import shared


def test_get_stations_without_locations():
    # Create a temporary station file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("# Comment line\n")
        f.write("172.5 -43.5 STAT1\n")
        f.write("173.0 -44.0 STAT2\n")
        f.write("174.5 -45.5 STAT3\n")
        station_file = f.name
    
    try:
        result = shared.get_stations(station_file, locations=False)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["STAT1", "STAT2", "STAT3"]
    finally:
        Path(station_file).unlink()


def test_get_stations_with_locations():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("172.5 -43.5 STAT1\n")
        f.write("173.0 -44.0 STAT2\n")
        station_file = f.name
    
    try:
        stations, lats, lons = shared.get_stations(station_file, locations=True)
        assert isinstance(stations, list)
        assert isinstance(lats, list)
        assert isinstance(lons, list)
        assert len(stations) == 2
        assert len(lats) == 2
        assert len(lons) == 2
        assert stations == ["STAT1", "STAT2"]
        assert lats == pytest.approx([-43.5, -44.0])
        assert lons == pytest.approx([172.5, 173.0])
    finally:
        Path(station_file).unlink()


def test_get_corners():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("# Model params\n")
        f.write("  c1= 172.164 -41.414\n")
        f.write("  c2= 172.500 -41.200\n")
        f.write("  c3= 173.000 -41.800\n")
        f.write("  c4= 172.700 -42.100\n")
        model_params_file = f.name
    
    try:
        result = shared.get_corners(model_params_file, gmt_format=False)
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0] == pytest.approx((172.164, -41.414))
        assert result[1] == pytest.approx((172.500, -41.200))
        assert result[2] == pytest.approx((173.000, -41.800))
        assert result[3] == pytest.approx((172.700, -42.100))
    finally:
        Path(model_params_file).unlink()


def test_get_corners_with_gmt_format():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("  c1= 172.164 -41.414\n")
        f.write("  c2= 172.500 -41.200\n")
        f.write("  c3= 173.000 -41.800\n")
        f.write("  c4= 172.700 -42.100\n")
        model_params_file = f.name
    
    try:
        corners, cnr_str = shared.get_corners(model_params_file, gmt_format=True)
        assert isinstance(corners, list)
        assert isinstance(cnr_str, str)
        assert len(corners) == 4
        assert "172.164 -41.414" in cnr_str
        # Check for the coordinate - may be formatted differently (172.5 vs 172.500)
        assert "172.5" in cnr_str or "172.500" in cnr_str
        assert "-41.2" in cnr_str or "-41.200" in cnr_str
    finally:
        Path(model_params_file).unlink()


@pytest.mark.parametrize(
    "station_name,expected",
    [
        ("abcdef1", True),
        ("1234567", True),
        ("abc1234", True),
        ("ABCDEF1", False),  # Contains capital
        ("abc123", False),  # Too short (only 6 chars)
        ("STAT01", False),  # Contains capitals
    ],
)
def test_is_virtual_station(station_name: str, expected: bool):
    result = shared.is_virtual_station(station_name)
    assert result == expected

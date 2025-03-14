import itertools

import numpy as np
import pytest

from qcore import coordinates, grid


def test_grid_corners():
    """Test the grid_corners function to verify corner calculations"""
    centroid = np.array([-43.5, 172.5])
    strike = 90
    dip_dir = 0
    dtop = 2.0  # km
    dbottom = 4.0  # km
    length = 10  # km
    projected_width = 5  # km
    dip = np.arctan((dbottom - dtop) / projected_width)

    corners = grid.grid_corners(
        centroid, strike, dip_dir, dtop, dbottom, length, projected_width
    )
    assert corners.shape == (4, 3)  # Should return 4 corners with (lat, lon, depth)
    assert corners[0, 2] == pytest.approx(
        dtop * 1000, abs=1e-2
    )  # Depth in meters for top corners
    assert corners[2, 2] == pytest.approx(
        dbottom * 1000, abs=1e-2
    )  # Depth in meters for bottom corners
    assert coordinates.distance_between_wgs_depth_coordinates(
        corners[1], corners[0]
    ) == pytest.approx(length * 1000, abs=1e-2)
    assert coordinates.distance_between_wgs_depth_coordinates(
        corners[2], corners[0]
    ) == pytest.approx(projected_width / np.cos(dip) * 1000, abs=1e-2)
    assert np.allclose(np.mean(corners, axis=0)[:2], centroid)


def test_coordinate_meshgrid():
    """Test the coordinate_meshgrid function to verify grid creation"""
    origin = np.array([-43.5, 172.5, 5000])  # (lat, lon, depth in m)
    x_upper = np.array([-43.45498004, 172.50037108, 5000.0])
    y_bottom = np.array([-43.50025372, 172.56184531, 5000.0])
    resolution = 1000  # 1 km resolution

    meshgrid = grid.coordinate_meshgrid(origin, x_upper, y_bottom, resolution)
    assert meshgrid.shape[2] == 3  # Should have shape (ny, nx, 3) for (lat, lon, depth)
    for i, j in itertools.product(
        range(meshgrid.shape[0] - 1), range(meshgrid.shape[1] - 1)
    ):
        assert coordinates.distance_between_wgs_depth_coordinates(
            meshgrid[i + 1, j], meshgrid[i, j]
        ) == pytest.approx(resolution, abs=1e-2)
        assert coordinates.distance_between_wgs_depth_coordinates(
            meshgrid[i, j + 1], meshgrid[i, j]
        ) == pytest.approx(resolution, abs=1e-2)

    ny, nx = meshgrid.shape[:2]
    assert nx > 1 and ny > 1  # There should be multiple grid points in both directions

    # Check if the depths are consistent
    assert np.all(meshgrid[:, :, 2] == pytest.approx(origin[2], rel=1e-2))


def test_coordinate_meshgrid_x_boundary():
    """Test the coordinate_meshgrid function when you only have one patch in x and y direction"""
    origin = np.array([-43.5, 172.5, 5000])  # (lat, lon, depth in m)
    x_upper = np.array([-43.45498004, 172.50037108, 5000.0])
    y_bottom = np.array([-43.50025372, 172.56184531, 5000.0])
    resolution = coordinates.distance_between_wgs_depth_coordinates(
        x_upper, origin
    )  # 1 km resolution

    meshgrid = grid.coordinate_patchgrid(origin, x_upper, y_bottom, resolution)
    assert meshgrid.shape[2] == 3  # Should have shape (ny, nx, 3) for (lat, lon, depth)
    ny, nx = meshgrid.shape[:2]
    assert meshgrid.shape == (1, 1, 3)
    # This one grid point should be the centre of the trapezium
    origin_nztm = coordinates.wgs_depth_to_nztm(origin)
    x_upper_nztm = coordinates.wgs_depth_to_nztm(x_upper)
    y_bottom_nztm = coordinates.wgs_depth_to_nztm(y_bottom)
    centre = x_upper_nztm + y_bottom_nztm - origin_nztm

    assert meshgrid[0, 0] == pytest.approx(
        coordinates.nztm_to_wgs_depth(centre), rel=1e-2
    )


def test_coordinate_patchgrid_different_nx_ny():
    """Test the coordinate_meshgrid function to verify nx/ny parameters"""
    origin = np.array([-43.5, 172.5, 5000])  # (lat, lon, depth in m)
    x_upper = np.array([-43.45498004, 172.50037108, 5000.0])
    y_bottom = np.array([-43.50025372, 172.56184531, 5000.0])
    ny = 3
    nx = 5
    meshgrid = grid.coordinate_patchgrid(origin, x_upper, y_bottom, nx=nx, ny=ny)
    assert meshgrid.shape == (ny, nx, 3)


def test_coordinate_patchgrid():
    """Test the coordinate_meshgrid function to verify grid creation"""
    origin = np.array([-43.5, 172.5, 5000])  # (lat, lon, depth in m)
    x_upper = np.array([-43.45498004, 172.50037108, 5000.0])
    y_bottom = np.array([-43.50025372, 172.56184531, 5000.0])
    resolution = 1000  # 1 km resolution

    meshgrid = grid.coordinate_patchgrid(origin, x_upper, y_bottom, resolution)
    assert meshgrid.shape[2] == 3  # Should have shape (ny, nx, 3) for (lat, lon, depth)
    for i, j in itertools.product(
        range(meshgrid.shape[0] - 1), range(meshgrid.shape[1] - 1)
    ):
        assert coordinates.distance_between_wgs_depth_coordinates(
            meshgrid[i + 1, j], meshgrid[i, j]
        ) == pytest.approx(resolution, abs=1e-2)
        assert coordinates.distance_between_wgs_depth_coordinates(
            meshgrid[i, j + 1], meshgrid[i, j]
        ) == pytest.approx(resolution, abs=1e-2)

    ny, nx = meshgrid.shape[:2]
    width = coordinates.distance_between_wgs_depth_coordinates(y_bottom, origin)
    length = coordinates.distance_between_wgs_depth_coordinates(x_upper, origin)
    assert ny == round(length / resolution)
    assert nx == round(width / resolution)

    assert nx > 1 and ny > 1  # There should be multiple grid points in both directions

    # Check if the depths are consistent
    assert np.all(meshgrid[:, :, 2] == pytest.approx(origin[2], rel=1e-2))


@pytest.mark.parametrize(
    "length,resolution,expected",
    [(10, 5, 3), (10, 3, 5), (10000, 1000, 11), (1500, 500, 4)],
)
def test_gridpoint_count_in_length(length: float, resolution: float, expected: int):
    """Test gridpoint_count_in_length function"""
    assert grid.gridpoint_count_in_length(length, resolution) == expected

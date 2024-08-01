"""
This module provides functions for working with planar regions defined by geographical coordinates.

Functions
---------
grid_corners
    Returns the corners of a plane from a series of parameters.

coordinate_meshgrid
    Creates a meshgrid of points in a bounded plane region.

gridpoint_count_in_length
    Calculate the number of gridpoints that fit into a given length.
"""

import numpy as np
import scipy as sp

from qcore import coordinates


def grid_corners(
    centroid: np.ndarray,
    strike: float,
    dip_dir: float,
    dtop: float,
    dbottom: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Returns the corners of a plane from a series of parameters.

    Parameters
    ----------
    centroid : np.ndarray
        The centroid of the plane.
    strike : float
        The strike of the plane (degrees).
    dip_dir : float
        The dip direction of the plane (degrees).
    dtop : float
        The top depth of the plane (in km).
    dbottom : float
        The bottom depth of the plane (in km).
    length : float
        The length of the plane (in km).
    width : float
        The width of the plane (in km).

    Returns
    -------
    np.ndarray
        The corners of the plane in the following order.

                       strike
             0      ----------->     1
              ┌─────────────────────┐
              │                     │
              │                     │
              └─────────────────────┘
             2                       3

        Corners are given in (lat, lon, depth) format.
        The output depth is in metres.
    """
    strike_direction = np.array(
        [np.cos(np.radians(strike)), np.sin(np.radians(strike))]
    )
    dip_direction = np.array([np.cos(np.radians(dip_dir)), np.sin(np.radians(dip_dir))])
    centroid_nztm = coordinates.wgs_depth_to_nztm(centroid)
    basis = (
        np.array(
            [
                [-1 / 2, -1 / 2],  # origin
                [1 / 2, -1 / 2],  # x_upper
                [-1 / 2, 1 / 2],  # y_bottom
                [1 / 2, 1 / 2],  # The last corner
            ]
        )
        * np.array([length, width])
        * 1000
    )
    corners = centroid_nztm + basis @ np.array([strike_direction, dip_direction])
    return coordinates.nztm_to_wgs_depth(
        np.c_[corners, np.array([dtop, dtop, dbottom, dbottom]) * 1000]
    )


def coordinate_meshgrid(
    origin: np.ndarray,
    x_upper: np.ndarray,
    y_bottom: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Creates a meshgrid of points in a bounded plane region.

    Given the bounds of a rectangular planar region, create a meshgrid of
    (lat, lon, depth) coordinates spaced at close to resolution metres apart
    in the strike and dip directions.

    origin                       x_upper
          ┌─────────────────────┐
          │. . . . . . . . . . .│
          │                     │
          │. . . . . . . . . . .│
          │                     │
          │. . . . . . . . . . .│
          │                     │
          │. . . . . . . . . . .│
          │                     │
          │. . . . . . . . . . .│
          │            ∧ ∧      │
          └────────────┼─┼──────┘
     y_bottom          └─┘
                    resolution

    Parameters
    ----------
    origin : np.ndarray
        Coordinates of the origin point (lat, lon, depth).
    x_upper : np.ndarray
        Coordinates of the upper x boundary (lat, lon, depth).
    y_bottom : np.ndarray
        Coordinates of the bottom y boundary (lat, lon, depth).
    resolution : float
        Resolution of the meshgrid (in metres).

    Returns
    -------
    np.ndarray
        The meshgrid of the rectangular planar region. Has shape (ny, nx), where
        ny is the number of points in the origin->y_bottom direction and nx the number of
        points in the origin->x_upper direction.
    """
    # These calculations are easier to do if the coordinates are in NZTM
    # rather than (lat, lon, depth).
    origin = coordinates.wgs_depth_to_nztm(origin)
    x_upper = coordinates.wgs_depth_to_nztm(x_upper)
    y_bottom = coordinates.wgs_depth_to_nztm(y_bottom)

    length_x = np.linalg.norm(x_upper - origin)
    length_y = np.linalg.norm(y_bottom - origin)

    nx = gridpoint_count_in_length(length_x, resolution)
    ny = gridpoint_count_in_length(length_y, resolution)

    # We first create a meshgrid of coordinates across a flat rectangle like the following
    #
    #  (0, 0)       (length_x, 0)
    #    ┌─────────┐
    #    │         │
    #    │         │
    #    │         │
    #    │         │
    #    │         │
    #    │         │
    #    │         │
    #    └─────────┘
    # (0, length_y)

    x = np.linspace(0, length_x, nx)
    y = np.linspace(0, length_y, ny)
    xv, yv = np.meshgrid(x, y)
    subdivision_coordinates = np.vstack([xv.ravel(), yv.ravel()])

    # The subdivision coordinates lie on a rectangle that has the right size,
    # but is not in the right orientation or position.  The job of the
    # transformation matrix is to rotate or shear the meshgrid to fit a plane
    # with the same orientation as the desired plane.
    # Diagramatically:
    #
    #                         ╱╲
    # ┌─────────┐            ╱  ╲
    # │         │           ╱    ╲
    # │         │          ╱      ╲
    # │         │          ╲       ╲
    # │         │  ─────>   ╲       ╲
    # │         │ tr. matrix ╲       ╲
    # │         │             ╲      ╱
    # │         │              ╲    ╱
    # └─────────┘               ╲  ╱
    #                            ╲╱
    transformation_matrix = np.vstack(
        [(x_upper - origin) / length_x, (y_bottom - origin) / length_y]
    ).T
    nztm_meshgrid = (transformation_matrix @ subdivision_coordinates).T

    # nztm_meshgrid is a grid of points along a plane with the same orientation
    # as the desired plane, but it needs to be translated back to the origin.
    nztm_meshgrid += origin

    return coordinates.nztm_to_wgs_depth(nztm_meshgrid).reshape((ny, nx, 3))


def coordinate_patchgrid(
    origin: np.ndarray, x_upper: np.ndarray, y_bottom: np.ndarray, resolution: float
) -> np.ndarray:
    """Creates a grid of patches in a bounded plane region.

    Given the bounds of a rectangular planar region, create a grid of
    (lat, lon, depth) coordinates spaced at close to resolution metres apart
    in the strike and dip directions. These coordinates are the centre of
    patches with area resolution * resolution m^2.

    Parameters
    ----------
    origin : np.ndarray
        Coordinates of the origin point (lat, lon, depth).
    x_upper : np.ndarray
        Coordinates of the upper x boundary (lat, lon, depth).
    y_bottom : np.ndarray
        Coordinates of the bottom y boundary (lat, lon, depth).
    resolution : float
        Resolution of the meshgrid (in metres).

    Returns
    -------
    np.ndarray
        The patch grid of the rectangular planar region. Has shape (ny, nx), where
        ny is the number of points in the origin->y_bottom direction and nx the number of
        points in the origin->x_upper direction.
    """
    meshgrid = coordinate_meshgrid(origin, x_upper, y_bottom, resolution)
    ny, nx = meshgrid.shape[:2]
    meshgrid = coordinates.wgs_depth_to_nztm(meshgrid.reshape((-1, 3))).reshape(
        (ny, nx, 3)
    )
    kernel = np.full((2, 2), 1 / 2)
    patch_grid = sp.spatial.convolve2d(meshgrid, kernel, mode="valid")
    return coordinates.nztm_to_wgs_depth(patch_grid.reshape((-1, 3))).reshape(
        (ny, nx, 3)
    )


def gridpoint_count_in_length(length: float, resolution: float) -> int:
    """Calculate the number of gridpoints that fit into a given length.

    Computes the number of gridpoints that fit into a given length, if each
    gridpoint is roughly resolution metres apart, and if the gridpoints
    includes the endpoints. If length = 10, and resolution = 5, then the
    function returns 3 grid points spaced as follows:

      5m    5m
    +─────+─────+

    Parameters
    ----------
    length : float
        Length to distribute grid points along.
    resolution : float
        Resolution of the grid.

    Returns
    -------
    int
        The number of gridpoints that fit into length.
    """
    return int(np.round(length / resolution + 2))

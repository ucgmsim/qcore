"""This module provides functions for working with and generating GSF files.

Functions:
- gridpoint_count_in_length:
    Calculates the number of gridpoints that fit into a given length.
- coordinate_meshgrid:
    Creates a meshgrid of points in a bounded plane region.
- write_fault_to_gsf_file:
    Writes geometry data to a GSF file.
- read_gsf:
    Parses a GSF file into a pandas DataFrame.

The GSF file format is used to define the grid of the source model for a fault.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the GSF format.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from qcore import coordinates


def gridpoint_count_in_length(length: float, resolution: float) -> int:
    """Calculate the number of gridpoints that fit into a given length.

    Computes the number of gridpoints that fit into a given length, if each
    gridpoint is roughly resolution metres apart, and if the gridpoints
    includes the endpoints. If length = 10, and resolution = 5, then the
    function returns 3 grid points spaced as follows:

      5m    5m
    +-----+-----+

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
        Resolution of the meshgrid.

    Returns
    -------
    np.ndarray
        The meshgrid of the rectangular planar region. Has shape (ny, nx), where
        ny is the number of points in the origin->y_bottom direction and nx the number of
        points in the origin->x_upper direction.
    """
    # These calculations are easier to do if the coordinates are in NZTM rather
    # than (lat, lon, depth).
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


def write_fault_to_gsf_file(
    gsf_filepath: Path,
    gsf_df: pd.DataFrame,
    resolution: int,
):
    """Writes geometry data to a GSF file.

    This code assumes that the dip is constant across all faults.

    Parameters
    ----------
    gsf_filepath : Path
        The file path pointing to the GSF file to write to.
    gsf_df : pd.DataFrame
        The GSF dataframe to write. This dataframe must have the columns length,
        width, strike, dip, rake, and meshgrid. Each row corresponds to one
        fault plane, with the meshgrid column being the discretisation of the
        fault planes.
    resolution : int
        Resolution of the meshgrid.
    """
    with open(gsf_filepath, "w", encoding="utf-8") as gsf_file_handle:
        gsf_file_handle.write(
            "# LON  LAT  DEP(km)  SUB_DX  SUB_DY  LOC_STK  LOC_DIP  LOC_RAKE  SLIP(cm)  INIT_TIME  SEG_NO\n"
        )
        number_of_points = gsf_df.apply(
            lambda row: np.prod(row["meshgrid"].shape[:2]), axis=1
        ).sum()

        # Get the number of dip gridpoints by looking at the first dimension of
        # the meshgrid of the first fault plane. See coordinate_meshgrid for an
        # explanation of meshgrid dimensions.
        n_dip = gsf_df.iloc[0]["meshgrid"].shape[0]

        gsf_file_handle.write(f"{number_of_points}\n")
        for dip_index in range(n_dip):
            for plane_index, plane in gsf_df.iterrows():
                length = plane["length"]
                width = plane["width"]
                strike = plane["strike"]
                dip = plane["dip"]
                rake = plane["rake"]
                meshgrid = plane["meshgrid"]
                strike_step = length / gridpoint_count_in_length(
                    length * 1000, resolution
                )
                dip_step = width / gridpoint_count_in_length(width * 1000, resolution)
                for point in meshgrid[dip_index]:
                    gsf_file_handle.write(
                        f"{point[1]:11.5f} {point[0]:11.5f} {point[2] / 1000:11.5e} {strike_step:11.5e} {dip_step:11.5e} {strike:6.1f} {dip:6.1f} {rake:6.1f} {-1.0:8.2f} {-1.0:8.2f} {plane_index:3d}\n"
                    )


def read_gsf(gsf_filepath: Path) -> pd.DataFrame:
    """Parse a GSF file into a pandas DataFrame.

    Parameters
    ----------
    gsf_filepath : Path
        The file handle pointing to the GSF file to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all the points in the GSF file. The DataFrame's columns are
        - lon (longitude)
        - lat (latitude)
        - depth (Kilometres below ground, i.e. depth = -10 indicates a point 10km underground).
        - sub_dx (The subdivision size in the strike direction)
        - sub_dy (The subdivision size in the dip direction)
        - strike
        - dip
        - rake
        - slip (nearly always -1)
        - init_time (nearly always -1)
        - seg_no (the fault segment this point belongs to)
    """
    with open(gsf_filepath, mode="r", encoding="utf-8") as gsf_file_handle:
        # we could use pd.read_csv with the skiprows argument, but it's not
        # as versatile as simply skipping the first n rows with '#'
        while gsf_file_handle.readline()[0] == "#":
            pass
        # NOTE: This skips the line after the last line beginning with #.
        # This is ok as this line is always the number of points in the GSF
        # file, which we do not need.
        return pd.read_csv(
            gsf_file_handle,
            sep=r"\s+",
            names=[
                "lon",
                "lat",
                "depth",
                "sub_dx",
                "sub_dy",
                "strike",
                "dip",
                "rake",
                "slip",
                "init_time",
                "seg_no",
            ],
        )

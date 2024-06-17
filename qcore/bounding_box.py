"""
Module for handling bounding boxes in 2D space.

This module provides classes and functions for working with bounding boxes in
2D space, including calculating axis-aligned and minimum area bounding boxes,
and computing various properties such as area and bearing. The bounding box
dimensions are in metres except where otherwise mentioned.

Classes:
    - BoundingBox: Represents a 2D bounding box with properties and methods for calculations.

Functions:
    - axis_aligned_bounding_box: Returns an axis-aligned bounding box containing points.
    - rotation_matrix: Returns the 2D rotation matrix for a given angle.
    - minimum_area_bounding_box: Returns the smallest rectangle bounding points.
    - minimum_area_bounding_box_for_polygons_masked: Returns a bounding box around masked polygons.

References:
    - BoundingBox wiki page: https://github.com/ucgmsim/qcore/wiki/BoundingBox
"""

import dataclasses

import numpy as np
import scipy as sp
import shapely
from qcore import geo
from shapely import Polygon


@dataclasses.dataclass
class BoundingBox:
    """Represents a 2D bounding box with properties and methods for calculations.

    Attributes
    ----------
        corners : np.ndarray
            The corners of the bounding box. The order of the corners is
            clock-wise from the top-left point with respect to latitude
            and longitude.
    """

    corners: np.ndarray

    @property
    def origin(self):
        """Returns the origin of the bounding box."""
        return np.mean(self.corners, axis=0)

    @property
    def extent_x(self):
        """Returns the extent along the x-axis of the bounding box (in km)."""
        return np.linalg.norm(np.linalg.norm(self.corners[2] - self.corners[1]) / 1000)

    @property
    def extent_y(self):
        """Returns the extent along the y-axis of the bounding box (in km)."""
        return np.linalg.norm(np.linalg.norm(self.corners[1] - self.corners[0]) / 1000)

    @property
    def bearing(self):
        """Returns the bearing of the bounding box."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        horizontal_direction = np.append(self.corners[1] - self.corners[0], 0)
        return geo.oriented_bearing_wrt_normal(
            north_direction, horizontal_direction, up_direction
        )

    @property
    def area(self):
        """Returns the area of the bounding box."""
        return self.extent_x * self.extent_y

    @property
    def polygon(self):
        """Returns a shapely geometry for the bounding box."""
        return Polygon(np.append(self.corners, np.atleast_2d(self.corners[0]), axis=0))


def axis_aligned_bounding_box(points: np.ndarray) -> BoundingBox:
    """Returns an axis-aligned bounding box containing points.

    Parameters
    ----------
    points : np.ndarray
        The points to bound.

    Returns:
        BoundingBox: The axis-aligned bounding box.
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    corners = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    return BoundingBox(corners)


def rotation_matrix(angle: float) -> np.ndarray:
    """Returns the 2D rotation matrix for a given angle.

    Parameters
    ----------
    angle : float
        The angle to rotate by in radians.

    Returns
    -------
    np.ndarray
        The 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def minimum_area_bounding_box(points: np.ndarray) -> BoundingBox:
    """Returns the smallest rectangle bounding points. The rectangle may be rotated.

    Parameters
    ----------
    points : np.ndarray
        The points to bound.

    Returns
    -------
    BoundingBox
        The minimum area bounding box.
    """
    # This is a somewhat brute-force method to obtain the minimum-area bounding
    # box of a set of points, where the bounding box is not axis-aligned and is
    # instead allowed to be rotated. The idea is to reduce the problem to the
    # far simpler axis-aligned bounding box by observing that the minimum
    # area bounding box must have a side parallel with *some* edge of the
    # convex hull of the points. By rotating the picture so that the shared
    # edge is axis-aligned, the problem is reduced to that of finding the
    # axis-aligned bounding box. Because we do not know this edge apriori,
    # we simply try it for all the edges and then take the smallest area
    # box at the end.
    convex_hull = sp.spatial.ConvexHull(points).points
    segments = np.array(
        [
            convex_hull[(i + 1) % len(convex_hull)] - convex_hull[i]
            for i in range(len(convex_hull))
        ]
    )
    # This finds the slope of each segment with respect to the axes.
    rotation_angles = -np.arctan2(segments[:, 1], segments[:, 0])

    # Create a list of rotated bounding boxes by rotating each rotation angle,
    # and then finding the axis-aligned bounding box of the convex hull. This
    # creates a list of boxes that are each parallel to a different segment.
    bounding_boxes = [
        axis_aligned_bounding_box(convex_hull @ geo.rotation_matrix(angle).T)
        for angle in rotation_angles
    ]

    minimum_rotation_angle, minimum_bounding_box = min(
        zip(rotation_angles, bounding_boxes), key=lambda rot_box: rot_box[1].area
    )
    return BoundingBox(
        # rotating by -minimum_rotation_angle we undo the rotation applied
        # to obtain bounding_boxes.
        minimum_bounding_box.corners
        @ geo.rotation_matrix(-minimum_rotation_angle).T
    )


def minimum_area_bounding_box_for_polygons_masked(
    must_include: list[Polygon], may_include: list[Polygon], mask: Polygon
) -> BoundingBox:
    """
    Return the minimum area bounding box for a list of polygons masked by
    another polygon.

    Parameters
    ----------
    must_include : list[Polygon]
        List of polygons the bounding box must include.
    may_include : list[Polygon]
        List of polygons the bounding box will include portions of, when inside of mask.
    mask : Polygon
        The masking polygon.

    Returns
    -------
    BoundingBox
        The smallest box containing all the points of `must_include`, and all the
        points of `may_include` that lie within the bounds of `mask`.

    """
    may_include_polygon = shapely.normalize(shapely.union_all(may_include))
    must_include_polygon = shapely.normalize(shapely.union_all(must_include))
    bounding_polygon = shapely.normalize(
        shapely.union(
            must_include_polygon, shapely.intersection(may_include_polygon, mask)
        )
    )

    if isinstance(bounding_polygon, Polygon):
        return bounding_box.minimum_area_bounding_box(
            np.array(bounding_polygon.exterior.coords)
        )
    return bounding_box.minimum_area_bounding_box(
        np.vstack([np.array(geom.exterior.coords) for geom in bounding_polygon.geoms])
    )

import copy

import numpy as np
import pytest
import shapely

from qcore import bounding_box, geo
from qcore.bounding_box import BoundingBox


@pytest.fixture
def dummy_box() -> BoundingBox:
    return BoundingBox(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


def test_bounding_box_extent_x(dummy_box: BoundingBox):
    np.testing.assert_allclose(dummy_box.extent_x, 1 / 1000)


def test_bounding_box_extent_y(dummy_box: BoundingBox):
    np.testing.assert_allclose(dummy_box.extent_y, 1 / 1000)


def test_bounding_box_rotational_invariance(dummy_box: BoundingBox):
    # if we rotate past 90 degrees, x and y will flip.
    for rotation_angle in range(90):
        rotation_transformation = geo.rotation_matrix(np.radians(rotation_angle))
        bounding_box = BoundingBox(dummy_box.corners @ rotation_transformation.T)
        np.testing.assert_allclose(bounding_box.extent_x, dummy_box.extent_x)
        np.testing.assert_allclose(bounding_box.extent_y, dummy_box.extent_y)
        np.testing.assert_allclose(bounding_box.area, dummy_box.area)


def test_bounding_box_bearing(dummy_box: BoundingBox):
    for rotation_angle in range(90):
        rotation_transformation = geo.rotation_matrix(np.radians(rotation_angle))
        bounding_box = BoundingBox(dummy_box.corners @ rotation_transformation.T)
        np.testing.assert_allclose(bounding_box.bearing, rotation_angle)


def test_axis_aligned_boxes(dummy_box):
    # basic check, all the default boxes should be axis aligned
    np.testing.assert_allclose(
        dummy_box.corners,
        bounding_box.axis_aligned_bounding_box(dummy_box.corners).corners,
    )
    # if we add points inside the bounding box we do not change the bounding box.
    grid = np.mgrid[0:1:0.1, 0:1:0.1].reshape((2, -1)).T
    np.testing.assert_allclose(
        dummy_box.corners,
        bounding_box.axis_aligned_bounding_box(
            np.append(dummy_box.corners, grid, axis=0)
        ).corners,
    )


def test_minimum_area_bounding_boxes(dummy_box):
    # basic check, all the default boxes should be minimum area bounding boxes
    np.testing.assert_allclose(
        dummy_box.corners,
        bounding_box.minimum_area_bounding_box(dummy_box.corners).corners,
    )

    grid = np.mgrid[0:1:0.1, 0:1:0.1].reshape((2, -1)).T
    points = np.append(dummy_box.corners, grid, axis=0)
    # testing minimum area bounding boxes work at different rotation angles
    for rotation_angle in range(90):
        rotation_transformation = geo.rotation_matrix(np.radians(rotation_angle))
        rotated_points = points @ rotation_transformation.T
        rotated_bounding_box = bounding_box.minimum_area_bounding_box(rotated_points)
        # the orientation of rotated_points may not be the same as the
        # bounding box, so we sort the points in both cases before comparing.
        np.testing.assert_allclose(
            np.sort(rotated_bounding_box.corners, axis=0),
            np.sort(rotated_points[:4], axis=0),
            atol=1e-16,
        )


def test_minimum_area_bounding_box_for_polygons_masked():
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    must_include = shapely.Polygon(corners)
    may_include = shapely.Point(2, 2)
    mask = copy.deepcopy(must_include)
    np.testing.assert_allclose(
        np.sort(
            bounding_box.minimum_area_bounding_box_for_polygons_masked(
                must_include, may_include, mask
            ).corners,
            axis=0,
        ),
        np.sort(corners[:4], axis=0),
        atol=1e-16,
    )

import numpy as np

from qcore import point_in_polygon


def test_is_inside_postgis_point_inside():
    # Square polygon from (0,0) to (10,10)
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    point = np.array([5.0, 5.0])

    result = point_in_polygon.is_inside_postgis(polygon, point)
    assert result == 1


def test_is_inside_postgis_point_outside():
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    point = np.array([15.0, 15.0])

    result = point_in_polygon.is_inside_postgis(polygon, point)
    assert result == 0


def test_is_inside_postgis_point_on_edge():
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # Use a point that's exactly on an edge with proper alignment
    point = np.array([5.0, 0.0])

    result = point_in_polygon.is_inside_postgis(polygon, point)
    assert result == 2  # On edge


def test_is_inside_postgis_parallel_single_point():
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array([[5.0, 5.0]])

    result = point_in_polygon.is_inside_postgis_parallel(points, polygon)
    assert result.shape == (1,)
    assert result[0]


def test_is_inside_postgis_parallel_multiple_points():
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array([[5.0, 5.0], [15.0, 15.0], [0.0, 5.0]])

    result = point_in_polygon.is_inside_postgis_parallel(points, polygon)
    assert result.shape == (3,)
    assert result[0]  # inside
    assert not result[1]  # outside
    assert result[2]  # on edge (returns 2, which is truthy)


def test_is_inside_postgis_triangle():
    # Triangle polygon
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    point_inside = np.array([5.0, 3.0])
    point_outside = np.array([15.0, 15.0])  # Clearly outside

    assert point_in_polygon.is_inside_postgis(polygon, point_inside) == 1
    assert point_in_polygon.is_inside_postgis(polygon, point_outside) == 0

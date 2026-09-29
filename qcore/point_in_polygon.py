"Vectorised numpy routines for point-in-polygon checks."

import itertools
from typing import Literal

import numpy as np
import numpy.typing as npt

from qcore.typing import TNFloat


def _edge_terms(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray | float,
    y1: np.ndarray | float,
    x2: np.ndarray | float,
    y2: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the ray-casting test of points against polygon edges.

    All arguments broadcast together. The edge runs from (x1, y1) to
    (x2, y2).

    Parameters
    ----------
    px, py : np.ndarray
        Point coordinates.
    x1, y1, x2, y2 : np.ndarray | float
        Edge endpoint coordinates.

    Returns
    -------
    on_boundary : np.ndarray
        True where the point lies on the edge.
    crossing : np.ndarray
        +1 / -1 where the edge crosses the horizontal ray through the point
        (by edge orientation), 0 otherwise.
    """
    dx = px - x1
    dy = py - y1
    dx2 = px - x2
    dy2 = py - y2

    f = (dx - dx2) * dy - dx * (dy - dy2)
    on_boundary = (f == 0.0) & (dx * dx2 <= 0) & (dy * dy2 <= 0)

    crosses = ((dy >= 0) & (dy2 < 0)) | ((dy2 >= 0) & (dy < 0))
    crossing = (crosses & (f > 0)).astype(np.int64) - (crosses & (f < 0))
    return on_boundary, crossing


def is_inside_postgis(
    polygon: npt.NDArray[TNFloat], point: npt.NDArray[TNFloat]
) -> Literal[0, 1, 2]:
    """Function that checks if a point is inside a polygon.

    Parameters
    ----------
    polygon : np.ndarray
        List of points that define the polygon e.g. [[x1, y1], [x2, y2], ...].
    point : np.ndarray
        Point to test [x, y].

    Returns
    -------
    int
        0 if the point is outside the polygon
        1 if the point is inside the polygon
        2 if the point is on the polygon
    """
    polygon = np.asarray(polygon)
    point = np.asarray(point)
    # Edges run between consecutive vertices (the polygon is not implicitly
    # closed), vectorised over edges.
    on_boundary, crossing = _edge_terms(
        point[0],
        point[1],
        polygon[:-1, 0],
        polygon[:-1, 1],
        polygon[1:, 0],
        polygon[1:, 1],
    )
    if on_boundary.any():
        return 2
    return 1 if crossing.sum() != 0 else 0


def is_inside_postgis_parallel(
    points: npt.NDArray[TNFloat], polygon: npt.NDArray[TNFloat]
) -> npt.NDArray[np.bool_]:
    """
    Function that checks if a set of points is inside a polygon (vectorised).

    Parameters
    ----------
    points : np.ndarray
        List of points that define the polygon e.g. [[x1, y1], [x2, y2], ...]
    polygon : np.ndarray
        List of points that define the point e.g. [x, y]

    Returns
    -------
    np.ndarray
        List of boolean values that indicate if the point is inside the polygon
    """
    polygon = np.asarray(polygon)
    points = np.asarray(points)
    n_points = len(points)
    on_boundary = np.zeros(n_points, dtype=np.bool_)
    intersections = np.zeros(n_points, dtype=np.int64)

    # Sort points by y so that, for each edge, only the points whose y lies
    # within the edge's y-extent are tested. Points outside that band can
    # neither cross the edge's ray nor lie on it, so the result is identical
    # to testing every point against every edge.
    order = np.argsort(points[:, 1], kind="stable")
    px = points[order, 0]
    py = points[order, 1]

    for (x1, y1), (x2, y2) in itertools.pairwise(polygon):
        start = np.searchsorted(py, min(y1, y2), side="left")
        stop = np.searchsorted(py, max(y1, y2), side="right")
        if start == stop:
            continue
        band = slice(start, stop)
        edge_boundary, crossing = _edge_terms(px[band], py[band], x1, y1, x2, y2)
        on_boundary[band] |= edge_boundary
        intersections[band] += crossing

    result = np.empty(n_points, dtype=np.bool_)
    # Both inside (1) and on-boundary (2) points count as inside.
    result[order] = on_boundary | (intersections != 0)
    return result

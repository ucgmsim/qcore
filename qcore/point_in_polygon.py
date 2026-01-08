"Numba routines for point-in-polygon checks."

from typing import Literal

import numba
import numpy as np
import numpy.typing as npt
from numba import jit, njit

from qcore.typing import TNFloat


@jit(nopython=True)
def is_inside_postgis(
    polygon: npt.NDArray[TNFloat], point: npt.NDArray[TNFloat]
) -> Literal[0, 1, 2]:  # pragma: no cover
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
    length = len(polygon)
    intersections = 0

    dx2 = point[0] - polygon[0][0]
    dy2 = point[1] - polygon[0][1]
    jj = 1

    while jj < length:
        dx = dx2
        dy = dy2
        dx2 = point[0] - polygon[jj][0]
        dy2 = point[1] - polygon[jj][1]

        # Check if the point is on the polygon
        F = (dx - dx2) * dy - dx * (dy - dy2)  # noqa: N806
        if 0.0 == F and dx * dx2 <= 0 and dy * dy2 <= 0:
            return 2

        if (dy >= 0 > dy2) or (dy2 >= 0 > dy):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1

        jj += 1

    return 1 if intersections != 0 else 0


@njit(parallel=True)
def is_inside_postgis_parallel(
    points: npt.NDArray[TNFloat], polygon: npt.NDArray[TNFloat]
) -> npt.NDArray[np.bool_]:  # pragma: no cover
    """
    Function that checks if a set of points is inside a polygon in parallel

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
    ln = len(points)
    D = np.empty(ln, dtype=np.bool_)  # noqa: N806
    for i in numba.prange(ln):  # type: ignore
        D[i] = is_inside_postgis(polygon, points[i])
    return D

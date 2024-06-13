from numba import jit, njit
import numba
import numpy as np


@jit(nopython=True)
def is_inside_postgis(polygon: np.ndarray, point: np.ndarray):
    """
    Function that checks if a point is inside a polygon
    Based on solutions found here
    (https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python)

    Parameters
    ----------
    polygon : np.ndarray
        List of points that define the polygon e.g. [[x1, y1], [x2, y2], ...]
    point : np.ndarray
        List of points that define the point e.g. [x, y]

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
        F = (dx - dx2) * dy - dx * (dy - dy2)
        if 0.0 == F and dx * dx2 <= 0 and dy * dy2 <= 0:
            return 2

        if (dy >= 0 > dy2) or (dy2 >= 0 > dy):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1

        jj += 1

    return intersections != 0


@njit(parallel=True)
def is_inside_postgis_parallel(points: np.ndarray, polygon: np.ndarray):
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
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_postgis(polygon, points[i])
    return D

"""
Module for coordinate conversions between WGS84 (latitude and longitude) and
NZTM (New Zealand Transverse Mercator) coordinate systems.

References
----------
This module provides functions for converting coordinates between WGS84 and NZTM coordinate systems.
See LINZ[0] for a description of the NZTM coordinate system.

[0]: https://www.linz.govt.nz/guidance/geodetic-system/coordinate-systems-used-new-zealand/projections/new-zealand-transverse-mercator-2000-nztm2000
"""

import numpy as np
import numpy.typing as npt
import pyproj

from qcore import geo

# Module level conversion constants for internal use only.
_WGS_CODE = 4326
_NZTM_CODE = 2193

_WGS2NZTM = pyproj.Transformer.from_crs(_WGS_CODE, _NZTM_CODE)
_NZTM2WGS = pyproj.Transformer.from_crs(_NZTM_CODE, _WGS_CODE)


def wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.

    Parameters
    ----------
    wgs_depth_coordinates : np.ndarray
        An array of shape (N, 3), (N, 2), (2,) or (3,) containing WGS84
        coordinates latitude, longitude and, optionally, depth.

    Returns
    -------
    np.ndarray
        An array with the same shape as wgs_depth_coordinates containing NZTM
        coordinates y, x and, optionally, depth.

    Raises
    ------
    ValueError
        If the given coordinates are not in the valid range to be converted
        to NZTM coordinates.

    Examples
    --------
    >>> wgs_depth_to_nztm(np.array([-43.5320, 172.6366]))
    array([5180040.61473068, 1570636.6812821])
    >>> wgs_depth_to_nztm(np.array([-43.5320, 172.6366, 1]))
    array([5180040.61473068, 1570636.6812821 ,       1.        ])
    >>> wgs_depth_to_nztm(np.array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]]))
    array([[5.92021456e+06, 1.75731133e+06, 1.00000000e+02],
           [5.42725716e+06, 1.74893148e+06, 0.00000000e+00]])
    """
    nztm_coordinates = np.array(_WGS2NZTM.transform(*wgs_depth_coordinates.T)).T
    if not np.all(np.isfinite(nztm_coordinates)):
        raise ValueError(
            "Latitude and longitude coordinates given are invalid (did you input lon, lat instead of lat, lon?)"
        )

    return nztm_coordinates


def nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    """
    Convert NZTM coordinates (y, x, depth) to WGS84 coordinates.

    Parameters
    ----------
    nztm_coordinates : np.ndarray
        An array of shape (N, 3), (N, 2), (2,) or (3,) containing NZTM
        coordinates y, x and, optionally, depth.

    Returns
    -------
    np.ndarray
        An array with the same shape as nztm_coordinates containing WGS84
        coordinates latitude, longitude and, optionally, depth.

    Raises
    ------
    ValueError
        If the given NZTM coordinates are not valid.

    Examples
    --------
    >>> nztm_to_wgs_depth(np.array([5180040.61473068, 1570636.6812821]))
    array([-43.5320, 172.6366])
    >>> nztm_to_wgs_depth(np.array([5180040.61473068, 1570636.6812821, 0]))
    array([-43.5320, 172.6366, 0])
    >>> wgs_depth_to_nztm(array([[5.92021456e+06, 1.75731133e+06, 100],
                                 [5.42725716e+06, 1.74893148e+06, 0]]))
    array([[-36.8509, 174.7645, 100], [-41.2924, 174.7787, 100]])
    """
    wgs_coordinates = np.array(_NZTM2WGS.transform(*nztm_coordinates.T)).T
    if not np.all(np.isfinite(wgs_coordinates)):
        raise ValueError(
            "NZTM coordinates given are invalid (did you input x, y instead of y, x?)"
        )
    return wgs_coordinates


def distance_between_wgs_depth_coordinates(
    point_a: npt.ArrayLike, point_b: npt.ArrayLike
) -> npt.ArrayLike:
    """Return the distance between two points in lat, lon, depth format.

    Valid only for points that can be converted into NZTM format.

    Parameters
    ----------
    point_a : np.ndarray
        The first point (lat, lon, optionally depth). May have shape
        (2,), (3,), (n, 2), (n, 3).

    point_b : np.ndarray
        The second point (lat, lon, optionally depth). May have shape
        (2, ), (3,), (n, 2), (n, 3)

    Returns
    -------
    float or np.ndarray
        The distance (in metres) between point_a and point_b. Will
        return an array of floats if input contains multiple points
    """
    if len(point_a.shape) > 1:
        return np.linalg.norm(
            wgs_depth_to_nztm(point_a) - wgs_depth_to_nztm(point_b), axis=1
        )
    return np.linalg.norm(wgs_depth_to_nztm(point_a) - wgs_depth_to_nztm(point_b))


def nztm_bearing_to_great_circle_bearing(
    origin: np.ndarray, distance: float, nztm_bearing: float
) -> float:
    """Correct a NZTM bearing to match a great circle bearing.

    This function can be used to translate bearings computed from NZTM
    coordinates into equivalent bearings in great-circle geometry.
    Primarily useful in ensuring tools like NZVM and EMOD3D agree on
    domain corners with newer code using NZTM. This correction is
    larger as the origin moves farther south, as the nztm bearing varies,
    and (slightly) as distance increases.

    Parameters
    ----------
    origin : np.ndarray
        The origin point to compute the bearing from.
    distance : float
        The distance to shift.
    nztm_bearing : float
        The NZTM bearing for the final point.

    Returns
    -------
    float
        The equivalent bearing such that:
        `geo.ll_shift`(*`origin`, `distance`, bearing) ≅ `nztm_heading`.
    """
    nztm_heading = nztm_to_wgs_depth(
        wgs_depth_to_nztm(origin)
        + distance
        * 1000
        * np.array([np.cos(np.radians(nztm_bearing)), np.sin(np.radians(nztm_bearing))])
    )
    return geo.ll_bearing(*origin[::-1], *nztm_heading[::-1])


def great_circle_bearing_to_nztm_bearing(
    origin: np.ndarray, distance: float, great_circle_bearing: float
) -> float:
    """Correct a great circle bearing to match a NZTM bearing.

    This function can be used to translate bearings computed from
    great-circle geometry to equivalent bearings in NZTM coordinates.
    Primarily useful in ensuring tools like NZVM and EMOD3D agree on
    domain corners with newer code using NZTM. This correction is
    larger as the origin moves farther south, as the nztm bearing
    varies, and (slightly) as distance increases.

    Parameters
    ----------
    origin : np.ndarray
        The origin point to compute the bearing from.
    distance : float
        The distance to shift.
    great_circle_bearing : float
        The great circle bearing for the final point.

    Returns
    -------
    float
        The equivalent bearing such that:
        `geo.ll_shift`(*`origin`, `distance`, `ll_bearing`) ≅ nztm_heading.
    """
    x, y = origin
    great_circle_heading = np.array(geo.ll_shift(x, y, distance, great_circle_bearing))

    return geo.oriented_bearing_wrt_normal(
        np.array([1, 0, 0]),
        np.append(
            wgs_depth_to_nztm(great_circle_heading) - wgs_depth_to_nztm(origin), 0
        ),
        np.array([0, 0, 1]),
    )


R_EARTH = 6378.139


class SphericalProjection:
    """
    Performs forward and inverse azimuthal equidistant projection for a spherical Earth
    with a customisable centre and a 2D rotation of the projected coordinates.

    The projection is centred at (`mlat`, `mlon`). The `mrot` parameter applies
    a rotation in the projected (x, y) plane *after* the base azimuthal equidistant projection,
    and the y-axis is inverted after this rotation.

    Parameters
    ----------
    mlon : float
        Longitude of the projection centre in degrees.
    mlat : float
        Latitude of the projection centre in degrees.
    mrot : float
        Rotation angle in the projected plane in degrees.
    radius : float, optional
        Radius of the spherical Earth in kilometres. Default is `R_EARTH` (6378.0 km).

    Attributes
    ----------
    mlon : float
        Longitude of the projection centre in degrees.
    mlat : float
        Latitude of the projection centre in degrees.
    mrot : float
        Rotation angle in the projected plane in degrees.
    """

    def __init__(self, mlon: float, mlat: float, mrot: float, radius: float = R_EARTH):  # noqa: D107 # numpydoc ignore=GL08
        self.mlon = mlon
        self.mlat = mlat
        self.mrot = mrot
        self.radius = radius

        arg = np.radians(mrot)
        cos_a = np.cos(arg)
        sin_a = np.sin(arg)

        arg = np.radians(90.0 - mlat)
        cos_t = np.cos(arg)
        sin_t = np.sin(arg)

        arg = np.radians(mlon)
        cos_p = np.cos(arg)
        sin_p = np.sin(arg)

        self.amat = np.array(
            [
                [
                    cos_a * cos_t * cos_p + sin_a * sin_p,
                    sin_a * cos_t * cos_p - cos_a * sin_p,
                    sin_t * cos_p,
                ],
                [
                    cos_a * cos_t * sin_p - sin_a * cos_p,
                    sin_a * cos_t * sin_p + cos_a * cos_p,
                    sin_t * sin_p,
                ],
                [-cos_a * sin_t, -sin_a * sin_t, cos_t],
            ],
            dtype=np.float64,
        )

    @property
    def geod(self) -> pyproj.Geod:  # numpydoc ignore=RT01
        """pyproj.Geod: A pyproj representation of the EMOD3D earth as a Geod."""
        return pyproj.Geod(ellps="sphere", a=self.radius, b=self.radius)

    def cartesian(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
    ) -> np.ndarray:
        """
        Converts geodetic coordinates to ECEF (Earth-Centred, Earth-Fixed) coordinates.

        Parameters
        ----------
        lat : array-like
            Latitude(s) in degrees.
        lon : array-like
            Longitude(s) in degrees.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (N, 3) representing the ECEF coordinates (x, y, z)
            in metres, where N is the number of input points.
            If the input was a single float, the output is a 1D array (3,).
        """
        lon = np.radians(np.asarray(lon))
        lat = np.radians(90.0 - np.asarray(lat))

        return np.array(
            [np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)]
        )

    def inverse_cartesian(
        self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Converts ECEF (Earth-Centred, Earth-Fixed) coordinates back to geodetic coordinates.

        Parameters
        ----------
        x : array-like
            ECEF x-coordinate(s) in kilometres.
        y : array-like
            ECEF y-coordinate(s) in kilometres.
        z : array-like
            ECEF z-coordinate(s) in kilometres.

        Returns
        -------
        np.ndarray
            Geodetic coordinates (lat, lon) corresponding to the cartesian coordinates (x, y, z).
        """

        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(y, x))

        return np.array((lat, lon))

    def project(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        z: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Performs forward projection from geographic coordinates (`lat`, `lon`)
        to rotated projected coordinates (`x`, `y`).

        Parameters
        ----------
        lat : array-like
            Latitude(s) in degrees.
        lon : array-like
            Longitude(s) in degrees.
        z : array-like, optional
            Depth or height coordinate(s) in kilometres. Not used in the projection.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (N, 2) representing the projected and rotated
            coordinates (x, y) in kilometres, where N is the number of input points.
            If the input was a single float, the output is a 1D array (2,).
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        # Convert lat, lon into normalised spherical coordinates.
        ecef = self.cartesian(lat, lon)
        # Rotate the coordinates
        x, y, w = np.linalg.solve(self.amat, ecef)
        x, y = (
            R_EARTH * np.arctan2(y, w),
            R_EARTH * np.arctan2(x, w),
        )
        if z is not None:
            out = np.column_stack((x, y, np.asarray(z)))
        else:
            out = np.column_stack((x, y))

        if lon.ndim == 0 and lat.ndim == 0:
            return out.flatten()
        return out

    __call__ = project

    def inverse(
        self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike | None = None
    ) -> np.ndarray:
        """Performs inverse azimuthal equidistant projection from rotated projected coordinates (`x`, `y`)
        back to geographic coordinates (`lat`, `lon`).

        Parameters
        ----------
        x : array-like
            Rotated projected x-coordinate(s) in kilometres.
        y : array-like
            Rotated projected y-coordinate(s) in kilometres.
        z : array-like, optional
            Depth or height coordinate(s) in kilometres. Not used in the projection.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (N, 2) representing the geographic coordinates
            (lat, lon) in degrees, where N is the number of input points.
            If the input was a single float, the output is a 1D array (2,).
        """

        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        x /= R_EARTH
        y /= R_EARTH
        tan_x = np.tan(x)
        tan_y = np.tan(y)
        w = 1 / np.sqrt(1 + tan_x**2 + tan_y**2)
        x = w * tan_y
        y = w * tan_x
        x, y, w = np.clip(self.amat @ np.array([x, y, w]), -1.0, 1.0)

        lat, lon = self.inverse_cartesian(x, y, w)

        lon = np.where(np.isclose(np.abs(lat), 90.0), 0.0, lon)

        if z is not None:
            out = np.column_stack((np.asarray(lat), np.asarray(lon), np.asarray(z)))
        else:
            out = np.column_stack((np.asarray(lat), np.asarray(lon)))
        if x.ndim == 0 and y.ndim == 0:
            return out.flatten()
        return out

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SphericalProjection(mlon={self.mlon}, mlat={self.mlat}, mrot={self.mrot})"
        )

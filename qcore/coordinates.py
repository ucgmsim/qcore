"""
Module for coordinate conversions between WGS84 (latitude and longitude) and
NZTM (New Zealand Transverse Mercator) coordinate systems.

Functions
----------
- wgs_depth_to_nztm(wgs_depth_coordinates: np.ndarray) -> np.ndarray:
    Converts WGS84 coordinates (latitude, longitude, depth) to NZTM coordinates.
- nztm_to_wgs_depth(nztm_coordinates: np.ndarray) -> np.ndarray:
    Converts NZTM coordinates (y, x, depth) to WGS84 coordinates.

References
----------
This module provides functions for converting coordinates between WGS84 and NZTM coordinate systems.
See LINZ[0] for a description of the NZTM coordinate system.

[0]: https://www.linz.govt.nz/guidance/geodetic-system/coordinate-systems-used-new-zealand/projections/new-zealand-transverse-mercator-2000-nztm2000
"""

from typing import Union

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
    point_a: np.ndarray, point_b: np.ndarray
) -> Union[float, np.ndarray]:
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
    ll_bearing : float
        The great circle bearing for the final point.

    Returns
    -------
    float
        The equivalent bearing such that:
        `geo.ll_shift`(*`origin`, `distance`, `ll_bearing`) ≅ nztm_heading.
    """
    great_circle_heading = np.array(
        geo.ll_shift(*origin, distance, great_circle_bearing)
    )

    return geo.oriented_bearing_wrt_normal(
        np.array([1, 0, 0]),
        np.append(
            wgs_depth_to_nztm(great_circle_heading) - wgs_depth_to_nztm(origin), 0
        ),
        np.array([0, 0, 1]),
    )


R_EARTH_METRES = 6378139.0


class SphericalProjection:
    """
    Performs forward and inverse gnomonic projection for a spherical Earth
    with a customisable center and a 2D rotation of the projected coordinates.

    The projection is centred at (`mlat`, `mlon`). The `mrot` parameter applies
    a rotation in the projected (x, y) plane *after* the base gnomonic projection,
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
        Radius of the spherical Earth in meters. Default is `R_EARTH_METRES` (6378139.0 m).

    Attributes
    ----------
    mlon : float
        Longitude of the projection centre in degrees.
    mlat : float
        Latitude of the projection centre in degrees.
    mrot : float
        Rotation angle in the projected plane in degrees.
    """

    def __init__(
        self, mlon: float, mlat: float, mrot: float, radius: float = R_EARTH_METRES
    ):  # noqa: D107
        self.mlon = mlon
        self.mlat = mlat
        self.mrot = mrot

        # Define the source CRS (spherical geographic coordinates)
        # Using +proj=latlong for geographic coordinates (lat/lon)
        # and specifying the spherical radius.
        _source_crs = pyproj.CRS(f"+proj=latlong +R={radius} +units=m +no_defs")

        # +proj=gnom: Gnomonic projection
        # +lon_0: Central longitude
        # +lat_0: Central latitude
        # +R: Radius of the sphere
        # +no_defs: Use parameters from the string, not default files
        _target_crs_base = pyproj.CRS(
            f"+proj=gnom +lon_0={self.mlon} +lat_0={self.mlat} +R={radius} +units=m +no_defs"
        )

        self._transformer = pyproj.Transformer.from_crs(
            _source_crs,
            _target_crs_base,
            always_xy=True,
        )

        _mrot_rad: float = np.radians(self.mrot)
        self._cos_mrot: float = np.cos(_mrot_rad)
        self._sin_mrot: float = np.sin(_mrot_rad)

    def __call__(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        z: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Performs forward gnomonic projection from geographic coordinates (`lat`, `lon`)
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

        x_base, y_base = self._transformer.transform(lon, lat)
        y_base *= -1  # Invert y-axis

        if np.isnan(x_base).any() or np.isnan(y_base).any():
            raise ValueError(
                "Latitude and longitude coordinates given are invalid (did you input lon, lat instead of lat, lon?)."
                " Check the coordinates are in the same hemisphere as the projection centre."
            )

        x_base = np.asarray(x_base)
        y_base = np.asarray(y_base)

        x_rotated = x_base * self._cos_mrot + y_base * self._sin_mrot
        y_rotated = -x_base * self._sin_mrot + y_base * self._cos_mrot

        if z is not None:
            out = np.column_stack((x_rotated, y_rotated, np.asarray(z)))
        else:
            out = np.column_stack((x_rotated, y_rotated))

        if lon.ndim == 0 and lat.ndim == 0:
            return out.flatten()
        return out

    def inverse(
        self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike | None = None
    ) -> np.ndarray:
        """Performs inverse gnomonic projection from rotated projected coordinates (`x`, `y`)
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

        x = np.asarray(x)
        y = np.asarray(y)

        x_base = x * self._cos_mrot - y * self._sin_mrot
        y_base = x * self._sin_mrot + y * self._cos_mrot
        y_base = y * -1
        lon, lat = self._transformer.transform(
            x_base, y_base, direction=pyproj.enums.TransformDirection.INVERSE
        )
        if z is not None:
            out = np.column_stack((np.asarray(lat), np.asarray(lon), np.asarray(z)))
        else:
            out = np.column_stack((np.asarray(lat), np.asarray(lon)))
        if x.ndim == 0 and y.ndim == 0:
            return out.flatten()
        return out

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SphericalProjection(mlon={self.mlon}, mlat={self.mlat}, "
            f"mrot={self.mrot})"
        )

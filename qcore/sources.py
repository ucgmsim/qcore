"""Module for representing the geometry seismic sources: point sources, fault planes and faults.

This module provides classes and functions for representing fault planes and
faults, along with methods for calculating various properties such as
dimensions, orientation, and coordinate transformations.

Classes
-------
PointSource:
    A representation of a point source.

Plane:
    A representation of a single plane of a Fault.

Fault:
    A representation of a fault, consisting of one or more Planes.
"""

import dataclasses
from typing import Optional, Protocol

import numpy as np
import scipy as sp

from qcore import coordinates, geo, grid

_KM_TO_M = 1000


@dataclasses.dataclass
class Point:
    """A representation of point source."""

    point_coordinates: np.ndarray
    # used to approximate point source as a small planar patch (metres).
    length_m: float
    # The usual strike, dip, dip direction, etc cannot be calculated
    # from a point source and so must be provided by the user.
    strike: float
    dip: float
    dip_dir: float

    @property
    def length(self) -> float:
        """
        Returns
        -------
        float
            The length of the approximating planar patch (in kilometres).
        """
        return self.length_m / _KM_TO_M

    @property
    def width_m(self) -> float:
        """
        Returns
        -------
        float
            The width of the approximating planar patch (in metres).
        """
        return self.length_m

    @property
    def width(self) -> float:
        """
        Returns
        -------
        float
            The width of the approximating planar patch (in kilometres).
        """
        return self.width_m / _KM_TO_M

    def fault_coordinates_to_wgs_depth_coordinates(
        self, fault_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert fault-local coordinates to global (lat, lon, depth) coordinates.

        Parameters
        ----------
        fault_coordinates : np.ndarray
            The local fault coordinates

        Returns
        -------
        np.ndarray
            The global coordinates for these fault-local
            coordinates. Because this is a point-source, the global
            coordinates are just the location of the point source.
        """

        return self.point_coordinates

    def wgs_depth_coordinates_to_fault_coordinates(
        self, wgs_depth_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates into fault-local coordinates.

        Parameters
        ----------
        wgs_depth_coordinates : np.ndarray
            The global coordinates to convert.

        Returns
        -------
        np.ndarray
            The fault-local coordinates. Because this is a
            point-source, the local coordinates are simply (1/2, 1/2)
            near the source point and undefined elsewhere.

        Raises
        ------
        ValueError
            If the point is not near the source point.
        """
        nztm_coordinates = coordinates.wgs_depth_to_nztm(wgs_depth_coordinates)
        if np.all(
            np.abs(nztm_coordinates - self.point_coordinates)[:2] / _KM_TO_M
            < self.length
        ):
            return np.array([1 / 2, 1 / 2])  # Point is in the centre of the small patch
        raise ValueError("Given global coordinates out of bounds for point source.")


@dataclasses.dataclass
class Plane:
    """A representation of a single plane of a Fault.

    This class represents a single plane of a fault, providing various
    properties and methods for calculating its dimensions, orientation, and
    converting coordinates between different reference frames.

    Attributes
    ----------
    corners_nztm : np.ndarray
        The corners of the fault plane, in NZTM format. The order of the
        corners is given clockwise from the top left (according to strike
        and dip). See the diagram below.

         0            1
          ┌──────────┐
          │          │
          │          │
          │          │
          │          │
          │          │
          │          │
          │          │
          └──────────┘
         3            2
    """

    corners_nztm: np.ndarray

    @property
    def corners(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The corners of the fault plane in (lat, lon, depth) format. The
            corners are the same as in corners_nztm.
        """
        return coordinates.nztm_to_wgs_depth(self.corners_nztm)

    @property
    def length_m(self) -> float:
        """
        Returns
        -------
        float
            The length of the fault plane (in metres).
        """
        return np.linalg.norm(self.corners_nztm[1] - self.corners_nztm[0])

    @property
    def width_m(self) -> float:
        """
        Returns
        -------
        float
            The width of the fault plane (in metres).
        """
        return np.linalg.norm(self.corners_nztm[-1] - self.corners_nztm[0])

    @property
    def bottom_m(self) -> float:
        """
        Returns
        -------
        float
            The bottom depth (in metres).
        """
        return self.corners_nztm[-1, -1]

    @property
    def width(self) -> float:
        """
        Returns
        -------
        float
            The width of the fault plane (in kilometres).
        """
        return self.width_m / _KM_TO_M

    @property
    def length(self) -> float:
        """
        Returns
        -------
        float
            The length of the fault plane (in kilometres).
        """
        return self.length_m / _KM_TO_M

    @property
    def projected_width_m(self) -> float:
        """
        Returns
        -------
        float
            The projected width of the fault plane (in metres).
        """
        return self.length_m * np.cos(np.radians(self.dip))

    @property
    def projected_width(self) -> float:
        """
        Returns
        -------
        float
            The projected width of the fault plane (in kilometres).
        """
        return self.projected_width_m / _KM_TO_M

    @property
    def strike(self) -> float:
        """
        Returns
        -------
        float
            The bearing of the strike direction of the fault
            (from north; in degrees)
        """

        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        strike_direction = self.corners_nztm[1] - self.corners_nztm[0]
        return geo.oriented_bearing_wrt_normal(
            north_direction, strike_direction, up_direction
        )

    @property
    def dip_dir(self) -> float:
        """
        Returns
        -------
        float
            The bearing of the dip direction (from north; in degrees).
        """
        if np.isclose(self.dip, 90):
            return 0  # TODO: Is this right for this case?
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        dip_direction = self.corners_nztm[-1] - self.corners_nztm[0]
        dip_direction[-1] = 0
        return geo.oriented_bearing_wrt_normal(
            north_direction, dip_direction, up_direction
        )

    @property
    def dip(self) -> float:
        """
        Returns
        -------
        float
            The dip angle of the fault.
        """
        return np.degrees(np.arcsin(np.abs(self.bottom_m) / self.width_m))

    @staticmethod
    def from_centroid_strike_dip(
        centroid: np.ndarray,
        strike: float,
        dip_dir: Optional[float],
        top: float,
        bottom: float,
        length: float,
        width: float,
    ) -> "Plane":
        """Create a fault plane from the centroid, strike, dip_dir, top, bottom, length, and width

        This is used for older descriptions of sources. Internally
        converts everything to corners so self.strike ~ strike (but
        not exactly due to rounding errors).

        Parameters
        ----------
        centroid : np.ndarray
            The centre of the fault plane in lat, lon coordinate.s
        strike : float
            The strike of the fault (in degrees).
        dip_dir : Optional[float]
            The dip direction of the fault (in degrees). If None this is assumed to be strike + 90 degrees.
        top : float
            The top depth of the plane (in km).
        bottom : float
            The bottom depth of the plane (in km).
        length : float
            The length of the fault plane (in km).
        width : float
            The width of the fault plane (in km).

        Returns
        -------
        Plane
            The fault plane with centre at `centroid`, and where the
            parameters strike, dip_dir, top, bottom, length and width
            match what is passed to this function.
        """
        corners = grid.grid_corners(
            centroid,
            strike,
            dip_dir if dip_dir is not None else (strike + 90),
            top,
            bottom,
            length,
            width,
        )
        return Plane(corners)

    def fault_coordinates_to_wgs_depth_coordinates(
        self, plane_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert plane coordinates to nztm global coordinates.

        Parameters
        ----------
        plane_coordinates : np.ndarray
            Plane coordinates to convert. Plane coordinates are
            2D coordinates (x, y) given for a fault plane (a plane), where x
            represents displacement along the strike, and y
            displacement along the dip (see diagram below). The
            origin for plane coordinates is the centre of the fault.

                          +x
             0 0   ─────────────────>
                ┌─────────────────────┐ │
                │      < strike >     │ │
                │                 ^   │ │
                │                dip  │ │ +y
                │                 v   │ │
                │                     │ │
                └─────────────────────┘ ∨
                                       1,1

        Returns
        -------
        np.ndarray
            An 3d-vector of (lat, lon, depth) transformed coordinates.
        """
        origin = self.corners_nztm[0]
        top_right = self.corners_nztm[1]
        bottom_left = self.corners_nztm[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))

        return coordinates.nztm_to_wgs_depth(origin + plane_coordinates @ frame)

    def wgs_depth_coordinates_to_fault_coordinates(
        self,
        global_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Convert coordinates (lat, lon, depth) to plane coordinates (x, y).

        See plane_coordinates_to_global_coordinates for a description of
        plane coordinates.

        Parameters
        ----------
        global_coordinates : np.ndarray
            Global coordinates to convert.

        Returns
        -------
        np.ndarray
            The plane coordinates (x, y) representing the position of
            global_coordinates on the fault plane.

        Raises
        ------
        ValueError
            If the given coordinates do not lie in the fault plane.
        """
        origin = self.corners_nztm[0]
        top_right = self.corners_nztm[1]
        bottom_left = self.corners_nztm[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))
        offset = coordinates.wgs_depth_to_nztm(global_coordinates) - origin
        plane_coordinates, residual, _, _ = np.linalg.lstsq(frame.T, offset, rcond=None)
        if not np.isclose(residual[0], 0, atol=1e-02):
            raise ValueError("Coordinates do not lie in fault plane.")
        return np.clip(plane_coordinates, 0, 1)

    def wgs_depth_coordinates_in_plane(self, global_coordinates: np.ndarray) -> bool:
        """Test if some global coordinates lie in the bounds of a plane.

        Parameters
        ----------
        global_coordinates : np.ndarray
            The global coordinates to check

        Returns
        -------
        bool
            True if the given global coordinates (lat, lon, depth) lie on the
            fault plane.
        """

        try:
            plane_coordinates = self.wgs_depth_coordinates_to_fault_coordinates(
                global_coordinates
            )
            return np.all(
                np.logical_or(
                    np.abs(plane_coordinates) < 1 / 2,
                    np.isclose(np.abs(plane_coordinates), 1 / 2, atol=1e-3),
                )
            )
        except ValueError:
            return False

    def centroid(self) -> np.ndarray:
        """Returns the centre of the fault plane.

        Returns
        -------
        np.ndarray
            A 1 x 3 dimensional vector representing the centroid of the fault
            plane in (lat, lon, depth) format.

        """

        return coordinates.nztm_to_wgs_depth(
            np.mean(self.corners_nztm, axis=0).reshape((1, -1))
        ).ravel()


@dataclasses.dataclass
class Fault:
    """A representation of a fault, consisting of one or more Planes.

    This class represents a fault, which is composed of one or more Planes.
    It provides methods for computing the area of the fault, getting the widths and
    lengths of all fault planes, retrieving all corners of the fault, converting
    global coordinates to fault coordinates, converting fault coordinates to global
    coordinates, generating a random hypocentre location within the fault, and
    computing the expected fault coordinates.

    Attributes
    ----------
    planes : list[Plane]
        A list containing all the Planes that constitute the fault.

    Methods
    -------
    area:
        Compute the area of a fault.
    widths:
        Get the widths of all fault planes.
    lengths:
        Get the lengths of all fault planes.
    corners:
        Get all corners of a fault.
    global_coordinates_to_fault_coordinates:
        Convert global coordinates to fault coordinates.
    fault_coordinates_to_wgsdepth_coordinates:
        Convert fault coordinates to global coordinates.
    """

    planes: list[Plane]

    def area(self) -> float:
        """Compute the area of a fault.

        Returns
        -------
        float
            The area of the fault.
        """
        return self.width * np.sum(self.lengths)

    @property
    def lengths(self) -> np.ndarray:
        """The lengths of each plane in the fault.

        Returns
        -------
        np.ndarray
           A numpy array of each plane length (in km).
        """
        return np.array([fault.length for fault in self.planes])

    @property
    def length(self) -> float:
        """The length of the fault.

        Returns
        -------
        float
            The total length of each fault plane.
        """

        return self.lengths.sum()

    @property
    def width(self) -> float:
        """The width of the fault.

        Returns
        -------
        float
            The width of the first fault plane (A fault is assumed to
            have planes of constant width).
        """
        return self.planes[0].width

    @property
    def dip_dir(self) -> float:
        """The dip direction of the fault.

        Returns
        -------
        float
            The dip direction of the first fault plane (A fault is
            assumed to have planes of constant dip direction).
        """
        return self.planes[0].dip_dir

    def corners(self) -> np.ndarray:
        """Get all corners of a fault.

        Returns
        -------
        np.ndarray of shape (4n x 3)
            The corners in (lat, lon, depth) format of each fault plane in the
            fault, stacked vertically.
        """
        return np.vstack([plane.corners for plane in self.planes])

    def corners_nztm(self) -> np.ndarray:
        """Get all corners of a fault.

        Returns
        -------
        np.ndarray of shape (4n x 3)
            The corners in NZTM format of each fault plane in the fault, stacked vertically.
        """
        return np.vstack([plane.corners_nztm for plane in self.planes])

    def wgs_depth_coordinates_to_fault_coordinates(
        self, global_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates in (lat, lon, depth) format to fault coordinates.

        Fault coordinates are a tuple (s, d) where s is the distance
        from the top left, and d the distance from the top of the
        fault (refer to the diagram). The coordinates are normalised
        such that (0, 0) is the top left and (1, 1) the bottom right.

        (0, 0)
          ┌──────────────────────┬──────┐
          │          |           │      │
          │          |           │      │
          │          | d         │      │
          │          |           │      │
          ├----------*           │      │
          │    s     ^           │      │
          │          |           │      │
          │          |           │      │
          │          |           │      │
          └──────────|───────────┴──────┘
                     +                    (1, 1)
                  point: (s, d)

        Parameters
        ----------
        global_coordinates : np.ndarray of shape (1 x 3)
            The global coordinates to convert.

        Returns
        -------
        np.ndarray
            The fault coordinates.

        Raises
        ------
        ValueError
            If the given point does not lie on the fault.

        """
        # the right edges as a cumulative proportion of the fault length (e.g. [0.1, ..., 0.8])
        right_edges = self.lengths.cumsum() / self.length
        right_edges = np.append(right_edges, 1)
        for i, plane in enumerate(self.planes):
            if plane.wgs_depth_coordinates_in_plane(global_coordinates):
                plane_coordinates = plane.wgs_depth_coordinates_to_fault_coordinates(
                    global_coordinates
                )
                return np.array([right_edges[i], 0]) + plane_coordinates * np.array(
                    [right_edges[i + 1] - right_edges[i], 1]
                )
        raise ValueError("Given coordinates are not on fault.")

    def fault_coordinates_to_wgs_depth_coordinates(
        self, fault_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert fault coordinates to global coordinates.

        See global_coordinates_to_fault_coordinates for a description of fault
        coordinates.

        Parameters
        ----------
        fault_coordinates : np.ndarray
            The fault coordinates of the point.

        Returns
        -------
        np.ndarray
            The global coordinates (lat, lon, depth) for this point.
        """

        # the right edges as a cumulative proportion of the fault length (e.g. [0.1, ..., 0.8])
        right_edges = self.lengths.cumsum() / self.length
        fault_segment_index = np.searchsorted(right_edges, fault_coordinates[0])
        left_proportion = (
            right_edges[fault_segment_index - 1] if fault_segment_index > 0 else 0
        )
        right_proportion = (
            right_edges[fault_segment_index + 1]
            if fault_segment_index < len(right_edges) - 1
            else 1
        )
        segment_proportion = (fault_coordinates[0] - left_proportion) / (
            right_proportion - left_proportion
        )
        return self.planes[
            fault_segment_index
        ].fault_coordinates_to_wgs_depth_coordinates(
            np.array([segment_proportion, fault_coordinates[1]])
        )


class HasCoordinates(Protocol):
    """Type definition for a source with local coordinates."""

    def fault_coordinates_to_wgs_depth_coordinates(
        self,
        fault_coordinates: np.ndarray,
    ) -> np.ndarray: ...

    def wgs_depth_coordinates_to_fault_coordinates(
        self,
        fault_coordinates: np.ndarray,
    ) -> np.ndarray: ...


def closest_point_between_sources(
    source_a: HasCoordinates, source_b: HasCoordinates
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point between two sources that have local coordinates.

    Parameters
    ----------
    source_a : HasCoordinates
        The first source. Must have a two-dimensional fault coordinate system.
    source_b : HasCoordinates
        The first source. Must have a two-dimensional fault coordinate system.

    Raises
    ------
    ValueError
        Raised when we are unable to converge on the closest points between sources.

    Returns
    -------
    source_a_coordinates : np.ndarray
        The source-local coordinates of the closest point on source a.
    source_b_coordinates : np.ndarray
        The source-local coordinates of the closest point on source b.
    """

    def fault_coordinate_distance(fault_coordinates: np.ndarray) -> float:
        source_a_global_coordinates = (
            source_a.fault_coordinates_to_wgs_depth_coordinates(fault_coordinates[:2])
        )
        source_b_global_coordinates = (
            source_b.fault_coordinates_to_wgs_depth_coordinates(fault_coordinates[2:])
        )
        return coordinates.distance_between_wgs_depth_coordinates(
            source_a_global_coordinates, source_b_global_coordinates
        )

    res = sp.optimize.minimize(
        fault_coordinate_distance,
        np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]),
        bounds=[(0, 1)] * 4,
    )

    if not res.success:
        raise ValueError(
            f"Optimisation failed to converge for provided sources: {res.message}"
        )

    return res.x[:2], res.x[2:]

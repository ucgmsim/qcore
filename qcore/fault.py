"""Module for representing fault segments and faults.

This module provides classes and functions for representing fault segments and
faults, along with methods for calculating various properties such as dimensions,
orientation, and coordinate transformations.

Classes
-------
TectType:
    An enumeration of all the different kinds of fault types.

FaultSegment:
    A representation of a single segment of a Fault.

Fault:
    A representation of a fault, consisting of one or more FaultSegments.
"""

import dataclasses
from enum import Enum

import numpy as np
import scipy as sp

import qcore.coordinates
import qcore.geo
from qcore.uncertainties import distributions


class TectType(Enum):
    """An enumeration of all the different kinds of fault types."""

    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4


_KM_TO_M = 1000


@dataclasses.dataclass
class FaultSegment:
    """A representation of a single segment of a Fault.

    This class represents a single segment of a fault, providing various properties and methods
    for calculating its dimensions, orientation, and converting coordinates between different
    reference frames.

    Attributes
    ----------
    corners : np.ndarray
        An array containing the coordinates of the corners of the fault segment.
    rake : float
        The rake angle of the fault segment.
    """

    corners: np.ndarray
    rake: float

    def __init__(self, corners: np.ndarray, rake: float):
        self.corners = qcore.coordinates.wgs_depth_to_nztm(corners)
        self.rake = rake

    @property
    def length_m(self) -> float:
        """
        Returns
        -------
        float
            The length of the fault segment (in metres).
        """
        return np.linalg.norm(self.corners[1] - self.corners[0])

    @property
    def width_m(self) -> float:
        """
        Returns
        -------
        float
            The width of the fault segment (in metres).
        """
        return np.linalg.norm(self.corners[-1] - self.corners[0])

    @property
    def bottom_m(self) -> float:
        """
        Returns
        -------
        float
            The bottom depth (in metres).
        """
        return self.corners[-1, -1]

    @property
    def width(self) -> float:
        """
        Returns
        -------
        float
            The width of the fault segment (in kilometres).
        """
        return self.width_m / _KM_TO_M

    @property
    def length(self) -> float:
        """
        Returns
        -------
        float
            The length of the fault segment (in kilometres).
        """
        return self.length_m / _KM_TO_M

    @property
    def projected_width_m(self) -> float:
        """
        Returns
        -------
        float
            The projected width of the fault segment (in metres).
        """
        return self.length_m * np.cos(np.radians(self.dip))

    @property
    def projected_width(self) -> float:
        """
        Returns
        -------
        float
            The projected width of the fault segment (in kilometres).
        """
        return self.projected_width / _KM_TO_M

    @property
    def strike(self) -> float:
        """
        Returns
        -------
        float
            The bearing of the strike direction of the fault (from north; in degrees)
        """

        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        strike_direction = self.corners[1] - self.corners[0]
        return (
            np.degrees(
                qcore.geo.oriented_angle_wrt_normal(
                    north_direction, strike_direction, up_direction
                )
            )
            % 360
        )

    @property
    def dip_dir(self) -> float:
        """
        Returns
        -------
        float
            The bearing of the dip direction (from north; in degrees).
        """
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        dip_direction = self.corners[-1] - self.corners[0]
        dip_direction[-1] = 0
        return (
            np.degrees(
                qcore.geo.oriented_angle_wrt_normal(
                    north_direction, dip_direction, up_direction
                )
            )
            % 360
        )

    @property
    def dip(self) -> float:
        """
        Returns
        -------
        float
            The dip angle of the fault.
        """
        return np.degrees(np.arcsin(self.bottom_m / self.width_m))

    @staticmethod
    def from_centroid_parameters(
        centroid: np.ndarray,
        strike: float,
        dip: float,
        dip_dir: float,
        top: float,
        length: float,
        width: float,
        rake: float,
    ):
        """Create a fault segment from a centroid, and other supplied parameters.

        This is useful for legacy realisation types (types 1-4), where faults
        are instantiated with centroid, strike, dip, rake, etc instead of
        computing these values from fault bounds.

        Parameters
        ----------
        centroid : np.ndarray
        strike : float
        dip : float
        dip_dir : float
        top : float
        length : float
        width : float
        rake : float
        """
        corners = []
        for segment_coordinates in [
            [-1 / 2, -1 / 2],
            [1 / 2, -1 / 2],
            [1 / 2, 1 / 2],
            [-1 / 2, 1 / 2],
        ]:
            depth = top + (
                (segment_coordinates[0] + 1 / 2)
                * _KM_TO_M
                * width
                * -np.sin(np.radians(dip))
            )
            dip_coord_dir = dip_dir if segment_coordinates[0] > 0 else 180 + dip_dir
            projected_width = width * np.cos(np.radians(dip))
            width_shift_lat, width_shift_lon = qcore.geo.ll_shift(
                *centroid,
                projected_width * np.abs(segment_coordinates[0]),
                dip_coord_dir,
            )
            strike_coord_dir = strike if segment_coordinates[1] > 0 else 180 + strike
            length_shift_lat, length_shift_lon = qcore.geo.ll_shift(
                width_shift_lat,
                width_shift_lon,
                length * np.abs(segment_coordinates[1]),
                strike_coord_dir,
            )
            corners.append([length_shift_lat, length_shift_lon, depth])
        return FaultSegment(np.ndarray(corners), rake)

    def segment_coordinates_to_global_coordinates(
        self, segment_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert segment coordinates to nztm global coordinates.

        Parameters
        ----------
        segment_coordinates : np.ndarray
            Segment coordinates to convert. Segment coordinates are
            2D coordinates (x, y) given for a fault segment (a plane), where x
            represents displacement along the length of the fault, and y
            displacement along the width of the fault (see diagram below). The
            origin for segment coordinates is the centre of the fault.

                          +x
          -1/2,-1/2 ─────────────────>
                ┌─────────────────────┐ │
                │      < width >      │ │
                │                 ^   │ │
                │               length│ │ +y
                │                 v   │ │
                │                     │ │
                └─────────────────────┘ ∨
                                     1/2,1/2

        Returns
        -------
        np.ndarray
            An 3d-vector of (lat, lon, depth) transformed coordinates.
        """
        origin = self.corners[0]
        top_right = self.corners[1]
        bottom_left = self.corners[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))
        offset = np.array([1 / 2, 1 / 2])

        return qcore.coordinates.nztm_to_wgs_depth(
            origin + (segment_coordinates + offset) @ frame
        )

    def global_coordinates_to_segment_coordinates(
        self,
        global_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Convert coordinates (lat, lon, depth) to segment coordinates (x, y).

        See segment_coordinates_to_global_coordinates for a description of segment
        coordinates.

        Parameters
        ----------
        global_coordinates : np.ndarray
            Global coordinates to convert.

        Returns
        -------
        np.ndarray
            The segment coordinates (x, y) representing the position of
            global_coordinates on the fault segment.

        Raises
        ------
        ValueError
            If the given coordinates do not lie in the fault plane.
        """
        origin = self.corners[0]
        top_right = self.corners[1]
        bottom_left = self.corners[-1]
        frame = np.vstack((top_right - origin, bottom_left - origin))
        offset = qcore.coordinates.wgs_depth_to_nztm(global_coordinates) - origin
        segment_coordinates, residual, _, _ = np.linalg.lstsq(frame.T, offset)
        if not np.isclose(residual[0], 0):
            raise ValueError("Coordinates do not lie in fault plane.")
        return segment_coordinates - np.array([1 / 2, 1 / 2])

    def global_coordinates_in_segment(self, global_coordinates: np.ndarray) -> bool:
        """Test if some global coordinates lie in the bounds of a segment.

        Parameters
        ----------
        global_coordinates : np.ndarray
            The global coordinates to check

        Returns
        -------
        bool
            True if the given global coordinates (lat, lon, depth) lie on the fault segment.
        """

        segment_coordinates = self.global_coordinates_to_segment_coordinates(
            global_coordinates
        )
        return np.all(
            np.logical_or(
                np.abs(segment_coordinates) < 1 / 2,
                np.isclose(np.abs(segment_coordinates), 1 / 2, atol=1e-4),
            )
        )

    def centroid(self) -> np.ndarray:
        """Returns the centre of the fault segment.

        Returns
        -------
        np.ndarray
            A 1 x 3 dimensional vector representing the centroid of the fault
            plane in (lat, lon, depth) format.

        """

        return qcore.coordinates.nztm_to_wgs_depth(
            np.mean(self.corners, axis=0).reshape((1, -1))
        ).ravel()


@dataclasses.dataclass
class Fault:
    """A representation of a fault, consisting of one or more FaultSegments.

    This class represents a fault, which is composed of one or more FaultSegments.
    It provides methods for computing the area of the fault, getting the widths and
    lengths of all fault segments, retrieving all corners of the fault, converting
    global coordinates to fault coordinates, converting fault coordinates to global
    coordinates, generating a random hypocentre location within the fault, and
    computing the expected fault coordinates.

    Attributes
    ----------
    name : str
        The name of the fault.
    tect_type : str
        The type of fault this is (e.g. crustal, volcanic, subduction).
    segments : list[FaultSegment]
        A list containing all the FaultSegments that constitute the fault.

    Methods
    -------
    area():
        Compute the area of a fault.
    widths():
        Get the widths of all fault segments.
    lengths():
        Get the lengths of all fault segments.
    corners():
        Get all corners of a fault.
    global_coordinates_to_fault_coordinates(global_coordinates: np.ndarray) -> np.ndarray:
        Convert global coordinates to fault coordinates.
    fault_coordinates_to_wgsdepth_coordinates(fault_coordinates: np.ndarray) -> np.ndarray:
        Convert fault coordinates to global coordinates.
    """

    name: str
    tect_type: str
    segments: list[FaultSegment]

    def area(self) -> float:
        """Compute the area of a fault.

        Returns
        -------
        float
            The area of the fault.
        """
        return sum(segment.width * segment.length for segment in self.segments)

    def widths(self) -> np.ndarray:
        """Get the widths of all fault segments.

        Returns
        -------
        np.ndarray of shape (1 x n)
            The widths of all fault segments contained in this fault.
        """
        return np.array([seg.width for seg in self.segments])

    def lengths(self) -> np.ndarray:
        """Get the lengths of all fault segments.

        Returns
        -------
        np.ndarray of shape (1 x n)
            The lengths of all fault segments contained in this fault.
        """
        return np.array([seg.length for seg in self.segments])

    def corners(self) -> np.ndarray:
        """Get all corners of a fault.

        Returns
        -------
        np.ndarray of shape (4n x 3)
            The corners of each fault segment in the fault, stacked vertically.
        """

        return np.vstack([segment.corners() for segment in self.segments])

    def global_coordinates_to_fault_coordinates(
        self, global_coordinates: np.ndarray
    ) -> np.ndarray:
        """Convert global coordinates in (lat, lon, depth) format to fault coordinates.

        Fault coordinates are a tuple (s, d) where s is the distance (in
        kilometres) from the top centre, and d the distance from the top of the
        fault (refer to the diagram).

        ┌─────────┬──────────────┬────┐
        │         │      ╎       │    │
        │         │      ╎       │    │
        │         │    d ╎       │    │
        │         │      ╎       │    │
        │         │      └╶╶╶╶╶╶╶╶╶╶+ │
        │         │           s  │  ∧ │
        │         │              │  │ │
        │         │              │  │ │
        └─────────┴──────────────┴──┼─┘
                                    │
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

        running_length = 0.0
        midpoint = np.sum(self.lengths()) / 2
        for segment in self.segments:
            if segment.global_coordinates_in_segment(global_coordinates):
                segment_coordinates = segment.global_coordinates_to_segment_coordinates(
                    global_coordinates
                )
                strike_length = segment_coordinates[0] + 1 / 2
                dip_length = segment_coordinates[1] + 1 / 2
                return np.array(
                    [
                        running_length + strike_length * segment.length - midpoint,
                        max(dip_length * segment.width, 0),
                    ]
                )
            running_length += segment.length
        raise ValueError("Specified coordinates not contained on fault.")

    def fault_coordinates_to_wgsdepth_coordinates(
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

        Raises
        ------
        ValueError
            If the fault coordinates are out of bounds.
        """
        midpoint = np.sum(self.lengths()) / 2
        remaining_length = fault_coordinates[0] + midpoint
        for segment in self.segments:
            segment_length = segment.length
            if remaining_length < segment_length:
                return segment.segment_coordinates_to_global_coordinates(
                    np.array(
                        [
                            remaining_length / segment_length - 1 / 2,
                            fault_coordinates[1] / segment.width - 1 / 2,
                        ]
                    ),
                )
            remaining_length -= segment_length
        raise ValueError("Specified fault coordinates out of bounds.")

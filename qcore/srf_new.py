"""
Module for handling SRF (Standard Rupture Format) files.

This module provides classes and functions for reading and writing SRF files,
as well as representing their contents.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the SRF format.


Classes:
- SrfSegment: A representation of a single segment within an SRF file.
- SrfPoint: A single point in an SRF file.
- SrfFile: Representation of an SRF file.

Exceptions:
- SrfParseError: Exception raised for errors in parsing SRF files.

Functions:
- read_srf_file: Read an SRF file into memory.
- write_srf_header: Write a list of SRF plane segments as a header.
- write_point_count: Write out a POINTS declaration for the number of points in an SRF file.
- write_slip: Write out slip values to an SRF file.
- write_srf_point: Write out a single SRF point.
- read_version: Read the version value from an SRF file.
- write_version: Write version value to an SRF file.
- read_srf_headers: Read the header section of an SRF file.
- read_float: Read a float from an SRF file.
- read_int: Read an int from an SRF file.
- read_points_count: Read the number of points contained in an SRF file.
- read_srf_n_points: Read a number of points from an SRF file.
"""

import dataclasses
import re
from pathlib import Path
from typing import TextIO, Optional


@dataclasses.dataclass
class SrfSegment:
    """
    A representation of a single segment within an SRF file.

    Attributes
    ----------
    elon : float
        The longitude of the segment's centroid.
    elat : float
        The latitude of the segment's centroid.
    nstk : int
        The number of subsegments along the strike direction.
    ndip : int
        The number of subsegments along the dip direction.
    len : float
        The length of the segment in the strike direction (km).
    wid : float
        The width of the segment in the dip direction (km).
    stk : float
        The strike angle of the segment (degrees).
    dip : float
        The dip angle of the segment (degrees).
    dtop : float
        The depth to the top of the segment (km).
    shyp : float
        The distance of the hypocentre along the strike direction from the centre.
    dhyp : float
        The distance of the hypocentre along the dip direction from the top.
    """

    elon: float
    elat: float
    nstk: int
    ndip: int
    len: float
    wid: float
    stk: float
    dip: float
    dtop: float
    shyp: float
    dhyp: float


@dataclasses.dataclass
class SrfPoint:
    """
    A single point in an SRF file.

    Attributes
    ----------
    lon : float
        The longitude of the point.
    lat : float
        The latitude of the point.
    dep : float
        The depth of the point (km).
    stk : float
        The strike angle of the point (degrees).
    dip : float
        The dip angle of the point (degrees).
    area : float
        The area of the point (cm^2).
    tinit : float
        The initial rupture time of the point (s).
    dt : float
        The time step in slip velocity function.
    rake : float
        The rake angle of the point (degrees). Represents the direction of u1.
    slip1 : float
        The total slip (cm) in the u1 direction
    slip2 : float
        The total slip (cm) in the u2 direction
    slip3 : float
        The total slip (cm) in the u3 direction
    sr1 : list[float]
        Slip velocity at each time step for u1 direction
    sr2 : list[float]
        Slip velocity at each time step for u2 direction
    sr3 : list[float]
        Slip velocity at each time step for u3 direction
    """

    lon: float
    lat: float
    dep: float
    stk: float
    dip: float
    area: float
    tinit: float
    dt: float
    rake: float
    slip1: float
    slip2: float
    slip3: float
    sr1: list[float]
    sr2: list[float]
    sr3: list[float]


@dataclasses.dataclass
class SrfFile:
    """
    Representation of an SRF file.

    Attributes
    ----------
    header : list[SrfSegment]
        A list of SrfSegment objects representing the header of the SRF file.
    points : list[SrfPoint]
        A list of SrfPoint objects representing the points in the SRF file.
    """

    header: list[SrfSegment]
    points: list[SrfPoint]


class SrfParseError(Exception):
    """Exception raised for errors in parsing SRF files."""

    pass


PLANE_COUNT_RE = r"PLANE (\d+)"


def read_version(srf_file: TextIO):
    """Read the version value from an srf file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file object to read from.
    """
    return float(srf_file.readline())


def write_version(srf_file: TextIO):
    """Write version value to an srf file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file object to write to.
    """
    srf_file.write("1.0\n")


def read_srf_headers(srf_file: TextIO) -> list[SrfSegment]:
    """Read the header section of an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file object to read the headers from.

    Raises
    ------
    SrfParseError
        An SrfParseError is raised if an error occurs parsing the SRF file headers.

    Returns
    -------
    list[SrfSegment]
        A list of all the planes listed in the header of the SRF file.
    """
    plane_count_line = srf_file.readline().strip()
    plane_count_match = re.match(PLANE_COUNT_RE, plane_count_line)
    if not plane_count_match:
        raise SrfParseError(f'Expecting PLANE header line, got: "{plane_count_line}"')
    plane_count = int(plane_count_match.group(1))
    segments = []
    for _ in range(plane_count):
        elon = read_float(srf_file)
        elat = read_float(srf_file)
        nstk = read_int(srf_file)
        ndip = read_int(srf_file)
        len = read_float(srf_file)
        wid = read_float(srf_file)
        stk = read_float(srf_file)
        dip = read_float(srf_file)
        dtop = read_float(srf_file)
        shyp = read_float(srf_file)
        dhyp = read_float(srf_file)

        segments.append(
            SrfSegment(
                elon,
                elat,
                nstk,
                ndip,
                len,
                wid,
                stk,
                dip,
                dtop,
                shyp,
                dhyp,
            )
        )
    return segments


POINT_COUNT_RE = r"POINTS (\d+)"


def read_float(srf_file: TextIO, label: Optional[str] = None) -> float:
    """Read a float from an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to read from.
    label : str | None
        A human friendly label for the floating point (for debugging purposes).

    Raises
    ------
    SrfParseError
        If there is an error reading the float value from the SRF file.

    Returns
    -------
    float
        The float read from the SRF file.
    """
    while (cur := srf_file.read(1)).isspace():
        pass
    float_str = cur
    while not (cur := srf_file.read(1)).isspace():
        float_str += cur
    try:
        return float(float_str)
    except ValueError:
        if label:
            raise SrfParseError(f'Expecting float ({label}), got: "{float_str}"')
        else:
            raise SrfParseError(f'Expecting float, got: "{float_str}"')


def read_int(srf_file: TextIO, label: Optional[str] = None) -> int:
    """Read a int from an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to read from.
    label : str | None
        A human friendly label for the int (for debugging purposes).

    Raises
    ------
    SrfParseError
        If there is an error reading the int value from the SRF file.

    Returns
    -------
    int
        The int read from the SRF file.
    """
    while (cur := srf_file.read(1)).isspace():
        pass
    int_str = cur
    while not (cur := srf_file.read(1)).isspace():
        int_str += cur
    try:
        return int(int_str)
    except ValueError:
        if label:
            raise SrfParseError(f'Expecting int ({label}), got: "{int_str}"')
        else:
            raise SrfParseError(f'Expecting int, got: "{int_str}"')


def read_points_count(srf_file: TextIO) -> int:
    """Read the number of points contained in an srf file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file object to read from.

    Raises
    ------
    SrfParseError
        If the current line does not match the expected syntax for the number
        of points in an SRF file (see POINT_COUNT_RE).

    Returns
    -------
    int
        The number of points contained in the SRF file.
    """
    points_count_line = srf_file.readline().strip()
    points_count_match = re.match(POINT_COUNT_RE, points_count_line)
    if not points_count_match:
        raise SrfParseError(f'Expecting POINTS header line, got: "{points_count_line}"')
    return int(points_count_match.group(1))


def read_srf_n_points(point_count: int, srf_file: TextIO) -> list[SrfPoint]:
    """Read a number of points from an SRF file.

    Parameters
    ----------
    point_count : int
        The number of points to read.
    srf_file : TextIO
        The SRF file object to read from.

    Raises
    ------
    SrfParseError
        If there is an error whilst parsing an SRF point.

    Returns
    -------
    list[SrfPoint]
        A list of point_count SrfPoints read from srf_file.
    """
    points = []
    for _ in range(point_count):
        lon = read_float(srf_file, label="lon")
        lat = read_float(srf_file, label="lat")
        dep = read_float(srf_file, label="dep")
        stk = read_float(srf_file, label="stk")
        dip = read_float(srf_file, label="dip")
        area = read_float(srf_file, label="area")
        tinit = read_float(srf_file, label="tinit")
        dt = read_float(srf_file, label="dt")
        rake = read_float(srf_file, label="rake")
        slip1 = read_float(srf_file, label="slip1")
        nt1 = read_int(srf_file, label="nt1")
        slip2 = read_float(srf_file, label="slip2")
        nt2 = read_int(srf_file, label="nt2")
        slip3 = read_float(srf_file, label="slip3")
        nt3 = read_int(srf_file, label="nt3")
        slipt1 = [read_float(srf_file, label="slipt1") for _ in range(nt1)]
        slipt2 = [read_float(srf_file, label="slipt2") for _ in range(nt2)]
        slipt3 = [read_float(srf_file, label="slipt3") for _ in range(nt3)]
        points.append(
            SrfPoint(
                lon,
                lat,
                dep,
                stk,
                dip,
                area,
                tinit,
                dt,
                rake,
                slip1,
                slip2,
                slip3,
                sr1=slipt1,
                sr2=slipt2,
                sr3=slipt3,
            )
        )
    return points


def read_srf_file(srf_filepath: Path) -> SrfFile:
    """Read an SRF file into memory.

    Parameters
    ----------
    srf_filepath : Path
        The filepath of the SRF file to read.

    Raises
    ------
    SrfParseError
        If an error occurs whilst parsing the SRF file.

    Returns
    -------
    SrfFile
        The parsed contents of the SRF file.
    """
    with open(srf_filepath, mode="r", encoding="utf-8") as srf_file:
        srf_file.readline()  # skip version
        header = read_srf_headers(srf_file)
        count = read_points_count(srf_file)
        points = read_srf_n_points(count, srf_file)
    return SrfFile(header, points)


def write_srf_header(srf_file: TextIO, header: list[SrfSegment]):
    """Write a list of SRF plane segments as a header.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    header : list[SrfSegment]
        The list of SrfSegments to write.
    """
    srf_file.write(f"PLANE {len(header)}\n")
    for segment in header:
        srf_file.write(
            f"{segment.elon:.6f} {segment.elat:.6f} {segment.nstk} {segment.ndip} {segment.len:.4f} {segment.wid:.4f}\n"
        )
        srf_file.write(
            f"{segment.stk:g} {segment.dip:g} {segment.dtop:.4f} {segment.shyp:.4f} {segment.dhyp:.4f}\n"
        )


def write_point_count(srf_file: TextIO, point_count: int):
    """Write out a POINTS declaration for the number of points in an SRF file..

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    point_count : int
        The number of points to write.
    """
    srf_file.write(f"POINTS {point_count}\n")


def write_slip(srf_file: TextIO, slips: list[float]):
    """Write out slip values to an SRF file.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    slips : list[float]
        The slip values to write.
    """
    for i in range(0, len(slips), 6):
        srf_file.write(
            "  "
            + "  ".join(f"{slips[j]:.5E}" for j in range(i, min(len(slips), i + 6)))
            + "\n"
        )


def write_srf_point(srf_file: TextIO, point: SrfPoint):
    """Write out a single SRF point.

    Parameters
    ----------
    srf_file : TextIO
        The SRF file to write to.
    point : SrfPoint
        The point to write.
    """
    srf_file.write(
        f"{point.lon:.6f} {point.lat:.6f} {point.dep:g} {point.stk:g} {point.dip:g} {point.area:.4E} {point.tinit:.4f} {point.dt:.6E}\n"
    )
    srf_file.write(
        f"{point.rake:g} {point.slip1:.4f} {len(point.sr1)} {point.slip2:.4f} {len(point.sr2)} {point.slip3:.4f} {len(point.sr3)}\n"
    )
    if point.sr1:
        write_slip(srf_file, point.sr1)
    if point.sr2:
        write_slip(srf_file, point.sr2)
    if point.sr3:
        write_slip(srf_file, point.sr3)

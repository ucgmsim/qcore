"""
This module provides functionality to read xyts and xyzts files.

Extended Summary
----------------
This module includes the XYTSFile class, which represents an XYTS or XYZTS
file.  It allows users to load metadata, retrieve data, and calculate PGV
(Peak Ground Velocity) and MMI (Modified Mercalli Intensity) values from the
XYTS file.

Two file layouts are supported:

1. **Standard XYTS** (60-byte ``tsheader`` header, ``nz=1``):
   Surface XY-plane timeslices with exactly 3 velocity components.
   Data shape: ``(nt, 3, ny, nx)``.

2. **Proc-local timeslice** (72-byte ``tsheader_procP`` / ``tsheader_procP3``
   header):
   Written per MPI rank.  Two sub-variants are auto-detected from the header:

   a. **Proc-local XYTS** – ``local_nz == 1``, ``ncomp == 3``.
      Data shape: ``(nt, 3, local_ny, local_nx)``.
      Requires ``proc_local_file=True`` (backwards-compatible flag).

   b. **Proc-local XYZTS** (EMOD3D ≥ v3.0.13, ``ts_xyz`` output) –
      ``local_nz > 1``, ``ncomp ∈ {3, 6, 9}``.
      Detected automatically without any user flag.
      Data shape: ``(nt, ncomp, local_nz, local_ny, local_nx)``.
      Filenames typically follow the ``*_xyzts-*`` pattern.

For proc-local files ``ncomp`` is derived from the file size so that no
explicit ``ncomp`` field is required in the header.

Classes
----------------
- XYTSFile: Represents an XYTS / XYZTS file and provides methods to interact
  with it.

Notes
-----
- This module assumes that the simulation domain is flat, and the timeseries
  contained inside the xyts files begins at t = 0.

References
----------
- XYTS file format documentation:
    https://wiki.canterbury.ac.nz/x/FAGnAw#FileFormatsUsedInGroundMotionSimulation-XYTS.e3dbinaryformat
- C struct metadata definition:
    merge_ts/structure.h (in the EMOD3D repository)

Examples
--------
# Load a standard XYTS file
xyts_file = XYTSFile("example.e3d")

# Load a proc-local XYTS file (backwards-compatible flag)
xyts_file = XYTSFile("example_xyts-000000", proc_local_file=True)

# Load a volumetric XYZTS file (auto-detected – no flag needed)
xyzts_file = XYTSFile("example_xyzts-000000")

# Retrieve corners of the simulation domain
corners = xyts_file.corners()

# Retrieve PGV map
pgv_map = xyts_file.pgv()

# Calculate MMI values
_, mmi_values = xyts_file.pgv(mmi=True)
"""

import dataclasses
from enum import Enum
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from qcore import geo

# Maximum plausible grid dimension.  Values up to this threshold are treated
# as valid local_nx / local_ny during endianness detection; byte-swapped
# representations of small integers exceed this bound by several orders of
# magnitude (e.g. 100 byte-swapped → 1 677 721 600).
_MAX_GRID_DIM = 0xFFFF


class Component(Enum):
    """Timestep component."""

    MAGNITUDE = -1
    X = 0
    Y = 1
    Z = 2


@dataclasses.dataclass
class XYTSFile:
    """
    Represents an XYTS or XYZTS file.

    Supports three file layouts – see module docstring for details.

    Assumptions:
    - dip = 0: Simulation domain is flat.
    - t0 = 0: Complete timeseries from t = 0.

    Attributes:
        x0: Starting x-coordinate.
        y0: Starting y-coordinate.
        z0: Starting z-coordinate.
        t0: Starting time.
        local_nx: Number of local x-coordinates (proc-local files only).
        local_ny: Number of local y-coordinates (proc-local files only).
        local_nz: Number of local z-coordinates (proc-local files only).
        nx: Total number of x-coordinates.
        ny: Total number of y-coordinates.
        nz: Total number of z-coordinates.
        nt: Total number of time steps.
        dx: Grid spacing in the x-direction.
        dy: Grid spacing in the y-direction.
        hh: Grid spacing in the z-direction.
        dt: Time step size.
        mrot: Rotation angle for model origin.
        mlat: Latitude of the model origin.
        mlon: Longitude of the model origin.
        ncomp: Number of components per grid point.  Always 3 for standard
            XYTS and proc-local XYTS files.  For XYZTS files this is 3, 6,
            or 9 as set by ``ts_xyz_ncomp`` in the EMOD3D configuration.
        dxts: Original simulation grid spacing in the x-direction.
        dyts: Original simulation grid spacing in the y-direction.
        nx_sim: Original simulation size in the x-direction.
        ny_sim: Original simulation size in the y-direction.
        dip: Dip angle.
        comps: Orientation of components (X, Y, Z).
        cosR: Cosine of the rotation angle.
        sinR: Sine of the rotation angle.
        cosP: Cosine of the dip angle.
        sinP: Sine of the dip angle.
        rot_matrix: Rotation matrix for components.
        data: Memory-mapped array containing the data.
            Shape is ``(nt, ncomp, ny, nx)`` for standard XYTS,
            ``(nt, ncomp, local_ny, local_nx)`` for proc-local XYTS, and
            ``(nt, ncomp, local_nz, local_ny, local_nx)`` for XYZTS.
        ll_map: Longitude-latitude map for data.

    Methods:
        __init__(xyts_path, meta_only=False, proc_local_file=False):
            Initializes the XYTSFile object by loading metadata and memmapping
            data sections.

        corners(gmt_format=False):
            Retrieves the corners of the simulation domain.

        region(corners=None):
            Returns the simulation region as a tuple (x_min, x_max, y_min, y_max).

        tslice_get(step, comp=-1, outfile=None):
            Retrieves timeslice data (standard and proc-local XYTS only).

        pgv(mmi=False, pgvout=None, mmiout=None):
            Retrieves PGV map and optionally calculates MMI (standard XYTS
            only).
    """

    # Header values
    x0: int
    y0: int
    z0: int
    t0: int
    #######################
    nx: int
    ny: int
    nz: int
    nt: int
    dx: float
    dy: float
    hh: float
    dt: float
    mrot: float
    mlat: float
    mlon: float
    # Derived values
    ncomp: int
    dxts: int
    dyts: int
    nx_sim: int
    dip: float
    comps: dict[str, float]
    cosR: float
    sinR: float
    cosP: float
    sinP: float
    rot_matrix: np.ndarray
    # proc-local files only
    local_nx: int | None = None
    local_ny: int | None = None
    local_nz: int | None = None

    # contents
    data: np.memmap | None = (
        None  # NOTE: this is distinct (but nearly identical to) a np.ndarray
    )

    ll_map: np.ndarray | None = None

    def __init__(
        self,
        xyts_path: Path | str,
        meta_only: bool = False,
        proc_local_file: bool = False,
        round_dt: bool = True,
    ):
        """Initializes the XYTSFile object.

        Parameters
        ----------
        xyts_path : Path | str
            Path to the xyts / xyzts file.
        meta_only : bool
            If True, only loads metadata and doesn't prepare gridpoint datum
            locations (slower).
        proc_local_file : bool
            If True, forces reading with a 72-byte proc-local header.  This
            is still required for proc-local XYTS files whose ``local_nz``
            equals 1, because those cannot be distinguished from standard XYTS
            files by header inspection alone.  XYZTS files (``local_nz > 1``)
            are detected automatically and do not need this flag.
        round_dt : bool
            If True, round the dt value to 4dp (present only for backwards
            compatibility).

        Raises
        ------
        ValueError
            If the file layout cannot be determined from the header bytes.
        """

        xytf = open(xyts_path, "rb")

        self.xyts_path = xyts_path

        # ---------------------------------------------------------------
        # Step 1 – determine endianness and whether this is a proc-local
        # (72-byte header) file.
        #
        # Layout of the first 7 32-bit integers:
        #   Standard XYTS (60-byte header):
        #     [x0, y0, z0, t0, nx, ny, nz]  →  nz == 1
        #   Proc-local XYTS / XYZTS (72-byte header):
        #     [x0, y0, z0, t0, local_nx, local_ny, local_nz]
        #
        # For standard XYTS the 7th word is nz=1, giving a reliable
        # endianness probe.  For XYZTS files local_nz > 1, so the probe
        # fails and we fall through to the proc-local branch below.
        # ---------------------------------------------------------------
        raw7_be = np.fromfile(xytf, dtype=">i4", count=7)
        seventh_be = int(raw7_be[-1])

        if seventh_be == 0x00000001:
            endian = ">"
            proc_local = proc_local_file
        elif seventh_be == 0x01000000:
            endian = "<"
            proc_local = proc_local_file
        else:
            # The 7th word is neither 1 nor byte-swapped 1.  The only
            # known cause is a proc-local XYZTS file where local_nz > 1.
            # Determine endianness from local_nx (5th word, index 4):
            # small-positive big-endian reads are implausible as
            # little-endian because byte-swapping maps e.g. 100 → 1677721600.
            local_nx_be = int(raw7_be[4])
            raw7_le = np.frombuffer(raw7_be.tobytes(), dtype="<i4")
            local_nx_le = int(raw7_le[4])

            if 1 <= local_nx_be <= _MAX_GRID_DIM and not (1 <= local_nx_le <= _MAX_GRID_DIM):
                endian = ">"
            elif 1 <= local_nx_le <= _MAX_GRID_DIM and not (1 <= local_nx_be <= _MAX_GRID_DIM):
                endian = "<"
            elif 1 <= local_nx_be <= _MAX_GRID_DIM:
                # Both readings look plausible; use the big-endian
                # interpretation of local_ny as a tiebreaker.
                local_ny_be = int(raw7_be[5])
                local_ny_le = int(raw7_le[5])
                if 1 <= local_ny_be <= _MAX_GRID_DIM and not (1 <= local_ny_le <= _MAX_GRID_DIM):
                    endian = ">"
                else:
                    endian = "<"
            else:
                xytf.close()
                raise ValueError(
                    "File is not an XY timeslice file: %s" % (xyts_path)
                )
            proc_local = True
        xytf.seek(0)

        # ---------------------------------------------------------------
        # Step 2 – read the header fields.
        # ---------------------------------------------------------------
        (self.x0, self.y0, self.z0, self.t0) = np.fromfile(
            xytf, dtype="%si4" % (endian), count=4
        )
        if proc_local:
            (self.local_nx, self.local_ny, self.local_nz) = np.fromfile(
                xytf, dtype="%si4" % (endian), count=3
            )

        (
            self.nx,
            self.ny,
            self.nz,
            self.nt,
        ) = np.fromfile(xytf, dtype="%si4" % (endian), count=4)
        self.dx, self.dy, self.hh, self.dt, self.mrot, self.mlat, self.mlon = (
            np.fromfile(xytf, dtype="%sf4" % (endian), count=7)
        )
        xytf.close()
        # dt is sensitive to float error eg 0.2 stores as 0.199999 (dangerous)
        if round_dt:
            self.dt = np.around(self.dt, decimals=4)

        # ---------------------------------------------------------------
        # Step 3 – derive ncomp.
        #
        # For standard XYTS files ncomp is always 3.
        # For proc-local files it is derived from the file size so that no
        # explicit ncomp field is required in the header:
        #   ncomp = (file_bytes - 72) / (4 * nt * local_nz * local_ny * local_nx)
        # ---------------------------------------------------------------
        if proc_local:
            header_bytes = 72
            file_bytes = Path(xyts_path).stat().st_size
            data_floats = (file_bytes - header_bytes) // 4
            vol = int(self.nt) * int(self.local_nz) * int(self.local_ny) * int(self.local_nx)
            if vol == 0:
                raise ValueError(
                    "Proc-local header dimensions are zero; cannot derive ncomp."
                )
            ncomp, remainder = divmod(data_floats, vol)
            if remainder != 0 or ncomp not in {3, 6, 9}:
                raise ValueError(
                    f"Cannot determine a valid ncomp (got {ncomp}) from file size "
                    f"{file_bytes} and local dimensions "
                    f"{int(self.local_nz)}×{int(self.local_ny)}×{int(self.local_nx)}, "
                    f"nt={int(self.nt)}.  Expected ncomp ∈ {{3, 6, 9}}."
                )
            self.ncomp = int(ncomp)
        else:
            self.ncomp = 3

        # determine original sim parameters
        self.dxts = int(round(self.dx / self.hh))
        self.dyts = int(round(self.dy / self.hh))
        self.nx_sim = self.nx * self.dxts
        self.ny_sim = self.ny * self.dyts

        # orientation of components
        self.dip = 0
        self.comps = {
            "X": radians(90 + self.mrot),
            "Y": radians(self.mrot),
            "Z": radians(90 - self.dip),
        }
        # rotation of components so Y is true north
        self.cosR = cos(self.comps["X"])
        self.sinR = sin(self.comps["X"])
        # simulation plane always flat, dip = 0
        self.cosP = 0  # cos(self.comps['Z'])
        self.sinP = 1  # sin(self.comps['Z'])
        # xy dual component rotation matrix
        # must also flip vertical axis
        theta = radians(self.mrot)
        self.rot_matrix = np.array(
            [[cos(theta), -sin(theta), 0], [-sin(theta), -cos(theta), 0], [0, 0, -1]]
        )

        # save speed when only loaded to read metadata section
        if meta_only:
            return

        # ---------------------------------------------------------------
        # Step 4 – memory-map the data payload.
        #
        # Shapes:
        #   Standard XYTS             : (nt, ncomp, ny,       nx)
        #   Proc-local XYTS (nz == 1) : (nt, ncomp, local_ny, local_nx)
        #   XYZTS        (nz  > 1)   : (nt, ncomp, local_nz, local_ny, local_nx)
        # ---------------------------------------------------------------
        if proc_local:
            if int(self.local_nz) > 1:
                # Volumetric XYZTS: 5-D tensor
                shape: tuple[int, ...] = (
                    int(self.nt),
                    self.ncomp,
                    int(self.local_nz),
                    int(self.local_ny),
                    int(self.local_nx),
                )
            else:
                # Surface proc-local XYTS: 4-D tensor (backwards-compatible)
                shape = (
                    int(self.nt),
                    self.ncomp,
                    int(self.local_ny),
                    int(self.local_nx),
                )
            self.data = np.memmap(
                xyts_path,
                dtype="%sf4" % (endian),
                mode="r",
                offset=72,
                shape=shape,
            )
        else:
            # memory map for data section
            self.data = np.memmap(
                xyts_path,
                dtype="%sf4" % (endian),
                mode="r",
                offset=60,
                shape=(int(self.nt), self.ncomp, int(self.ny), int(self.nx)),
            )

        # create longitude, latitude map for data
        grid_points = (
            np.mgrid[0 : self.nx_sim : self.dxts, 0 : self.ny_sim : self.dyts]
            .reshape(2, -1, order="F")
            .T
        )
        amat = geo.gen_mat(self.mrot, self.mlon, self.mlat)[0]
        ll_map = geo.xy2ll(
            geo.gp2xy(grid_points, self.nx_sim, self.ny_sim, self.hh), amat
        ).reshape(self.ny, self.nx, 2)
        if np.min(ll_map[:, :, 0]) < -90 and np.max(ll_map[:, :, 0]) > 90:
            # assume crossing over 180 -> -180, extend past 180
            ll_map[ll_map[:, :, 0] < 0, 0] += 360
        self.ll_map = ll_map

    def corners(
        self, gmt_format: bool = False
    ) -> list[list[float]] | tuple[list[list[float]], str]:
        """Retrieves the corners of the simulation domain.

        Parameters
        ----------
        gmt_format : bool
            If True, returns corners in GMT string format alongside the corner
            list.

        Returns
        -------
        List[List[float]] | Tuple[List[List[float]], str]
            List of corners and (optionally) GMT string.
        """
        # compared with model_params format:
        # c1 =   x0   y0
        # c2 = xmax   y0
        # c3 = xmax ymax
        # c4 =   x0 ymax
        # cannot just use self.ll_map as xmax, ymax for simulation domain
        # may have been decimated. sim nx 1400 (xmax 1399) with dxts 5 = 1395
        gp_cnrs = np.array(
            [
                [0, 0],
                [self.nx_sim - 1, 0],
                [self.nx_sim - 1, self.ny_sim - 1],
                [0, self.ny_sim - 1],
            ]
        )
        amat = geo.gen_mat(self.mrot, self.mlon, self.mlat)[0]
        ll_cnrs = geo.xy2ll(geo.gp2xy(gp_cnrs, self.nx_sim, self.ny_sim, self.hh), amat)
        if np.min(ll_cnrs[:, 0]) < -90 and np.max(ll_cnrs[:, 0]) > 90:
            # assume crossing over 180 -> -180, extend past 180
            ll_cnrs[ll_cnrs[:, 0] < 0, 0] += 360

        if not gmt_format:
            return ll_cnrs.tolist()

        gmt_cnrs = "\n".join([" ".join(map(str, cnr)) for cnr in ll_cnrs])
        return ll_cnrs.tolist(), gmt_cnrs

    def region(
        self, corners: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """Returns simulation region.

        Parameters
        ----------
        corners : Optional[np.ndarray]
            If not None, use the given precalculated corners.

        Returns
        -------
        Tuple[float, float, float, float]
            The simulation region as a tuple (x_min, x_max, y_min, y_max).
        """
        if corners is None:
            corners = self.corners()
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)

        return (float(x_min), float(x_max), float(y_min), float(y_max))

    def tslice_get(
        self,
        step: int,
        comp: Component = Component.MAGNITUDE,
    ) -> np.ndarray:
        """Retrieves timeslice data.

        This method operates on the surface XY plane and is intended for
        standard XYTS and proc-local XYTS files (``local_nz == 1``).  For
        volumetric XYZTS files access ``self.data`` directly.

        Parameters
        ----------
        step : int
            Timestep to retrieve data for.
        comp : Component
            Timestep component.

        Returns
        -------
        np.ndarray
            Retrieved timeslice data.

        Raises
        ------
        ValueError
            If called on a volumetric XYZTS file (``local_nz > 1``).
        """
        if self.local_nz is not None and int(self.local_nz) > 1:
            raise ValueError(
                "tslice_get() is not supported for volumetric XYZTS files "
                "(local_nz > 1).  Access self.data directly."
            )
        match comp:
            case Component.MAGNITUDE:
                return np.linalg.norm(self.data[step, :3, :, :], axis=0)
            case Component.X:
                return (
                    self.data[step, 0, :, :] * self.sinR
                    + self.data[step, 1, :, :] * self.cosR
                )
            case Component.Y:
                return (
                    self.data[step, 0, :, :] * self.cosR
                    - self.data[step, 1, :, :] * self.sinR
                )
            case Component.Z:
                return self.data[step, 2, :, :] * -1

    def pgv(
        self,
        mmi: bool = False,
        pgvout: Path | str | None = None,
        mmiout: Path | str | None = None,
    ) -> None | np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Retrieves PGV and/or MMI map.

        This method is intended for standard XYTS files.  It is not applicable
        to volumetric XYZTS files.

        Parameters
        ----------
        mmi : bool
            If True, also calculates MMI.
        pgvout : Optional[Path | str]
            File to store PGV or None to return it.
        mmiout : Optional[Path | str]
            File to store MMI or None to return it.

        Returns
        -------
        None | np.ndarray | Tuple[np.ndarray, np.ndarray]
            PGV map or tuple of (PGV map, MMI map) or None (if both are written to a
            file).

        Raises
        ------
        ValueError
            If called on a volumetric XYZTS file (``local_nz > 1``).
        """
        if self.local_nz is not None and int(self.local_nz) > 1:
            raise ValueError(
                "pgv() is not supported for volumetric XYZTS files "
                "(local_nz > 1)."
            )
        # PGV as timeslices reduced to maximum value at each point
        pgv = np.zeros(self.nx * self.ny)
        for ts in range(self.t0, self.nt):
            pgv = np.maximum(
                np.sqrt(
                    np.sum(
                        np.power(
                            np.dot(
                                self.data[ts, :, :, :].reshape(3, -1).T, self.rot_matrix
                            ),
                            2,
                        ),
                        axis=1,
                    )
                ),
                pgv,
            )

        # modified marcalli intensity formula
        if mmi:
            mmiv = np.where(
                np.log10(pgv) < 0.53,
                3.78 + 1.47 * np.log10(pgv),
                2.89 + 3.16 * np.log10(pgv),
            )

        # transform to give longitude, latitude, pgv value
        pgv = np.vstack((self.ll_map.reshape(-1, 2).T, pgv)).T
        if mmi:
            mmiv = np.vstack((self.ll_map.reshape(-1, 2).T, mmiv)).T

        # store / output
        if pgvout is not None:
            pgv.astype(np.float32).tofile(pgvout)
        if mmi and mmiout is not None:
            mmiv.astype(np.float32).tofile(mmiout)

        if pgvout is None:
            if not mmi:
                return pgv
            elif mmiout is None:
                return pgv, mmiv

"""NHM Model Module."""

import datetime
import math
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import pandas as pd
import pooch

from qcore import geo
from qcore.uncertainties.distributions import truncated_normal as sample_trunc_norm_dist

NHM_HEADER = f"""FAULT SOURCES - (created {datetime.datetime.now().strftime("%d-%b-%Y")})
Row 1: FaultName
Row 2: TectonicType , FaultType
Row 3: LengthMean , LengthSigma (km)
Row 4: DipMean , DipSigma (deg)
Row 5: DipDir
Row 6: Rake (deg)
Row 7: RupDepthMean , RupDepthSigma (km)
Row 8: RupTopMean, RupTopMin RupTopMax  (km)
Row 9: SlipRateMean , SlipRateSigma (mm/yr)
Row 10: CouplingCoeff , CouplingCoeffSigma (mm/yr)
Row 11: MwMedian , RecurIntMedian  (yr)
Row 12: Num Locations on Fault Surface
Row 13+: Location Coordinates (Long, Lat)
"""

# This mu is used to calculated Moment Rate
MU = 3.0 * 10.0**10.0

POINTS_PER_KILOMETER = (
    1 / 0.1
)  # 1km divided by distance between points (1km/0.1km gives 100m grid)


# In the /QuakeCoRE/Public/Cybershake_200m directory
NHM_MODEL_URL = "https://www.dropbox.com/scl/fi/q93swheg6pqs0fiwviohg/NZ_FLTmodel_2010_v18p6.txt?rlkey=gnotizv4kx50un73y9176gzhk&st=667dlt19&dl=1"
NHM_MODEL_HASH = "3e70bf86f00d89d8191b7e0d27d052e3c5784900e7bdc4963d3772e692305d7a"


def mag2mom_nm(mw: float) -> float:
    """Converts magnitude to moment - Newton-metre

    Parameters
    ----------
    mw : float
        The magnitude to convert.

    Returns
    -------
    float
        The moment in newton-metres.
    """
    return 10 ** (9.05 + 1.5 * mw)


@dataclass
class NHMFault:
    """
    Contains the information for a single fault from a NHM file.

    Notes
    -----
    This class stores geometric, kinematic, and statistical parameters
    describing an individual fault, as defined in the New Zealand
    National Hazard Model (NHM) fault database.
    """

    name: str
    """Name of the fault."""

    tectonic_type: str
    """Tectonic setting type of the fault."""

    fault_type: str
    """Fault style of the rupture (e.g., REVERSE, NORMAL, STRIKE-SLIP)."""

    length: float
    """Fault length in kilometers."""

    length_sigma: float
    """Uncertainty (standard deviation) in fault length, in kilometers."""

    dip: float
    """Fault dip angle in degrees."""

    dip_sigma: float
    """Uncertainty (standard deviation) in dip angle, in degrees."""

    dip_dir: float
    """Dip direction (azimuth) in degrees."""

    rake: float
    """Slip rake angle in degrees."""

    dbottom: float
    """Depth to the bottom of the fault plane, in kilometers."""

    dbottom_sigma: float
    """Uncertainty (standard deviation) in bottom depth, in kilometers."""

    dtop: float
    """Mean depth to the top of the fault plane, in kilometers."""

    dtop_min: float
    """Minimum depth to the top of the fault plane, in kilometers."""

    dtop_max: float
    """Maximum depth to the top of the fault plane, in kilometers."""

    slip_rate: float
    """Slip rate along the fault, in millimeters per year."""

    slip_rate_sigma: float
    """Uncertainty (standard deviation) in slip rate, in millimeters per year."""

    coupling_coeff: float
    """Fault coupling coefficient."""

    coupling_coeff_sigma: float
    """Uncertainty (standard deviation) in coupling coefficient."""

    mw: float
    """Moment magnitude of the fault."""

    recur_int_median: float
    """Median recurrence interval in years."""

    trace: np.ndarray
    """Fault surface trace as an array of (longitude, latitude) pairs for the top edge of the fault."""

    # TODO: add x y z fault plane data as in SRF info
    # TODO: add leonard mw function

    def sample_2012(
        self, mw_area_scaling: bool = True, mw_perturbation: bool = True
    ) -> "NHMFault":
        """
        Perturb the current NHM fault according to the OpenSHA implementation,
        using the same Mw scaling relations as Stirling (2012).

        The top depth is perturbed uniformly between `dtop_min` and `dtop_max`.
        All other parameters are perturbed using a truncated normal distribution
        within two standard deviations.

        Parameters
        ----------
        mw_area_scaling : bool, optional
            If True, apply magnitude scaling based on rupture area. Default is True.
        mw_perturbation : bool, optional
            If True, apply a random perturbation to the magnitude. Default is True.

        Returns
        -------
        NHMFault
            A new `NHMFault` instance with perturbed parameters, where all
            sigma values are set to zero.
        """
        mw = self.mw

        dtop = self.dtop_min + (self.dtop_max - self.dtop_min) * np.random.uniform()
        length = sample_trunc_norm_dist(self.length, self.length_sigma)
        dbot = sample_trunc_norm_dist(self.dbottom, self.dbottom_sigma)
        dip = sample_trunc_norm_dist(self.dip, self.dip_sigma)
        slip_rate = sample_trunc_norm_dist(self.slip_rate, self.slip_rate_sigma)
        coupling_coeff = sample_trunc_norm_dist(
            self.coupling_coeff, self.coupling_coeff_sigma
        )

        if mw_area_scaling:
            mw_sigma = 0.2

            if mw_perturbation:
                mw = sample_trunc_norm_dist(self.mw, mw_sigma, std_dev_limit=1)

        moment = mag2mom_nm(mw)
        moment_base = mag2mom_nm(self.mw)
        moment_rate_base = moment_base * 1 / self.recur_int_median

        # if the slip rate is 0, then the moment rate does not need scaling
        if self.slip_rate > 0:
            slip_factor = slip_rate / self.slip_rate
        else:
            slip_factor = 1

        moment_rate = moment_rate_base * slip_factor
        recur_int_median = moment / moment_rate

        return NHMFault(
            name=self.name,
            tectonic_type=self.tectonic_type,
            fault_type=self.fault_type,
            length=length,
            length_sigma=0,
            dip=dip,
            dip_sigma=0,
            dip_dir=self.dip_dir,
            rake=self.rake,
            dbottom=dbot,
            dbottom_sigma=0,
            dtop=dtop,
            dtop_min=dtop,
            dtop_max=dtop,
            slip_rate=slip_rate,
            slip_rate_sigma=0,
            coupling_coeff=coupling_coeff,
            coupling_coeff_sigma=0,
            mw=mw,
            recur_int_median=recur_int_median,
            trace=self.trace,
        )

    def write(self, out_fp: TextIO, header: bool = False) -> None:
        """
        Write the fault parameters to an NHM file.

        Parameters
        ----------
        out_fp : TextIO
            File pointer for the open file (for writing).
        header : bool, optional
            If True, write the NHM header at the start of the file. Default is False.
        """
        if header:
            out_fp.write(NHM_HEADER)

        out_fp.write("\n")
        out_fp.write(f"{self.name}\n")
        out_fp.write(f"{self.tectonic_type} {self.fault_type}\n")
        out_fp.write(f"{self.length:10.3f}{self.length_sigma:10.3f}\n")
        out_fp.write(f"{self.dip:10.3f}{self.dip_sigma:10.3f}\n")
        out_fp.write(f"{self.dip_dir:10.3f}\n")
        out_fp.write(f"{self.rake:10.3f}\n")
        out_fp.write(f"{self.dbottom:10.3f}{self.dbottom_sigma:10.3f}\n")
        out_fp.write(f"{self.dtop:10.3f}{self.dtop_min:10.3f}{self.dtop_max:10.3f}\n")
        out_fp.write(f"{self.slip_rate:10.3f}{self.slip_rate_sigma:10.3f}\n")
        out_fp.write(f"{self.coupling_coeff:10.3f}{self.coupling_coeff_sigma:10.3f}\n")
        out_fp.write(f"{self.mw:10.3f}{self.recur_int_median:10.3e}\n")
        out_fp.write(f"{len(self.trace):10d}\n")
        for lat, lon in self.trace:
            out_fp.write(f"{lat:10.5f} {lon:10.5f}\n")


def load_nhm(
    nhm_path: str | None = None, skiprows: int = len(NHM_HEADER.splitlines()) + 1
) -> dict[str, NHMFault]:
    """Reads the nhm_path and returns a dictionary of NHMFault by fault name.

    Parameters
    ----------
    nhm_path : str, optional
        NHM file to load. If not provided, a default will be downloaded from the QuakeCoRE Dropbox.
    skiprows : int, optional
        Skip the first skiprows lines; default: 15.

    Returns
    -------
    dict[str, NHMFault]
        NHMFault by name
    """
    if not nhm_path:
        nhm_path = pooch.retrieve(url=NHM_MODEL_URL, known_hash=NHM_MODEL_HASH)
    with open(nhm_path, "r") as f:
        rows = "".join(f.readlines()[skiprows:])

    faults = {}
    for entry in rows.split("\n\n"):
        rows = list(map(str.strip, entry.split("\n")))

        def str2floats(line: str):
            return list(map(float, line.split()))

        tectonic_type, fault_type = rows[1].split()
        length, length_sigma = str2floats(rows[2])
        dip, dip_sigma = str2floats(rows[3])
        dbottom, dbottom_sigma = str2floats(rows[6])
        dtop, dtop_min, dtop_max = str2floats(rows[7])
        slip_rate, slip_rate_sigma = str2floats(rows[8])
        coupling_coeff, coupling_coeff_sigma = str2floats(rows[9])
        mw, recur_int_median = str2floats(rows[10])

        nhm_fault = NHMFault(
            name=rows[0],
            tectonic_type=tectonic_type,
            fault_type=fault_type,
            length=length,
            length_sigma=length_sigma,
            dip=dip,
            dip_sigma=dip_sigma,
            dip_dir=float(rows[4]),
            rake=float(rows[5]),
            dbottom=dbottom,
            dbottom_sigma=dbottom_sigma,
            dtop=dtop,
            dtop_min=dtop_min,
            dtop_max=dtop_max,
            slip_rate=slip_rate,
            slip_rate_sigma=slip_rate_sigma,
            coupling_coeff=coupling_coeff,
            coupling_coeff_sigma=coupling_coeff_sigma,
            mw=mw,
            recur_int_median=recur_int_median,
            trace=np.array(list(map(float, " ".join(rows[12:]).split()))).reshape(
                (-1, 2)
            ),
        )
        nhm_fault = nhm_fault
        faults[nhm_fault.name] = nhm_fault

    return faults


def load_nhm_df(nhm_ffp: str, erf_name: str | None = None):
    """Creates a standardised pandas dataframe for the
    ruptures in the given erf file.

    Parameters
    ----------
    nhm_ffp : str
        Path to the ERF file
    erf_name : str or None
        name to identify faults from an ERF

    Returns
    -------
    DataFrame
        The NHM2010 model loaded as a dataframe.
    """
    nhm_infos = load_nhm(nhm_ffp)

    erf_suffix = ""
    if erf_name:
        erf_suffix = f"_{erf_name}"

    rupture_dict = {
        f"{info.name}{erf_suffix}": {
            "name": info.name,
            "tectonic_type": info.tectonic_type,
            "length": info.length,
            "length_sigma": info.length_sigma,
            "dip": info.dip,
            "dip_sigma": info.dip_sigma,
            "dip_dir": info.dip_dir,
            "rake": info.rake,
            "dbottom": info.dbottom,
            "dbottom_sigma": info.dbottom_sigma,
            "dtop": info.dtop,
            "dtop_min": info.dtop_min,
            "dtop_max": info.dtop_max,
            "slip_rate": info.slip_rate,
            "slip_rate_sigma": info.slip_rate_sigma,
            "coupling_coeff": info.coupling_coeff,
            "coupling_coeff_sigma": info.coupling_coeff_sigma,
            "mw": info.mw,
            "recur_int_median": info.recur_int_median
            if info.recur_int_median > 0
            else float("nan"),
            "exceedance": 1 / info.recur_int_median
            if info.recur_int_median > 0
            else float("nan"),
        }
        for info in nhm_infos.values()
    }

    return pd.DataFrame.from_dict(rupture_dict, orient="index").sort_index()


def get_fault_header_points(
    fault: NHMFault,
) -> tuple[list[dict[str, int | float]], np.ndarray]:
    """
    Calculates and produces fault information such as the entire trace and fault header info per plane

    Parameters
    ----------
    fault : NHMFault
        A fault object from an NHM file

    Returns
    -------
    list of dict
        SRF Header values.
    np.ndarray
        SRF points.
    """
    srf_points = []
    srf_header: list[dict[str, int | float]] = []
    lon1, lat1 = fault.trace[0]
    lon2, lat2 = fault.trace[1]
    strike = geo.ll_bearing(lon1, lat1, lon2, lat2, midpoint=True)

    # If the dip direction is not to the right of the strike, turn the fault around
    indexes = (
        np.arange(len(fault.trace))
        if 180 > fault.dip_dir - strike >= 0
        else np.flip(np.arange(len(fault.trace)))
    )

    plane_offset = 0
    for i, i2 in zip(indexes[:-1], indexes[1:]):
        lon1, lat1 = fault.trace[i]
        lon2, lat2 = fault.trace[i2]

        strike = geo.ll_bearing(lon1, lat1, lon2, lat2, midpoint=True)
        plane_point_distance = geo.ll_dist(lon1, lat1, lon2, lat2)

        nstrike = round(plane_point_distance * POINTS_PER_KILOMETER)
        strike_dist = plane_point_distance / nstrike

        end_strike = geo.ll_bearing(lon1, lat1, lon2, lat2)
        for j in range(nstrike):
            top_lat, top_lon = geo.ll_shift(lat1, lon1, strike_dist * j, end_strike)
            srf_points.append([top_lon, top_lat, fault.dtop])

        height = fault.dbottom - fault.dtop

        width = abs(height / np.tan(np.deg2rad(fault.dip)))
        dip_dist = height / np.sin(np.deg2rad(fault.dip))

        ndip = int(round(dip_dist * POINTS_PER_KILOMETER))
        hdip_dist = width / ndip
        vdip_dist = height / ndip

        for j in range(1, ndip):
            hdist = j * hdip_dist
            vdist = j * vdip_dist + fault.dtop
            for local_lon, local_lat, _ in srf_points[
                plane_offset : plane_offset + nstrike
            ]:
                new_lat, new_lon = geo.ll_shift(
                    local_lat, local_lon, hdist, fault.dip_dir
                )
                srf_points.append([new_lon, new_lat, vdist])

        plane_offset += nstrike * ndip
        srf_header.append(
            {
                "nstrike": nstrike,
                "ndip": ndip,
                "strike": strike,
                "length": plane_point_distance,
                "dip": fault.dip,
                "dtop": fault.dtop,
                "width": fault.dbottom / math.sin(math.radians(fault.dip)),
                "dhyp": -999.9,
                "shyp": -999.9,
            }
        )

    return srf_header, np.asarray(srf_points)

from typing import List
import datetime

import numpy as np
from scipy.stats import truncnorm

from srf_generation.source_parameter_generation.uncertainties import mag_scaling


NHM_HEADER = f"""FAULT SOURCES - New Zealand National Seismic Hazard Model 2010 (created {datetime.datetime.now().strftime("%d-%b-%Y")}) 
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


MU = 3.0 * 10.0 ** 10.0


def sample_trunc_norm_dist(mean, sigma, sigma_limit=2):
    return float(truncnorm(-sigma_limit, sigma_limit, loc=mean, scale=sigma).rvs())


class NHMFault:
    """Contains the information for a single fault from a NHM file

    Attributes
    ----------
    name : str
    tectonic_type : str
    fault_type : str
    length : float, (km)
    length_sigma : float (km)
    dip : float  (deg)
    dip_sigma : float (deg)
    dip_dir : float
    rake : float
    dbottom : float (km)
    dbottom_sigma : float (km)
    dtop_mean : float (km)
    dtop_min : float (km)
    dtop_max : float (km)
    slip_rate : float (mm/yr)
    slip_rate_sigma : float (mm/yr)
    coupling_coeff : float (mm/yr)
    coupling_coeff_sigma : float (mm/yr)
    mw : float
    recur_int_median : float (yr)
    trace: np.ndarray
        fault surface trace (lon, lat)
    """

    def __init__(
        self,
        name,
        tectonic_type,
        fault_type,
        length,
        length_sigma,
        dip,
        dip_sigma,
        dip_dir,
        rake,
        dbottom,
        dbottom_sigma,
        dtop,
        dtop_min,
        dtop_max,
        slip_rate,
        slip_rate_sigma,
        coupling_coeff,
        coupling_coeff_sigma,
        mw,
        recur_int_median,
        trace,
    ):
        """Creates an NHMFault instance from the given NHM text.

        Parameters
        ----------
        kwargs of all the required parameters for a nhm fault

        It is up to the caller to make sure all the required parameters are available
        """
        self.name = name
        self.tectonic_type = tectonic_type
        self.fault_type = fault_type
        self.length = length
        self.length_sigma = length_sigma
        self.dip = dip
        self.dip_sigma = dip_sigma
        self.dip_dir = dip_dir
        self.rake = rake
        self.dbottom = dbottom
        self.dbottom_sigma = dbottom_sigma
        self.dtop = dtop
        self.dtop_min = dtop_min
        self.dtop_max = dtop_max
        self.slip_rate = slip_rate
        self.slip_rate_sigma = slip_rate_sigma
        self.coupling_coeff = coupling_coeff
        self.coupling_coeff_sigma = coupling_coeff_sigma
        self.mw = mw
        self.recur_int_median = recur_int_median
        self.trace = trace

        # TODO: add x y z fault plane data as in SRF info
        # TODO: add leonard mw function

    def sample_2012(self):
        """
        Permutates the current NHM fault as per the OpenSHA implementation. This uses the same Mw scaling relations
        in Stirling 2012
        Dtop is peturbated with a uniform distribution between min and max.
        The remaining parameters are perturburbated with a truncated normal distribution (2 standard deviations)

        :return: new NHM object with the perturbated parameters containing 0 in all sigma sections
        """
        rake = self.rake
        strike = self.dip_dir
        fault_name = self.name
        tectonic_type = self.tectonic_type
        fault_type = self.fault_type

        dtop = self.dtop_min + (self.dtop_max - self.dtop_min) * np.random.uniform()

        length = sample_trunc_norm_dist(self.length, self.length_sigma)
        dbot = sample_trunc_norm_dist(self.dbottom, self.dbottom_sigma)
        dip = sample_trunc_norm_dist(self.dip, self.dip_sigma)
        slip_rate = sample_trunc_norm_dist(self.slip_rate, self.slip_rate_sigma)
        coupling_coeff = sample_trunc_norm_dist(
            self.coupling_coeff, self.coupling_coeff_sigma
        )

        width = (dbot - dtop) / np.sin(np.radians(dip))
        if tectonic_type == "VOLCANIC" or (
            tectonic_type == "ACTIVE_SHALLOW" and fault_type == "NORMAL_FAULTING"
        ):
            mw_scaling_rel = mag_scaling.MagnitudeScalingRelations.VILLAMORETAL2007
        elif tectonic_type == "SUBDUCTION_INTERFACE":
            ### SUBDUCTION INTERFACE RELATION IS TO BE CONFIRMED BY BB leaving equations here for reference

            mw_old = 4.441 + 0.846 * np.log10(length * width)
            mom = (
                self.recur_int_median
                * MU
                * length
                * width
                * self.slip_rate
                * 10e3
                * self.coupling_coeff
            )
            mw = (np.log10(mom) - 9.05) / 1.5

            mw_scaling_rel = mag_scaling.MagnitudeScalingRelations.SKARLATOUDIS2016
        elif fault_type == "PLATE_BOUNDARY":
            mw_scaling_rel = mag_scaling.MagnitudeScalingRelations.HANKSBAKUN2002
        elif fault_type == "OTHER_CRUSTAL_FAULTING":
            mw_scaling_rel = mag_scaling.MagnitudeScalingRelations.STIRLING2008
        else:
            raise (
                ValueError,
                f"Invalid combination of tectonic type: {tectonic_type} and fault type: {fault_type}",
            )

        mw_median, mw_sigma = mag_scaling.lw_2_mw_sigma_scaling_relation(
            length, width, mw_scaling_rel, rake
        )
        mw = sample_trunc_norm_dist(mw_median, mw_sigma)

        moment = 10 ** (9.05 + 1.5 * mw)
        momentRate = mu * (length) * (width) * (slip_rate * 1000.0) * coupling_coeff

        if momentRate > 0:
            recur_int_median = moment / momentRate
        else:
            recur_int_median = self.recur_int_median

        return NHMFault(
            name=fault_name,
            tectonic_type=tectonic_type,
            fault_type=fault_type,
            length=length,
            length_sigma=0,
            dip=dip,
            dip_sigma=0,
            dip_dir=strike,
            rake=rake,
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

    def write(self, out_fp, header=False):
        """
        Writes a section of the NHM file

        :param out_fp: file pointer for open file (for writing)
        :param header: flag to write the header at the start of the file
        :return:
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
        out_fp.write(f"{self.slip_rate:10.3f}{0:10.3f}\n")
        out_fp.write(f"{self.coupling_coeff:10.3f}{0:10.3f}\n")
        out_fp.write(f"{self.mw:10.3f}{self.recur_int_median:10.3e}\n")
        out_fp.write(f"{len(self.trace):10d}\n")
        for lat, lon in self.trace:
            out_fp.write(f"{lat:10.5f} {lon:10.5f}\n")


def load_nhm(nhm_path: str, skiprows: int = len(NHM_HEADER.splitlines()) + 1):
    """Reads the nhm_path and returns a dictionary of NHMFault by fault name.

    Parameters
    ----------
    nhm_path: str
        NHM file to load
    skiprows: int
        Skip the first skiprows lines; default: 15.

    Returns
    -------
    dict of NHMFault by name
    """
    with open(nhm_path, "r") as f:
        rows = "".join(f.readlines()[skiprows:])

    faults = {}
    for entry in rows.split("\n\n"):
        rows = list(map(str.strip, entry.split("\n")))

        def str2floats(line):
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

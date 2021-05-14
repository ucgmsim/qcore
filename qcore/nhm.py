from typing import List
import datetime

import numpy as np

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

    def __init__(self, entry: List[str]):
        """Creates an NHMFault instance from the given NHM text.

        Parameters
        ----------
        entry : List of str
            The rows of an NHM file of one fault.
            Format:
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
        rows = list(map(str.strip, entry.split("\n")))

        def str2floats(line):
            return list(map(float, line.split()))

        self.name = rows[0]
        self.tectonic_type, self.fault_type = rows[1].split()
        self.length, self.length_sigma = str2floats(rows[2])
        self.dip, self.dip_sigma = str2floats(rows[3])
        self.dip_dir = float(rows[4])
        self.rake = float(rows[5])
        self.dbottom, self.dbottom_sigma = str2floats(rows[6])
        self.dtop, self.dtop_min, self.dtop_max = str2floats(rows[7])
        self.slip_rate, self.slip_rate_sigma = str2floats(rows[8])
        self.coupling_coeff, self.coupling_coeff_sigma = str2floats(rows[9])
        self.mw, self.recur_int_median = str2floats(rows[10])
        self.trace = np.array(list(map(float, " ".join(rows[12:]).split()))).reshape((-1, 2))
        # TODO: add x y z fault plane data as in SRF info
        # TODO: add leonard mw function


def load_nhm(nhm_path: str, skiprows: int=15):
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
        nhm_fault = NHMFault(entry)
        faults[nhm_fault.name] = nhm_fault

    return faults


def write_nhm_section(out_fp, coupling_coeff, dbot, dip, dtop, fault_data, fault_name, length, mw, rake,
                      recurance_interval, slip_rate, strike, trace):
    out_fp.write("\n")
    out_fp.write(f"{fault_name}\n")
    out_fp.write(f"{fault_data.tect_type} {fault_data.fault_type}\n")
    out_fp.write(f"{length:10.3f}{0:10.3f}\n")
    out_fp.write(f"{dip:10.3f}{0:10.3f}\n")
    out_fp.write(f"{strike:10.3f}\n")
    out_fp.write(f"{rake:10.3f}\n")
    out_fp.write(f"{dbot:10.3f}{0:10.3f}\n")
    out_fp.write(f"{dtop:10.3f}{dtop:10.3f}{dtop:10.3f}\n")
    out_fp.write(f"{slip_rate:10.3f}{0:10.3f}\n")
    out_fp.write(f"{coupling_coeff:10.3f}{0:10.3f}\n")
    out_fp.write(f"{mw:10.3f}{recurance_interval:10.3e}\n")
    out_fp.write(f"{len(trace):10d}\n")
    for lat, lon in trace:
        out_fp.write(f"{lat:10.5f} {lon:10.5f}\n")
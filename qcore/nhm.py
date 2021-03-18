from typing import List

import numpy as np


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

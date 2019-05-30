from typing import List

import numpy as np


class NHMInfo:
    """Contains the information for a single fault from an nhm file

    Attributes
    ----------
    fault_name : str
    tectonic_type : str
    fault_type : str
    length_mean : float, (km)
    length_sigma : float (km)
    dip_mean : float  (deg)
    dip_sigma : float (deg)
    dip_dir : float
    rup_depth_mean : float (km)
    rup_depth_sigma : float (km)
    slip_rate_mean : float (mm/yr)
    slip_rate_sigma : float (mm/yr)
    coupling_coeff : float (mm/yr)
    coupling_coeff_sigma : float (mm/yr)
    mw_median : float
    recur_int_median : float (yr)
    n_locations : int
        Number of locations on fault surface
    locations: np.ndarray
        The fault surface locations, shape [n_locations, 2], format (lon, lat)
    """

    def __init__(self):
        """Do not use the constructor, instances should only be
        created via the from_nhm_section method"""
        pass

    @classmethod
    def from_nhm_section(cls, rows: List[str]):
        """Creates an NHMInfo instance from the given nhm file section

        Parameters
        ----------
        rows : List of str
            The rows of an nhm file corresponding to one specific fault.
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

        Returns
        -------
        NHMInfo
        """
        # Not sure of a tidier of doing this (without adding all these to the init..)
        info = NHMInfo()

        info.fault_name = rows[0].strip()
        info.tectonic_type, info.fault_type = cls.read_row(rows[1], convert=False)
        info.length_mean, info.length_sigma = cls.read_row(rows[2])
        info.dip_mean, info.dip_sigma = cls.read_row(rows[3])
        info.dip_dir = float(rows[4].strip())
        info.rake = float(rows[5].strip())
        info.rup_depth_mean, info.rup_depth_sigma = cls.read_row(rows[6])
        info.rup_top_mean, info.rup_top_min, info.rup_top_max = cls.read_row(rows[7])
        info.slip_rate_mean, info.slip_rate_sigma = cls.read_row(rows[8])
        info.coupling_coeff, info.coupling_coeff_sigma = cls.read_row(rows[9])
        info.mw_median, info.recur_int_median = cls.read_row(rows[10])
        info.n_locations = int(rows[11].strip())

        # Read the locations
        info.locations = np.asarray([cls.read_row(row) for row in rows[12:]])

        return info

    @staticmethod
    def read_row(row: str, sep: str = None, convert: bool = True):
        return [
            float(entry.strip()) if convert else entry.strip()
            for entry in row.split(sep)
        ]


def read_nhm_file(nhm_file: str):
    """Reads the nhm file and returns a list of NHMInfo, one for each fault

    Parameters
    ----------
    nhm_file: str
        The nhm file to read

    Returns
    -------
    List of NHMInfo
    """
    # Read the file and skip the first 15 lines (format info)
    with open(nhm_file, "r") as f:
        rows = f.readlines()[15:]

    cur_rows, result = [], []
    for row in rows:
        # Empty row, i.e. separates faults
        if len(row.strip()) == 0:
            result.append(NHMInfo.from_nhm_section(cur_rows))
            cur_rows = []
        else:
            cur_rows.append(row)

    return result

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
    rake : float
    rup_depth_mean : float (km)
    rup_depth_sigma : float (km)
    rup_top_mean : float (km)
    rup_top_min : float (km)
    rup_top_max : float (km)
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

    def __init__(
        self,
        fault_name: str,
        tectonic_type: str,
        fault_type: str,
        length_mean: float,
        length_sigma: float,
        dip_mean: float,
        dip_sigma: float,
        dip_dir: float,
        rake: float,
        rup_depth_mean: float,
        rup_depth_sigma: float,
        rup_top_mean: float,
        rup_top_min: float,
        rup_top_max: float,
        slip_rate_mean: float,
        slip_rate_sigma: float,
        coupling_coeff: float,
        coupling_coeff_sigma: float,
        mw_median: float,
        recur_int_median: float,
        n_locations: int,
        locations: np.ndarray,
    ):
        self.fault_name = fault_name
        self.tectonic_type = tectonic_type
        self.fault_type = fault_type
        self.length_mean = length_mean
        self.length_sigma = length_sigma
        self.dip_mean = dip_mean
        self.dip_sigma = dip_sigma
        self.dip_dir = dip_dir
        self.rake = rake
        self.rup_depth_mean = rup_depth_mean
        self.rup_depth_sigma = rup_depth_sigma
        self.rup_top_mean = rup_top_mean
        self.rup_top_min = rup_top_min
        self.rup_top_max = rup_top_max
        self.slip_rate_mean = slip_rate_mean
        self.slip_rate_sigma = slip_rate_sigma
        self.coupling_coeff = coupling_coeff
        self.coupling_coeff_sigma = coupling_coeff_sigma
        self.mw_median = mw_median
        self.recur_int_median = recur_int_median
        self.n_locations = n_locations
        self.locations = locations

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
        fault_name = rows[0].strip()
        tectonic_type, fault_type = cls.__read_row(rows[1], convert=False)
        length_mean, length_sigma = cls.__read_row(rows[2])
        dip_mean, dip_sigma = cls.__read_row(rows[3])
        dip_dir = float(rows[4].strip())
        rake = float(rows[5].strip())
        rup_depth_mean, rup_depth_sigma = cls.__read_row(rows[6])
        rup_top_mean, rup_top_min, rup_top_max = cls.__read_row(rows[7])
        slip_rate_mean, slip_rate_sigma = cls.__read_row(rows[8])
        coupling_coeff, coupling_coeff_sigma = cls.__read_row(rows[9])
        mw_median, recur_int_median = cls.__read_row(rows[10])
        n_locations = int(rows[11].strip())

        return NHMInfo(
            fault_name,
            tectonic_type,
            fault_type,
            length_mean,
            length_sigma,
            dip_mean,
            dip_sigma,
            dip_dir,
            rake,
            rup_depth_mean,
            rup_depth_sigma,
            rup_top_mean,
            rup_top_min,
            rup_top_max,
            slip_rate_mean,
            slip_rate_sigma,
            coupling_coeff,
            coupling_coeff_sigma,
            mw_median,
            recur_int_median,
            n_locations,
            np.asarray([cls.__read_row(row) for row in rows[12:]]))

    @staticmethod
    def __read_row(row: str, sep: str = None, convert: bool = True):
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

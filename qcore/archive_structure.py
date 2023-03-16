"""
Gives access to the folder structure of archived cybershake directories
"""
from pathlib import Path

from .simulation_structure import get_fault_from_realisation


def get_IM_csv_from_root(archive_root: Path, realisation: str):
    """Gets the full path to the im_csv file given the archive root dir and the realistion name"""
    fault_name = get_fault_from_realisation(realisation)
    return (
        archive_root
        / fault_name
        / f"{fault_name}_IM"
        / f"{realisation}.csv"
    )

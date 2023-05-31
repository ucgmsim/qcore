"""
Gives access to the folder structure of archived cybershake directories
"""
from pathlib import Path

from .simulation_structure import get_fault_from_realisation


def get_fault_source_dir(fault_dir: Path):
    """Gets the Source directory for the given fault directory"""
    return fault_dir / "Source"


def get_fault_im_dir(fault_dir: Path):
    """Gets the IM directory for the given fault directory"""
    return fault_dir / "IM"


def get_fault_bb_dir(fault_dir: Path):
    """Gets the BB directory for the given fault directory"""
    return fault_dir / "BB"


def get_IM_csv_from_root(archive_root: Path, realisation: str):
    """Gets the full path to the im_csv file given the archive root dir and the realistion name"""
    fault_name = get_fault_from_realisation(realisation)
    return get_fault_im_dir(archive_root / fault_name) / f"{realisation}.csv"

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


def get_fault_source_dir(fault_dir: Path):
    """Gets the Source directory for the given fault directory"""
    return fault_dir / f"{fault_dir.stem}_Source"

def get_fault_im_dir(fault_dir: Path):
    """Gets the IM directory for the given fault directory"""
    return fault_dir / f"{fault_dir.stem}_IM"

def get_fault_bb_dir(fault_dir: Path):
    """Gets the BB directory for the given fault directory"""
    return fault_dir / f"{fault_dir.stem}_BB"
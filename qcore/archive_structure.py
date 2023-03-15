"""
Gives access to the folder structure of archived cybershake directories
"""
from pathlib import Path


def get_fault_from_realisation(realisation: str):
    """ Gets the fault name from the realisation name"""
    return realisation.rsplit("_REL", 1)[0]


def get_IM_csv_from_root(archive_root: Path, realisation: str):
    """ Gets the full path to the im_csv file given the archive root dir and the realistion name"""
    return archive_root / get_fault_from_realisation(realisation) / "IM" / f"{realisation}.csv"
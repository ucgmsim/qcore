"""
Gives access to the folder structure of archived cybershake directories
"""

from pathlib import Path

from .simulation_structure import get_fault_from_realisation


def get_fault_source_dir(fault_dir: Path) -> Path:
    """
    Get the Source directory for a given fault directory.

    Parameters
    ----------
    fault_dir : Path
        Path to the fault directory.

    Returns
    -------
    Path
        Path to the Source directory within the given fault directory.
    """
    return fault_dir / "Source"


def get_fault_im_dir(fault_dir: Path) -> Path:
    """
    Get the IM directory for a given fault directory.

    Parameters
    ----------
    fault_dir : Path
        Path to the fault directory.

    Returns
    -------
    Path
        Path to the IM directory within the given fault directory.
    """
    return fault_dir / "IM"


def get_IM_csv_from_root(archive_root: Path, realisation: str) -> Path:  # noqa: N802
    """
    Get the full path to the IM CSV file given the archive root and realisation name.

    Parameters
    ----------
    archive_root : Path
        Path to the root directory of the Cybershake archive.
    realisation : str
        Name of the realisation to locate.

    Returns
    -------
    Path
        Full path to the IM CSV file for the specified realisation.
    """
    fault_name = get_fault_from_realisation(realisation)
    return get_fault_im_dir(archive_root / fault_name) / f"{realisation}.csv"

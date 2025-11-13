from pathlib import Path

import pytest

from qcore import archive_structure


def test_get_fault_source_dir():
    fault_dir = Path("/path/to/fault")
    result = archive_structure.get_fault_source_dir(fault_dir)
    assert result == Path("/path/to/fault/Source")
    assert isinstance(result, Path)


def test_get_fault_im_dir():
    fault_dir = Path("/path/to/fault")
    result = archive_structure.get_fault_im_dir(fault_dir)
    assert result == Path("/path/to/fault/IM")
    assert isinstance(result, Path)


def test_get_IM_csv_from_root():
    archive_root = Path("/archive/root")
    realisation = "SomeFault_REL01"
    result = archive_structure.get_IM_csv_from_root(archive_root, realisation)
    expected = Path("/archive/root/SomeFault/IM/SomeFault_REL01.csv")
    assert result == expected
    assert isinstance(result, Path)


def test_get_IM_csv_from_root_different_realisation():
    archive_root = Path("/archive/root")
    realisation = "AlpineF2K3T1_REL03"
    result = archive_structure.get_IM_csv_from_root(archive_root, realisation)
    expected = Path("/archive/root/AlpineF2K3T1/IM/AlpineF2K3T1_REL03.csv")
    assert result == expected

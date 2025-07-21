"""Test module for XYTS file processing using pytest fixtures."""

from pathlib import Path
from typing import Optional
from urllib import request

import numpy as np
import pytest

from qcore import xyts
from qcore.test.tool import utils


@pytest.fixture(scope="session")
def xyts_file() -> xyts.XYTSFile:
    """Provide path to the XYTS test file."""
    # Assuming the test file exists in the test directory
    test_file = Path(__file__).parent / "sample1" / "xyts.e3d"
    if not test_file.exists():
        request.urlretrieve(
            "https://www.dropbox.com/s/zge70zvntzxatpo/xyts.e3d?dl=1", test_file
        )
    return xyts.XYTSFile(str(test_file))


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    """Provide path to sample output directory."""
    return Path(__file__).parent / "sample1" / "output"


@pytest.mark.parametrize(
    "gmt_format,expected_corners",
    [
        (
            False,
            [
                [170.9143829345703, -43.291595458984375],
                [172.26246643066406, -44.217742919921875],
                [171.01295471191406, -45.15299987792969],
                [169.66343688964844, -44.21214294433594],
            ],
        ),
        (
            True,
            (
                [
                    [170.9143829345703, -43.291595458984375],
                    [172.26246643066406, -44.217742919921875],
                    [171.01295471191406, -45.15299987792969],
                    [169.66343688964844, -44.21214294433594],
                ],
                "170.914382935 -43.291595459\n"
                "172.262466431 -44.2177429199\n"
                "171.012954712 -45.1529998779\n"
                "169.66343689 -44.2121429443",
            ),
        ),
    ],
)
def test_corners(
    xyts_file: xyts.XYTSFile,
    gmt_format: bool,
    expected_corners: list[list[float]] | tuple[list[list[float]], str],
) -> None:
    """Test the corners method of XYTS file."""
    computed_corners = xyts_file.corners(gmt_format=gmt_format)

    if gmt_format:
        expected_array, expected_string = expected_corners
        computed_array, computed_string = computed_corners

        assert np.array(computed_array) == pytest.approx(np.array(expected_array))

        computed_coords = np.array(
            [line.split() for line in computed_string.split("\n")], dtype=float
        )
        expected_coords = np.array(
            [line.split() for line in expected_string.split("\n")], dtype=float
        )
        assert computed_coords == pytest.approx(expected_coords)
    else:
        assert np.array(computed_corners) == pytest.approx(np.array(expected_corners))


@pytest.mark.parametrize(
    "corners,expected_region",
    [
        (
            None,
            (
                169.66343688964844,
                172.26246643066406,
                -45.15299987792969,
                -43.291595458984375,
            ),
        ),
        (
            np.array(
                [
                    [170.91449, -43.2916],
                    [172.26257, -44.21773],
                    [171.01305, -45.15298],
                    [169.66354, -44.21213],
                ]
            ),
            (
                169.66354000000001,
                172.26257000000001,
                -45.152979999999999,
                -43.291600000000003,
            ),
        ),
    ],
)
def test_region(
    xyts_file: xyts.XYTSFile,
    corners: np.ndarray | None,
    expected_region: tuple,
) -> None:
    """Test the region method of XYTS file."""
    result = xyts_file.region(corners)
    assert result == pytest.approx(expected_region)


@pytest.mark.parametrize(
    "mmi,pgvout_name,mmiout_name",
    [
        (True, None, None),
        (False, None, None),
        (True, "test_pgv_path1", "test_mmi_path1"),
        (False, "test_pgv_path2", None),
    ],
)
def test_pgv(
    xyts_file: xyts.XYTSFile,
    sample_data_dir: Path,
    tmp_path: Path,
    mmi: bool,
    pgvout_name: str | None,
    mmiout_name: str | None,
) -> None:
    """Test the PGV (Peak Ground Velocity) processing method."""
    sample_pgv = sample_data_dir / "sample_pgvout"
    sample_mmi = sample_data_dir / "sample_mmiout"

    # Set up output paths if specified
    pgvout = str(tmp_path / pgvout_name) if pgvout_name else None
    mmiout = str(tmp_path / mmiout_name) if mmiout_name else None

    # Execute PGV processing
    xyts_output = xyts_file.pgv(mmi=mmi, pgvout=pgvout, mmiout=mmiout)

    # Validate results
    if pgvout:
        sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
        test_pgv_array = np.fromfile(pgvout, dtype="3<f4")
        assert test_pgv_array == pytest.approx(sample_pgv_array)

        if mmiout:
            sample_mmi_array = np.fromfile(sample_mmi, dtype="3<f4")
            test_mmi_array = np.fromfile(mmiout, dtype="3<f4")
            assert test_mmi_array == pytest.approx(sample_mmi_array)
    else:
        if not mmi:
            sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
            assert xyts_output == pytest.approx(sample_pgv_array)
        elif mmiout is None:
            pgv_result, mmi_result = xyts_output
            sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
            assert pgv_result == pytest.approx(sample_pgv_array)
            sample_mmi_array = np.fromfile(sample_mmi, dtype="3<f4")
            assert mmi_result == pytest.approx(sample_mmi_array)


@pytest.mark.parametrize(
    "step,comp,sample_outfile",
    [
        (10, xyts.Component.MAGNITUDE, "out_tslice-1"),
        (10, xyts.Component.X, "out_tslice0"),
        (10, xyts.Component.Y, "out_tslice1"),
        (10, xyts.Component.Z, "out_tslice2"),
    ],
)
def test_tslice_get(
    xyts_file: xyts.XYTSFile,
    sample_data_dir: Path,
    step: int,
    comp: xyts.Component,
    sample_outfile: str,
) -> None:
    """Test the time slice extraction method."""
    sample_file = sample_data_dir / sample_outfile
    test_output = xyts_file.tslice_get(step, comp=comp)
    sample_array = np.fromfile(sample_file, dtype="3<f4")
    assert test_output == pytest.approx(sample_array[:, -1].reshape(test_output.shape))

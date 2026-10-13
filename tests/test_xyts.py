"""Test module for XYTS file processing using pytest fixtures."""

from pathlib import Path
from urllib import request

import numpy as np
import pytest

from qcore import xyts


@pytest.fixture(scope="session")
def xyts_file_path() -> Path:
    """Provide path to the XYTS test file, downloading it if required."""
    # Assuming the test file exists in the test directory
    test_file = Path(__file__).parent / "sample1" / "xyts.e3d"
    if not test_file.exists():
        request.urlretrieve(
            "https://www.dropbox.com/s/zge70zvntzxatpo/xyts.e3d?dl=1", test_file
        )
    return test_file


@pytest.fixture(scope="session")
def xyts_file(xyts_file_path: Path) -> xyts.XYTSFile:
    """Provide an XYTSFile instance."""
    return xyts.XYTSFile(str(xyts_file_path))


@pytest.fixture(scope="session")
def meta_only_xyts_file(xyts_file_path: Path) -> xyts.XYTSFile:
    """Provide a meta-only XYTSFile instance."""
    return xyts.XYTSFile(str(xyts_file_path), meta_only=True)


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
                (
                    "170.914382935 -43.291595459\n"
                    "172.262466431 -44.2177429199\n"
                    "171.012954712 -45.1529998779\n"
                    "169.66343689 -44.2121429443"
                ),
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

        assert isinstance(computed_string, str)
        assert isinstance(expected_string, str)
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
            assert isinstance(xyts_output, tuple)
            assert len(xyts_output) == 2
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


def test_xyts_invalid_file(tmp_path: Path) -> None:
    """Test that ValueError is raised for invalid XYTS file."""
    invalid_file = tmp_path / "invalid.e3d"
    # Create a file with invalid header
    with open(invalid_file, "wb") as f:
        f.write(b"\x00" * 100)

    with pytest.raises(ValueError, match="File is not an XY timeslice file"):
        xyts.XYTSFile(str(invalid_file))


def test_tslice_get_meta_only(meta_only_xyts_file: xyts.XYTSFile) -> None:
    """Test that AttributeError is raised when tslice_get is called on meta-only instance."""
    with pytest.raises(
        AttributeError, match="The data attribute must be set to use `tslice_get`"
    ):
        meta_only_xyts_file.tslice_get(10, comp=xyts.Component.MAGNITUDE)


def test_pgv_meta_only(meta_only_xyts_file: xyts.XYTSFile) -> None:
    """Test that AttributeError is raised when pgv is called on meta-only instance."""
    with pytest.raises(
        AttributeError, match="The data and ll_map attributes must be set to use `pgv`"
    ):
        meta_only_xyts_file.pgv()


def _write_xyts(
    path: Path,
    data: np.ndarray,
    endian: str = "<",
    mrot: float = 30.0,
    mlat: float = -43.5,
    mlon: float = 172.5,
    hh: float = 0.1,
    dxts: int = 5,
    local_shape: tuple[int, int] | None = None,
) -> None:
    """Write a synthetic XYTS file with the given data and header values.

    `data` has shape (nt, 3, ny, nx). If `local_shape` (local_nx, local_ny) is
    given, a process-local header is written instead of a merged one.
    """
    nt, _, ny, nx = data.shape
    ints = [0, 0, 0, 0]
    if local_shape is not None:
        ints += [*local_shape, 1]
    ints += [nx, ny, 1, nt]
    floats = [hh * dxts, hh * dxts, hh, 0.1, mrot, mlat, mlon]
    with open(path, "wb") as f:
        np.asarray(ints, dtype=f"{endian}i4").tofile(f)
        np.asarray(floats, dtype=f"{endian}f4").tofile(f)
        data.astype(f"{endian}f4").tofile(f)


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """Provide deterministic (nt, 3, ny, nx) velocity data."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 2.0, size=(4, 3, 3, 5)).astype(np.float32)


def test_endianness_detected(tmp_path: Path, synthetic_data: np.ndarray) -> None:
    """Little- and big-endian files with the same contents read identically."""
    little = tmp_path / "little.e3d"
    big = tmp_path / "big.e3d"
    _write_xyts(little, synthetic_data, endian="<")
    _write_xyts(big, synthetic_data, endian=">")
    # The two files really are byte-swapped versions of each other.
    assert little.read_bytes() != big.read_bytes()

    little_file = xyts.XYTSFile(little)
    big_file = xyts.XYTSFile(big)

    for xyts_file in (little_file, big_file):
        assert (xyts_file.nx, xyts_file.ny, xyts_file.nz, xyts_file.nt) == (5, 3, 1, 4)
        assert xyts_file.hh == pytest.approx(0.1)
        assert xyts_file.dxts == 5
        assert xyts_file.mlon == pytest.approx(172.5)
        assert xyts_file.data is not None
        assert np.array_equal(xyts_file.data, synthetic_data)
        assert xyts_file.tslice_get(2) == pytest.approx(
            np.linalg.norm(synthetic_data[2], axis=0)
        )

    assert little_file.ll_map is not None and big_file.ll_map is not None
    assert np.array_equal(little_file.ll_map, big_file.ll_map)
    assert little_file.pgv() == pytest.approx(big_file.pgv())


def test_proc_local_file(tmp_path: Path, synthetic_data: np.ndarray) -> None:
    """A process-local file uses the local dimensions for its data section."""
    local_data = synthetic_data[:, :, :2, :3]
    path = tmp_path / "local.e3d"
    _write_xyts(path, local_data, local_shape=(3, 2))
    # Make the header claim the global (merged) domain is 5x3.
    with open(path, "r+b") as f:
        f.seek(28)
        np.asarray([5, 3], dtype="<i4").tofile(f)

    xyts_file = xyts.XYTSFile(path, proc_local_file=True)

    assert (xyts_file.local_nx, xyts_file.local_ny, xyts_file.local_nz) == (3, 2, 1)
    assert (xyts_file.nx, xyts_file.ny, xyts_file.nt) == (5, 3, 4)
    assert xyts_file.data is not None
    assert xyts_file.data.shape == (4, 3, 2, 3)
    assert np.array_equal(xyts_file.data, local_data)
    assert xyts_file.tslice_get(1, comp=xyts.Component.Z) == pytest.approx(
        -local_data[1, 2]
    )
    # The lon/lat map still covers the global domain.
    assert xyts_file.ll_map is not None
    assert xyts_file.ll_map.shape == (3, 5, 2)


def test_proc_local_file_missing_local_dims(
    tmp_path: Path, synthetic_data: np.ndarray
) -> None:
    """A process-local file with zero local nx is rejected unless meta only."""
    path = tmp_path / "local.e3d"
    _write_xyts(path, synthetic_data, local_shape=(0, 3))

    with pytest.raises(ValueError, match="Local nx, ny must be set"):
        xyts.XYTSFile(path, proc_local_file=True)

    meta = xyts.XYTSFile(path, proc_local_file=True, meta_only=True)
    assert (meta.local_nx, meta.local_ny) == (0, 3)
    assert meta.data is None


def test_date_line_crossing(tmp_path: Path, synthetic_data: np.ndarray) -> None:
    """Domains crossing the antimeridian report continuous longitudes past 180."""
    path = tmp_path / "dateline.e3d"
    _write_xyts(path, synthetic_data, mrot=0.0, mlat=-40.0, mlon=180.0, hh=1.0)

    xyts_file = xyts.XYTSFile(path)

    assert xyts_file.ll_map is not None
    lons = xyts_file.ll_map[:, :, 0]
    assert np.all((lons > 179) & (lons < 181))
    assert lons.min() < 180 < lons.max()

    corners = np.array(xyts_file.corners())
    assert np.all((corners[:, 0] > 179) & (corners[:, 0] < 181))
    assert corners[:, 0].min() < 180 < corners[:, 0].max()

    x_min, x_max, _, _ = xyts_file.region()
    assert 179 < x_min < 180 < x_max < 181

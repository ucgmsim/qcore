"""Test module for XYTS file processing using pytest fixtures."""

import struct
from pathlib import Path
from urllib import request

import numpy as np
import pytest

from qcore import xyts


# ---------------------------------------------------------------------------
# Helpers for constructing synthetic binary timeslice files
# ---------------------------------------------------------------------------

def _write_standard_xyts_header(
    buf: bytearray,
    endian: str,
    x0: int,
    y0: int,
    z0: int,
    t0: int,
    nx: int,
    ny: int,
    nz: int,
    nt: int,
    dx: float,
    dy: float,
    hh: float,
    dt: float,
    mrot: float,
    mlat: float,
    mlon: float,
) -> None:
    """Write a 60-byte standard XYTS header into *buf*."""
    fmt = f"{endian}4i4i7f"
    data = struct.pack(
        fmt.replace("4i", "iiii").replace("7f", "fffffff"),
        x0, y0, z0, t0,
        nx, ny, nz, nt,
        dx, dy, hh, dt, mrot, mlat, mlon,
    )
    buf[:60] = data


def _make_proc_local_header(
    endian: str,
    x0: int, y0: int, z0: int, t0: int,
    local_nx: int, local_ny: int, local_nz: int,
    nx: int, ny: int, nz: int, nt: int,
    dx: float, dy: float, hh: float, dt: float,
    mrot: float, mlat: float, mlon: float,
) -> bytes:
    """Return a 72-byte proc-local (tsheader_procP3) header as bytes."""
    pfx = ">" if endian == ">" else "<"
    ints = struct.pack(
        f"{pfx}11i",
        x0, y0, z0, t0,
        local_nx, local_ny, local_nz,
        nx, ny, nz, nt,
    )
    floats = struct.pack(
        f"{pfx}7f",
        dx, dy, hh, dt, mrot, mlat, mlon,
    )
    return ints + floats  # 44 + 28 = 72 bytes


def _make_xyzts_file(
    path: Path,
    endian: str,
    local_nx: int,
    local_ny: int,
    local_nz: int,
    nx: int,
    ny: int,
    nt: int,
    ncomp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a synthetic XYZTS proc-local file and return the payload array."""
    header = _make_proc_local_header(
        endian,
        x0=0, y0=0, z0=0, t0=0,
        local_nx=local_nx, local_ny=local_ny, local_nz=local_nz,
        nx=nx, ny=ny, nz=local_nz, nt=nt,
        dx=0.4, dy=0.4, hh=0.1, dt=0.02,
        mrot=0.0, mlat=-43.5, mlon=172.0,
    )
    payload = rng.random(
        (nt, ncomp, local_nz, local_ny, local_nx), dtype=np.float32
    )
    dtype = f"{endian}f4"
    path.write_bytes(header + payload.astype(dtype).tobytes())
    return payload


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


# ---------------------------------------------------------------------------
# XYZTS-specific tests (use synthetic data; no network required)
# ---------------------------------------------------------------------------

# Common dimensions used across XYZTS tests
_LOCAL_NX, _LOCAL_NY, _LOCAL_NZ = 4, 5, 3
_GLOBAL_NX, _GLOBAL_NY = 20, 20
_NT = 6


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Fixed-seed random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.mark.parametrize("endian,ncomp", [
    (">", 3),
    ("<", 3),
    ("<", 6),
    ("<", 9),
])
def test_xyzts_auto_detection(
    tmp_path: Path,
    rng: np.random.Generator,
    endian: str,
    ncomp: int,
) -> None:
    """XYTSFile should auto-detect an XYZTS file without proc_local_file flag."""
    fpath = tmp_path / f"test_xyzts-{endian[0]}_ncomp{ncomp}"
    _make_xyzts_file(
        fpath,
        endian=endian,
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=ncomp,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath)

    assert int(xf.local_nz) == _LOCAL_NZ
    assert int(xf.local_ny) == _LOCAL_NY
    assert int(xf.local_nx) == _LOCAL_NX
    assert xf.ncomp == ncomp
    assert int(xf.nt) == _NT


@pytest.mark.parametrize("endian", [">", "<"])
def test_xyzts_data_shape(
    tmp_path: Path,
    rng: np.random.Generator,
    endian: str,
) -> None:
    """Data memmap for XYZTS should be 5-D (nt, ncomp, nz, ny, nx)."""
    ncomp = 3
    fpath = tmp_path / f"shape_xyzts-{endian[0]}"
    _make_xyzts_file(
        fpath,
        endian=endian,
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=ncomp,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath)

    assert xf.data is not None
    assert xf.data.ndim == 5
    assert xf.data.shape == (_NT, ncomp, _LOCAL_NZ, _LOCAL_NY, _LOCAL_NX)


def test_xyzts_payload_values(
    tmp_path: Path,
    rng: np.random.Generator,
) -> None:
    """Values read from the XYZTS memmap match the written payload."""
    endian = "<"
    ncomp = 6
    fpath = tmp_path / "values_xyzts-0"
    payload = _make_xyzts_file(
        fpath,
        endian=endian,
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=ncomp,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath)

    assert xf.data == pytest.approx(payload)


def test_xyzts_tslice_get_raises(
    tmp_path: Path,
    rng: np.random.Generator,
) -> None:
    """tslice_get should raise ValueError for volumetric XYZTS files."""
    fpath = tmp_path / "tslice_xyzts-0"
    _make_xyzts_file(
        fpath,
        endian="<",
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=3,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath)

    with pytest.raises(ValueError, match="tslice_get"):
        xf.tslice_get(0)


def test_xyzts_pgv_raises(
    tmp_path: Path,
    rng: np.random.Generator,
) -> None:
    """pgv() should raise ValueError for volumetric XYZTS files."""
    fpath = tmp_path / "pgv_xyzts-0"
    _make_xyzts_file(
        fpath,
        endian="<",
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=3,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath)

    with pytest.raises(ValueError, match="pgv"):
        xf.pgv()


def test_xyzts_meta_only(
    tmp_path: Path,
    rng: np.random.Generator,
) -> None:
    """meta_only=True should work for XYZTS files and leave data=None."""
    fpath = tmp_path / "meta_xyzts-0"
    _make_xyzts_file(
        fpath,
        endian="<",
        local_nx=_LOCAL_NX,
        local_ny=_LOCAL_NY,
        local_nz=_LOCAL_NZ,
        nx=_GLOBAL_NX,
        ny=_GLOBAL_NY,
        nt=_NT,
        ncomp=3,
        rng=rng,
    )

    xf = xyts.XYTSFile(fpath, meta_only=True)

    assert xf.data is None
    assert int(xf.local_nz) == _LOCAL_NZ
    assert xf.ncomp == 3

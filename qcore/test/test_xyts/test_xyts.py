import errno
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from qcore import shared, xyts
from qcore.test.tool import utils

XYTS_DOWNLOAD_PATH = "https://www.dropbox.com/s/zge70zvntzxatpo/xyts.e3d?dl=0"
XYTS_STORE_PATH = os.path.join(Path.home(), "xyts.e3d")
DOWNLOAD_CMD = f"wget -O {XYTS_STORE_PATH} {XYTS_DOWNLOAD_PATH}"

if not os.path.isfile(XYTS_STORE_PATH):
    out, err = shared.exe(DOWNLOAD_CMD, debug=False)
    if "failed" in err:
        os.remove(XYTS_STORE_PATH)
        sys.exit(f"{err} failted to download xyts benchmark file")
    else:
        print("Successfully downloaded benchmark xyts.e3d")
else:
    print("Benchmark xyts.e3d already exits")

SAMPLE_OUT_DIR_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "sample1/output"
)
XYTS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sample1/xyts.e3d")
try:
    os.symlink(XYTS_STORE_PATH, XYTS_FILE)
except:
    pass

OBJ_XYTS = xyts.XYTSFile(XYTS_FILE)
SAMPLE_PGV = os.path.join(SAMPLE_OUT_DIR_PATH, "sample_pgvout")
SAMPLE_MMI = os.path.join(SAMPLE_OUT_DIR_PATH, "sample_mmiout")
TMP_DIR_NAME = os.path.join(
    Path.home(),
    (
        "tmp_"
        + os.path.basename(__file__)[:-3]
        + "_"
        + "".join(str(datetime.now()).split())
    ).replace(".", "_"),
).replace(":", "_")


def setup_module(scope="module"):
    """create a tmp directory for storing output from test"""
    print("----------setup_module----------")
    try:
        os.mkdir(TMP_DIR_NAME)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def teardown_module():
    """delete the symbolic link
    delete the tmp directory if it is empty"""
    print("---------teardown_module------------")
    if os.path.isfile(XYTS_FILE):
        os.remove(XYTS_FILE)
    if len(os.listdir(TMP_DIR_NAME)) == 0:
        try:
            shutil.rmtree(TMP_DIR_NAME)
        except (IOError, OSError) as e:
            sys.exit(e)


@pytest.mark.parametrize(
    "gmt_format, expected_corners",
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
                "170.914382935 -43.291595459\n172.262466431 -44.2177429199\n171.012954712 -45.1529998779\n169.66343689 -44.2121429443",
            ),
        ),
    ],
)
def test_corners(gmt_format, expected_corners):
    computed_corners = OBJ_XYTS.corners(gmt_format=gmt_format)
    if gmt_format:
        utils.compare_np_array(
            np.array(computed_corners[0]), np.array(expected_corners[0])
        )
        cc = np.array([x.split() for x in computed_corners[1].split("\n")], dtype=float)
        ec = np.array([x.split() for x in expected_corners[1].split("\n")], dtype=float)
        utils.compare_np_array(cc, ec)

    else:
        utils.compare_np_array(np.array(computed_corners), np.array(expected_corners))


@pytest.mark.parametrize(
    "corners, expected_region",
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
            [
                [170.91449, -43.2916],
                [172.26257, -44.21773],
                [171.01305, -45.15298],
                [169.66354, -44.21213],
            ],
            (
                169.66354000000001,
                172.26257000000001,
                -45.152979999999999,
                -43.291600000000003,
            ),
        ),
    ],
)
def test_region(corners, expected_region):
    assert OBJ_XYTS.region(corners) == pytest.approx(expected_region)


@pytest.mark.parametrize(
    "mmi, pgvout, mmiout, sample_pgv, sample_mmi",
    [
        (True, None, None, SAMPLE_PGV, SAMPLE_MMI),
        (False, None, None, SAMPLE_PGV, SAMPLE_MMI),
        (True, "test_pgv_path1", "test_mmi_path1", SAMPLE_PGV, SAMPLE_MMI),
        (False, "test_pgv_path2", None, SAMPLE_PGV, SAMPLE_MMI),
    ],
)
def test_pgv(mmi, pgvout, mmiout, sample_pgv, sample_mmi):
    files_to_del = []
    if pgvout:
        pgvout = os.path.join(TMP_DIR_NAME, pgvout)
        files_to_del.append(pgvout)
    if mmiout:
        mmiout = os.path.join(TMP_DIR_NAME, mmiout)
        files_to_del.append(mmiout)

    xyts_test_output_array = OBJ_XYTS.pgv(mmi=mmi, pgvout=pgvout, mmiout=mmiout)

    if pgvout:
        sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
        test_pgvout_array = np.fromfile(pgvout, dtype="3<f4")
        utils.compare_np_array(sample_pgv_array, test_pgvout_array)
        if mmiout:
            sample_mmi_array = np.fromfile(sample_mmi, dtype="3<f4")
            test_mmiout_array = np.fromfile(mmiout, dtype="3<f4")
            utils.compare_np_array(sample_mmi_array, test_mmiout_array)
    else:
        if not mmi:
            sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
            utils.compare_np_array(sample_pgv_array, xyts_test_output_array)
        elif mmiout == None:
            pgv, mmi = xyts_test_output_array
            sample_pgv_array = np.fromfile(sample_pgv, dtype="3<f4")
            utils.compare_np_array(sample_pgv_array, pgv)
            sample_mmi_array = np.fromfile(sample_mmi, dtype="3<f4")
            utils.compare_np_array(sample_mmi_array, mmi)

    for f in files_to_del:
        utils.remove_file(f)


@pytest.mark.parametrize(
    "step, comp, sample_outfile",
    [
        (10, xyts.Component.MAGNITUDE, "out_tslice-1"),
        (10, xyts.Component.MAGNITUDE, "out_tslice-1"),
        (10, xyts.Component.X, "out_tslice0"),
        (10, xyts.Component.Y, "out_tslice1"),
        (10, xyts.Component.Z, "out_tslice2"),
    ],
)
def test_tslice_get(step, comp, sample_outfile):
    files_to_del = []
    sample_outfile = os.path.join(SAMPLE_OUT_DIR_PATH, sample_outfile)
    test_tslice_output_array = OBJ_XYTS.tslice_get(step, comp=comp)
    sample_array = np.fromfile(sample_outfile, dtype="3<f4")
    utils.compare_np_array(sample_array, test_tslice_output_array)
    for f in files_to_del:
        utils.remove_file(f)

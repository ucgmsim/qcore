import os
import pickle
import shutil
import sys

import numpy as np
import pytest

from qcore import geo, src_site_dist as ssd
from qcore import shared

INPUT = "input"
OUTPUT = "output"
REALISATIONS = [
    (
        "PangopangoF29_HYP01-10_S1244",
        "https://www.dropbox.com/scl/fi/xn6pf4myfl95rnzmh0yji/calc_rrup_test.zip?rlkey=wy51zzxmfoocz1qc0ul3h7l91&st=iuvkw3q5&dl=1",
    )
]


@pytest.yield_fixture(scope="session", autouse=True)
def set_up(request):
    test_data_save_dirs = []
    for i, (REALISATION, DATA_DOWNLOAD_PATH) in enumerate(REALISATIONS):

        data_store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample" + str(i))
        zip_download_path = os.path.join(data_store_path, REALISATION + ".zip")
        print(f"Downloading {DATA_DOWNLOAD_PATH} to {zip_download_path}")

        download_cmd = "wget --no-check-certificate -O {} \"{}\"".format(zip_download_path, DATA_DOWNLOAD_PATH)
        unzip_cmd = "unzip {} -d {}".format(zip_download_path, data_store_path)
        print(data_store_path)
        test_data_save_dirs.append(data_store_path)
        if not os.path.isdir(data_store_path):
            os.makedirs(data_store_path, exist_ok=True)
            out, err = shared.exe(download_cmd, debug=False)
            if "error" in err:
                shutil.rmtree(data_store_path)
                sys.exit("{} failed to retrieve test data".format(err))
            # download_via_ftp(DATA_DOWNLOAD_PATH, zip_download_path)
            if not os.path.isfile(zip_download_path):
                sys.exit(
                    "File failed to download from {}. Exiting".format(
                        DATA_DOWNLOAD_PATH
                    )
                )
            out, err = shared.exe(unzip_cmd, debug=False)
            os.remove(zip_download_path)
            if "error" in err:
                shutil.rmtree(data_store_path)
                sys.exit("{} failed to extract data folder".format(err))

        else:
            print("Benchmark data folder already exists: ", data_store_path)

    # Run all tests
    yield test_data_save_dirs

    # Remove the test data directory
    for PATH in test_data_save_dirs:
        if os.path.isdir(PATH):
            shutil.rmtree(PATH, ignore_errors=True)


def test_calc_rrub_rjb(set_up):
    function = "calc_rrup_rjb"
    for root_path in set_up:
        srf_points = np.load(
            os.path.join(root_path, INPUT, function + "_srf_points.npy")
        )
        locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

        out_rrup = np.load(os.path.join(root_path, OUTPUT, function + "_rrup.npy"))
        out_rjb = np.load(os.path.join(root_path, OUTPUT, function + "_rjb.npy"))

        rrup, rjb = ssd.calc_rrup_rjb(srf_points, locations)

        assert np.all(np.isclose(out_rrup, rrup))
        assert np.all(np.isclose(out_rjb, rjb))


BASIC_SRF_POINTS = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [1, -1, 1]])
BASIC_SRF_HEADER = [{"nstrike": 2, "ndip": 2, "strike": 90.0, "length": 2}]
BASIC_STATIONS = np.asarray(
    [
        [0, 0, 0],
        [0.5, 0, 0],
        [*geo.ll_shift(0, 0, 50, 0)[::-1], 0],
        [*geo.ll_shift(*geo.ll_shift(0, 0, 50, 270), 100, 0)[::-1], 0],
    ]
)
BASIC_RX = np.asarray([0, 0, -50, -100])
BASIC_RY = np.asarray([-1, geo.ll_dist(0, 0, 0.5, 0) - 1, -1, -51])

HOSSACK_SRF_POINTS = np.asarray(
    [
        [176.2493, -38.3301, 0.0431],
        [176.2202, -38.3495, 0.0431],
        [176.2096, -38.3033, 7.886],
        [176.1814, -38.3221, 7.886],
    ]
)
HOSSACK_SRF_HEADER = [{"nstrike": 2, "ndip": 2, "strike": 230.0, "length": 0.2}]
HOSSACK_STATIONS = np.asarray(
    [
        [
            176.16718461,
            -38.27736689,
            0,
        ],  # location 9.118km down dip of the top centre point
        [
            176.30243581,
            -38.40219607,
            0,
        ],  # location 9.118km up dip of the top centre point
        [
            176.17626422,
            -38.35520564,
            0,
        ],  # Location 3.0383km to the South West of the fault
    ]
)
HOSSACK_RX = np.asarray([9.1180, -9.1180, 1.9987])
HOSSACK_RY = np.asarray([1.56658241, 1.5680196, 6.56920416])

RELATIVE_TOLERANCE = 0.001  # 1m tolerance


@pytest.mark.parametrize(
    ["srf_points", "srf_header", "station_location", "rx_bench", "ry_bench"],
    [
        (BASIC_SRF_POINTS, BASIC_SRF_HEADER, BASIC_STATIONS, BASIC_RX, BASIC_RY),
        (
            HOSSACK_SRF_POINTS,
            HOSSACK_SRF_HEADER,
            HOSSACK_STATIONS,
            HOSSACK_RX,
            HOSSACK_RY,
        ),
    ],
)
def test_calc_rx_ry_basic(srf_points, srf_header, station_location, rx_bench, ry_bench):
    rx, ry = ssd.calc_rx_ry(srf_points, srf_header, station_location)
    assert np.all(np.isclose(rx, rx_bench, rtol=RELATIVE_TOLERANCE))
    assert np.all(np.isclose(ry, ry_bench, rtol=RELATIVE_TOLERANCE))


def test_calc_rx_ry(set_up):
    function = "calc_rx_ry"
    for root_path in set_up:
        srf_points = np.load(
            os.path.join(root_path, INPUT, function + "_srf_points.npy")
        )
        srf_header = pickle.load(
            open(os.path.join(root_path, INPUT, function + "_srf_header.P"), "rb")
        )
        locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

        out_rx = np.load(os.path.join(root_path, OUTPUT, function + "_rx.npy"))
        out_ry = np.asarray(
            [-711.41646299]
        )  # np.load(os.path.join(root_path, OUTPUT, function + "_ry.npy"))

        rx, ry = ssd.calc_rx_ry(srf_points, srf_header, locations)

        assert np.all(np.isclose(out_rx, rx))
        assert np.all(np.isclose(out_ry, ry))

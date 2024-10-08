"""
Originally written for IM_calculation, but is relocated and modified for qcore.
History of this file:
https://github.com/ucgmsim/IM_calculation/commits/0dac6f172ddbb11e9b4faadf2e3ab27746e8f0d3/IM_calculation/test/test_source_site_dist/test_src_site_dist_calc.py
"""
import os
import pickle

import numpy as np
import pytest

from qcore import geo, src_site_dist as ssd

INPUT = "input"
OUTPUT = "output"

SAMPLE_NAMES = [("sample0")]


@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_calc_rrub_rjb(sample_name):
    function = "calc_rrup_rjb"

    root_path = os.path.join(os.path.dirname(__file__), sample_name)

    srf_points = np.load(os.path.join(root_path, INPUT, function + "_srf_points.npy"))
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


@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_calc_rx_ry(sample_name):
    function = "calc_rx_ry"

    root_path = os.path.join(os.path.dirname(__file__), sample_name)
    srf_points = np.load(os.path.join(root_path, INPUT, function + "_srf_points.npy"))
    srf_header = pickle.load(
        open(os.path.join(root_path, INPUT, function + "_srf_header.P"), "rb")
    )
    locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

    out_rx = np.load(os.path.join(root_path, OUTPUT, function + "_rx.npy"))
    out_ry = np.load(os.path.join(root_path, OUTPUT, function + "_ry.npy"))

    rx, ry = ssd.calc_rx_ry(srf_points, srf_header, locations)

    assert np.all(np.isclose(out_rx, rx))
    assert np.all(np.isclose(out_ry, ry))

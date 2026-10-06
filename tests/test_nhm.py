import dataclasses
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qcore import nhm


def test_mag2mom_nm() -> None:
    # Test magnitude to moment conversion
    mw = 7.0
    result = nhm.mag2mom_nm(mw)

    # Expected: 10^(9.05 + 1.5 * 7.0) = 10^19.55
    expected = 10 ** (9.05 + 1.5 * 7.0)
    assert result == pytest.approx(expected)


def test_mag2mom_nm_small_magnitude() -> None:
    mw = 5.0
    result = nhm.mag2mom_nm(mw)
    expected = 10 ** (9.05 + 1.5 * 5.0)
    assert result == pytest.approx(expected)


def test_nhm_fault_creation() -> None:
    # Test creating an NHMFault instance
    trace = np.array([[172.0, -43.0], [172.1, -43.1]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=0.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    assert fault.name == "TestFault"
    assert fault.mw == 7.0
    assert fault.length == 50.0
    assert fault.dip == 45.0
    assert np.array_equal(fault.trace, trace)


def test_load_nhm_single_fault(tmp_path: Path):
    # Minimal fake NHM content with one fault entry
    content = """Header line 1
Header line 2
Header line 3
Header line 4
Header line 5
Header line 6
Header line 7
Header line 8
Header line 9
Header line 10
Header line 11
Header line 12
Header line 13
Header line 14
Header line 15
FakeFault
TECTONIC FAULTTYPE
10.0 0.5
45.0 5.0
270.0
90.0
15.0 1.0
5.0 4.0 6.0
1.2 0.1
0.9 0.05
7.1 1200.0
100.0 200.0
101.0 201.0"""

    nhm_file = tmp_path / "fake_faults.nhm"
    nhm_file.write_text(content)

    # Run loader with skiprows=15 to skip header lines
    faults = nhm.load_nhm(str(nhm_file), skiprows=15)

    # Basic structure check
    assert isinstance(faults, dict)
    assert "FakeFault" in faults
    fault = faults["FakeFault"]
    assert isinstance(fault, nhm.NHMFault)

    # Sanity check on parsed values
    assert np.isclose(fault.length, 10.0)
    assert np.isclose(fault.dip, 45.0)
    assert fault.trace.shape == (1, 2)


def test_nhm_fault_write() -> None:
    # Test writing an NHMFault to a file
    trace = np.array([[172.0, -43.0], [172.1, -43.1]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=0.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    output = StringIO()
    fault.write(output, header=False)
    content = output.getvalue()

    # Check that key information is written
    assert "TestFault" in content
    assert "ACTIVE_SHALLOW" in content
    assert "REVERSE" in content
    assert "2\n" in content  # Number of trace points


def test_nhm_fault_write_with_header() -> None:
    trace = np.array([[172.0, -43.0]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=0.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    output = StringIO()
    fault.write(output, header=True)
    content = output.getvalue()

    # Check that header is included
    assert "FAULT SOURCES" in content
    assert "TestFault" in content


def test_nhm_fault_sample_2012() -> None:
    # Test sampling/perturbation of fault parameters
    np.random.seed(42)  # Set seed for reproducibility

    trace = np.array([[172.0, -43.0], [172.1, -43.1]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=2.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    sampled_fault = fault.sample_2012(mw_area_scaling=True, mw_perturbation=True)

    # Check that a new fault is returned
    assert isinstance(sampled_fault, nhm.NHMFault)
    assert sampled_fault.name == fault.name

    # Sigmas should be set to 0 in sampled fault
    assert sampled_fault.length_sigma == 0
    assert sampled_fault.dip_sigma == 0
    assert sampled_fault.dbottom_sigma == 0
    assert sampled_fault.slip_rate_sigma == 0
    assert sampled_fault.coupling_coeff_sigma == 0

    # Trace should be preserved
    assert np.array_equal(sampled_fault.trace, fault.trace)


def test_nhm_fault_sample_2012_without_mw_perturbation() -> None:
    np.random.seed(42)

    trace = np.array([[172.0, -43.0]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=2.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    sampled_fault = fault.sample_2012(mw_area_scaling=True, mw_perturbation=False)

    # Without perturbation, Mw should remain the same
    assert sampled_fault.mw == fault.mw


def test_get_fault_header_points() -> None:
    # Test getting fault header and points
    trace = np.array([[172.0, -43.0], [172.1, -43.1]])

    fault = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=0.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=trace,
    )

    header, points = nhm.get_fault_header_points(fault)

    # Header should be a list of dictionaries
    assert isinstance(header, list)
    assert len(header) > 0
    assert isinstance(header[0], dict)
    assert "nstrike" in header[0]
    assert "ndip" in header[0]

    # Points should be a numpy array
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 3  # lon, lat, depth


def _make_fault(**overrides: Any) -> nhm.NHMFault:
    base = nhm.NHMFault(
        name="TestFault",
        tectonic_type="ACTIVE_SHALLOW",
        fault_type="REVERSE",
        length=50.0,
        length_sigma=5.0,
        dip=45.0,
        dip_sigma=5.0,
        dip_dir=90.0,
        rake=90.0,
        dbottom=20.0,
        dbottom_sigma=2.0,
        dtop=2.0,
        dtop_min=0.0,
        dtop_max=5.0,
        slip_rate=5.0,
        slip_rate_sigma=1.0,
        coupling_coeff=0.9,
        coupling_coeff_sigma=0.1,
        mw=7.0,
        recur_int_median=1000.0,
        trace=np.array([[172.0, -43.0], [172.1, -43.1]]),
    )
    return dataclasses.replace(base, **overrides)


def test_nhm_fault_sample_2012_zero_slip_rate_keeps_recurrence() -> None:
    # With a zero mean slip rate the moment rate is not rescaled, so with an
    # unperturbed magnitude the recurrence interval must be unchanged even
    # though a non-zero slip rate is sampled.
    np.random.seed(0)
    fault = _make_fault(slip_rate=0.0, slip_rate_sigma=1.0)

    sampled = fault.sample_2012(mw_area_scaling=False)

    assert sampled.mw == fault.mw
    assert sampled.slip_rate != 0.0
    assert sampled.recur_int_median == pytest.approx(fault.recur_int_median)


def test_nhm_fault_sample_2012_positive_slip_rate_scales_recurrence() -> None:
    # A positive slip rate rescales the recurrence interval inversely with
    # the sampled slip rate.
    np.random.seed(0)
    fault = _make_fault(slip_rate=5.0, slip_rate_sigma=1.0)

    sampled = fault.sample_2012(mw_area_scaling=False)

    assert sampled.recur_int_median == pytest.approx(
        fault.recur_int_median * fault.slip_rate / sampled.slip_rate
    )


def _write_nhm(path: Path, faults: list[nhm.NHMFault]) -> None:
    with open(path, "w") as f:
        for i, fault in enumerate(faults):
            fault.write(f, header=i == 0)


def test_load_nhm_df_round_trip(tmp_path: Path) -> None:
    faults = [
        _make_fault(name="FaultB", mw=6.5, recur_int_median=500.0),
        _make_fault(name="FaultA", mw=7.2, recur_int_median=2000.0, dip=60.0),
    ]
    nhm_file = tmp_path / "faults.nhm"
    _write_nhm(nhm_file, faults)

    df = nhm.load_nhm_df(str(nhm_file))

    # Index is the bare fault name (no ERF suffix) and is sorted
    assert list(df.index) == ["FaultA", "FaultB"]
    assert list(df["name"]) == ["FaultA", "FaultB"]
    assert df.loc["FaultA", "tectonic_type"] == "ACTIVE_SHALLOW"
    assert df.loc["FaultA", "dip"] == pytest.approx(60.0)
    assert df.loc["FaultA", "mw"] == pytest.approx(7.2)
    assert df.loc["FaultB", "mw"] == pytest.approx(6.5)
    assert df.loc["FaultA", "recur_int_median"] == pytest.approx(2000.0)
    assert df.loc["FaultA", "exceedance"] == pytest.approx(1 / 2000.0)
    assert df.loc["FaultB", "exceedance"] == pytest.approx(1 / 500.0)


def test_load_nhm_df_erf_name_suffix(tmp_path: Path) -> None:
    nhm_file = tmp_path / "faults.nhm"
    _write_nhm(nhm_file, [_make_fault(name="FaultA")])

    df = nhm.load_nhm_df(str(nhm_file), erf_name="NHM2010")

    assert list(df.index) == ["FaultA_NHM2010"]
    # The name column keeps the bare fault name
    assert df.loc["FaultA_NHM2010", "name"] == "FaultA"


def test_load_nhm_df_zero_recurrence_is_nan(tmp_path: Path) -> None:
    nhm_file = tmp_path / "faults.nhm"
    _write_nhm(
        nhm_file,
        [
            _make_fault(name="ZeroRecur", recur_int_median=0.0),
            _make_fault(name="Recur", recur_int_median=100.0),
        ],
    )

    df = nhm.load_nhm_df(str(nhm_file))

    # A zero recurrence interval is undefined, rather than an infinite rate
    assert np.isnan(df.loc["ZeroRecur", "recur_int_median"])
    assert np.isnan(df.loc["ZeroRecur", "exceedance"])
    assert df.loc["Recur", "exceedance"] == pytest.approx(0.01)

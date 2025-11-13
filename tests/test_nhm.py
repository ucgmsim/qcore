from io import StringIO

import numpy as np
import pytest

from qcore import nhm


def test_mag2mom_nm():
    # Test magnitude to moment conversion
    mw = 7.0
    result = nhm.mag2mom_nm(mw)

    # Expected: 10^(9.05 + 1.5 * 7.0) = 10^19.55
    expected = 10 ** (9.05 + 1.5 * 7.0)
    assert result == pytest.approx(expected)


def test_mag2mom_nm_small_magnitude():
    mw = 5.0
    result = nhm.mag2mom_nm(mw)
    expected = 10 ** (9.05 + 1.5 * 5.0)
    assert result == pytest.approx(expected)


def test_nhm_fault_creation():
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


def test_nhm_fault_write():
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


def test_nhm_fault_write_with_header():
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


def test_nhm_fault_sample_2012():
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


def test_nhm_fault_sample_2012_without_mw_perturbation():
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


def test_get_fault_header_points():
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

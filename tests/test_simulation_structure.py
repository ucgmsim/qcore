import os

from qcore import simulation_structure


def test_get_fault_from_realisation_simple():
    realisation = "AlpineF2K3T1_REL01"
    result = simulation_structure.get_fault_from_realisation(realisation)
    assert result == "AlpineF2K3T1"


def test_get_fault_from_realisation_with_path():
    realisation = "/path/to/simulation/SomeFault_REL05"
    result = simulation_structure.get_fault_from_realisation(realisation)
    assert result == "SomeFault"


def test_get_realisation_name():
    result = simulation_structure.get_realisation_name("MyFault", 3)
    assert result == "MyFault_REL03"


def test_get_realisation_name_single_digit():
    result = simulation_structure.get_realisation_name("TestFault", 1)
    assert result == "TestFault_REL01"


def test_get_realisation_name_double_digit():
    result = simulation_structure.get_realisation_name("TestFault", 15)
    assert result == "TestFault_REL15"


def test_get_srf_info_location():
    realisation = "AlpineF2_REL01"
    result = simulation_structure.get_srf_info_location(realisation)
    expected = os.path.join("AlpineF2", "Srf", "AlpineF2_REL01.info")
    assert result == expected


def test_get_srf_dir():
    cybershake_root = "/cybershake"
    realisation = "MyFault_REL02"
    result = simulation_structure.get_srf_dir(cybershake_root, realisation)
    expected = os.path.join("/cybershake", "Data", "Sources", "MyFault", "Srf")
    assert result == expected


def test_get_srf_location():
    realisation = "TestFault_REL03"
    result = simulation_structure.get_srf_location(realisation)
    expected = os.path.join("TestFault", "Srf", "TestFault_REL03.srf")
    assert result == expected


def test_get_srf_path():
    cybershake_root = "/cybershake"
    realisation = "AlpineF2_REL01"
    result = simulation_structure.get_srf_path(cybershake_root, realisation)
    expected = os.path.join(
        "/cybershake", "Data", "Sources", "AlpineF2", "Srf", "AlpineF2_REL01.srf"
    )
    assert result == expected


def test_get_fault_dir():
    cybershake_root = "/cybershake"
    fault_name = "MyFault"
    result = simulation_structure.get_fault_dir(cybershake_root, fault_name)
    expected = os.path.join("/cybershake", "Runs", "MyFault")
    assert result == expected


def test_get_sim_dir():
    cybershake_root = "/cybershake"
    realisation = "MyFault_REL01"
    result = simulation_structure.get_sim_dir(cybershake_root, realisation)
    expected = os.path.join("/cybershake", "Runs", "MyFault", "MyFault_REL01")
    assert result == expected


def test_get_im_calc_dir_no_realisation():
    sim_root = "/sim/root"
    result = simulation_structure.get_im_calc_dir(sim_root)
    assert result == "/sim/root/IM_calc"


def test_get_im_calc_dir_with_realisation():
    sim_root = "/sim/root"
    realisation = "MyFault_REL01"
    result = simulation_structure.get_im_calc_dir(sim_root, realisation)
    expected = os.path.join("/sim/root", "Runs", "MyFault", "MyFault_REL01", "IM_calc")
    assert result == expected


def test_get_IM_csv_from_root():  # noqa: N802
    cybershake_root = "/cybershake"
    realisation = "MyFault_REL01"
    result = simulation_structure.get_IM_csv_from_root(cybershake_root, realisation)
    expected = os.path.join(
        "/cybershake",
        "Runs",
        "MyFault",
        "MyFault_REL01",
        "IM_calc",
        "MyFault_REL01.csv",
    )
    assert result == expected


def test_get_fault_yaml_path_with_fault():
    sim_root = "/sim/root"
    fault_name = "MyFault"
    result = simulation_structure.get_fault_yaml_path(sim_root, fault_name)
    expected = os.path.join("/sim/root", "MyFault", "fault_params.yaml")
    assert result == expected


def test_get_fault_yaml_path_without_fault():
    sim_root = "/sim/root"
    result = simulation_structure.get_fault_yaml_path(sim_root)
    expected = os.path.join("/sim/root", "fault_params.yaml")
    assert result == expected

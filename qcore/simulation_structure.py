"""
Gives access to the folder structure of the cybershake directory
"""

import os


def get_fault_from_realisation(realisation: str) -> str:
    realisation = os.path.basename(realisation)  # if realisation is a fullpath
    return realisation.rsplit("_REL", 1)[0]


def get_realisation_name(fault_name: str, rel_no: int) -> str:
    return f"{fault_name}_REL{rel_no:0>2}"


def get_srf_info_location(realisation: str) -> str:
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".info")


def get_srf_dir(cybershake_root: str, realisation: str) -> str:
    return os.path.join(
        cybershake_root,
        "Data",
        "Sources",
        get_fault_from_realisation(realisation),
        "Srf",
    )


def get_srf_location(realisation: str) -> str:
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".srf")


def get_srf_path(cybershake_root: str, realisation: str) -> str:
    return os.path.join(
        cybershake_root, "Data", "Sources", get_srf_location(realisation)
    )


def get_fault_dir(cybershake_root: str, fault_name: str) -> str:
    return os.path.join(cybershake_root, "Runs", fault_name)


def get_sim_dir(cybershake_root: str, realisation: str) -> str:
    return os.path.join(
        get_fault_dir(cybershake_root, get_fault_from_realisation(realisation)),
        realisation,
    )


def get_im_calc_dir(sim_root: str, realisation: str | None = None) -> str:
    if realisation is None:
        return os.path.join(sim_root, "IM_calc")
    else:
        return get_im_calc_dir(get_sim_dir(sim_root, realisation))


def get_IM_csv_from_root(cybershake_root: str, realisation: str) -> str:  # noqa: N802
    return os.path.join(
        get_im_calc_dir(get_sim_dir(cybershake_root, realisation)),
        "{}.{}".format(realisation, "csv"),
    )


def get_fault_yaml_path(sim_root: str, fault_name: str | None = None) -> str:
    """
    Gets the fault_params.yaml for the specified simulation.
    Note: For the manual workflow set fault_name to None as the
    fault params are stored directly in the simulation directory.
    """
    return os.path.join(sim_root, fault_name or "", "fault_params.yaml")

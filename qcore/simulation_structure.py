"""
Gives access to the folder structure of the cybershake directory
"""

import os


def get_fault_from_realisation(realisation):
    realisation = os.path.basename(realisation)  # if realisation is a fullpath
    return realisation.rsplit("_REL", 1)[0]


def get_realisation_name(fault_name, rel_no):
    return f"{fault_name}_REL{rel_no:0>2}"


# SRF
def get_srf_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".srf")


def get_srf_info_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".info")


def get_srf_dir(cybershake_root, realisation):
    return os.path.join(
        cybershake_root,
        "Data",
        "Sources",
        get_fault_from_realisation(realisation),
        "Srf",
    )


def get_srf_path(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_srf_location(realisation)
    )


# Stoch
def get_stoch_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Stoch", realisation + ".stoch")


def get_stoch_path(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_stoch_location(realisation)
    )


def get_runs_dir(cybershake_root):
    """Gets the path to the Runs directory of a cybershake run"""
    return os.path.join(cybershake_root, "Runs")


def get_fault_dir(cybershake_root, fault_name):
    return os.path.join(get_runs_dir(cybershake_root), fault_name)


def get_sim_dir(cybershake_root, realisation):
    return os.path.join(
        get_fault_dir(cybershake_root, get_fault_from_realisation(realisation)),
        realisation,
    )


def get_im_calc_dir(sim_root, realisation=None):
    if realisation is None:
        return os.path.join(sim_root, "IM_calc")
    else:
        return get_im_calc_dir(get_sim_dir(sim_root, realisation))


def get_IM_csv_from_root(cybershake_root, realisation):
    return os.path.join(
        get_im_calc_dir(get_sim_dir(cybershake_root, realisation)),
        "{}.{}".format(realisation, "csv"),
    )


def get_fault_yaml_path(sim_root, fault_name=None):
    """
    Gets the fault_params.yaml for the specified simulation.
    Note: For the manual workflow set fault_name to None as the
    fault params are stored directly in the simulation directory.
    """
    fault_name = "" if fault_name is None else fault_name
    return os.path.join(sim_root, fault_name, "fault_params.yaml")

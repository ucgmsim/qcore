"""
Gives access to the folder structure of the cybershake directory
"""
import os

import qcore.constants as const

def get_fault_from_realisation(realisation):
    return realisation.split("_")[0]


def get_realisation_name(fault_name, rel_no):
    return "{}_REL{:0>2}".format(fault_name, rel_no)


# VM
def get_fault_VM_dir(cybershake_root, realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(get_VM_dir(cybershake_root), fault)


def get_VM_dir(cybershake_root):
    return os.path.join(cybershake_root, "Data", "VMs")


def get_realisation_VM_dir(cybershake_root, realisation):
    return os.path.join(get_fault_VM_dir(cybershake_root, realisation), realisation)


# SRF
def get_srf_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".srf")


def get_srf_info_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".info")


def get_srf_dir(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_fault_from_realisation(realisation), "Srf"
    )


def get_srf_path(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_srf_location(realisation)
    )


# Source_params
def get_sources_dir(cybershake_root):
    """Gets the cybershake sources directory"""
    return os.path.join(cybershake_root, "Data", "Sources")


def get_source_params_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Sim_params", realisation + ".yaml")


def get_source_params_dir(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_fault_from_realisation(realisation), "Sim_params"
    )


def get_source_params_path(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_source_params_location(realisation)
    )


# Stoch
def get_stoch_location(realisation):
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Stoch", realisation + ".stoch")


def get_stoch_dir(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_fault_from_realisation(realisation), "Stoch"
    )


def get_stoch_path(cybershake_root, realisation):
    return os.path.join(
        cybershake_root, "Data", "Sources", get_stoch_location(realisation)
    )


# Runs
def get_runs_dir(cybershake_root):
    """Gets the path to the Runs directory of a cybershake run"""
    return os.path.join(cybershake_root, "Runs")


# Cybershake
def get_cybershake_list(cybershake_root):
    """Gets the cybershake list, specifying the faults and number of realisation"""
    return os.path.join(cybershake_root, "list.txt")


def get_mgmt_db(cybershake_root):
    """Get the mgmt_db file"""
    return os.path.join(cybershake_root, const.SLURM_MGMT_DB_NAME)


def get_mgmt_db_queue(cybershake_root):
    return os.path.join(cybershake_root, "mgmt_db_queue")


def get_fault_dir(cybershake_root, fault_name):
    return os.path.join(get_runs_dir(cybershake_root), fault_name)


def get_sim_dir(cybershake_root, realisation):
    return os.path.join(
        get_fault_dir(cybershake_root, get_fault_from_realisation(realisation)),
        realisation,
    )


# LF
def get_lf_dir(sim_root):
    return os.path.join(sim_root, "LF")


def get_lf_outbin_dir(sim_root):
    return os.path.join(get_lf_dir(sim_root), "OutBin")

def get_lf_restart_dir(sim_root):
    return os.path.join(get_lf_dir(sim_root), "Restart")

# BB
def get_bb_dir(sim_root):
    return os.path.join(sim_root, "BB")


def get_bb_acc_dir(sim_root):
    return os.path.join(get_bb_dir(sim_root), "Acc")


def get_bb_bin_path(sim_root):
    return os.path.join(get_bb_acc_dir(sim_root), "BB.bin")


# HF
def get_hf_dir(sim_root):
    return os.path.join(sim_root, "HF")


def get_hf_acc_dir(sim_root):
    return os.path.join(get_hf_dir(sim_root), "Acc")


def get_hf_bin_path(sim_root):
    return os.path.join(get_hf_acc_dir(sim_root), "HF.bin")


# IM_calc
def get_im_calc_dir(sim_root):
    return os.path.join(sim_root, "IM_calc")


def get_IM_csv(sim_root):
    return os.path.join(
        get_im_calc_dir(sim_root),
        "{}.{}".format(os.path.basename(sim_root).split(".")[0], "csv"),
    )


def get_IM_info(sim_root):
    return os.path.join(
        get_im_calc_dir(sim_root),
        "{}{}".format(
            os.path.basename(sim_root).split(".")[0], const.IM_SIM_CALC_INFO_SUFFIX
        ),
    )


# yaml
def get_fault_yaml_path(sim_root, fault_name=None):
    """
    Gets the fault_params.yaml for the specified simulation. 
    Note: For the manual workflow set fault_name to None as the 
    fault params are stored directly in the simulation directory.
    """
    fault_name = "" if fault_name is None else fault_name
    return os.path.join(sim_root, fault_name, "fault_params.yaml")


def get_root_yaml_path(sim_root):
    """
    Gets the root_params.yaml for the specified simulation.
    """
    return os.path.join(sim_root, "root_params.yaml")


# verification
def get_verification_dir(sim_root):
    """
    Gets the folder for data used for verification etc.
    """
    return os.path.join(sim_root, "Verification")


def get_sources_plot_dir(cybershake_root, realisation):
    """
    Gets the folder for storing plots that can be generated 
    before installing a cybershake. eg. srf square & map plots.
    """
    return os.path.join(
        cybershake_root, "Data", "Sources", get_fault_from_realisation(realisation), "Verification"
    )


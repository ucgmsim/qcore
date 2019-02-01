"""
Gives access to the folder structure of the cybershake directory
"""

import os


def __get_fault_from_realisation(realisation):
    # Remove the part after the last underscore
    # If the station name contains underscores this will preserve the station name
    return '_'.join(realisation.split('_')[:-1])


def get_realisation_name(fault_name, rel_no):
    return "{}_REL{:0>2}".format(fault_name, rel_no)


# VM
def get_VM_dir(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'VMs', realisation)


# SRF
def get_srf_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Srf', realisation + '.srf')


def get_srf_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_srf_location(realisation))


# Source_params
def get_source_params_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Sim_params', realisation + '.yaml')


def get_source_params_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_source_params_location(realisation))


# Stoch
def get_stoch_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Stoch', realisation + '.stoch')


def get_stoch_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_stoch_location(realisation))


# LF
def get_lf_dir(sim_root):
    return os.path.join(sim_root, 'LF')


def get_lf_outbin_dir(sim_root):
    return os.path.join(get_lf_dir(sim_root), 'OutBin')


# BB
def get_bb_dir(sim_root):
    return os.path.join(sim_root, 'BB')


def get_bb_acc_dir(sim_root):
    return os.path.join(get_bb_dir(sim_root), 'Acc')


def get_bb_bin_path(sim_root):
    return os.path.join(get_bb_acc_dir(sim_root), 'BB.bin')


# HF
def get_hf_dir(sim_root):
    return os.path.join(sim_root, 'HF')


def get_hf_acc_dir(sim_root):
    return os.path.join(get_hf_dir(sim_root), 'ACC')

def get_hf_bin_path(sim_root):
    return os.path.join(get_hf_acc_dir(sim_root), 'HF.bin')


"""
Gives access to the folder structure of the cybershake directory
"""

import os


def __get_fault_from_realisation(realisation):
    return realisation.split('_')[0]


def get_realisation_name(fault_name, rel_no):
    return "{}_REL{:0>2}".format(fault_name, rel_no)


def get_srf_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Srf', realisation + '.srf')


def get_stoch_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Stoch', realisation + '.stoch')


def get_source_params_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Sim_params', realisation + '.yaml')


def get_srf_path(cybershake_root, realisation):
    return os.path.join(get_sources_dir(cybershake_root), get_srf_location(realisation))

def get_sources_dir(cybershake_root):
    """Gets the cybershake sources directory"""
    return os.path.join(cybershake_root, 'Data', 'Sources')

def get_stoch_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_stoch_location(realisation))

def get_source_params_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_source_params_location(realisation))

def get_VMs_dir(cybershake_root):
    """Gets the cybershake VMs directory"""
    return os.path.join(cybershake_root, "Data","VMs")

def get_runs_dir(cybershake_root):
    """Gets the path to the Runs directory of a cybershake run"""
    return os.path.join(cybershake_root, "Runs")

def get_cybershake_config(cybershake_root):
    """Gets the path to the cybershake config json file"""
    return os.path.join(cybershake_root, "cybershake_config.json")

def get_cybershake_list(cybershake_root):
    """Gets the cybershake list, specifying the faults and number of realisation"""
    return os.path.join(cybershake_root, "list.txt")
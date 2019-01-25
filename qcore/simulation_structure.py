"""
Gives access to the folder structure of the cybershake directory
"""

import os


def __get_fault_from_realisation(realisation):
    return realisation.split('_')[0]


def get_realisation_name(fault_name, rel_no):
    return "{}_REL{:0>2}".format(fault_name, rel_no)


def get_srf_path(cybershake_root, realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(cybershake_root, 'Data', 'Sources', fault, 'Srf', realisation + '.srf')


def get_stoch_path(cybershake_root, realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(cybershake_root, 'Data', 'Sources', fault, 'Stoch', realisation + '.stoch')


def get_source_params_path(cybershake_root, realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(cybershake_root, 'Data', 'Sources', fault, 'Sim_params', realisation + '.yaml')

"""
A class to retrieve versioned files.
Defaults are available from the config files
"""

from qcore.config import qconfig
from os import path

default_bin_loc = qconfig['tools_dir']

HF_BINMOD_NAME_STR = "hb_high_binmod_v{}"
HF_NP2MM_NAME_STR = "hb_high_np2mm+_v{}"
LF_EMOD3D_NAME_STR = "emod3d-mpi_v{}"
GENSLIP_NAME_STR = "genslip_v{}"


def get_hf_binmod(version, bin_loc=default_bin_loc):
    return path.join(bin_loc, HF_BINMOD_NAME_STR.format(version))


def get_hf_np2mm(version, bin_loc=default_bin_loc):
    return path.join(bin_loc, HF_NP2MM_NAME_STR.format(version))


def get_lf_bin(version, bin_loc=default_bin_loc):
    return path.join(bin_loc, LF_EMOD3D_NAME_STR.format(version))


def get_genslip_bin(version, bin_loc=default_bin_loc):
    return path.join(bin_loc,  GENSLIP_NAME_STR.format(version))


def get_unversioned_bin(bin_name, bin_loc=default_bin_loc):
    return path.join(bin_loc, bin_name)

